from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.loading import load_prompt
from loguru import logger
import logging
import orjson
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import time


from utils import read_jsonl, format_handler, refresh_handler


class ClassificationResult(BaseModel):
    pred: int = Field(
        description=" ".join((
            "If the news should be excluded, return 1.",
            "If the news should not be excluded, return 0.",

        ))
    )


class ClassificationResultWithReason(BaseModel):
    pred: int = Field(
        description=" ".join((
            "If the news should be excluded, return 1.",
            "If the news should not be excluded, return 0.",
        ))
    )
    reason: str = Field(
        description=" ".join((
            "Reason for why or why not it should be excluded",
            "for constructing EPU index.",
            "Use no more thant 30 words.",
        ))
    )


class Prompt:
    parser = PydanticOutputParser(pydantic_object=ClassificationResult)
    output_instructions = parser.get_format_instructions()

    parser_with_reason = PydanticOutputParser(
        pydantic_object=ClassificationResultWithReason
    )

    output_instructions_with_reason = parser_with_reason.get_format_instructions()

    question = "Should the news be excluded?"

    def __init__(self,
                 country: str = "Taiwan",
                 system_message_template_path: str = "prompt_template/system.json",
                 human_message_template_path: str = "prompt_template/human.json",
                ) -> None:

        system_prompt_template = load_prompt(system_message_template_path)
        human_prompt_template = load_prompt(human_message_template_path)

        self.country = country
        self.system_message = system_prompt_template.template
        self.human_message = human_prompt_template.template

    @property
    def zero_shot(self) -> ChatPromptTemplate:
        return (
            ChatPromptTemplate
            .from_messages([
                ("system", self.system_message),
                MessagesPlaceholder(variable_name = "chat_history"),
                ("human", self.human_message)
            ])
            .partial(
                country = self.country,
                output_instructions = self.output_instructions,
            )
        )

    @property
    def zero_shot_with_reason(self) -> ChatPromptTemplate:
        return (
            ChatPromptTemplate
            .from_messages([
                ("system", self.system_message),
                MessagesPlaceholder(variable_name = "chat_history"),
                ("human", self.human_message)]
            )
            .partial(
                country = self.country,
                output_instructions = self.output_instructions_with_reason,
            )
        )


    def reasoning_instance(self, chain: LLMChain, label: int, warning: str, i: int):
        chain.memory = ConversationBufferMemory(
            memory_key = "chat_history",
            return_messages = True
        )

        instruction = f"In this scenario, the label is known. {warning}"
        res = format_handler(chain, i, instruction)
        j = 0

        while res.get('pred') != label:
            time.sleep(2)

            if res.get('pred') == -1 or j == 5:
                logger.info(f"refresh the memory for the {i}th sample, inconsistent for: {j} times")
                chain, res = refresh_handler(chain, instruction, i)
                j = 1
            else:
                logger.info(f"incosistent reasoning for the {i}th sample, pred: {res['pred']}; label: {label}, re-generate")
                j += 1
                res = format_handler(chain, i, f'Incorrect classification. {warning}')

        return res


    def reasoning(self, n: int, example_path: Path, output_path: Path) -> Tuple[List[str], List[str]]:

        log_file = Path(f"log/reasoning_{n}_{str(example_path).split('/')[-1]}.log")
        if log_file.exists: log_file.unlink(missing_ok = True)
        logger.remove()
        logger.add(log_file, level = "INFO")
        logging.getLogger("openai").setLevel(logging.WARNING)

        example_list = read_jsonl(example_path, n)
        news_set = [i.get("news") for i in example_list]
        labels = [i.get("label") for i in example_list]

        llm = ChatOpenAI(
            model = "gpt-3.5-turbo",
            temperature = 0.6,
            request_timeout = 120
        )
        chat_prompt_template = (
            ChatPromptTemplate
            .from_messages([
                ("system", self.system_message),
                MessagesPlaceholder(variable_name = "chat_history"),
                ("human", self.human_message),
            ])
            .partial(
                country = self.country,
                output_instructions = self.output_instructions_with_reason
            )
        )

        examples = []
        warning = {
            1: 'This news should be excluded when constructing EPU index, and thus the "pred" key should be 1. Organize the information and generate reronable interpretations',
            0: 'This news should not be excluded when constructing EPU index, and thus the "pred" key should be 0. Organize the information and generate reronable interpretations',
        }
        with output_path.open("wb") as f:
            with get_openai_callback() as cb:
                for i, news in enumerate(tqdm(news_set)):
                    chain = LLMChain(
                        llm = llm,
                        prompt = chat_prompt_template.partial(news = news)
                    )
                    label = labels[i]
                    res = self.reasoning_instance(
                        chain,
                        label,
                        warning.get(label),
                        i
                    )

                    examples.append(orjson.dumps(res))
                    f.write(orjson.dumps(res, option = orjson.OPT_APPEND_NEWLINE))
                    time.sleep(1)


        logger.info(f"the cost for fine tuning: {cb.total_cost}")

        return news_set, examples

    def __create_fewshot_prompt(self,
                                article_example: List[str],
                                response_example: List[str],
                                output_instructions: str,
                               ) -> ChatPromptTemplate:
        few_shot_example = [
            {
                "news": article_example[i],
                "response": response_example[i],
                "correct_instructions": self.question,
                "output_instructions": "",
            }
            for i in range(len(article_example))
        ]

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", self.human_message),
            ("ai", "{response}"),
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt = example_prompt,
            examples = few_shot_example,
        )

        return (
            ChatPromptTemplate
            .from_messages([
                ("system", self.system_message),
                few_shot_prompt,
                MessagesPlaceholder(variable_name = "chat_history"),
                ("human", self.human_message),
            ])
            .partial(
                country = self.country,
                output_instructions = output_instructions,
            )
        )

    def few_shot(self, n: int, example_path: str) -> ChatPromptTemplate:
        self.num = n
        article_example = []
        response_example = []
        for i in Path(example_path).read_text().split("\n")[:n]:
            if i != "":
                item = orjson.loads(i)
                article_example.append(item["news"])
                response_example.append(f'{{"pred": {item["label"]}}}')

        return self.__create_fewshot_prompt(
            article_example,
            response_example,
            self.output_instructions
        )

    def few_shot_with_reason(self,
                             n: int,
                             example_path: str = "data/raw/fewshot_news/normal.jsonl",
                             output_dir: str = "data/processed/fewshot_reasons/normal"
                             ) -> ChatPromptTemplate:
        self.num = n
        _example_path = Path(example_path)
        _output_dir = Path(output_dir)
        output_path = _output_dir / f"{n}.jsonl"

        if not output_path.exists():
            logger.warning("reasoning example doesn't exist, start reasoning")
            article_example, response_example = self.reasoning(
                n, _example_path, output_path
            )
        else:
            article_example = [
                orjson.loads(i)["news"]
                for i in read_jsonl(_example_path, n, return_str = True)
            ]
            response_example = read_jsonl(output_path)

        return self.__create_fewshot_prompt(
            article_example, response_example, self.output_instructions_with_reason
        )
