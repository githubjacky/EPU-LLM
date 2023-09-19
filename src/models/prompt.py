from pathlib import Path
from typing import List
import orjson
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.loading import load_prompt
from loguru import logger
from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    pred: int = Field(
        description=" ".join(
            (
                "If the news should be excluded, return 1.",
                "If the news should not be excluded, return 0.",
            )
        )
    )


class ClassificationResultWithReason(BaseModel):
    pred: int = Field(
        description=" ".join(
            (
                "If the news should be excluded, return 1.",
                "If the news should not be excluded, return 0.",
            )
        )
    )
    reason: str = Field(
        description=" ".join(
            (
                "Reason for why or why not it should be excluded for constructing EPU index.",
                "Use no more thant 30 words.",
            )
        )
    )


class Prompt:
    parser = PydanticOutputParser(pydantic_object=ClassificationResult)
    output_instructions = parser.get_format_instructions()

    parser_with_reason = PydanticOutputParser(
        pydantic_object=ClassificationResultWithReason
    )

    output_instructions_with_reason = parser_with_reason.get_format_instructions()

    def __init__(
        self,
        country: str = "Taiwan",
        system_message_template_path: str = "prompt_template/system.json",
        human_message_template_path: str = "prompt_template/human.json",
    ) -> None:
        system_prompt_template = load_prompt(system_message_template_path)
        human_prompt_template = load_prompt(human_message_template_path)

        self.system_message = system_prompt_template.format(country=country)
        self.human_message = human_prompt_template.template

    @property
    def zero_shot(self):
        return ChatPromptTemplate.from_messages(
            [("system", self.system_message), ("human", self.human_message)]
        ).partial(correct_instructions="", output_instructions=self.output_instructions)

    @property
    def zero_shot_with_reason(self):
        return ChatPromptTemplate.from_messages(
            [("system", self.system_message), ("human", self.human_message)]
        ).partial(
            correct_instructions="",
            output_instructions=self.output_instructions_with_reason,
        )

    def reasoning(self, n: int, example_path: Path, output_path: Path):
        example_list = [
            orjson.loads(i) for i in example_path.read_text().split("\n")[:n] if i != ""
        ]
        news_set = [i["news"] for i in example_list]
        labels = [i["label"] for i in example_list]

        load_dotenv("env/.env")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", self.human_message),
            ]
        ).partial(output_instructions=self.output_instructions_with_reason)

        examples = []
        for i, news in enumerate(news_set):
            chain = LLMChain(
                prompt=chat_prompt_template.partial(news=news),
                llm=llm,
                memory=memory,
            )
            label = labels[i]

            instructions = (
                "This news should be excluded. Organize the information and re-classify again."
                if label == 1
                else "This news should not be excluded. Organize the information and re-classify again."
            )

            res = orjson.loads(chain.run(correct_instructions=""))
            while res["pred"] != label:
                logger.warning("incosistent reasoning, re-generate")
                res = orjson.loads(chain.run(correct_instructions=instructions))

            examples.append(
                {
                    "pred": label,
                    "reason": res["reason"],
                }
            )

        with output_path.open("wb") as f:
            for i in examples:
                f.write(orjson.dumps(i, option=orjson.OPT_APPEND_NEWLINE))

        return news_set, examples

    def __create_fewshot_prompt(
        self,
        article_example: List[str],
        response_example: List[str],
        output_instructions: str,
    ):
        few_shot_example = [
            {
                "news": article_example[i],
                "response": response_example[i],
                "correct_instructions": "",
                "output_instructions": "",
            }
            for i in range(len(article_example))
        ]

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", self.human_message),
                ("ai", "{response}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=few_shot_example,
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                few_shot_prompt,
                ("human", self.human_message),
            ]
        ).partial(
            correct_instructions="",
            output_instructions=output_instructions,
        )

    def few_shot(self, n: int, example_path: str):
        self.num = n
        article_example = []
        response_example = []
        for i in Path(example_path).read_text().split("\n")[:n]:
            if i != "":
                item = orjson.loads(i)
                article_example.append(item["news"])
                response_example.append(f'{{"pred": {item["label"]}}}')

        return self.__create_fewshot_prompt(
            article_example, response_example, self.output_instructions
        )

    def few_shot_with_reason(self, n: int, example_path: str, output_dir: str):
        self.num = n
        _example_path = Path(example_path)
        _output_dir = Path(output_dir)
        output_path = _output_dir / f"reason_example_{n}.jsonl"

        if not output_path.exists():
            logger.warning("reasoning example doesn't exist, start reasoning")
            article_example, response_example = self.reasoning(
                n, _example_path, output_path
            )
        else:
            article_example = [
                orjson.loads(i)["news"]
                for i in _example_path.read_text().split("\n")[:n]
                if i != ""
            ]

            response_example = [
                i for i in output_path.read_text().split("\n") if i != ""
            ]

        return self.__create_fewshot_prompt(
            article_example, response_example, self.output_instructions_with_reason
        )
