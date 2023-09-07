from pathlib import Path
from typing import Optional

import orjson
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import FewShotChatMessagePromptTemplate, PromptTemplate
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
        example_num: int = 3,
        example_path: Optional[str] = "data/raw/reason_example.jsonl",
        reason_example_dir: Optional[str] = "data/processed",
    ) -> None:
        system_prompt_template = load_prompt(system_message_template_path)
        human_prompt_template = load_prompt(human_message_template_path)

        self.system_message = system_prompt_template.format(country=country)
        self.human_message = human_prompt_template.template

        self.num = example_num

        self.example_path = example_path
        self.reason_example_path = (
            Path(reason_example_dir) / f"reason_example_{example_num}.jsonl"
        )

    @property
    def zero_shot(self):
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_message), ("human", self.human_message)]
        ).partial(correct_instructions="", output_instructions=self.output_instructions)

        return chat_prompt_template

    @property
    def zero_shot_with_reason(self):
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_message), ("human", self.human_message)]
        ).partial(
            correct_instructions="",
            output_instructions=self.output_instructions_with_reason,
        )

        return chat_prompt_template

    def reasoning(self):
        example_list = [
            orjson.loads(i)
            for i in Path(self.example_path).read_text().split("\n")[: self.num]
            if i != ""
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

        with open(self.reason_example_path, "wb") as f:
            for i in examples:
                f.write(orjson.dumps(i, option=orjson.OPT_APPEND_NEWLINE))

        return news_set, examples

    @property
    def few_shot(self):
        example_prompt_template = """Quesion: Should the following article be excluded when constructing EPU index?

        {news}

        Response: {response}
        """

        article_example = []
        response_example = []
        for i in Path(self.example_path).read_text().split("\n")[: self.num]:
            if i != "":
                item = orjson.loads(i)
                article_example.append(item["news"])
                response_example.append(f'{{"pred": {item["label"]}}}')

        few_shot_example = [
            {
                "news": article_example[i],
                "response": response_example[i],
                "correct_instructions": "",
                "output_instructions": "",
            }
            for i in range(len(article_example))
        ]
        example_prompt = PromptTemplate.from_template(example_prompt_template)

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

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                few_shot_prompt,
                ("human", self.human_message),
            ]
        ).partial(
            correct_instructions="",
            output_instructions=self.output_instructions_with_reason,
        )

        return final_prompt

    @property
    def few_shot_with_reason(self):
        example_prompt_template = """Quesion: Should the following article be excluded when constructing EPU index?

        {news}

        Response: {response}
        """

        if not Path(self.reason_example_path).exists():
            logger.warning("reasoning example doesn't exist, start reasoning")
            article_example, response_example = self.reasoning()
        else:
            article_example = [
                orjson.loads(i)["news"]
                for i in Path(self.example_path).read_text().split("\n")[: self.num]
                if i != ""
            ]

            response_example = [
                i
                for i in Path(self.reason_example_path).read_text().split("\n")
                if i != ""
            ]

        few_shot_example = [
            {
                "news": article_example[i],
                "response": response_example[i],
                "correct_instructions": "",
                "output_instructions": "",
            }
            for i in range(len(article_example))
        ]
        example_prompt = PromptTemplate.from_template(example_prompt_template)

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

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                few_shot_prompt,
                ("human", self.human_message),
            ]
        ).partial(
            correct_instructions="",
            output_instructions=self.output_instructions_with_reason,
        )

        return final_prompt
