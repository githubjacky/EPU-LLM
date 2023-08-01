from typing import List, Dict
from pathlib import Path

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from itertools import chain
from tqdm import trange
from sklearn.metrics import classification_report

import os
import json


class ClassificationResult(BaseModel):
    pred: int = Field(
        description="If the given article is to be included, return 0; if not, return 1.")
    reason: str = Field(
        description="reason for why or why not it should be included for constructing EPU index")


class Prompt():
    example_prompt_template = """Quesion: {question}

    {article}

    Response: {response}
    """

    # suffix of the example prompt
    suffix = """Quesion: {question}

    {article}

    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=ClassificationResult)

    def __init__(self,
                 country: str,
                 system_message: str,
                 question: str,
                 examples: List[Dict[str, str]],
                 ):

        example_prompt = PromptTemplate(
            input_variables=['article', 'response'],
            partial_variables={'question': question},
            template=self.example_prompt_template
        )

        fewshot_prompt = FewShotPromptTemplate(
            input_variables=['article'],
            examples=examples,
            example_prompt=example_prompt,
            suffix=self.suffix,
            partial_variables={
                'question': question,
                'format_instructions': self.parser.get_format_instructions()
            }
        )

        messages = [
            SystemMessagePromptTemplate.from_template(
                system_message).format(country=country),
            HumanMessagePromptTemplate(prompt=fewshot_prompt)
        ]

        self.chat_message = ChatPromptTemplate(
            input_variables=['article'],
            output_parser=self.parser,
            messages=messages
        )


class EPUClassifier:
    def __init__(self, prompt: Prompt, model: str, temperature: float, batch_size: int) -> None:
        self.batch_size = batch_size
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=temperature
        )
        self.chat = LLMChain(llm=llm, prompt=prompt.chat_message)

    def preprocess(self, source: str | List[Dict]) -> None:
        if isinstance(source, str):
            self.data = json.loads(Path(source).read_text())
        else:
            self.data = source

    def predict_instance(self, article: str) -> ClassificationResult:
        return self.chat.predict_and_parse(article=article)

    def predict_batch(self, batch_articles: List[str]) -> List[ClassificationResult]:
        input_list = [
            {'article': i}
            for i in batch_articles
        ]
        return self.chat.apply_and_parse(input_list)

    def predict(self) -> None:
        n = len(self.data)
        predictions = []

        if n >= self.batch_size:
            for idx in trange(0, n, self.batch_size):
                articles = [
                    i['article']
                    for i in self.data[idx:min(idx+self.batch_size, n)]
                ]
                predictions.append(self.predict_batch(articles))

            predictions = chain.from_iterable(predictions)
        else:
            for i in trange(n):
                predictions.append(
                    self.predict_instance(self.data[i]['article']))

        preds = [i.pred for i in predictions]
        labels = [i['label'] for i in self.data]
        print(classification_report(labels, preds, zero_division=1.))

        self.predictions = predictions

    def output(self, path: str) -> None:
        res = [json.loads(i.json()) for i in self.predictions]
        with open(path, 'w') as f:
            json.dump(res, f)


class Param(BaseModel):
    country = "Taiwan"

    system_message_template = '''\
    As an economist working on constructing {country}'s Economic Policy Uncertainty Index (EPU index), your task is to determine wheter the article should be included when measuring Economic Policy Uncertainty in {country}. 
    Articles related to {country}'s economic environment and introducing the policy uncertainty in {country} should be considered useful for coustructing {country}'s EPU index.
    Let me introduce you two criterion for determining wheter the given article should be included.
    
    Criterion1: 
    determine whether the main idea of the article is introducing news that will impact {country}'s economic environment directly.
    If it is, it should be included.
    While both historical accounts and abstract subjective inferences should not be included

    Criterion2:
    assess whether the events mentioned in the article actually occur within {country}. 
    If the events in article occur within {country}, it should be included. Others should not.
    
    Your response should maintain a consistent style and consist of two parts of json object. 
    The first part should indicate whether the given article is should be included. 
    Return 0 if it should be included while 1 for not. 
    The second part should contains your reason for such classification.
    '''

    # quesion + article = query(langchain.HumanMessage)
    question = "Should the following article be excluded when constructing EPU index?"
    # purpose of {{}} is to prevent error from formating the few shot example prompt
    examples = [
        {
            'article': "中國商業氣氛降至低點，習近平主導的中國市場「不再需要外國人了」",
            'response': "{{'pred': 1, 'reason': 'Althoug the event might relate to Taiwna, it does not occur within Taiwan.'}}"
        },
        {
            'article': "美國聯準會與台灣經濟都有一個「6月難題」",
            'response': "{{'pred': 0, 'reason': 'It describes the policy uncertainty of Taiwan.'}}"
        }
    ]
    model = "gpt-3.5-turbo"
    temperature = 0.
    batch_size = 128
    data_path = "./data/EPU_Noise_Test.json"
    output_path = "./data/pred_Test.json"
