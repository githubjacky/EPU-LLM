from typing import List, Dict, Optional
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from tqdm import trange
from sklearn.metrics import classification_report

import os
import json


class ClassificationResult(BaseModel):
    pred: int = Field(description="prediction of whether the article you be excluded")
    reason: str = Field(description="reason for such classification")


class Prompt():
    example_prompt_template = """Quesion: {question}

    {article}

    Answer: {classify}
    """

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

        self.system_message = SystemMessagePromptTemplate.from_template(system_message).format(country = country)

        example_prompt = PromptTemplate(
            input_variables = ['article', 'classify'],
            partial_variables = {'question': question},
            template = self.example_prompt_template
        )

        self.fewshot_prompt = FewShotPromptTemplate(
            examples = examples,
            example_prompt = example_prompt,
            prefix = "Below are some examples.",
            suffix = self.suffix,
            input_variables = ['article'],
            partial_variables = {'question': question, 'format_instructions': self.parser.get_format_instructions()}
        )



class EPUClassifier:
    def __init__(self, prompt: Prompt, openai_api_key:Optional[str], model: str, temperature: int) -> None:
        self.prompt = prompt
        self.chat = ChatOpenAI(
            openai_api_key = openai_api_key, model = model, temperature = temperature
        )


    def preprocess(self, path: str) -> List[Dict]:
        return json.loads( Path(path).read_text() )

    def predict_instance(self, obs: Dict) -> ClassificationResult:
        message  = [
            self.prompt.system_message,
            HumanMessage(content = self.prompt.fewshot_prompt.format(article = obs['article']))
        ]
        response = self.chat(message)
        
        return self.prompt.parser.parse(response.content)


    def predict(self, path: str) -> None:
        data = self.preprocess(path)
        predictions = []

        for i in trange(len(data)):
            predictions.append(self.predict_instance(data[i]))

        labels = [i['label'] for i in data]
        preds = [i.pred for i in predictions]
        print( classification_report(labels, preds, zero_division = 1.) )

        self.predictions = predictions


    def output(self, path: str) -> None:
        res = []
        for i in range(len(self.predictions)):
            res.append({
                'pred': self.predictions[i].pred,
                'response': self.predictions[i].reason
            })

        with open(path, 'w') as f:
            json.dump(res, f)



class Param(BaseModel):
    country = "Taiwan"

    system_message_template = '''\
    As an economist working on constructing {country}'s Economic Policy Uncertainty Index (EPU index), your task is to identify articles that contain EPU keywords but should not be included when measuring Economic Policy Uncertainty. 
    In other words, you need to identify the noise that should be excluded when constructing {country}'s EPU index. 
    There are two aspects you need to consider. 
    First, you need to determine whether the main idea of the article is introducing news that will impact {country}'s economic environment directly.
    Both historical accounts and abstract subjective inferences should be considered as the noise. 
    Second, you should assess whether the events mentioned in the article actually occur within {country}. 
    If the events do not occur within {country}, they should be excluded. 
    Your response should maintain a consistent style and consist of two parts separated by a semicolon. 
    The first part should indicate whether the given article is considered to be noise and thus should be excluded, with a value of 1 for yes and 0 for no. 
    The second part should provide your reason for such classification.
    '''

    question = "Should the following article be excluded when constructing EPU index?" # quesion + article = query(langchain.HumanMessage)
    examples = [
        {
            'article': "半工半讀掙錢幫家裡，減輕媽媽經濟負擔",
            'classify': "1; Yes, it should be excluded. Although the article contains keywords, it has nothing to do with the Taiwan's economic environment."
        },
        {
            'article': "中國商業氣氛降至低點，習近平主導的中國市場「不再需要外國人了」",
            'classify': "1; Yes, it should be excluded, as it does not mention any economic policy uncertainty events in Taiwan."

        },
        {
            'article': "美國聯準會與台灣經濟都有一個「6月難題」",
            'classify': "0; NO, it shouldn't be excluded. It introduces the policy uncertainty of Taiwan."
        }
    ]
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model = "gpt-3.5-turbo"
    temperature = 0
    data_path = "./data/EPU_Noise_Examples.json"
    output_path = "./data/pred_Examples.json"

