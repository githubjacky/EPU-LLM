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
