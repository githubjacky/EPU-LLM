from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from pydantic import BaseModel, Field


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

    system_message = '''\
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

    question = "Should the following article be excluded when constructing EPU index?"

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

    parser = PydanticOutputParser(pydantic_object=ClassificationResult)

    def __init__(self, country: str) -> None:

        example_prompt = PromptTemplate(
            input_variables=['article', 'response'],
            partial_variables={'question': self.question},
            template=self.example_prompt_template
        )

        fewshot_prompt = FewShotPromptTemplate(
            input_variables=['article'],
            examples=self.examples,
            example_prompt=example_prompt,
            suffix=self.suffix,
            partial_variables={
                'question': self.question,
                'format_instructions': self.parser.get_format_instructions()
            }
        )

        messages = [
            SystemMessagePromptTemplate.from_template(
                self.system_message
            ).format(country=country),
            HumanMessagePromptTemplate(prompt=fewshot_prompt)
        ]

        self.chat_message = ChatPromptTemplate(
            input_variables=['article'],
            output_parser=self.parser,
            messages=messages
        )
