import itertools
from typing import List, Dict

from sklearn.metrics import classification_report
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

from typing import List, Dict
from pathlib import Path
from tqdm import trange
from langchain.chat_models import ChatOpenAI
import json



class Prompt:
    def __init__(self, country, instructions) -> None:
        system_messag = f'''
        As an economist working on constructing {country}'s Economic Policy Uncertainty 
        Index (EPU index), your task is to identify articles that contain EPU keywords 
        but should not be included when measuring Economic Policy Uncertainty. In other 
        words, you need to identify the noise that should be excluded when constructing 
        {country}'s EPU index. There are two aspects you need to consider. First, you 
        need to determine whether the main idea of the article is introducing news that 
        will impact {country}'s economic environment directly rather than providing 
        historical accounts or subjective inference. Second, you should assess whether 
        the events mentioned in the article actually occur within {country}. 
        If the events do not occur within {country}, they should be excluded. Your 
        response should maintain a consistent style and consist of two parts separated 
        by a semicolon. The first part should indicate whether the given article is 
        considered to be noise and thus should be excluded, with a value of 1 for yes 
        and 0 for no. The second part should provide your reason for such classification.
        '''

        question = f"Should the following article be excluded when constructing {country}'s EPU index?"

        human_message_template = '''{question}
        """
        {news} 
        """
        '''
        def complete_human_message(news: str):
            return HumanMessagePromptTemplate.from_template(
                human_message_template
            ).format(question = question, news = news)

        instructions_prompt = [
            (
                complete_human_message(instruction['news']),
                AIMessage(content = instruction['response'])
            )
            for instruction in instructions
        ]
        instructions_prompt = list(itertools.chain.from_iterable(instructions_prompt))

        self.pad = question
        self.template = [SystemMessage(content = system_messag)] + instructions_prompt



class EPUClassifier:
    def __init__(self, prompt: List, pad: str, model: str) -> None:
        self.prompt = prompt
        self.pad = pad
        self.chat = ChatOpenAI(model = model, temperature = 0.)


    def preprocess(self, path: str) -> List[Dict]:
        return json.loads( Path(path).read_text() )


    def predict(self, path: str) -> None:
        data = self.preprocess(path)
        articles = [i['article'] for i in data]

        n = len(data)
        preds = list(itertools.repeat(0, n))
        responses = list(itertools.repeat('', n))

        for i in trange(n):
            content = f'{self.pad}\n"""\n{articles[i]}\n"""\n'
            messages = self.prompt + [HumanMessage(content = content)]

            response = self.chat(messages)
            response = response.content.split(';')
            preds[i] = int(response[0])
            responses[i] = response[1]

        labels = [i['label'] for i in data]
        print( classification_report(labels, preds, zero_division = 1.) )

        self.articles = articles
        self.labels = labels
        self.preds = preds
        self.responses = responses


    def output(self, path: str) -> None:
        res = []
        for i in range(len(self.articles)):
            res.append({
                "article": self.articles[i],
                "label": self.labels[i],
                "pred": self.preds[i],
                "response": self.responses[i]
            })

        with open(path, 'w') as f:
            json.dump(res, f)
