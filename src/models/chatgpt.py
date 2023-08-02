from itertools import chain
import json
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from sklearn.metrics import classification_report
from tqdm import trange
from typing import List, Dict

from .utils import Prompt, ClassificationResult


class ChatGPT:
    def __init__(self,
                 prompt: Prompt,
                 key: str,
                 model: str,
                 temperature: float,
                 batch_size: int,
                 data: Path | List[Dict]
                 ) -> None:
        llm = ChatOpenAI(
            openai_api_key=key,
            model=model,
            temperature=temperature
        )
        self.chat = LLMChain(
            llm=llm,
            prompt=prompt.chat_message,
            output_parser=prompt.parser
        )

        self.batch_size = batch_size

        if isinstance(data, Path):
            self.data = json.loads(data.read_text())
        else:
            self.data = data

    def predict_instance(self, article: str) -> ClassificationResult:
        return self.chat.run(article=article)

    def predict_batch(self,
                      batch_articles: List[str]
                      ) -> List[ClassificationResult]:
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

    def output(self, path: Path) -> None:
        res = [json.loads(i.json()) for i in self.predictions]
        with open(path, 'w') as f:
            json.dump(res, f)
