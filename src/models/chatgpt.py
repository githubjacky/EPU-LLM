from pathlib import Path
from typing import Dict, List

import orjson
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from loguru import logger
from prompt import Prompt
from sklearn.metrics import classification_report
from tqdm import trange


class ChatGPT:
    def __init__(
        self,
        prompt: Prompt,
        strategy: str,
        model: str,
        temperature: float,
        data: Path | List[Dict],
    ) -> None:
        load_dotenv("env/.env")
        llm = ChatOpenAI(model=model, temperature=temperature)

        match strategy:
            case "zero_shot":
                self.chat = LLMChain(
                    llm=llm, prompt=prompt.zero_shot, output_parser=prompt.parser
                )
            case "zero_shot_with_reason":
                self.chat = LLMChain(
                    llm=llm,
                    prompt=prompt.zero_shot_with_reason,
                    output_parser=prompt.parser_with_reason,
                )
            case "few_shot":
                self.chat = LLMChain(
                    llm=llm, prompt=prompt.few_shot, output_parser=prompt.parser
                )
            case "few_shot_with_reason":
                self.chat = LLMChain(
                    llm=llm,
                    prompt=prompt.few_shot_with_reason,
                    output_parser=prompt.parser_with_reason,
                )

        if isinstance(data, Path):
            self.data = [
                orjson.loads(i) for i in data.read_text().split("\n") if i != ""
            ]
        else:
            self.data = data

        log_file = Path("log/chatgpt_predict.log")
        log_file.unlink(missing_ok=True)
        self.log_file = log_file

    def predict_instance(self, news: str):
        return self.chat.run(news=news)

    def predict_batch(self, batch_news: List[str]):
        input_list = [{"news": i} for i in batch_news]
        return [i["text"] for i in self.chat.apply(input_list)]

    def predict(self) -> None:
        n = len(self.data)
        predictions = []
        logger.remove()
        logger.add(self.log_file, level="WARNING")

        for i in trange(n, position=0, leave=True):
            try:
                predictions.append(self.predict_instance(self.data[i]["news"]))
            except:
                logger.warning("change to gpt-3.5-turbo-16k")
                self.chat.llm = ChatOpenAI(
                    model="gpt-3.5-turbo-16k", temperature=self.chat.llm.temperature
                )
                predictions.append(self.predict_instance(self.data[i]["news"]))

                self.chat.llm = ChatOpenAI(
                    model="gpt-3.5-turbo", temperature=self.chat.llm.temperature
                )

        preds = [i.pred for i in predictions]
        labels = [i["label"] for i in self.data]
        print(classification_report(labels, preds, zero_division=1.0))

        self.predictions = predictions

    def output(self, path: Path) -> None:
        with open(path, "w") as f:
            for i in self.predictions:
                f.write(i.json() + "\n")
