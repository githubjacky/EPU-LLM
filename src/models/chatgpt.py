from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from loguru import logger
import mlflow
import matplotlib.pyplot as plt
import orjson
from prompt import Prompt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from typing import Dict, List
from tqdm import trange


class ChatGPT:
    def __init__(
        self,
        prompt: Prompt,
        strategy: str,
        model: str,
        temperature: float,
        data: Path,
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
        self.strategy = strategy
        self.input_name = str(data).split("/")[2]
        self.model_name = model

        self.n_example = (
            prompt.num
            if self.strategy == "few_shot" or self.strategy == "few_shot_with_reason"
            else 0
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
        logger.remove()
        logger.add(self.log_file, level="WARNING")

        n = len(self.data)
        predictions = []

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

        metric_dict = classification_report(labels, preds, output_dict=True)

        exper = mlflow.set_experiment(f"EPU denoise_{self.input_name}")

        if (
            self.strategy == "zero_shot_with_reason"
            or self.strategy == "few_shot_with_reason"
        ):
            table_dict = {
                "news": [i["news"] for i in self.data],
                "reason": [i.reason for i in predictions],
                "pred": preds,
                "label": labels,
            }
        else:
            table_dict = {
                "news": [i["news"] for i in self.data],
                "pred": preds,
                "label": labels,
            }

        with mlflow.start_run(
            experiment_id=exper.experiment_id,
            run_name="_".join((self.strategy, str(self.n_example), self.model_name)),
        ):
            mlflow.log_param("n_example", self.n_example)
            mlflow.log_param("input_file", self.input_name)

            mlflow.log_metric("precision_0", metric_dict["0"]["precision"])
            mlflow.log_metric("micro_f1", metric_dict["accuracy"])
            mlflow.log_metric("macro_f1", metric_dict["macro avg"]["f1-score"])
            mlflow.log_metric("weighted_f1", metric_dict["weighted avg"]["f1-score"])

            cm = confusion_matrix(labels, preds, normalize="pred")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig("confustion_matrix.png")

            mlflow.log_artifact("confustion_matrix.png")
            mlflow.log_table(table_dict, "news_pred_reason.json")

        self.predictions = predictions

    def output(self, output_dir: Path) -> None:
        output_path = output_dir / (
            "_".join((self.strategy, str(self.n_example), self.model_name)) + ".jsonl"
        )
        with open(output_path, "w") as f:
            for i in self.predictions:
                f.write(i.json() + "\n")
