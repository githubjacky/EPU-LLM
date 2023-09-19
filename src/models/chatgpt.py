from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from loguru import logger
import mlflow
import matplotlib.pyplot as plt
import orjson
import openai
from prompt import Prompt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from typing import List
from tqdm import trange


class ChatGPT:
    def __init__(
        self,
        prompt: Prompt,
        strategy: str,
        model: str,
        temperature: float,
        timeout: float,
        data: Path,
        n_example: int = 6,
        example_path: str = "data/raw/reason_example.jsonl",
        reason_output_dir: str = "data/processed",
    ) -> None:
        load_dotenv()
        llm = ChatOpenAI(model=model, temperature=temperature, request_timeout=timeout)

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
                    llm=llm,
                    prompt=prompt.few_shot(n_example, example_path),
                    output_parser=prompt.parser,
                )
            case "few_shot_with_reason":
                self.chat = LLMChain(
                    llm=llm,
                    prompt=prompt.few_shot_with_reason(
                        n_example, example_path, reason_output_dir
                    ),
                    output_parser=prompt.parser_with_reason,
                )
        self.strategy = strategy
        self.input_name = str(data).split("/")[2]
        self.model_name = model
        self.prompt = prompt

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

    def __format_examples_no_reason(self, tag: str, input_path: Path, n: int):
        output_dir = Path("data/processed/fine_tune/")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{n}_no_reason_{tag}.jsonl"

        if not output_path.exists():
            if tag == "train":
                raw_examples = [
                    orjson.loads(i)
                    for i in Path(input_path).read_text().split("\n")
                    if i != ""
                ][:n]
            else:
                raw_examples = [
                    orjson.loads(i)
                    for i in Path(input_path).read_text().split("\n")
                    if i != ""
                ]

            formatted = []
            for i in range(len(raw_examples)):
                human_message = self.prompt.human_message.format(
                    correct_instructions="",
                    output_instructions=self.prompt.output_instructions,
                    news=raw_examples[i]["news"],
                )
                formatted.append(
                    {
                        "messages": [
                            {"role": "system", "content": self.prompt.system_message},
                            {"role": "user", "content": human_message},
                            {
                                "role": "assistant",
                                "content": f'{{"pred": {raw_examples[i]["label"]}}}',
                            },
                        ]
                    }
                )

            with output_path.open("wb") as f:
                for i in formatted:
                    f.write(orjson.dumps(i, option=orjson.OPT_APPEND_NEWLINE))

    def finetune_train_test_split_no_reason(
        self,
        n: int = 50,
        training_path: str = "data/raw/example01.jsonl",
        testing_path: str = "data/raw/example02.jsonl",
    ):
        self.__format_examples_no_reason("train", Path(training_path), n)
        self.__format_examples_no_reason("val", Path(testing_path), n)
        self.formatted_training_path = (
            f"data/processed/fine_tune/{n}_no_reason_train.jsonl"
        )
        self.formatted_val_path = f"data/processed/fine_tune/{n}_no_reason_val.jsonl"

    def fine_tune(self, suffix: str):
        train = openai.File.create(
            file=open(self.formatted_training_path, "rb"), purpose="fine-tune"
        )

        val = openai.File.create(
            file=open(self.formatted_val_path, "rb"), purpose="fine-tune"
        )
        res = openai.FineTuningJob.create(
            training_file=train["id"],
            validation_file=val["id"],
            suffix=suffix,
            model="gpt-3.5-turbo",
        )

        return res
