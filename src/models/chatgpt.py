from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import get_openai_callback
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
from tqdm import trange
import time

from utils import format_handler, refresh_handler, read_jsonl


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

        match strategy:
            case "zero_shot":
                prompt_strategy = prompt.zero_shot
                output_parser = prompt.parser
            case "zero_shot_with_reason":
                prompt_strategy = prompt.zero_shot_with_reason
                output_parser = prompt.parser_with_reason
            case "few_shot":
                prompt_strategy = prompt.few_shot(n_example, example_path)
                output_parser = prompt.parser
            # few shot with reason
            case _:
                prompt_strategy = prompt.few_shot_with_reason(
                    n_example,
                    example_path,
                    reason_output_dir
                )
                output_parser = prompt.parser_with_reason

        self.llm = ChatOpenAI(
            model = model,
            temperature = temperature,
            request_timeout = timeout
        )
        self.prompt_strategy = prompt_strategy
        self.output_parser = output_parser

        self.strategy = strategy
        self.input_name = str(data).split("/")[2]
        self.model_name = model
        self.prompt = prompt

        self.n_example = (
            prompt.num
            if self.strategy in ["few_shot", "few_shot_with_reason"]
            else 0
        )
        self.data = read_jsonl(data) if isinstance(data, Path) else data

        log_file = Path("log/chatgpt_predict.log")
        log_file.unlink(missing_ok=True)
        self.log_file = log_file


    def predict_instance(self, chain: LLMChain, instruction: str, i: int):
        chain.memory = ConversationBufferMemory(
            memory_key = "chat_history",
            return_messages = True
        )
        res = format_handler(chain, i, instruction, self.strategy)
        if res.get('pred') == -1:
            logger.info(f"change to gpt-3.5-turbo-16k, for the {i}th sample")
            _, res = refresh_handler(chain, instruction, i, self.strategy)

        return res


    def predict(self) -> None:
        logger.remove()
        logger.add(self.log_file, level="INFO")

        labels = [i["label"] for i in self.data]
        news = [i["news"] for i in self.data]

        pred = []
        reason = []
        predictions = []
        with get_openai_callback() as cb:
            for i in trange(len(news), position = 0, leave = True):
                chain = LLMChain(
                    llm = self.llm,
                    prompt = self.prompt_strategy.partial(news = news[i]),
                )
                res = self.predict_instance(chain, self.prompt.question, i)
                predictions.append(res)
                pred.append(res['pred'])
                reason.append(res.get('reason'))
                time.sleep(1)

        metric_dict = classification_report(labels, pred, output_dict = True)
        table_dict = (
            {
                "news": news,
                "reason": reason,
                "pred": pred,
                "label": labels,
            }
            if self.strategy in ["zero_shot_with_reason", "few_shot_with_reason"]
            else
            {
                "news": news,
                "pred": pred,
                "label": labels,
            }
        )

        exper = mlflow.set_experiment(f"EPU denoise_{self.input_name}")
        with mlflow.start_run(
            experiment_id = exper.experiment_id,
            run_name = "_".join((self.strategy, str(self.n_example), self.model_name)),
        ):
            mlflow.log_param("model", self.model_name)
            mlflow.log_param("few_shot_n_example", self.n_example)
            mlflow.log_param("used_tokens", cb.total_tokens)

            if self.model_name[:2] == 'ft':
                mlflow.log_param("epochs", self.model_name.split(':')[-2].split('-')[2])
                mlflow.log_param("fine_tune_n_example", self.model_name.split(':')[-2].split('-')[0])
                cost = 0.012*(cb.prompt_tokens/1000) + 0.016*(cb.completion_tokens/1000)
                mlflow.log_param("cost", cost)
            else:
                mlflow.log_param("cost", cb.total_cost)

            mlflow.log_metric("precision_0", metric_dict["0"]["precision"])
            mlflow.log_metric("micro_f1", metric_dict["accuracy"])
            mlflow.log_metric("macro_f1", metric_dict["macro avg"]["f1-score"])
            mlflow.log_metric("weighted_f1", metric_dict["weighted avg"]["f1-score"])

            cm = confusion_matrix(labels, pred, normalize="pred")
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
        with open(output_path, 'wb') as f:
            for i in self.predictions:
                f.write(orjson.dumps(i, option = orjson.OPT_APPEND_NEWLINE))
