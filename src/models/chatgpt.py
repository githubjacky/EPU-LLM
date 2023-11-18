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

        log_file = Path(
            '/'.join((
                'log',
                'chatgpt_predict',
                self.input_name,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            ))
        )
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


    def predict(self, output_dir: Path) -> None:
        logger.remove()
        logger.add(self.log_file, level="INFO")
        logger.info(f'model: {self.llm.model_name}')
        logger.info(f'prompting strategy: {self.strategy}_{self.n_example}')

        labels = [i["label"] for i in self.data]
        news = [i["news"] for i in self.data]

        output_path = output_dir / (
            "_".join((self.strategy, str(self.n_example), self.model_name)) + ".jsonl"
        )

        with get_openai_callback() as cb:
            if not output_path.exists():
                reason = []
                pred = []
                with output_path.open('wb') as f:
                    for i in trange(len(news), position = 0, leave = True):
                        chain = LLMChain(
                            llm = self.llm,
                            prompt = self.prompt_strategy.partial(news = news[i]),
                            # verbose = True
                        )
                        res = self.predict_instance(chain, self.prompt.question, i)
                        pred.append(res['pred'])
                        reason.append(res.get('reason'))
                        f.write(orjson.dumps(res, option = orjson.OPT_APPEND_NEWLINE))
                        time.sleep(1)
            else:
                with output_path.open('ab') as f:
                    _res = read_jsonl(output_path)
                    reason = [i.get('reason') for i in _res]
                    pred = [i.get('pred') for i in _res]
                    _i = len(_res)
                    for i in trange(len(news) - _i, position = 0, leave = True):
                        chain = LLMChain(
                            llm = self.llm,
                            prompt = self.prompt_strategy.partial(news = news[i + _i]),
                        )
                        res = self.predict_instance(chain, self.prompt.question, i + _i)
                        pred.append(res['pred'])
                        reason.append(res.get('reason'))
                        f.write(orjson.dumps(res, option = orjson.OPT_APPEND_NEWLINE))
                        time.sleep(1)

            self.log_predict(cb, news, labels, pred, reason)


    def log_predict(self, cb, news, labels, pred, reason):
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
            mlflow.log_param("prediction_strategy", self.strategy)

            if self.model_name[:2] == 'ft':
                mlflow.log_param("epochs", self.model_name.split(':')[-2].split('-')[2])
                mlflow.log_param("fine_tune_n_example", self.model_name.split(':')[-2].split('-')[0])
                cost = 0.012*(cb.prompt_tokens/1000) + 0.016*(cb.completion_tokens/1000)
                mlflow.log_param("cost", cost)
            else:
                mlflow.log_param("fine_tune_n_example", 0)
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
