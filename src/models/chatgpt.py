from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import get_openai_callback
from langchain_core.runnables.base import RunnableSequence
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

from utils import format_handler, read_jsonl, add_memory


class ChatGPT:
    def __init__(self,
                 prompt: Prompt,
                 strategy: str = 'few_shot_with_reason',
                 model: str = 'gpt-3.5-turbo-1106',
                 temperature: float = 0.,
                 timeout: float = 120,
                 data: Path = Path('data/raw/test.jsonl'),
                 n_example: int = 6,
                 example_path: str = "data/raw/fewshot_news/normal.jsonl",
                 reason_output_dir: str = "data/processed/fewshot_reasons/normal",
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
            timeout = timeout
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

        self.reasoning_strategy = reason_output_dir.split('/')[-1]
        self.filename = "_".join((
            self.reasoning_strategy,
            strategy,
            str(n_example),
            model
        ))
        self.log_file = Path(f'log/chatgpt/{self.filename}.log')


    def predict_instance(self, chain: RunnableSequence, instruction: str, i: int):
        _chain = add_memory(chain)
        res = format_handler(_chain, i, instruction, self.strategy)

        return res


    def predict(self, output_dir: Path) -> None:
        labels = [i["label"] for i in self.data]
        news = [i["news"] for i in self.data]
        output_path = output_dir / f'{self.filename}.jsonl'

        if not output_path.exists():
            _i = 0
            n_news = len(news)
        else:
            _i = len(read_jsonl(output_path))
            n_news = len(news) - _i

        with get_openai_callback() as cb:
            with output_path.open('ab') as f:
                for i in trange(n_news, position = 0, leave = True):
                    chain = (
                        self.prompt_strategy.partial(news = news[i])
                        | self.llm
                        | self.output_parser
                    )
                    res = self.predict_instance(chain, "", i + _i)
                    f.write(orjson.dumps(res, option = orjson.OPT_APPEND_NEWLINE))

        _res = read_jsonl(output_path)
        reason = [i.get('reason') for i in _res]
        pred = [i.get('pred') for i in _res]
        self.log_predict(cb, news, labels, pred, reason, output_path)


    def log_predict(self, cb, news, labels, pred, reason, output_path):
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

        exper = mlflow.set_experiment(f"EPU denoise {self.input_name}")
        with mlflow.start_run(
            experiment_id = exper.experiment_id,
            run_name = self.filename,
        ):
            mlflow.log_param("model", self.model_name)
            mlflow.log_param("few_shot_n_example", self.n_example)
            mlflow.log_param("used_tokens", cb.total_tokens)
            mlflow.log_param("prediction_strategy", self.strategy)
            mlflow.log_param("reason_strategy", self.reasoning_strategy)

            if self.model_name[:2] == 'ft':
                mlflow.log_param("epochs", self.model_name.split(':')[-2].split('-')[2])
                mlflow.log_param("fine_tune_n_example", self.model_name.split(':')[-2].split('-')[0])
                cost = 0.003*(cb.prompt_tokens/1000) + 0.006*(cb.completion_tokens/1000)
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
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")

            mlflow.log_artifact(str(self.log_file))
            mlflow.log_artifact(output_path)
            mlflow.log_table(table_dict, "news_pred_reason.json")
