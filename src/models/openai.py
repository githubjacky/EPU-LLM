from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import get_openai_callback
from langchain_core.runnables.base import RunnableSequence
from loguru import logger
import mlflow
import matplotlib.pyplot as plt
import orjson
from os import getenv
import openai
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm, trange
import tiktoken
import time
from typing import List, Dict, Optional

from .prompt import Prompt
from .utils import format_handler, read_jsonl, add_memory, log_init



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
                 verbose = False
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
            model=model,
            temperature=temperature,
            timeout=timeout
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
        self.log_file = Path(f'log/openai/{self.filename}.log')
        self.fewshot_examples_path = example_path
        self.fewshot_reasons_path = f'{reason_output_dir}/{self.n_example}.jsonl'

        self.verbose = verbose


    def predict_instance(self, chain: RunnableSequence, instruction: str, i: int):
        _chain = add_memory(chain)
        res = format_handler(_chain, i, instruction, self.strategy, self.verbose)

        return res


    def predict(self, output_dir: Path) -> None:
        log_init(self.log_file)
        logger.info("start predicting")

        labels = [i["label"] for i in self.data]
        news = [i["news"] for i in self.data]
        output_path = output_dir / f'{self.filename}.jsonl'

        if not output_path.exists():
            _i = 0
            n_news = len(news)
        else:
            _i = len(read_jsonl(output_path))
            n_news = len(news) - _i
            logger.info(f"restart the process form the {_i+1}th news")

        with get_openai_callback() as cb:
            with output_path.open('ab') as f:
                for i in trange(n_news, position = 0, leave = True):
                    chain = (
                        self.prompt_strategy.partial(news = news[i])
                        | self.llm
                        | self.output_parser
                    )
                    res = self.predict_instance(chain, self.prompt.question, i + _i)
                    f.write(orjson.dumps(res, option = orjson.OPT_APPEND_NEWLINE))

        logger.info("finish the process")
        _res = read_jsonl(output_path)
        reason = [i.get('reason') for i in _res]
        pred = [i.get('pred') for i in _res]

        cost = (
            cb.total_cost
            if self.model_name[:2] != 'ft'
            else
            0.003*(cb.prompt_tokens/1000) + 0.006*(cb.completion_tokens/1000)
        )
        self.log_predict(cb.total_tokens, cost, news, labels, pred, reason, output_path)


    def log_predict(self,
                    n_tokens: int,
                    cost: float,
                    news: List[str],
                    labels: List[int],
                    pred: List[int],
                    reason: Optional[List[str]] = None,
                    output_path: Path = Path('data/predict/normal_few_shot_with_reason_8_gpt3.5-turbo-1106.josnl')
                    ):
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
            mlflow.log_artifact(self.prompt.system_message_path)
            mlflow.log_artifact(self.prompt.human_message_path)
            mlflow.log_artifact(self.fewshot_examples_path)

            mlflow.log_param("model", self.model_name)
            mlflow.log_param("few_shot_n_example", self.n_example)
            mlflow.log_param("used_tokens", n_tokens)
            mlflow.log_param("prediction_strategy", self.strategy)
            mlflow.log_param("cost", cost)

            if self.strategy in ["zero_shot_with_reason", "few_shot_with_reason"]:
                mlflow.log_param("reason_strategy", self.reasoning_strategy)
                mlflow.log_artifact(self.fewshot_reasons_path)

            if self.model_name[:2] == 'ft':
                mlflow.log_param("epochs", self.model_name.split(':')[-2].split('-')[2])
                mlflow.log_param("fine_tune_n_example", self.model_name.split(':')[-2].split('-')[0])
            else:
                mlflow.log_param("fine_tune_n_example", 0)

            mlflow.log_metric("precision_0", metric_dict["0"]["precision"])
            mlflow.log_metric("micro_f1", metric_dict["accuracy"])
            mlflow.log_metric("macro_f1", metric_dict["macro avg"]["f1-score"])
            mlflow.log_metric("weighted_f1", metric_dict["weighted avg"]["f1-score"])

            cm = confusion_matrix(labels, pred, normalize="pred")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig("confusion_matrix_prec.png")
            mlflow.log_artifact("confusion_matrix_prec.png")

            cm = confusion_matrix(labels, pred, normalize="true")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig("confusion_matrix_recall.png")
            mlflow.log_artifact("confusion_matrix_recall.png")



            mlflow.log_artifact(str(self.log_file))
            mlflow.log_artifact(output_path)
            mlflow.log_table(table_dict, "news_pred_reason.json")


class OpenAIFineTuner:
    def __init__(self,
                 prompt: Prompt,
                 strategy: str,
                 input_dir: str = "data/raw",
                 output_dir: str = f'data/processed/fine_tune',
                 train_file: str = "example01.jsonl",
                 val_file: Optional[str] = None,
                ):
        self.prompt = prompt
        self.strategy = strategy
        self.input_dir = Path(input_dir)
        self.output_dir = Path(f'{output_dir}/{strategy}_format')
        self.output_reason_dir = Path(f'{output_dir}/reason_example')



        record_dir = Path(output_dir) / getenv('OPENAI_API_KEY')
        record_dir.mkdir(exist_ok = True, parents = True)
        self.record_file_path = record_dir / 'upload_file_info.jsonl'
        self.record_model_path = record_dir / 'fine_models.jsonl'
        file_name = []
        file_id = []
        for j in read_jsonl(self.record_file_path):
            if j.get('strategy') == strategy:
                file_name.append(j.get("name"))
                file_id.append(j.get("id"))
        self.file_name = file_name
        self.file_id = file_id

        self.train_file = train_file
        self.val_file = val_file

    def __format_examples_no_reason(self,
                                    tag: str,
                                    input_path: Path,
                                    output_file: str
                                    ):
        output_path = self.output_dir / output_file

        raw_examples = (
            read_jsonl(input_path, self.n)
            if tag == "train"
            else
            read_jsonl(input_path)
        )

        with output_path.open("ab") as f:
            for i in range(len(raw_examples)):
                human_message = self.prompt.human_message.format(
                    correct_instructions = "",
                    output_instructions = self.prompt.output_instructions,
                    news = raw_examples[i].get("news"),
                )
                item = {
                    "messages": [
                        {"role": "system", "content": self.prompt.system_message},
                        {"role": "user", "content": human_message},
                        {
                            "role": "assistant",
                            "content": f'{{"pred": {raw_examples[i].get("label")}}}',
                        },
                    ]
                }
                f.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE))

    def __format_examples_with_reason(self,
                                      tag: str,
                                      input_path: Path,
                                      output_file: str,
                                      output_reason_file: str
                                      ):
        if tag == "train":
            _raw_examples = read_jsonl(input_path)
            total_n = len(_raw_examples)
            raw_examples = _raw_examples[:self.n]
            trunc_n = self.n
        else:
            raw_examples = read_jsonl(input_path)
            total_n = len(raw_examples)
            trunc_n = total_n

        output_reason_path = self.output_reason_dir / output_reason_file
        if not output_reason_path.exists():
            _ = self.prompt.reasoning(total_n, input_path, output_reason_path)
        reason_examples = read_jsonl(output_reason_path, trunc_n, return_str = True)

        system_message = self.prompt.system_message.format(country = self.prompt.country)
        output_path = self.output_dir / output_file
        with output_path.open("wb") as f:
            for i in range(trunc_n):
                human_message = self.prompt.human_message.format(
                    correct_instructions = self.prompt.question,
                    output_instructions = self.prompt.output_instructions,
                    news = raw_examples[i].get("news"),
                )
                item = {
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": human_message},
                        {
                            "role": "assistant",
                            "content": reason_examples[i],
                        },
                    ]
                }
                f.write(orjson.dumps(item, option = orjson.OPT_APPEND_NEWLINE))


    def __upload(self, file_path: Path, file: str):
        obj = openai.File.create(
            file = open(str(file_path), "rb"),
            purpose = "fine-tune",
            user_provided_filename = file,
        )

        with self.record_file_path.open('ab') as f:
            f.write(orjson.dumps(
                {
                    "name": file,
                    "id": obj.get('id'),
                    "strategy": self.strategy
                },
                option = orjson.OPT_APPEND_NEWLINE
            ))
        self.file_name.append(file)
        self.file_id.append(obj.get('id'))


    def format(self, n: int = 50):
        self.n = n
        self.formatted_file_path = {}
        for tag in ["train", "val"]:
            if eval(f"self.{tag}_file") is not None:
                prefix_n = (
                    self.n
                    if tag == 'train'
                    else
                    len((self.input_dir / self.val_file).read_text().split("\n")[:-1])
                )
                self.formatted_file_path[tag] = f'{prefix_n}_{eval(f"self.{tag}_file")}'

                input_path = self.input_dir / eval(f"self.{tag}_file")
                output_file = self.formatted_file_path.get(tag)

                if not (self.output_dir / output_file).exists():
                    logger.info(f"formatting {tag} file")
                    if self.strategy == "no_reason":
                        self.__format_examples_no_reason(
                            tag,
                            input_path,
                            output_file
                        )
                    else:
                        self.__format_examples_with_reason(
                            tag,
                            input_path,
                            output_file,
                            f"{eval(f'self.{tag}_file').split('.')[0]}.jsonl"
                        )
                    self.__upload(self.output_dir / output_file, output_file)
                else:
                    logger.info(f"{tag} file has already formatted")

    def __num_tokens_from_messages(self,
                                   messages: List[Dict[str, str]],
                                   tokens_per_message: int = 3,
                                   tokens_per_name: int = 1
                                   ):
        encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3

        return num_tokens

    def log_fine_tune_cost(self, data_path: str | Path, n_epochs: int):
        dataset = read_jsonl(data_path)

        convo_lens = []
        for ex in dataset:
            messages = ex["messages"]
            convo_lens.append(self.__num_tokens_from_messages(messages))

        MAX_TOKENS_PER_EXAMPLE = 4096
        n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
        cost = 0.008* (n_epochs * (n_billing_tokens_in_dataset / 1000))

        return cost


    def fine_tune(self, n_epochs: int):
        train_file_id = self.file_id[
            self.file_name.index(self.formatted_file_path.get('train'))
        ]
        val_file_id = (
            self.file_id[
                self.file_name.index(self.formatted_file_path.get('val'))
            ]
            if self.val_file is not None
            else
            -1
        )

        processed = False
        uploaded_file_id = [
            i["id"]
            for i in openai.File.list().get("data")
            if i["status"] == "uploaded"
        ]
        if (
            train_file_id not in uploaded_file_id and
            val_file_id not in uploaded_file_id
        ):
            processed = True
        else:
            logger.info("uploading the file")

        while not processed:
            time.sleep(10)
            uploaded_file_id = [
                i["id"]
                for i in openai.File.list().get("data")
                if i["status"] == "uploaded"
            ]
            if (
                train_file_id not in uploaded_file_id and
                val_file_id not in uploaded_file_id
            ):
                processed = True

        short_strategy = (
            'nor'
            if self.strategy == 'no_reason'
            else
            'withr'
        )
        suffix = f"{self.n}-{short_strategy}-{n_epochs}"
        logger.info(f"new fine tune job: {suffix}")

        if self.val_file is not None:
            train_cost = (
                self.log_fine_tune_cost(
                    self.output_dir / self.formatted_file_path.get('train'),
                    n_epochs
                )
            )
            val_cost = (
                self.log_fine_tune_cost(
                    self.output_dir / self.formatted_file_path.get('val'),
                    1
                )
            )
            logger.info(f"the cost for fine tuning: {train_cost + val_cost}")
            self.ft_job_id = (
                openai.FineTuningJob
                .create(
                    training_file = train_file_id,
                    validation_file = val_file_id,
                    suffix = suffix,
                    model = "gpt-3.5-turbo",
                    hyperparameters = {"n_epochs": n_epochs},
                )
                .get('id')
            )
        else:
            train_cost = (
                self.log_fine_tune_cost(
                    self.output_dir / self.formatted_file_path.get('train'),
                    n_epochs
                )
            )
            logger.info(f"the cost for fine tuning: {train_cost}")
            self.ft_job_id = (
                openai.FineTuningJob
                .create(
                    training_file = train_file_id,
                    suffix = suffix,
                    model = "gpt-3.5-turbo",
                    hyperparameters = {"n_epochs": n_epochs},
                )
                .get('id')
            )

    def log_fine_tune(self) -> None:
        status = ''
        logger.info(f'wait for fine tuning job to initiate')
        processed = [

        ]
        while status in processed:
            time.sleep(120)
            status = (
                openai.FineTuningJob
                .list_events(id = self.ft_job_id, limit = 1)
                .get('data')[0]
                .get('message')
            )

        event = (
            openai.FineTuningJob
            .list_events(id = self.ft_job_id, limit = 1)
            .get('data')[0]
        )
        step, total_step = event['step'].split('/')
        with tqdm(total = total_step) as pbar:
            while step != total_step:
                pbar.update(step)
                time.sleep(10)
                step = (
                    openai.FineTuningJob
                    .list_events(id = self.ft_job_id, limit = 1)
                )
            pbar.update(total_step)

        with self.record_model_path.open('ab') as f:
            f.write(orjson.dumps(
                {
                    'model':openai.FineTuningJob.retrieve(self.ft_job_id)['fine_tuned_model'],
                    'train': self.train_file,
                    'val': self.val_file if self.val_file is not None else ''
                }, 
                option = orjson.OPT_APPEND_NEWLINE
            ))
