import hydra
from loguru import logger
import orjson
import json
from omegaconf import DictConfig
import openai
from pathlib import Path
from prompt import Prompt
import time
from typing import List, Dict, Optional
import tiktoken

from utils import env_setup, read_jsonl


class OpenAIFineTuner:
    def __init__(self,
                 prompt: Prompt,
                 strategy: str,
                 input_dir: str = "data/raw",
                 output_dir: str = "data/processed/fine_tune",
                 train_file: str = "example01.jsonl",
                 val_file: Optional[str] = None,
                ):
        self.prompt = prompt
        self.strategy = strategy
        self.input_dir = Path(input_dir)
        self.output_dir = Path(f'{output_dir}/{strategy}_format')
        self.output_reason_dir = Path(f'{output_dir}/reason_example')


        record_path = Path(output_dir) / 'upload_file_info.jsonl'
        file_name = []
        file_id = []
        for j in [orjson.loads(i) for i in record_path.read_text().split("\n")[:-1]]:
            file_name.append(j.get("name"))
            file_id.append(j.get("id"))
        self.record_path = record_path
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

        with self.record_path.open('ab') as f:
            f.write(orjson.dumps(
                {
                    "name": file,
                    "id": obj.get('id')
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
            openai.FineTuningJob.create(
                training_file = train_file_id,
                validation_file = val_file_id,
                suffix = suffix,
                model = "gpt-3.5-turbo",
                hyperparameters = {"n_epochs": n_epochs},
            )
        else:
            train_cost = (
                self.log_fine_tune_cost(
                    self.output_dir / self.formatted_file_path.get('train'),
                    n_epochs
                )
            )
            logger.info(f"the cost for fine tuning: {train_cost}")
            openai.FineTuningJob.create(
                training_file = train_file_id,
                suffix = suffix,
                model = "gpt-3.5-turbo",
                hyperparameters = {"n_epochs": n_epochs},
            )



@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    env_setup()

    if cfg.model.name == "ChatGPT":
        finetuner = OpenAIFineTuner(
            Prompt(
                cfg.model.country,
                cfg.prompt.system_message_template_path,
                cfg.prompt.human_message_template_path,
            ),
            cfg.model.fine_tune.strategy,
            cfg.model.fine_tune.input_dir,
            cfg.model.fine_tune.output_dir,
            cfg.model.fine_tune.train_file,
            cfg.model.fine_tune.val_file,
        )

        finetuner.format(cfg.model.fine_tune.n)
        time.sleep(5)
        finetuner.fine_tune(cfg.model.fine_tune.n_epochs)


if __name__ == "__main__":
    main()
