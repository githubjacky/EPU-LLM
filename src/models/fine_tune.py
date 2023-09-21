import hydra
from loguru import logger
import orjson
from omegaconf import DictConfig
import openai
from pathlib import Path
from prompt import Prompt
import sys


class OpenAIFineTuner:
    def __init__(
        self,
        prompt: Prompt,
        strategy: str,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed/fine_tune",
        training_file: str = "example01.jsonl",
        val_file: str = "example02.jsonl",
    ):
        self.prompt = prompt
        self.strategy = strategy
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        record_path = Path(output_dir) / "upload_file_info.jsonl"
        file_name = []
        file_id = []
        for j in [orjson.loads(i) for i in record_path.read_text().split("\n")[:-1]]:
            file_name.append(j["name"])
            file_id.append(j["id"])

        self.existed_file_name = file_name
        self.existed_file_id = file_id
        self.record_path = record_path

        self.training_file = training_file
        self.val_file = val_file

    def __format_examples_no_reason(
        self, tag: str, input_path: Path, n: int, output_file: str
    ):
        output_path = self.output_dir / output_file
        if tag == "train":
            raw_examples = [
                orjson.loads(i) for i in input_path.read_text().split("\n") if i != ""
            ][:n]
        else:
            raw_examples = [
                orjson.loads(i) for i in input_path.read_text().split("\n") if i != ""
            ]
        with output_path.open("ab") as f:
            for i in range(len(raw_examples)):
                human_message = self.prompt.human_message.format(
                    correct_instructions="",
                    output_instructions=self.prompt.output_instructions,
                    news=raw_examples[i]["news"],
                )
                item = {
                    "messages": [
                        {"role": "system", "content": self.prompt.system_message},
                        {"role": "user", "content": human_message},
                        {
                            "role": "assistant",
                            "content": f'{{"pred": {raw_examples[i]["label"]}}}',
                        },
                    ]
                }
                f.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE))

    def __upload(self, file_path: Path, file: str):
        obj = openai.File.create(
            file=open(str(file_path), "rb"),
            purpose="fine-tune",
            user_provided_filename=file,
        )
        self.existed_file_name.append(file)
        self.existed_file_id.append(obj["id"])

        with self.record_path.open("ab") as f:
            f.write(
                orjson.dumps(
                    {"name": file, "id": obj["id"]}, option=orjson.OPT_APPEND_NEWLINE
                )
            )

    def format(self, n: int = 50):
        self.n = n
        for tag in ["training", "val"]:
            input_path = self.input_dir / eval(f"self.{tag}_file")
            output_file = f"{n}_{self.strategy}_{tag}.jsonl"
            if not (self.output_dir / output_file).exists():
                logger.info(f"formatting {tag} file")
                if self.strategy == "no_reason":
                    self.__format_examples_no_reason(tag, input_path, n, output_file)
                else:
                    pass
                self.__upload(self.output_dir / output_file, output_file)
            else:
                logger.info(f"{tag} file has already formatted")

    def fine_tune(self, n_epochs: int):
        train_file_id = self.existed_file_id[
            self.existed_file_name.index(f"{self.n}_{self.strategy}_training.jsonl")
        ]
        val_file_id = self.existed_file_id[
            self.existed_file_name.index(f"{self.n}_{self.strategy}_val.jsonl")
        ]
        uploaded_file_id = [
            i["id"]
            for i in openai.File.list()["data"]
            if i["status"] not in ["processed", "uploaded"]
        ]
        if val_file_id not in uploaded_file_id:
            logger.info("the validation file hasn't finished processing")
            if train_file_id not in uploaded_file_id:
                logger.info("the training file hasn't finished processing")
            sys.exit()
        else:
            res = openai.FineTuningJob.create(
                training_file=train_file_id,
                validation_file=val_file_id,
                suffix=f"{self.n}_train_samples_{n_epochs}_epochs",
                model="gpt-3.5-turbo",
                hyperparameters={"n_epochs": n_epochs},
            )

        return res


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
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

        if cfg.model.fine_tune.strategy == "no_reason":
            finetuner.format(cfg.model.fine_tune.n)
            finetuner.fine_tune(cfg.model.fine_tune.n_epochs)
        else:
            pass


if __name__ == "__main__":
    main()
