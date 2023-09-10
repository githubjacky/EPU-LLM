from dotenv import load_dotenv
import hydra
from loguru import logger
import logging
from omegaconf import DictConfig, OmegaConf
from os import getenv
from pathlib import Path

from chatgpt import ChatGPT
from prompt import Prompt


def env_setup() -> str:
    load_dotenv("env/.env")
    key = getenv("OPENAI_API_KEY")
    if key is None:
        key = input("input your OpenAI API key: ")
        file = Path("env/.env")
        if file.is_file():
            with open(file, "a") as f:
                f.write(f"\nOPENAI_API_KEY={key}")
        else:
            with open(file, "w") as f:
                f.write(f"OPENAI_API_KEY={key}")

    return key


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    logger.info("setting up the environment variable")
    env_setup()

    logging.getLogger("openai").setLevel(logging.WARNING)

    if cfg.model.strategy == "few_shot_with_reason":
        prompt = Prompt(
            cfg.model.country,
            cfg.prompt.system_message_template_path,
            cfg.prompt.human_message_template_path,
            cfg.model.few_shot_num_example,
            cfg.prompt.example_path,
            cfg.prompt.reason_example_dir,
        )
    elif cfg.model.strategy == "few_shot":
        prompt = Prompt(
            cfg.model.country,
            cfg.prompt.system_message_template_path,
            cfg.prompt.human_message_template_path,
            cfg.model.few_shot_num_example,
            cfg.prompt.example_path,
        )
    else:
        prompt = Prompt(
            cfg.model.country,
            cfg.prompt.system_message_template_path,
            cfg.prompt.human_message_template_path,
        )

    input_dir = Path(cfg.data.raw)

    # input = 'EPU_Noise_Test.json'
    input = cfg.model.input_file
    input_path = input_dir / input
    # data = json.loads(input_path.read_text())[0]

    if cfg.model.name == "ChatGPT":
        clf = ChatGPT(
            prompt,
            cfg.model.strategy,
            cfg.model.model,
            cfg.model.temperature,
            # data
            input_path,
        )
    else:
        pass

    cfg_detail = OmegaConf.to_object(cfg)
    print("\nscrapping parameters:\n")
    print(OmegaConf.to_yaml(cfg_detail["model"]))

    logger.info("start predicting")
    clf.predict()

    logger.info("start writing json output")
    output_dir = Path(cfg.data.predict) / input
    output_dir.mkdir(parents=True, exist_ok=True)
    clf.output(output_dir)

    logger.info("finish the process")


if __name__ == "__main__":
    main()
