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
    load_dotenv()
    key = getenv("OPENAI_API_KEY")

    if key is None:
        key = input("input your OpenAI API key: ")

        file = Path(".env")
        if file.is_file():
            with file.open("a") as f:
                f.write(f"\nOPENAI_API_KEY={key}")
        else:
            with file.open("w") as f:
                f.write(f"OPENAI_API_KEY={key}")

    return key


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    logger.info("setting up the environment variable")
    env_setup()

    logging.getLogger("openai").setLevel(logging.WARNING)
    prompt = Prompt(
        cfg.model.country,
        cfg.prompt.system_message_template_path,
        cfg.prompt.human_message_template_path,
    )

    # data = json.loads(cfg.model.test_path.read_text())[0]
    test_data = cfg.model.test_data
    test_path = Path(cfg.data.raw) / test_data
    if cfg.model.name == "ChatGPT":
        clf = ChatGPT(
            prompt,
            cfg.model.strategy,
            cfg.model.model,
            cfg.model.temperature,
            cfg.model.timeout,
            # data
            test_path,
            cfg.model.few_shot_n_example,
            cfg.model.example_path,
            cfg.model.reason_output_dir,
        )
    else:
        pass

    cfg_detail = OmegaConf.to_object(cfg)
    print("\nllm prediction parameters:\n")
    print(OmegaConf.to_yaml(cfg_detail["model"]))

    logger.info("start predicting")
    clf.predict()

    logger.info("start writing json output")
    output_dir = Path(cfg.data.predict) / test_data
    output_dir.mkdir(parents=True, exist_ok=True)
    clf.output(output_dir)

    logger.info("finish the process")


if __name__ == "__main__":
    main()
