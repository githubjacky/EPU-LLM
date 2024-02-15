import os, sys
sys.path.append(os.path.abspath(f"{os.getcwd()}"))

import hydra
from loguru import logger
import logging
from omegaconf import DictConfig
from pathlib import Path

from src.models.prompt import Prompt
from src.models.utils import env_setup
from src.models.openai import ChatGPT


@hydra.main(config_path="../config", config_name="main", version_base=None)
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
    clf = ChatGPT(
        prompt,
        cfg.model.strategy,
        cfg.model.model,
        cfg.model.temperature,
        cfg.model.timeout,
        # data
        test_path,
        cfg.model.few_shot_n_example,
        f'{cfg.prompt.fewshot_news_path}/{cfg.model.reasoning_strategy}.jsonl',
        f'{cfg.prompt.fewshot_reasons_path}/{cfg.model.reasoning_strategy}',
        cfg.model.verbose
    )

    # cfg_detail = OmegaConf.to_object(cfg)
    # print("\nllm prediction parameters:\n")
    # print(OmegaConf.to_yaml(cfg_detail["model"]))


    output_dir = Path(cfg.data.predict) / test_data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{clf.filename}.jsonl'

    clf.predict(output_path)


if __name__ == "__main__":
    main()
