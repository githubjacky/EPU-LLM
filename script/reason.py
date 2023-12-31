import os, sys
sys.path.append(os.path.abspath(f"{os.getcwd()}"))

import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.models.prompt import Prompt


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    prompt = Prompt(
        cfg.model.country,
        cfg.prompt.system_message_template_path,
        cfg.prompt.human_message_template_path,
    )

    n = cfg.model.few_shot_n_example
    prompt.reasoning(
        n,
        Path(f'{cfg.prompt.fewshot_news_path}/{cfg.model.reasoning_strategy}.jsonl'),
        Path(f'{cfg.prompt.fewshot_reasons_path}/{cfg.model.reasoning_strategy}/{n}.jsonl'),
    )


if __name__ == "__main__":
    main()
