from dotenv import load_dotenv
import hydra
import json
from loguru import logger
from models.utils import Prompt
from models.chatgpt import ChatGPT
from omegaconf import DictConfig, OmegaConf
from os import getenv

from pathlib import Path


def env_setup() -> str:
    load_dotenv()
    key = getenv("OPENAI_API_KEY")
    if key is None:
        key = input("input your OpenAI API key: ")
        file = Path('.env')
        if file.is_file():
            with open(file, 'a') as f:
                f.write(f"\nOPENAI_API_KEY={key}")
        else:
            with open(file, 'w') as f:
                f.write(f"OPENAI_API_KEY={key}")

    return key


@hydra.main(config_path='../config', config_name='main', version_base=None)
def main(cfg: DictConfig):
    logger.info("setting up the environment variable")
    key = env_setup()
    prompt = Prompt(cfg.model.country)

    input_dir = Path(cfg.data.raw) / 'examples_taiwanese'
    input = 'EPU_Noise_Examples.json'
    input_path = input_dir / input
    # data = json.loads(input_path.read_text())[:100]

    if cfg.model.name == 'ChatGPT':
        clf = ChatGPT(
            prompt,
            key,
            cfg.model.model,
            cfg.model.temperature,
            cfg.model.batch_size,
            # data
            input_path
        )
    else:
        pass

    cfg_detail = OmegaConf.to_object(cfg)
    print("\nscrapping parameters:\n")
    print(OmegaConf.to_yaml(cfg_detail['model']))

    logger.info("start predicting")
    clf.predict()
    logger.info("start writing json output")
    clf.output(Path(cfg.data.predict) / f'{cfg.model.model}_{input}')
    logger.info("finish the process")


if __name__ == "__main__":
    main()
