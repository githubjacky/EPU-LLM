from dotenv import load_dotenv
from loguru import logger
import hydra
from os import getenv
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, ListConfig

from korean_data import Scrapper


def env_setup() -> tuple[str, str]:
    logger.info("setting up the environment variable")
    load_dotenv()
    email = getenv("EMAIL")
    password = getenv("PASSWORD")
    if (email is None) or (password is None):
        email = input("input your email address of BigKinds: ")
        password = input("input your password of BigKinds: ")
        file = Path('.env')
        if file.is_file():
            with open(file, 'a') as f:
                f.write(f"\nEMAIL={email}\n")
                f.write(f"PASSWORD={password}")
        else:
            with open(file, 'w') as f:
                f.write(f"EMAIL={email}\n")
                f.write(f"PASSWORD={password}")

    return email, password


@ hydra.main(config_path='../../config', config_name='main', version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.data.raw) / 'bigkinds'
    output_dir.mkdir(parents=True, exist_ok=True)

    scrapper = Scrapper(
        cfg.process.headless,
        cfg.process.timeout,
        output_dir,
        cfg.process.begin_date,
        cfg.process.end_date,
        cfg.process.interval
    )

    cfg_detail = OmegaConf.to_object(cfg)
    print(OmegaConf.to_yaml(cfg_detail['process']))

    scrapper.create_page()

    match cfg.process.mode:
        case 'csv':
            email, password = env_setup()
            scrapper.login(email, password)
            match cfg.process.press:
                case ListConfig():
                    prev_press = None
                    for press in cfg.process.press:
                        scrapper.download_by_press(press, prev_press)
                        scrapper.merge(press)
                        prev_press = press
                case str():
                    batch_press = cfg.process.press.split('_')
                    scrapper.download_by_multi_press(batch_press)
                    scrapper.merge('_'.join(batch_press))
        case 'json':
            match cfg.process.press:
                case ListConfig():
                    for press in cfg.process.press:
                        data_id = scrapper.collect_data_id(press)
                        scrapper.get_news_request_batch(data_id, press)

                case str():
                    data_id = scrapper.collect_data_id(
                        cfg.process.press.split('_')
                    )
                    scrapper.get_news_request_batch(data_id, cfg.process.press)


if __name__ == "__main__":
    main()
