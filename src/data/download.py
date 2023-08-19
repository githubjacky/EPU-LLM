import calendar
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
import hydra
from os import getenv
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, ListConfig

from HttpxScraper import HttpxScraper
from PlayWrightScraper import PlayWrightScraper


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

    cfg_detail = OmegaConf.to_object(cfg)
    mode = cfg_detail['process']['mode']

    match cfg.process.mode:
        case 'playwright':
            print()
            print(OmegaConf.to_yaml(cfg_detail['process'][mode]))

            email, password = env_setup()
            scraper = PlayWrightScraper(
                cfg.process.playwright.begin_date,
                cfg.process.playwright.end_date,
                cfg.process.playwright.interval,
                cfg.process.playwright.headless,
                cfg.process.playwright.timeout,
                output_dir,
            )
            scraper.create_page()
            scraper.login(email, password)

            match cfg.process.press:
                case ListConfig():
                    prev_press = None
                    for press in cfg.process.press:
                        scraper.download_by_press(press, prev_press)
                        scraper.merge(press)
                        prev_press = press
                case str():
                    batch_press = cfg.process.press.split('_')
                    scraper.download_by_multi_press(batch_press)
                    scraper.merge('_'.join(batch_press))
        case 'httpx':
            print()
            print(OmegaConf.to_yaml(cfg_detail['process'][mode]))

            if cfg.process.httpx.month is not None:
                begin = cfg.process.httpx.month + '-01'
                begin_date = datetime.strptime(begin, '%Y-%m-%d')
                days = calendar.monthrange(
                    begin_date.year,
                    begin_date.month
                )[1]
                end = cfg.process.httpx.month + f'-{days}'
            else:
                begin = cfg.process.httpx.begin_date
                end = cfg.process.httpx.end_date

            scraper = HttpxScraper(
                begin,
                end,
                cfg.process.httpx.interval,
                cfg.process.httpx.timeout,
                cfg.process.httpx.proxy,
                cfg.process.httpx.async_max_rate,
                cfg.process.httpx.async_time_period,
                output_dir,
            )
            match cfg.process.httpx.press:
                case ListConfig():
                    for press in cfg.process.httpx.press:
                        data_id_list_cluster = scraper.collect_data_id(press)
                        scraper.collect_news(press, data_id_list_cluster)

                case str():
                    data_id_list_cluster = scraper.collect_data_id(
                        cfg.process.httpx.press.split('_')
                    )
                    scraper.collect_news(
                        cfg.process.httpx.press,
                        data_id_list_cluster
                    )


if __name__ == "__main__":
    main()
