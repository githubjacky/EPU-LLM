from datetime import datetime, timedelta
from dotenv import load_dotenv
import glob
import hydra
from loguru import logger
from os import getenv, path
from omegaconf import DictConfig, OmegaConf
from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path
from shutil import rmtree
from typing import List
from tqdm import trange
import warnings


class Scrapper():
    def __init__(self,
                 begin: str,
                 end: str,
                 interval: int,
                 headless: bool,
                 timeout: int,
                 output_dir: Path
                 ) -> None:
        self.begin = begin
        self.end = end
        self.interval = interval
        self.headless = headless
        self.timeout = timeout
        self.output_dir = output_dir

    @property
    def begin_date(self) -> datetime:
        return datetime.strptime(self.begin, '%Y-%m-%d')

    @property
    def end_date(self) -> datetime:
        return datetime.strptime(self.end, '%Y-%m-%d')

    def __add_zero(self, x: int) -> str:
        str_x = str(x)
        return "0"+str_x if len(str_x) == 1 else str_x

    def __datetime_to_str(self, d: datetime) -> str:
        return f"{d.year}-{self.__add_zero(d.month)}-{self.__add_zero(d.day)}"

    @property
    def period(self) -> dict[str, List[str]]:
        days = (self.end_date - self.begin_date).days + 1
        rem = days % self.interval
        valid = rem == 0
        t = days // self.interval if valid else days // self.interval + 1

        period_dict = {'begin': [], 'end': []}
        begin = self.end_date - timedelta(days=self.interval-1)
        end = self.end_date
        for _ in range(t):
            period_dict['begin'].append(self.__datetime_to_str(begin))
            period_dict['end'].append(self.__datetime_to_str(end))

            begin -= timedelta(days=self.interval)
            end -= timedelta(days=self.interval)

        return period_dict

    def create_page(self):
        p = sync_playwright().start()
        browser = p.chromium.launch(headless=self.headless)
        page = browser.new_page()
        self.page = page

        self.page.goto('https://www.bigkinds.or.kr/v2/news/index.do')

    def login(self, email: str, password: str):
        # note: do not use: button[type=button]
        self.page.click('div.login-area')
        self.page.fill('input#login-user-id', email)
        self.page.fill('input#login-user-password', password)
        self.page.click('button[type=submit]')

    def trace(self, temp_folder, press):
        self.page.click('a:has-text("기간")')
        for t in trange(len(self.period['begin'])):
            # select the period
            begin = self.period['begin'][t]
            end = self.period['end'][t]
            self.page.fill('input#search-begin-date', begin)
            self.page.fill('input#search-end-date', end)
            self.page.click('button.news-report-search-btn')

            # check if the number of articles exceed 20,000
            num_article = self.page.locator(
                '#news-results-tab > div:nth-child(3) > h3 > span.total-news-cnt'
            ).inner_text()
            if int(num_article.replace(',', '')) > 20000:
                logger.warning("the number of articles exceed the limit")

            # download
            self.page.click('button#collapse-step-3')
            res_path = temp_folder / f'{begin}_{end}_{press}.xlsx'
            with self.page.expect_download(timeout=self.timeout) as d:
                self.page.click(
                    '#analytics-data-download > div.btm-btn-wrp > button'
                )
            download = d.value
            download.save_as(res_path)

            # rerun the whole process
            self.page.click('button#collapse-step-1')

    def download_by_press(self, press: str, prev_press=None) -> None:
        temp_folder = self.output_dir / 'temp'
        if temp_folder.exists():
            rmtree(temp_folder)
        else:
            temp_folder.mkdir(parents=True, exist_ok=True)

        # select the press
        if prev_press is not None:
            self.page.click('a:has-text("언론사")')
            self.page.click(f'label:has-text("{prev_press}")')
        self.page.click(f'label:has-text("{press}")')

        # trace and dowload
        self.trace(temp_folder, press)

    def download_by_multi_press(self, batch_press: List[str]) -> None:
        temp_folder = self.output_dir / 'temp'
        if temp_folder.exists():
            rmtree(temp_folder)
        else:
            temp_folder.mkdir(parents=True, exist_ok=True)

        # select the press
        for press in batch_press:
            self.page.click(f'label:has-text("{press}")')
            # trace and dowload
        self.trace(temp_folder, '_'.join(batch_press))

    def merge(self, label) -> None:
        warnings.simplefilter("ignore")
        temp_dir = self.output_dir / 'temp'
        filenames = sorted(glob.glob(path.join(temp_dir, "*.xlsx")))
        df = pd.DataFrame()

        for file in filenames:
            df = pd.concat(
                [
                    df,
                    pd.read_excel(file, sheet_name="sheet", engine='openpyxl')
                ],
                ignore_index=True,
                sort=False
            )

        res_file = '_'.join((
            label,
            self.__datetime_to_str(self.begin_date),
            self.__datetime_to_str(self.end_date),
        )) + '.xlsx'

        df.sort_values(by=['일자']).to_excel(
            self.output_dir / res_file,
            index=False
        )

        rmtree(temp_dir, ignore_errors=True)


def env_setup() -> tuple[str, str]:
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


@hydra.main(config_path='../../config', config_name='main', version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("setting up the environment variable")
    email, password = env_setup()

    output_dir = Path(cfg.data.raw) / 'bigkinds'
    output_dir.mkdir(parents=True, exist_ok=True)

    scrapper = Scrapper(
        cfg.process.begin_date,
        cfg.process.end_date,
        cfg.process.interval,
        cfg.process.headless,
        cfg.process.timeout,
        output_dir
    )

    cfg_detail = OmegaConf.to_object(cfg)
    print("\nscrapping parameters:\n")
    print(OmegaConf.to_yaml(cfg_detail['process']))

    logger.info("start scrapping")
    scrapper.create_page()
    scrapper.login(email, password)

    if cfg.process.method == 'by_press':
        prev_press = None
        for press in cfg.process.press.split('_'):
            scrapper.download_by_press(press, prev_press)
            logger.info("start merging files")
            scrapper.merge(press)
            logger.info("finish the process")
            prev_press = press
    else:
        batch_press = cfg.process.press.split('_')
        scrapper.download_by_multi_press(batch_press)
        logger.info("start merging files")
        scrapper.merge('_'.join(batch_press))
        logger.info("finish the process")


if __name__ == "__main__":
    main()
