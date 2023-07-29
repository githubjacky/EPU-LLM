from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
from os import getenv, path
from typing import List
from argparse import ArgumentParser, Namespace
from datetime import datetime, timedelta
from tqdm import trange
from creat_env import write_env
import pandas as pd
import glob
from pathlib import Path
import warnings
from shutil import rmtree


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '--press',
        default="매일경제신문_한국경제",
        type=lambda s: s.split('_'),
    )
    parser.add_argument(
        '--begin_date',
        default="2023-07-26",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
    )
    parser.add_argument(
        '--end_date',
        default="2023-07-29",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
    )

    parser.add_argument(
        '--interval',
        default=2,
        type=int,
    )
    args = parser.parse_args()
    return args


def join_press(press_list: List[str]):
    res = "("
    for i in range(len(press_list)-1):
        res += (f"({press_list[i]})OR")

    res += f"({press_list[-1]}))"

    return res


def add_zero(x):
    str_x = str(x)
    return "0"+str_x if len(str_x) == 1 else str_x


def datetime_to_str(d: datetime):
    return f"{d.year}-{add_zero(d.month)}-{add_zero(d.day)}"


def create_period(begin_date: datetime, end_date: datetime, interval: int):
    days = (end_date - begin_date).days + 1
    rem = days % interval
    valid = rem == 0
    num_period = days // interval if valid else days // interval + 1

    begin_date_list = []
    end_date_list = []
    begin = end_date - timedelta(days=interval-1)
    end = end_date
    for _ in range(num_period):
        begin_date_list.append(datetime_to_str(begin))
        end_date_list.append(datetime_to_str(end))

        begin -= timedelta(days=interval)
        end -= timedelta(days=interval)

    return begin_date_list, end_date_list


def download(output_dir: Path,
             email: str,
             password: str,
             keyword: str,
             begin_date_list: List[str],
             end_date_list: List[str]
             ):

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto('https://www.bigkinds.or.kr/v2/news/index.do')

        # step1:login
        page.click('div.login-area')  # note: do not use: button[type=button]
        page.fill('input#login-user-id', email)
        page.fill('input#login-user-password', password)
        page.click('button[type=submit]')

        # step2-1: search with press as keywords
        page.fill('input#total-search-key', keyword)
        page.click('a:has-text("기간")')

        for t in trange(len(begin_date_list)):
            begin_date = begin_date_list[t]
            end_date = end_date_list[t]

            # step2-1: select the period
            page.fill('input#search-begin-date', begin_date)
            page.fill('input#search-end-date', end_date)
            page.click('button.news-report-search-btn')

            # step3: download(try 5 times at most)
            attempt = 0
            download_info = None
            page.click('button#collapse-step-3')
            while attempt < 5:
                try:
                    with page.expect_download(timeout=3000000) as download_info:
                        page.click(
                            'button.mobile-excel-download',
                            timeout=3000000
                        )
                    break
                except TimeoutError:
                    attempt += 1

            if download_info is not None:
                download = download_info.value
                download.save_as(
                    f'./{output_dir}/{keyword}_{begin_date}_{end_date}.xlsx'
                )

            # step4: rerun the whole process
            page.click('button#collapse-step-1')


def main(args):
    begin_date_list, end_date_list = create_period(
        args.begin_date, args.end_date, args.interval
    )

    load_dotenv()
    email = getenv("EMAIL")
    password = getenv("PASSWORD")
    if (email is not None) and (password is not None):
        pass
    else:
        email = input("input your email address of BigKinds: ")
        password = input("input your password of BigKinds: ")
        write_env(email, password)
        load_dotenv()
        email = getenv("EMAIL")
        password = getenv("PASSWORD")

    output_dir = Path('./data/scrap_bigkinds_temp')
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        rmtree(output_dir)
    keyword = join_press(args.press)
    download(
        output_dir,
        email,
        password,
        keyword,
        begin_date_list,
        end_date_list
    )

    # merge all file
    warnings.simplefilter("ignore")
    filenames = sorted(glob.glob(path.join(output_dir, "*.xlsx")))
    df = pd.DataFrame()
    dir = Path('./data/scrap_bigkinds')
    dir.mkdir(parents=True, exist_ok=True)

    for file in filenames:
        df = pd.concat(
            [df, pd.read_excel(
                file, sheet_name="sheet", engine='openpyxl')],
            ignore_index=True,
            sort=False
        )
    df.sort_values(by=['일자']).to_excel(
        f'{dir}/{keyword}_{datetime_to_str(args.begin_date)}_{datetime_to_str(args.end_date)}.xlsx',
        index=False
    )

    rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main(parse_args())
