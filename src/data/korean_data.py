from datetime import datetime, timedelta
import glob
from httpx import get
from loguru import logger
from os import path
import orjson
from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Dict
from tqdm import trange
import warnings


class Scrapper():
    def __init__(self,
                 headless: bool,
                 timeout: int,
                 output_dir: Path,
                 begin: Optional[str],
                 end: Optional[str],
                 interval: Optional[int]
                 ) -> None:
        self.begin = begin
        self.end = end
        self.interval = interval if interval is not None else 1
        self.headless = headless
        self.timeout = timeout
        self.output_dir = output_dir

    @property
    def begin_date(self) -> datetime:
        if self.begin is not None:
            return datetime.strptime(self.begin, '%Y-%m-%d')
        else:
            return datetime.strptime('1990-01-01', '%Y-%m-%d')

    @property
    def end_date(self) -> datetime:
        if self.end is not None:
            return datetime.strptime(self.end, '%Y-%m-%d')
        else:
            return datetime.strptime('2023-07-29', '%Y-%m-%d')

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

    def create_page(self) -> None:
        p = sync_playwright().start()
        browser = p.chromium.launch(headless=self.headless)
        page = browser.new_page()
        self.page = page

        self.page.goto(
            'https://www.bigkinds.or.kr/v2/news/index.do',
            timeout=self.timeout
        )

    def login(self, email: str, password: str):
        # note: do not use: button[type=button]
        self.page.click('div.login-area')
        self.page.fill('input#login-user-id', email)
        self.page.fill('input#login-user-password', password)
        self.page.click('button[type=submit]')

    def __input_period(self, begin, end):
        self.page.fill('input#search-begin-date', begin)
        self.page.fill('input#search-end-date', end)
        self.page.click('button.news-report-search-btn')

    def __trace(self, temp_folder, press) -> None:
        self.page.click('a:has-text("기간")')
        for t in trange(len(self.period['begin'])):
            # select the period
            begin = self.period['begin'][t]
            end = self.period['end'][t]
            self.__input_period(begin, end)

            # check if the number of articles exceed 20,000
            num_article = self.page \
                .locator('span.total-news-cnt') \
                .first \
                .inner_text()
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
        logger.info("start downloading")
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
        self.__trace(temp_folder, press)

    def download_by_multi_press(self, batch_press: List[str]) -> None:
        logger.info("start downloading")
        temp_folder = self.output_dir / 'temp'
        if temp_folder.exists():
            rmtree(temp_folder)
        else:
            temp_folder.mkdir(parents=True, exist_ok=True)

        # select the press
        for press in batch_press:
            self.page.click(f'label:has-text("{press}")')
            # trace and dowload
        self.__trace(temp_folder, '_'.join(batch_press))

    def collect_data_id(self, press: str | List[str]) -> List[str]:
        logger.info("start collecting data id")

        match press:
            case str():
                self.page.click(
                    f'label:has-text("{press}")',
                    timeout=self.timeout
                )
            case [*_]:
                for p in press:
                    self.page.click(
                        f'label:has-text("{p}")',
                        timeout=self.timeout
                    )

        self.page.click('a:has-text("기간")')
        if self.begin is None and self.end is None:
            self.page.click('label:has-text("전체")')
            self.page.click('button.news-report-search-btn')
        else:
            self.__input_period(
                self.__datetime_to_str(self.begin_date),
                self.__datetime_to_str(self.end_date)
            )

        num_page = self.page.locator('div.lastNum') \
            .first                                  \
            .get_attribute('data-page')

        data_id = []
        for i in trange(int(num_page)):
            self.page                     \
                .locator('div.news-item') \
                .first                    \
                .wait_for(timeout=self.timeout)
            data_id += [
                item.get_attribute('data-id')
                for item in self.page.locator('div.news-item').all()
            ]
            self.page.fill('input#paging_news_result', str(i+2))

        return data_id

    def get_news_request(self, data_id: str):

        url = "https://www.bigkinds.or.kr/news/detailView.do"

        querystring = {
            "docId": f"{data_id}",
            "returnCnt": "1",
            "sectionDiv": "1000"
        }

        # payload = {
        #     "indexName": "news",
        #     "searchKey": "",
        #     "searchKeys": [{}],
        #     "byLine": "",
        #     "searchFilterType": "1",
        #     "searchScopeType": "1",
        #     "searchSortType": "date",
        #     "sortMethod": "date",
        #     "mainTodayPersonYn": "",
        #     "startDate": "2023-08-06",
        #     "endDate": "2023-08-07",
        #     "newsIds": [],
        #     "categoryCodes": [],
        #     "providerCodes": [f"{id1}"],
        #     "incidentCodes": [],
        #     "networkNodeType": "",
        #     "topicOrigin": "",
        #     "dateCodes": [],
        #     "editorialIs": False,
        #     "startNo": 1,
        #     "resultNumber": 10,
        #     "isTmUsable": False,
        #     "isNotTmUsable": False
        # }

        headers = {
            "cookie": "Bigkinds=0BB26B74A16385BAE58D8B7A54652C20",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Content-Type": "application/json;charset=UTF-8",
            "Cookie": "Bigkinds=121EC1E4BA8514C21E96D482B6B00CB6; _ga_QWY27BS8JM=GS1.1.1691419850.4.0.1691419850.60.0.0; _ga=GA1.3.379045186.1689940956; _gid=GA1.3.423302910.1691419875",
            "Origin": "https://www.bigkinds.or.kr",
            "Referer": "https://www.bigkinds.or.kr/v2/news/index.do",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "sec-ch-ua": "'Not/A)Brand';v='99', 'Google Chrome';v='115', 'Chromium';v='115'",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "''macOS'"
        }

        response = get(
            url,
            headers=headers,
            params=querystring
        )

        return orjson.loads(response.text)['detail']

    def __parse_sim(self, sim: str) -> Dict[str, float]:
        res = {}
        for word_score in sim.split(' OR '):
            word, score = word_score.split('^')
            res[word] = float(score)

        return res

    def get_news_request_batch(self, batch_data_id: List[str], label: str) -> None:
        res = []
        logger.info("parse the request")
        for id in batch_data_id:
            response = self.get_news_request(id)
            res.append({
                'title': response['TITLE'],
                'content': response['CONTENT'],
                'date': response['DATE'],
                'location': response['TMS_NE_LOCATION'].split('\n'),
                'category': response['CATEGORY_MAIN'].split('>'),
                'relevant': self.__parse_sim(response['TMS_SIMILARITY'])
            })

        res_file = '_'.join((
            label,
            self.__datetime_to_str(self.begin_date),
            self.__datetime_to_str(self.end_date),
        )) + '.json'
        with open(self.output_dir / res_file, 'wb') as f:
            f.write(orjson.dumps(res, option=orjson.OPT_APPEND_NEWLINE))

    def merge(self, label) -> None:
        logger.info("start merging files")
        warnings.simplefilter("ignore")
        temp_dir = self.output_dir / 'temp'
        filenames = glob.glob(path.join(temp_dir, "*.xlsx"))

        df_list = [
            pd.read_excel(file, sheet_name="sheet")
            for file in filenames
        ]
        df = pd.concat(df_list, ignore_index=True).sort_values(by='일자')

        res_file = '_'.join((
            label,
            self.__datetime_to_str(self.begin_date),
            self.__datetime_to_str(self.end_date),
        )) + '.csv'

        df.to_csv(self.output_dir / res_file, index=False)

        rmtree(temp_dir, ignore_errors=True)
        logger.info("finish the process")
