from playwright.sync_api import sync_playwright
from selectolax.parser import HTMLParser
from dotenv import load_dotenv
from time import sleep


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        page = browser.new_page()
        page.goto('https://www.bigkinds.or.kr/v2/news/index.do')
        page.click('buttion[type=buttion]')
        sleep(3)


if __name__ == "__main__":
    main()
