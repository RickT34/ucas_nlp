import requests
import re
from bs4 import BeautifulSoup
import time
from datetime import date, timedelta
from pathlib import Path
import multiprocessing.dummy as mp
import pickle

OUTDIR = Path("data/ppnews")

def dates_range(start_date, end_date):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += timedelta(days=1)

def get_url(url, header=None, cookies:dict|None=None):
    if header is None:
        header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
        }
    if cookies is not None:
        header["Cookie"] = ";".join([f"{k}={v}" for k,v in cookies.items()])
    ree = ""
    for i in range(3):
        try:
            re = requests.get(url, timeout=2, headers=header, cookies=cookies)
            return re
        except Exception as e:
            ree = str(e)
            time.sleep(1)
    raise Exception("Cannot get " + url + " " + ree)

# http://paper.people.com.cn/rmrb/pc/content/202503/17/content_30062224.html
# http://paper.people.com.cn/rmrb/html/2023-03/08/nw.D110000renmrb_20230308_1-01.htm

url_pattern = re.compile(r'(.*)content_(\d+).html$')

def get_news(url:str):
    re = get_url(url)
    soup = BeautifulSoup(re.content, 'lxml')
    atc = soup.select_one('.article')
    assert atc, "Cannot find article: "+url
    return atc.text.strip().replace('\u3000', '')

def get_news_of_day(start_url:str):
    pt = url_pattern.match(start_url)
    assert pt, "Invalid start_url: "+start_url
    base_url = pt.group(1)
    id = int(pt.group(2))
    news = []
    while True:
        url = base_url + "content_" + str(id) + ".html"
        try:
            text = get_news(url)
            news.append(text)
            id += 1
        except:
            break
    return news

# http://paper.people.com.cn/rmrb/pc/layout/202503/18/node_01.html
# http://paper.people.com.cn/rmrb/html/2023-03/08/nbs.D110000renmrb_01.htm
def get_start_url(year:int, month:int, day:int):
    ymdstr = f"{str(year)+str(month).zfill(2)}/{str(day).zfill(2)}/"
    url = f"http://paper.people.com.cn/rmrb/pc/layout/{ymdstr}node_01.html"
    re = get_url(url)
    soup = BeautifulSoup(re.content, 'lxml')
    start_url_a = soup.select_one(".news-list > li:nth-child(1) > a:nth-child(2)")
    if start_url_a:
        urlr = start_url_a["href"]
        assert isinstance(urlr, str)
        # ../../../content/202503/18/content_30062409.html
        pt = url_pattern.match(urlr)
        assert pt, "Invalid start_url: "+urlr
        id = pt.group(2)
        return f"http://paper.people.com.cn/rmrb/pc/content/{ymdstr}content_{id}.html"
    else:
        return None
    
# http://paper.people.com.cn/rmrb/html/2023-03/08/nw.D110000renmrb_20230308_1-01.htm
def get_news_of_day_old(year:int, month:int, day:int):
    i = 1
    news = []
    url = None
    while True:
        j = 1
        while True:
            url = f"http://paper.people.com.cn/rmrb/html/{str(year)}-{str(month).zfill(2)}/{str(day).zfill(2)}/nw.D110000renmrb_{str(year)}{str(month).zfill(2)}{str(day).zfill(2)}_{j}-{str(i).zfill(2)}.htm"
            re = get_url(url)
            soup = BeautifulSoup(re.content, 'lxml')
            atc = soup.select_one('.article')
            if not atc:
                break
            txt = atc.text.strip().replace('\u3000', '')
            news.append(txt)
            j += 1
        if j == 1:
            break
        i += 1
    # assert len(news) > 0, "No news found: "+url
    return news

def get_news_of_day_by_date(dat:date):
    path = OUTDIR / f"{dat.strftime('%Y-%m-%d')}.pkl"
    if path.exists():
        return
    re = None
    if dat >= date(2024, 12, 1):
        start_url = get_start_url(dat.year, dat.month, dat.day)
        assert start_url, f"No start_url found for {dat.strftime('%Y-%m-%d')}"
        re = get_news_of_day(start_url)
    else:
        re = get_news_of_day_old(dat.year, dat.month, dat.day)
    if len(re) == 0:
        print(f"{dat.strftime('%Y-%m-%d')} no news found")
        return
    pickle.dump(re, open(path, "wb"))
    print(f"{dat.strftime('%Y-%m-%d')} done, len={len(re)}")


# print(get_news_of_day("http://paper.people.com.cn/rmrb/pc/content/202503/17/content_30062224.html"))
# print(get_start_url(2025, 3, 18))

if __name__ == "__main__":
    OUTDIR.mkdir(exist_ok=True)
    p = mp.Pool(32)
    rg = dates_range(date(2021, 1, 1), date(2025,3,20))
    p.map(get_news_of_day_by_date, rg)
    p.close()
