import requests
import re
from bs4 import BeautifulSoup
import time
from pathlib import Path
import multiprocessing.dummy as mp
import PyPDF2
from io import BytesIO

OUTDIR = Path("data/arxivpapers")


def get_url(url, header=None, cookies: dict | None = None):
    if header is None:
        header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
        }
    if cookies is not None:
        header["Cookie"] = ";".join([f"{k}={v}" for k, v in cookies.items()])
    ree = ""
    for i in range(3):
        try:
            re = requests.get(url, timeout=5, headers=header, cookies=cookies)
            return re
        except Exception as e:
            ree = str(e)
            time.sleep(1)
    raise Exception("Cannot get " + url + " " + ree)


def get_urls(homeurl: str):
    re = get_url(homeurl)
    soup = BeautifulSoup(re.content, "lxml")
    ol = soup.select_one("ol.breathe-horizontal")
    assert ol
    urls = []
    for li in ol.find_all("li"):
        a = li.select_one(
            "div:nth-child(1) > p:nth-child(1) > span:nth-child(2) > a:nth-child(1)"
        )
        if a:
            urls.append(a["href"])
    return urls

def pdf_to_text(pdf:bytes):
    pdfio = BytesIO(pdf)
    pdfObj = PyPDF2.PdfReader(pdfio)
    txt = []
    for p in pdfObj.pages:
        txt.append(p.extract_text())
    return "\n".join(txt)

urlp = re.compile(r'^https://arxiv.org/pdf/(.+)$')

def get_pdf(url:str):
    name = urlp.match(url)
    assert name
    name = name.group(1)
    path = OUTDIR / (name + ".txt")
    if path.exists():
        return
    try:
        re = get_url(url)
        text = pdf_to_text(re.content)
        path.write_text(text)
        print(f"Downloaded {name}")
    except Exception as e:
        print(f"Failed to download {name} {e}")

OUTDIR.mkdir(exist_ok=True)

urls = []
for page in range(10):
    url = 'https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=llm&terms-0-field=title&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=hide&size=200&order='
    if page > 0:
        url+='&start='+str(page*200)
    print(f"Getting urls from {url}")
    urls += get_urls(url)

print(f"Got {len(urls)} urls")
pool = mp.Pool(32)
pool.map(get_pdf, urls)
pool.close()
