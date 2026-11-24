"""Closed 2024–2025 ECB policy releases, and the official effective-date schedule."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
import re
import time

RELEASES = '''240125~f738889bde 240307~a5fa52b82b 240411~1345644915 240606~2148ecdb3c
240718~b9e0ddd9d5 240912~67cb23badb 241017~aa366eaf20 241212~2acab6e51e
250130~530b29e622 250306~d4340800b3 250417~42727d0735 250605~3b5f67d007
250724~50bc70e13f 250911~6afb7a9490 251030~cf0540b5c0 251218~58b0e415a6'''.split()
LANGUAGES = ('en', 'es', 'pt')
RATES = 'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html'


def links():
    return [f'https://www.ecb.europa.eu/press/pr/date/20{release[:2]}/html/ecb.mp{release}.{lang}.html'
            for release in RELEASES for lang in LANGUAGES]


def hash_bytes(data):
    return sha256(data).hexdigest()


def fetch(cache, url):
    path = cache / (hash_bytes(url.encode()) + '.html')
    if not path.exists():
        for attempt in range(3):
            try:
                req = Request(url, headers={'User-Agent': 'MEGALAB closed-releases research/1'})
                with urlopen(req, timeout=25) as response:
                    if response.url != url:
                        raise ValueError(f'unexpected redirect: {url} -> {response.url}')
                    data = response.read()
                path.write_bytes(data)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    return path.read_text(encoding='utf-8')


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_main = False
        self.parts = None
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == 'main':
            self.in_main = True
        if self.in_main and tag == 'p' and not dict(attrs).get('class'):
            self.parts = []

    def handle_endtag(self, tag):
        if tag == 'main':
            self.in_main = False
        if tag == 'p' and self.parts is not None:
            value = ' '.join(''.join(self.parts).split())
            if value:
                self.paragraphs.append(value)
            self.parts = None

    def handle_data(self, content):
        if self.parts is not None:
            self.parts.append(content)


def paragraphs(html):
    p = Paragraphs()
    p.feed(html)
    if len(p.paragraphs) < 2:
        raise ValueError('release body is absent')
    return p.paragraphs


def get_all(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    urls = links() + [RATES]
    with ThreadPoolExecutor(max_workers=4) as pool:
        documents = dict(zip(urls, pool.map(lambda url: fetch(cache, url), urls)))
    return documents


if __name__ == '__main__':
    import sys
    docs = get_all(sys.argv[1])
    for url in links():
        pars = paragraphs(docs[url])
        print(url.rsplit('/', 1)[-1], len(pars), len(' '.join(pars)), ' / '.join(p for p in pars if re.search(r'\d[.,]\d{2}\s*%', p))[:290])
