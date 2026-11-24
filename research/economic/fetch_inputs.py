"""Fetch only public economic data. Preserve exact bytes and publication vintages."""
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "raw"
SERIES = "BAMLH0A0HYM2"
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

def origins():
    for year in range(2018, 2026):
        for month in range(1, 13):
            if (year,month) > (2025,1):
                return
            start = date(year,month,1)
            yield start + timedelta(days=(-start.weekday()) % 7)

def fetch(desc):
    name,url,header=desc
    path=DATA/name
    tries=[]
    for attempt in range(2):
        try:
            if path.exists():
                body=path.read_bytes()
            else:
                req=Request(url,headers={"User-Agent":"Mozilla/5.0 (economic research; historical series)"})
                with urlopen(req,timeout=17) as response:
                    body=response.read(8_000_001)
            if len(body)>8_000_000:
                raise ValueError('size limit')
            r=csv.DictReader(io.StringIO(body.decode('utf-8-sig')))
            if header not in (r.fieldnames or []):
                raise ValueError('missing requested vintage/column')
            if len(list(r))<150:
                raise ValueError('insufficient rows')
            path.write_bytes(body)
            return {"file":name,"url":url,"sha256":hashlib.sha256(body).hexdigest()}
        except Exception as e:
            tries.append(f"{type(e).__name__}: {e}")
            if isinstance(e, HTTPError) and e.code == 404:
                break
    return {"file":name,"url":url,"unavailable":tries}

def main():
    DATA.mkdir(exist_ok=True)
    inputs=[('vix.csv',CBOE,'HIGH')]
    for o in origins():
        v=o-timedelta(days=2)
        args=urlencode({'id':SERIES,'cosd':(v-timedelta(days=810)).isoformat(),'vintage_date':v.isoformat()})
        inputs.append((f'{v}.csv','https://alfred.stlouisfed.org/graph/alfredgraph.csv?'+args,SERIES+'_'+v.strftime('%Y%m%d')))
    # On 2026-11-24 ALFRED's ICE series documentation restricted retrieval to a
    # rolling three years; the observed earliest downloadable value was 2023-11-27.
    # Earlier origin vintages cannot contain a single usable pre-origin observation.
    # Keep the denominator, URL and explicit reason rather than impute today's values.
    floor='2023-11-27'
    early=[x for x in inputs if x[0]!='vix.csv' and x[0][:10]<floor]
    available=[x for x in inputs if x not in early]
    with ThreadPoolExecutor(max_workers=4) as pool:
        items=list(pool.map(fetch,available))
    items.extend({'file':name,'url':u,'unavailable':['before observed 2023-11-27 ICE archive coverage on study date']} for name,u,_ in early)
    items.sort(key=lambda item:item['file'])
    (ROOT/'input_manifest.json').write_text(json.dumps(items,indent=2)+'\n')
    print('Verified economic sources:',sum('sha256' in x for x in items),'of',len(items))
    print('Example errors:',[(x['file'],x['unavailable']) for x in items if 'unavailable' in x][:3])

if __name__=='__main__': main()
