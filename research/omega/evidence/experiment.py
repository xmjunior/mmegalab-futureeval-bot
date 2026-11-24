"""Reproduce the ECB provenance, novelty and resolution-criteria falsification."""

import argparse
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit
import zlib

from sources import LANGUAGES, RELEASES, RATES, get_all, links, paragraphs

ROOT = Path(__file__).resolve().parent
SLUG = re.compile(r'/press/pr/date/(20\d\d)/html/ecb\.(mp(\d{6})~[a-f0-9]{10})\.(en|es|pt)\.html')
NUMBER = re.compile(r'(?<![\d.])\d+[.,]\d\d\s*%')
MONTHS = {name: i for i, name in enumerate('January February March April May June July August September October November December'.split(), 1)}


def sha(text):
    return sha256(text.encode()).hexdigest()


def detail(url):
    parts = urlsplit(url)
    if parts.scheme != 'https' or parts.netloc != 'www.ecb.europa.eu':
        return None
    match = SLUG.fullmatch(parts.path)
    if not match or match[1][2:] != match[3][:2]:
        return None
    return match[2], datetime.strptime(match[3], '%y%m%d').date(), match[4]


def norm(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def ncd(a, b):
    a, b = a.encode(), b.encode()
    ca, cb = [len(zlib.compress(data, 9)) for data in (a, b)]
    joint = min(len(zlib.compress(x + b'\0' + y, 9)) for x, y in [(a, b), (b, a)])
    return (joint - min(ca, cb)) / max(ca, cb)


def shingles(text):
    tokens = text.split()
    return set(zip(*(tokens[i:] for i in range(5))))


def fact(body):
    for p in body:
        if not re.search(r'deposit facility|facilidad de dep[óo]sito|facilidade permanente de dep[óo]sito', p, re.I):
            continue
        rates = NUMBER.findall(p)
        if len(rates) == 3:
            values = sorted(Decimal(x.strip(' %').replace(',', '.')) for x in rates)
            return [str(x) for x in values], p
    return None, ''


def decision_effective(paragraph, meeting):
    match = re.search(r'(?:effect from|effective from) (\d{1,2}) ([A-Za-z]+) (20\d\d)', paragraph)
    if match:
        return date(int(match[3]), MONTHS[match[2]], int(match[1]))
    if 'unchanged' in paragraph.casefold():
        return meeting
    return None


def provenance(url_a, url_b, fact_a, fact_b):
    a, b = detail(url_a), detail(url_b)
    if not a or not b:
        return 'unknown'
    if a[0] != b[0]:
        return 'distinct'
    if not fact_a or not fact_b:
        return 'unverified'
    return 'same_release' if fact_a == fact_b else 'conflict'


def schedule(html):
    table = html.split('Date (with effect from)</th>', 1)[1].split('</table>', 1)[0]
    rows = []
    for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>', table, re.S):
        tr = re.sub(r'<sup>.*?</sup>', '', tr)
        cells = [re.sub('<[^>]+>', '', x).strip() for x in re.findall(r'<td\b[^>]*>(.*?)</td>', tr, re.S)]
        if len(cells) != 6 or cells[0] not in ['2023', '2024', '2025']:
            continue
        day, month = cells[1].replace('.', '').split()
        month_no = next(v for k, v in MONTHS.items() if k.startswith(month))
        rows.append((date(int(cells[0]), month_no, int(day)), str(Decimal(cells[2]))))
    return sorted(rows)


def scores(rows, key):
    c = Counter(('tp' if r['same'] else 'fp') if r[key] else ('fn' if r['same'] else 'tn') for r in rows)
    return dict((k, c[k]) for k in ['tp', 'fp', 'fn', 'tn'])


def evaluate(measurements, design):
    report = {}
    for year in (2024, 2025):
        year_rows = [r for r in measurements['pairs'] if r['year'] == year]
        rows = []
        for r in year_rows:
            r = dict(r, ncd_echo=r['ncd'] <= design['numerical_thresholds']['ncd'], shingle_echo=r['jaccard'] >= design['numerical_thresholds']['shingle_jaccard'], official_echo=r['status'] == 'same_release', url_echo=False)
            rows.append(r)
        criteria = [r for r in measurements['resolution'] if r['meeting'].startswith(str(year))]
        report[str(year)] = {
            'editions':sum(1 for s in measurements['sources'] if s['meeting'].startswith(str(year))),
            'meetings':len(criteria), 'pairs':len(rows), 'confusion':{k:scores(rows,k) for k in ['url_echo','ncd_echo','shingle_echo','official_echo']},
            'distinct_different_rate_false_ncd_echoes':sum(r['different_rate'] and r['ncd_echo'] for r in rows),
            'example_different_rate_false_ncd_echo':next(([r['left'],r['right'],r['ncd']] for r in rows if r['different_rate'] and r['ncd_echo']),None),
            'cut_count':sum(r['cut'] for r in criteria),
            'announced_only_correct':sum(r['announced'] == r['gold'] for r in criteria),
            'effective_dated_correct':sum(r['rule'] == r['gold'] for r in criteria),
            'adverse_subset_correct':{k:sum(r['cut'] and r[k] == r['gold'] for r in criteria) for k in ['announced','rule']},
        }
    return report


def collect(raw):
    output = {'sources': [], 'pairs': [], 'resolution': []}
    texts, dates = {}, {}
    effective_rates = schedule(raw[RATES])
    output['gold_schedule'] = [[d.isoformat(), v] for d,v in effective_rates]
    # Before the first study meeting the last effective rate was published in 2023.
    last_rate = max((d,rate) for d,rate in effective_rates if d < date(2024,1,1))[1]
    for url in links():
        release, day, lang = detail(url)
        body = paragraphs(raw[url]); rates, operative = fact(body)
        text = norm(' '.join(body)); name = f'{day}_{lang}'
        # Keep editions whose operative rate triple cannot be parsed; pairwise provenance abstains.
        metadata = {'id':name, 'url':url, 'meeting':day.isoformat(), 'language':lang, 'rates_low_to_high':rates, 'normalized_sha256':sha(text), 'tokens':len(text.split())}
        output['sources'].append(metadata)
        texts[name] = text, shingles(text)
        if lang != 'en':
            continue
        if not rates:
            raise ValueError(f'absent English settlement fields {day}')
        effective = decision_effective(operative, day)
        target = day + timedelta(days=1)
        gold = max((d,v) for d,v in effective_rates if d <= target)[1]
        forecast = (rates[0] if effective <= target else last_rate) if effective else None
        output['resolution'].append({'meeting':str(day),'target':str(target),'effective_in_release':str(effective) if effective else None,'old':last_rate,'announced':rates[0],'gold':gold,'rule':forecast,'cut':Decimal(rates[0]) < Decimal(last_rate)})
        last_rate = rates[0]
    for a,b in combinations(output['sources'],2):
        if a['meeting'][:4] != b['meeting'][:4]:
            continue
        x, sx = texts[a['id']]; y, sy = texts[b['id']]
        fa, fb = a['rates_low_to_high'],b['rates_low_to_high']
        output['pairs'].append({'left':a['id'],'right':b['id'],'year':int(a['meeting'][:4]),'same':a['meeting']==b['meeting'],'different_rate':bool(fa and fb and fa[0]!=fb[0]),'ncd':round(ncd(x,y),9),'jaccard':round(len(sx&sy)/len(sx|sy),9),'status':provenance(a['url'],b['url'],fa,fb)})
    first = output['sources'][0]; txt = texts[first['id']][0]
    changed_text = txt.replace('4 00', '4 01',1)
    if changed_text == txt:
        raise ValueError('mutation did not occur')
    rates = first['rates_low_to_high']; changed_rates = ['4.01']+rates[1:]
    output['adversarial_controls'] = {'one_digit_mutation_ncd':round(ncd(txt,changed_text),9),'same_id_different_rate':provenance(first['url'],first['url'],rates,changed_rates),'impostor_domain':provenance(first['url'],first['url'].replace('www.ecb.europa.eu','www.ecb.europa.eu.invalid'),rates,rates)}
    return output


def dump(value):
    return json.dumps(value,indent=2,ensure_ascii=False)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fetch',metavar='CACHE');p.add_argument('--write',action='store_true');args=p.parse_args()
    design=json.loads((ROOT/'design.json').read_bytes())
    if args.fetch:
        inputs=collect(get_all(args.fetch))
        text=dump(inputs)
        if args.write:
            (ROOT/'measurements.json').write_text(text)
        elif text != (ROOT/'measurements.json').read_text():
            raise SystemExit('Source/measurement drift. Investigate before replacing the frozen measurements.')
    else:
        inputs=json.loads((ROOT/'measurements.json').read_bytes())
    summary=evaluate(inputs,design)
    summary['synthetic_controls']=inputs['adversarial_controls']
    summary['design_sha256']=sha((ROOT/'design.json').read_text())
    summary['inputs_sha256']=sha((ROOT/'measurements.json').read_text())
    if args.write:
        (ROOT/'result.json').write_text(dump(summary))
    elif (ROOT/'result.json').read_text()!=dump(summary):
        raise SystemExit('Reproduction mismatch')
    print(dump(summary),end='')
