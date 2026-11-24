"""Read official public snapshots, issue a local forecast set and score proven on-time seals."""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from statistics import mean
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from model import HERE, canonical, horizon, metrics, prediction, sha

sys.path.insert(0, str(HERE.parent / 'forecastbench_export'))
from export import build, question_index

START = date(2026, 12, 6)
STOP = date(2027, 3, 31)
SCORE_END = date(2028, 3, 31)
UPSTREAM = 'forecastingresearch/forecastbench-datasets'
MODEL_SHA = '075db14a4e630609ed71eb2e701bf6c9664f62acadc4ca4bda4da1cabb7bd090'
UTC = timezone.utc


def now():
    return datetime.now(UTC)


def utc(value):
    moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if moment.tzinfo is None:
        raise ValueError('timestamp lacks timezone')
    return moment.astimezone(UTC)


def active(day):
    return START <= day <= STOP and (day - START).days % 14 == 0


def get(url, github_api=False):
    headers = {'User-Agent': 'MEGALAB-forecastbench-shadow-v1'}
    if github_api and os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['GITHUB_TOKEN']
    with urlopen(Request(url, headers=headers), timeout=12) as response:
        return response.read(16_000_001)


def snapshot(path, observed):
    query = urlencode({'sha':'main','path':path,'until':observed.isoformat(), 'per_page':1})
    history = json.loads(get('https://api.github.com/repos/' + UPSTREAM + '/commits?' + query, True))
    if not isinstance(history, list):
        raise ValueError('unexpected GitHub response')
    if not history:
        return None
    item = history[0]
    revision = item['sha']
    committed = utc(item['commit']['committer']['date'])
    if not re.fullmatch('[a-f0-9]{40}', revision) or committed > observed:
        raise ValueError('invalid or future upstream revision')
    url = 'https://raw.githubusercontent.com/' + UPSTREAM + '/' + revision + '/' + path
    try:
        raw = get(url)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if len(raw) > 16_000_000:
        raise ValueError('upstream document exceeds size bound')
    return {'revision':revision, 'committed_at_utc':committed.isoformat(), 'sha256':sha(raw), 'url':url}, json.loads(raw)


def publish_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(data)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def seal(state, day, fetched, issued):
    rd = day.isoformat()
    provenance, questions = fetched
    if not active(day) or issued.date() != day or utc(provenance['committed_at_utc']) > issued:
        raise ValueError('round outside issuance window')
    if questions['forecast_due_date'] != rd or questions['question_set'] != rd+'-llm.json':
        raise ValueError('wrong question set')
    model = json.loads((HERE / 'results.json').read_text())['model']
    if sha(canonical(model)) != MODEL_SHA:
        raise ValueError('fitted model digest differs')
    groups, required = question_index(questions)
    produced = []
    methods = Counter()
    for (source, ident), targets in sorted(required.items()):
        for target in sorted(targets):
            days = horizon(rd, target)
            if days is not None and days <= 0:
                raise ValueError('nonprospective dataset question')
            p, route = prediction(model, source, days)
            methods[groups[source, ident]+'/'+route] += 1
            produced.append({'source':source, 'id':ident, 'resolution_date':target, 'forecast':p})
    result, coverage = build(questions, {'organization':'MEGALAB', 'model':'Frozen source-horizon prior v1',
        'model_organization':'MEGALAB', 'question_set':questions['question_set'], 'forecasts':produced})
    if not all(x['pass'] for x in coverage.values()):
        raise ValueError('insufficient coverage')
    record = {'round':rd, 'issued_at_utc':issued.isoformat(), 'upstream_questions':provenance, 'model_sha256':MODEL_SHA,
              'forecast_sha256':sha(canonical(result)), 'forecast_set':result, 'coverage':coverage,
              'routes':dict(methods), 'channel':'internal shadow only'}
    publish_once(state / (rd+'.json'), record)


def read_seal(path):
    seal = json.loads(path.read_bytes())
    rd = path.stem
    if (seal['round'] != rd or not active(date.fromisoformat(rd)) or utc(seal['issued_at_utc']).date().isoformat() != rd
        or seal['model_sha256'] != MODEL_SHA or sha(canonical(seal['forecast_set'])) != seal['forecast_sha256']):
        raise ValueError('invalid seal: ' + rd)
    return seal


def receipt_ok(path, receipt_dir, issue):
    receipt = receipt_dir / path.name
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_bytes())
    published = utc(record['published_at_utc'])
    return (sha(path.read_bytes()) == record['issue_sha256'] and published.date().isoformat() == issue['round']
            and utc(issue['issued_at_utc']) <= published)


def record_scores(seal, prev, receipt_hash, payload, observed):
    rd = seal['round']
    upstream, doc = payload
    if doc.get('forecast_due_date') != rd or doc.get('question_set') != rd + '-llm.json':
        raise ValueError('different resolution round')
    if utc(upstream['committed_at_utc']) > observed:
        raise ValueError('future resolution commit')
    values = {(r['source'],r['id'],r['resolution_date']): r['forecast'] for r in seal['forecast_set']['forecasts']}
    ledger = prev or {'round':rd, 'forecast_sha256':seal['forecast_sha256'], 'issue_sha256':receipt_hash,
                      'batches':[], 'first_outcomes':[]}
    if ledger['forecast_sha256'] != seal['forecast_sha256'] or ledger['issue_sha256'] != receipt_hash:
        raise ValueError('score belongs to another issue')
    old = {tuple(r[:3]) for r in ledger['first_outcomes']}
    new = {}
    for r in doc['resolutions']:
        if (r.get('resolved') is not True or type(r.get('resolved_to')) not in (int,float)
            or r['resolved_to'] not in (0,1) or r.get('direction') or not isinstance(r['id'], str)):
            continue
        resolved = date.fromisoformat(r['resolution_date'][:10])
        if not date.fromisoformat(rd) < resolved <= min(observed.date(), utc(upstream['committed_at_utc']).date()):
            continue
        dated = r['source'], r['id'], resolved.isoformat()
        key = dated if dated in values else (r['source'],r['id'],None)
        if key in old or key not in values:
            continue
        if key in new:
            raise ValueError('ambiguous resolution for one question and horizon')
        new[key] = int(r['resolved_to']), resolved.isoformat()
    if not new:
        return prev
    batch = len(ledger['batches'])
    ledger['batches'].append({'observed_at_utc':observed.isoformat(), **upstream})
    ledger['first_outcomes'].extend([*key,y,resolved,batch] for key,(y,resolved) in sorted(new.items()))
    series = defaultdict(list)
    pairs = []
    for source, ident, target, y, _, _ in ledger['first_outcomes']:
        pair = values[source, ident, target], y
        series[source, ident].append(pair)
        pairs.append(pair)
    by_source = defaultdict(list)
    for (source, _), observations in series.items():
        by_source[source].append(metrics(observations)['brier'])
    ledger['scores'] = {'forecast':metrics(pairs), 'neutral':metrics([(0.5,y) for _,y in pairs]),
        'eligible_scored':len(pairs), 'issued':len(values), 'series':len(series),
        'source_series_balanced_brier':mean(mean(vals) for vals in by_source.values()),
        'source_series_brier':{s:mean(vals) for s,vals in sorted(by_source.items())}}
    return ledger


def run(state, receipts):
    checked = now()
    day = checked.date()
    if active(day):
        path = state / (day.isoformat()+'.json')
        if path.exists():
            read_seal(path)
        else:
            data = snapshot('datasets/question_sets/' + day.isoformat() + '-llm.json', checked)
            if data is None:
                print('Official round file pending', day)
            else:
                seal(state, day, data, now())
                print('Sealed internal round', day)
    if day > SCORE_END:
        return
    for path in sorted(state.glob('????-??-??.json')):
        issue = read_seal(path)
        if (day-date.fromisoformat(issue['round'])).days < 1 or not receipt_ok(path, receipts, issue):
            continue
        output = state / 'scores' / path.name
        previous = json.loads(output.read_bytes()) if output.exists() else None
        if previous and previous['scores']['eligible_scored'] == previous['scores']['issued']:
            continue
        data = snapshot('datasets/resolution_sets/'+issue['round']+'_resolution_set.json', checked)
        if data is None or (previous and data[0]['revision'] == previous['batches'][-1]['revision']):
            continue
        result = record_scores(issue, previous, sha(path.read_bytes()), data, checked)
        if result is not None:
            payload = canonical(result)
            if not output.exists() or output.read_bytes() != payload:
                output.parent.mkdir(parents=True, exist_ok=True)
                tmp = output.with_suffix('.tmp')
                tmp.write_bytes(payload)
                tmp.replace(output)
                print('Scored internal round', issue['round'], result['scores']['eligible_scored'])
    print('Checked', checked.isoformat())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    cmd = sub.add_parser('run')
    cmd.add_argument('--state-dir', type=Path, required=True)
    cmd.add_argument('--receipts-dir', type=Path)
    args = parser.parse_args()
    run(args.state_dir, args.receipts_dir or args.state_dir/'receipts')
