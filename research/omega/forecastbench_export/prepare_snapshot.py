"""Derive the minimal, offline historical fixture from the two pinned official input files."""
from datetime import date
from hashlib import sha256
from pathlib import Path
import argparse
import gzip
import json
from export import IDENTITY, question_index

ROUND = '2025-10-26'
AS_OF = date(2026, 11, 24)


def prepare(qpath, ppath, output):
    inputs = json.loads((Path(__file__).parent / 'sources.json').read_text())
    qraw, praw = qpath.read_bytes(), ppath.read_bytes()
    if [sha256(raw).hexdigest() for raw in (qraw, praw)] != [inputs['questions_sha256'], inputs['forecasts_member_sha256']]:
        raise ValueError('input hash differs from the pinned official snapshot')
    qs, ps = json.loads(qraw), json.loads(praw)
    if qs['forecast_due_date'] != ROUND or ps['forecast_due_date'] != ROUND or ps['question_set'] != qs['question_set']:
        raise ValueError('unexpected round')
    schema = {k: qs[k] for k in ('question_set', 'forecast_due_date')}
    schema['questions'] = [{k: item[k] for k in ('id', 'source', 'resolution_dates')} for item in qs['questions']]
    groups, _ = question_index(schema)
    archive = {k: ps[k] for k in IDENTITY}
    rows = []
    for row in ps['forecasts']:
        if row['resolved'] is not True or row['imputed'] is not False:
            continue
        resolved = date.fromisoformat(row['resolution_date'][:10])
        if resolved > AS_OF:
            continue
        key = row['source'], row['id']
        if row['direction'] or row['forecast_due_date'][:10] != ROUND:
            raise ValueError('unexpected direction or issue date in a standard round')
        rows.append({'id': key[1], 'source': key[0], 'resolution_date': resolved.isoformat() if groups[key] == 'dataset' else None, 'forecast': row['forecast']})
    archive['forecasts'] = rows
    for filename, data in [('round-index.json.gz', schema), ('resolved-archive.json.gz', archive)]:
        packed = json.dumps(data, separators=(',', ':'), sort_keys=True).encode()
        (output / filename).write_bytes(gzip.compress(packed, mtime=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('questions', type=Path)
    parser.add_argument('archived_forecasts', type=Path)
    args = parser.parse_args()
    prepare(args.questions, args.archived_forecasts, Path(__file__).parent)
