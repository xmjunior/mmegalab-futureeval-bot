"""Offline ForecastBench export and coverage verification for standard, post-2025-10-26 rounds."""
import argparse
from collections import defaultdict
from datetime import date
import gzip
import json
import math
from pathlib import Path
import sys

IDENTITY = ('organization', 'model', 'model_organization', 'question_set')
FIXTURE_PREFIX = 'HISTORICAL-FIXTURE-ONLY-'


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as source:
        return json.load(source)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def iso_date(value):
    return isinstance(value, str) and date.fromisoformat(value).isoformat() == value


def question_index(question_set):
    groups, required = {}, {}
    for question in question_set['questions']:
        key = question['source'], question['id']
        if not all(nonempty(v) for v in key) or key in required:
            raise ValueError('duplicate or invalid question identity')
        dates = question['resolution_dates']
        if dates == 'N/A':
            required[key], groups[key] = {None}, 'market'
        elif isinstance(dates, list) and dates and all(iso_date(v) for v in dates):
            required[key], groups[key] = set(dates), 'dataset'
            if len(dates) != len(required[key]):
                raise ValueError('duplicate resolution date')
        else:
            raise ValueError('invalid requested resolution dates')
    return groups, required


def build(question_set, predictions):
    if not all(nonempty(predictions.get(field)) for field in IDENTITY):
        raise ValueError('missing forecast-set metadata')
    if predictions['question_set'] != question_set['question_set']:
        raise ValueError('different question sets')
    groups, required = question_index(question_set)
    if not predictions['question_set'].startswith(FIXTURE_PREFIX):
        if not iso_date(question_set['forecast_due_date']) or predictions['question_set'] != f"{question_set['forecast_due_date']}-llm.json":
            raise ValueError('unexpected standard-round name or due date')
        if any(sum(g == group for g in groups.values()) != 250 for group in ('market', 'dataset')):
            raise ValueError('incomplete official question index: expected 250 market and 250 dataset questions')
    observed, rows = defaultdict(set), []
    if not isinstance(predictions.get('forecasts'), list):
        raise ValueError('missing forecasts array')
    for row in predictions['forecasts']:
        key = row['source'], row['id']
        if key not in required:
            raise ValueError('unknown question')
        target, probability = row['resolution_date'], row['forecast']
        if target not in required[key]:
            raise ValueError('unknown horizon or non-null market resolution date')
        if target in observed[key]:
            raise ValueError('duplicate forecast')
        if type(probability) not in (int, float) or not 0 <= probability <= 1 or not math.isfinite(probability):
            raise ValueError('forecast must be finite and in [0,1]')
        if row.get('reasoning') is not None and not isinstance(row['reasoning'], str):
            raise ValueError('invalid reasoning')
        observed[key].add(target)
        rows.append({f: row.get(f) for f in ('id', 'source', 'forecast', 'resolution_date', 'reasoning')})
    coverage = {}
    for group in ('market', 'dataset'):
        keys = [key for key in required if groups[key] == group]
        complete = sum(observed[key] == required[key] for key in keys)
        needed = (95 * len(keys) + 99) // 100
        coverage[group] = {'complete_questions': complete, 'total_questions': len(keys), 'required_95pct': needed,
                           'present_slots': sum(len(observed[key]) for key in keys),
                           'requested_slots': sum(len(required[key]) for key in keys),
                           'pass': bool(keys) and complete >= needed}
    result = {field: predictions[field] for field in IDENTITY}
    result['forecasts'] = rows
    return result, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('questions', type=Path)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, help='New local JSON file. Created only when both groups pass.')
    args = parser.parse_args()
    try:
        result, coverage = build(read_json(args.questions), read_json(args.predictions))
        print(json.dumps(coverage, sort_keys=True))
        if not all(group['pass'] for group in coverage.values()):
            return 1
        if args.output:
            with args.output.open('x', encoding='utf-8') as output:
                json.dump(result, output, ensure_ascii=False, allow_nan=False)
                output.write('\n')
    except (KeyError, ValueError, TypeError, OSError) as error:
        print(f'Invalid export: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
