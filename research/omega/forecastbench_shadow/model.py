"""Source and horizon baseline, trained only on the frozen OMEGA training rows."""
from collections import defaultdict
from datetime import date
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROW_SHA = "6d7b464141d5c0c5a141df2fdc8b3ec25dcb0ab1f2299950324795c5b6776860"
SHRINK = 4


def canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def horizon(issue, target):
    return None if target is None else (date.fromisoformat(target) - date.fromisoformat(issue)).days


def rows():
    payload = gzip.decompress((HERE.parent / 'binary' / 'rows.jsonl.gz').read_bytes())
    if sha(payload) != ROW_SHA:
        raise ValueError('training panel digest differs')
    return [json.loads(line) for line in payload.splitlines()]


def fit(panel):
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for split, due, source, _, target, _, *tail in panel:
        if split != 'train':
            continue
        outcome, family = tail[-2:]
        for bucket in ('all', str(horizon(due, target))):
            cells[source, bucket][due][family].append(outcome)
    rates = {}
    for (source, bucket), rounds in sorted(cells.items(), key=lambda kv: kv[0][1] != 'all'):
        round_means = [mean(mean(ys) for ys in families.values()) for families in rounds.values()]
        by_source = rates.setdefault(source, {})
        prior = by_source.get('all', 0.5)
        by_source[bucket] = (sum(round_means) + SHRINK * prior) / (len(round_means) + SHRINK)
    return {'source_horizon': rates, 'train_panel_sha256': ROW_SHA, 'pseudo_rounds': SHRINK}


def prediction(model, source, days):
    group = model['source_horizon'].get(source, {})
    key = str(days)
    if key in group:
        return group[key], 'source_horizon'
    if 'all' in group:
        return group['all'], 'source'
    return 0.5, 'neutral'


def metrics(pairs):
    if not pairs:
        return None
    bins = defaultdict(list)
    for p, y in pairs:
        bins[min(9, int(10 * p))].append(p - y)
    return {'n': len(pairs), 'brier': mean((p-y)**2 for p,y in pairs),
            'log_loss': mean(-math.log(p if y else 1-p) for p,y in pairs),
            'ece10': sum(abs(sum(items)) for items in bins.values()) / len(pairs)}


def replay():
    panel = rows()
    model = fit(panel)
    stats = {}
    for split in ('valid', 'test'):
        pairs, families, rounds = [], defaultdict(list), defaultdict(list)
        for part, due, source, _, target, _, *tail in panel:
            if part != split:
                continue
            p, _ = prediction(model, source, horizon(due, target))
            y, family = tail[-2:]
            pairs.append((p, y))
            rounds[due].append((p, y))
            families[due, family].append((p, y))
        balanced = defaultdict(list)
        for (due, _), pp in families.items():
            balanced[due].append(metrics(pp)['brier'])
        stats[split] = {'forecast': metrics(pairs), 'neutral': metrics([(0.5,y) for _,y in pairs]),
                        'rounds': {day: metrics(pp) for day,pp in sorted(rounds.items())},
                        'families': len({f for _,f in families}),
                        'round_family_balanced_brier': mean(mean(b) for b in balanced.values())}
    return {'model': model, 'model_sha256': sha(canonical(model)), 'training_rows': sum(r[0]=='train' for r in panel),
            'training_rounds': len({r[1] for r in panel if r[0]=='train'}), 'comparison': stats}


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True))
