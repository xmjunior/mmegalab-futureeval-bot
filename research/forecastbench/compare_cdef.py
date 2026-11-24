"""Evaluate the frozen ForecastBench C/D/E/F design using archived records. © CC BY-SA 4.0."""
import argparse
import hashlib
import itertools
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compare import ARCHIVE, CUT, PIN, key, sha

ROUNDS = ('2026-08-02', '2026-08-16', '2026-08-30', '2026-09-13', '2026-09-27')
PRIOR_ROUNDS = ('2026-05-24', '2026-06-07', '2026-06-21', '2026-07-05', '2026-07-19')
FILES = {'gemini': '.Google.Google DeepMind.gemini-3.5-flash', 'gpt': '.OpenAI.OpenAI.gpt-5.4-mini-2026-03-17', 'opus': '.Anthropic.Anthropic.claude-opus-4-7'}
MODELS = tuple(FILES) + ('C', 'D', 'E', 'F')
CONTRASTS = {'C': 'gemini', 'D': 'gemini', 'E': 'D', 'F': 'gemini'}
CROWD = {'manifold', 'polymarket', 'infer'}
MARKETS = CROWD | {'kalshi'}


def probability(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1


def native(row):
    return row is not None and row.get('imputed') is False and probability(row.get('forecast'))


def source_type(source):
    return 'markets' if source in MARKETS else 'datasets'


def scores(p, y):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.stack(((p - y) ** 2, -np.log(np.where(y == 1, q, 1-q))), axis=-1)


def round_sign_test(deltas):
    # Under equal independent round signs; ties carry no sign information.
    pos = sum(d > 1e-14 for d in deltas)
    neg = sum(d < -1e-14 for d in deltas)
    n = pos + neg
    return min(1, 2 * sum(math.comb(n, k) for k in range(min(pos, neg) + 1)) / (2 ** n))


def summarize(rows, infer=False):
    if not rows:
        return {'n': 0}
    p = np.asarray([r['p'] for r in rows])
    y = np.asarray([r['y'] for r in rows])[:, None]
    loss = scores(p, y)
    rd = np.array([r['round'] for r in rows])
    dates = sorted(set(rd))
    members = [loss[rd == dt] for dt in dates]
    means = loss.mean(axis=0)
    trials = len(dates) ** len(dates)
    out = dict(n=len(rows), dates=dates, unique_source_id=len({r['key'][:2] for r in rows}), unique_targets=len({r['key'] for r in rows}), mean=dict(zip(MODELS, means.tolist())), gains={}, E_triggers=sum(r['E_trigger'] for r in rows), F_uses_market=sum(r['F_trigger'] for r in rows))
    if infer:
        sample = np.array(list(itertools.product(range(len(dates)), repeat=len(dates))))
        counts = np.array([len(a) for a in members]); sums = np.array([a.sum(axis=0) for a in members]); bl = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)[:, None, None]
        out['bootstrap_round_block_resamples'] = trials
        residual = p[:, :3] - y
        out['residual_correlations_gemini_gpt_opus'] = np.corrcoef(residual.T).tolist()
    for model, baseline in CONTRASTS.items():
        x, b = MODELS.index(model), MODELS.index(baseline)
        delta = [r[:, b].mean(axis=0) - r[:, x].mean(axis=0) for r in members]
        stat = dict(vs=baseline, brier_and_log=(means[b]-means[x]).tolist(), by_round={d:a.tolist() for d,a in zip(dates,delta)})
        if infer:
            boots = bl[:,b] - bl[:,x]
            stat['ci95_by_metric'] = np.quantile(boots,[.025,.975],axis=0).T.tolist()
            stat['brier_bonferroni_ci9875'] = np.quantile(boots[:,0],[.00625,.99375]).tolist()
            stat['brier_sign_test_p'] = round_sign_test([a[0] for a in delta])
        out['gains'][model] = stat
    if infer:
        values=sorted((v['brier_sign_test_p'], k) for k,v in out['gains'].items())
        largest=0
        for i,(pv,k) in enumerate(values):
            largest=max(largest,pv*(len(values)-i));out['gains'][k]['brier_sign_test_holm_p']=min(1,largest)
    return out


def analyze(archive, resdir, outpath):
    files = defaultdict(dict); manifest = {}; resolution_hashes = {}; prior_targets = set()
    suffixes={v+'-run-variant-01-zero-shot.json': k for k,v in FILES.items()}
    with tarfile.open(archive, 'r|gz') as arc:
        for member in arc:
            name = Path(member.name).name
            if not member.isfile() or name[:10] not in ROUNDS: continue
            n = next((k for s,k in suffixes.items() if name.endswith(s)), None)
            if n is None: continue
            data=arc.extractfile(member).read(); manifest[name]=hashlib.sha256(data).hexdigest()
            parsed=json.loads(data); dt=parsed['forecast_due_date']; assert dt==name[:10] and parsed['leaderboard_eligible'] and n not in files[dt]
            items=parsed['forecasts']; assert all(e['forecast_due_date'][:10]==dt for e in items)
            index={key(e):e for e in items};assert len(index)==len(items);files[dt][n]=index
    assert set(files)==set(ROUNDS) and all(set(v)==set(FILES) for v in files.values())
    def resolutions(dt):
        path=resdir/f'{dt}_resolution_set.json'; resolution_hashes[path.name]=sha(path)
        data=json.loads(path.read_text()); assert data['forecast_due_date'][:10]==dt
        entries=data['resolutions']; hashed={key(e):e for e in entries}; assert len(hashed)==len(entries);return hashed
    for dt in PRIOR_ROUNDS: prior_targets.update(resolutions(dt))
    coverage=defaultdict(Counter); exclusions=Counter(); rows=[]
    for dt in ROUNDS:
        model=files[dt]; resolved=resolutions(dt)
        for k in sorted(set(resolved).union(*(set(m) for m in model.values()))):
            if k[0]=='metaculus': exclusions['metaculus']+=1; continue
            y=resolved.get(k)
            if not y or not y.get('resolved') or y['resolved_to'] not in (0,1) or not dt<k[2]<=CUT:
                exclusions['missing_resolution_nonbinary_or_out_of_time']+=1;continue
            truth=y['resolved_to']; el={n: m.get(k) for n,m in model.items()}
            dims=('all','round/'+dt,'source/'+k[0],'round_source/'+dt+'/'+k[0])
            def count(item):
                for dim in dims:coverage[dim][item]+=1
            count('eligible')
            for n,e in el.items():
                count(n+('_native' if native(e) else '_missing_or_imputed'))
            if any(e and e.get('resolved') and e.get('resolved_to')!=truth for e in el.values()):
                count('resolution_conflict');continue
            usable={n:native(e) for n,e in el.items()}
            if not (usable['gemini'] and usable['gpt']):continue
            count('D_available')
            g,h=el['gemini']['forecast'],el['gpt']['forecast']; triggered=abs(g-h)>=.20-1e-12
            if triggered:count('E_triggers_among_D')
            if triggered and not usable['opus']:count('E_missing_frontier_when_triggered')
            if not usable['opus']:continue
            count('complete'); o=el['opus']['forecast']; d=(g+h)/2
            market=el['gemini'].get('market_value_on_due_date_minus_one'); use_market=k[0] in CROWD and probability(market)
            rows.append(dict(round=dt,key=k,y=truth,p=[g,h,o,.9*g+.05,d,o if triggered else d,market if use_market else g],E_trigger=triggered,F_trigger=use_market))
    report={'archive_url':ARCHIVE,'archive_sha256':sha(archive),'resolution_revision':PIN,'design_sha256':sha(Path(__file__).with_name('design-cdef.md')),'script_sha256':sha(Path(__file__)),'cutoff':CUT,'input_forecast_hashes':manifest,'input_resolution_hashes':resolution_hashes,'license':'ForecastBench data © Forecasting Research Institute, CC BY-SA 4.0; derived comparison likewise. https://www.forecastbench.org/datasets/ and https://creativecommons.org/licenses/by-sa/4.0/','metric_order':['brier','logloss'],'forecast_date':'Official round due date only, individual prediction creation/upload times absent in processed source. Strictly earlier than target resolution.','score_gain':'Positive = variant has lower loss; row weighted. Four primary Brier comparisons, all on same complete panel.','filters_global':exclusions,'coverage':{},'primary':summarize(rows,True),'by_round':{},'by_source':{},'by_kind':{},'by_trigger':{}}
    for dim,c in sorted(coverage.items()):
        z=dict(c)
        for n in FILES:z[n+'_missing_rate']=1-c[n+'_native']/c['eligible']
        z['complete_missing_rate']=1-c['complete']/c['eligible']; report['coverage'][dim]=z
    for dt in ROUNDS:report['by_round'][dt]=summarize([r for r in rows if r['round']==dt]);
    for src in sorted({r['key'][0] for r in rows}):report['by_source'][src]=summarize([r for r in rows if r['key'][0]==src]);
    for kind in ('datasets','markets'):report['by_kind'][kind]=summarize([r for r in rows if source_type(r['key'][0])==kind]);
    for cond in ('E_trigger','F_trigger'):report['by_trigger'][cond]=summarize([r for r in rows if r[cond]])
    seen=set(prior_targets); fresh=[]
    for row in rows:
        if row['key'] in seen: continue
        fresh.append(row); seen.add(row['key'])
    report['sensitivity_first_per_target_not_in_prior_rounds']=summarize(fresh,True)
    outpath.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(len(rows),len(fresh),dict(coverage['all']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);ap.add_argument('resolutions',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args();analyze(args.archive,args.resolutions,args.output)
