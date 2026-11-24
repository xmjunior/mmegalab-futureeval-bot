"""ForecastBench paired retrospective analysis. Run with the official processed tar and pinned resolutions directory."""
import argparse, csv, hashlib, itertools, json, math, re, statistics, tarfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from datetime import date

PIN='86b92ba06ed34d606d633bff3e99af54dd447d3a'
ARCHIVE='https://www.forecastbench.org/assets/data/processed-forecast-sets/processed_forecast_sets.tar.gz'
CUT='2026-11-24'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def slot(name):
    n=name.lower()
    if n.endswith('.forecastbench.naive-forecaster.json'):return 'naive'
    if not n.endswith(('_zero_shot.json','-run-variant-01-zero-shot.json')):return
    if re.search(r'gemma-4-31b(?:-it)?(?:_|-run)',n):return 'gemma'
    if re.search(r'gemini-3\.5-flash(?:_|-run)',n):return 'flash'
def key(e):
    return (e['source'], str(e['id']),str(e['resolution_date'])[:10])
def proper(p,y):
    assert 0<=p<=1 and y in (0,1)
    pc=min(1-1e-6,max(1e-6,p))
    return np.array(((p-y)**2,-math.log(pc if y else 1-pc)))
def summary(records):
    if not records:return {}
    # Nonparametric block bootstrap over whole rounds, with row weighting.
    rounds=sorted({r[0] for r in records})
    groups=[np.asarray([r[1:] for r in records if r[0]==rd],dtype=float) for rd in rounds]
    metrics=np.concatenate(groups).mean(axis=0)
    n=np.asarray([len(g) for g in groups]); sums=np.asarray([g.sum(axis=0) for g in groups])
    ix=np.asarray(list(itertools.product(range(len(rounds)),repeat=len(rounds))))
    boot=sums[ix].sum(axis=1)/n[ix].sum(axis=1)[:,None]
    intervals=np.quantile(boot,[0.025,0.975],axis=0)
    labels=['flash_brier','flash_logloss','gemma_brier','gemma_logloss','naive_brier','naive_logloss','gain_brier_flash_vs_gemma','gain_log_flash_vs_gemma','gain_brier_flash_vs_naive','gain_log_flash_vs_naive']
    return dict(n=len(records),rounds=rounds,rows_per_round=n.tolist(),round_block_resamples=len(boot),pooled=dict(zip(labels,metrics)),round_block_95pct={label:intervals[:,j].tolist() for j,label in enumerate(labels) if label.startswith('gain_')},
        by_round={rd:dict(zip(labels,groups[j].mean(axis=0))) for j,rd in enumerate(rounds)})
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path);parser.add_argument('resolutions',type=Path);parser.add_argument('out',type=Path)
    a=parser.parse_args()
    files=defaultdict(dict); manifest={}
    with tarfile.open(a.archive,'r|gz') as tar:
        for m in tar:
            name=Path(m.name).name;model=slot(name)
            if not m.isfile() or not model:continue
            raw=tar.extractfile(m).read();d=json.loads(raw)
            assert d['leaderboard_eligible']
            rnd=d['forecast_due_date']
            assert rnd not in files or model not in files[rnd]
            files[rnd][model]=d
            manifest[name]=hashlib.sha256(raw).hexdigest()
    rounds=sorted(r for r in files if set(files[r])=={'gemma','flash','naive'})
    keep=[];crowd=[]; counters=Counter(); coverage=Counter(); unique=set()
    sources=Counter(); mfile={}
    for r in rounds:
        rp=a.resolutions/f'{r}_resolution_set.json';mfile[rp.name]=sha(rp)
        raw=json.loads(rp.read_text())['resolutions']; rr={key(e):e for e in raw};assert len(raw)==len(rr)
        models={n:{key(e):e for e in d['forecasts']} for n,d in files[r].items()}
        assert all(len(o)==len(files[r][n]['forecasts']) for n,o in models.items())
        for k in sorted(set.union(*(set(o) for o in models.values()))):
            if k[0]=='metaculus':counters['metaculus_source_excluded']+=1;continue
            y=rr.get(k)
            if not y or not y.get('resolved') or y['resolved_to'] not in (0,1) or not r<k[2]<=CUT:
                counters['unresolved_nonbinary_bad_date_or_not_in_pinned_label']+=1;continue
            if any(k not in m for m in models.values()):counters['not_in_all_three_files']+=1;continue
            el={model:d[k] for model,d in models.items()}
            if any(e.get('resolved') and e.get('resolved_to')!=y['resolved_to'] for e in el.values()):
                counters['resolution_conflict_skipped']+=1;continue
            assert all(e['forecast_due_date'][:10]==r for e in el.values())
            coverage['eligible_rows']+=1
            for n,e in el.items():
                if not e.get('imputed') and e.get('forecast') is not None:coverage[n+'_native']+=1
            if el['gemma']['imputed'] or el['flash']['imputed']:
                counters['one_or_both_models_imputed']+=1;continue
            if any(e.get('forecast') is None for e in el.values()):
                counters['missing_probability']+=1;continue
            y=y['resolved_to']
            f,g=proper(el['flash']['forecast'],y),proper(el['gemma']['forecast'],y)
            if el['naive']['imputed']:counters['naive_imputed']+=1
            else:
                n=proper(el['naive']['forecast'],y)
                keep.append([r,*f,*g,*n,*(g-f),*(n-f)]);sources[k[0]]+=1;unique.add(k[:2])
            if k[0] in ('manifold','polymarket','infer'):
                p=el['flash'].get('market_value_on_due_date_minus_one')
                if p is not None and 0<=p<=1:
                    # Historical values supplied in licensed FRI records, no market API request.
                    c=proper(p,y);crowd.append([r,*f,*g,*c,*(g-f),*(c-f)])
    report=dict(revision=PIN,archive_url=ARCHIVE,archive_sha256=sha(a.archive),cutoff=CUT,comparators='Primary: same zero-shot variants without freeze and non-imputed forecasts from both models and official naive. Excludes source=metaculus; only binary labels in pinned release resolving after due. Secondary: prior-day crowd on other market-sourced records; naive_* fields there represent prior-day crowd. Gains positive means first model improved.',date_semantics='ForecastBench due dates supplied as prospective forecast rounds; individual request timestamps unavailable in processed files. No new inference.',unique_source_question_ids_primary=len(unique),common_forecast_rounds=rounds,eligibility=coverage,primary_omissions=counters,source_counts_primary=sources,primary=summary(keep),market_prior_day_secondary=summary(crowd),forecast_file_sha256={k:v for k,v in manifest.items() if k[:10] in rounds},resolution_file_sha256=mfile,license='Forecasting Research Institute, ForecastBench Datasets, CC BY-SA 4.0. Derived comparisons; no affiliation.',caveat='Exact percentile bootstrap over few forecast rounds; repeated underlying series and questions may induce correlation between rounds. Row-weighted custom paired metrics, distinct from official adjusted leaderboard. Coverage conditional on model returning a genuine non-imputed probability.')
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for keyname in ('primary','market_prior_day_secondary'):
        group=report[keyname]; print(keyname,group.get('n'),group.get('pooled'),group.get('round_block_95pct'))
    print('coverage',coverage,'rounds',rounds)
if __name__=='__main__':main()
