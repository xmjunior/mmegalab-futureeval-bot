"""Paired ForecastBench archive audit. No model calls. Derived data CC BY-SA 4.0."""
import argparse
import collections
import hashlib
import itertools
import json
import math
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DESIGN = '0e45fc20334fa381a89f38620b2cbcf47923c88fd58ddcc0f31f11d99955c6ca'
CUT = '2026-11-24'
PIN = '86b92ba06ed34d606d633bff3e99af54dd447d3a'
CONTENT = 'https://www.forecastbench.org/assets/data/processed-forecast-sets/processed_forecast_sets.tar.gz'
E = ('2026-05-24','2026-06-07','2026-06-21','2026-07-05','2026-07-19')
M = ('2026-08-02','2026-08-16','2026-08-30','2026-09-13','2026-09-27')
L = ('2026-09-13','2026-09-27','2026-10-11')
O = ('2026-10-25',)
ALLOWED = {'dbnomics','fred','yfinance','acled','wikipedia'}
FRONT = ('6A','6S','6L','6.1S')
PANELS = [
    ('early',E,('naive','gemini35','gemma','5.5','5.4','5.4mini'),[(o,c) for o in ('5.5','5.4','5.4mini') for c in ('gemini35','gemma')]),
    ('middle',M,('naive','gemini35','5.5','5.6webcode'),[('5.6webcode',c) for c in ('gemini35','5.5')]),
    *[('late_'+m,L,('naive','gemini38web',m),[(m,'gemini38web')]) for m in FRONT],
    ('oct25',O,('naive','gemini38web','5.5web','5.6web','6A','6S','6L','6.1S'),[(m,'gemini38web') for m in ('5.5web','5.6web')+FRONT]),
]

def filehash(p):
    with p.open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def classify(d):
    s=d['model'].lower().replace(' (zero shot)','')
    # Modern records have explicit configuration, allowing tool/effort matching.
    s=d.get('model_run_slug',s)
    lut={'naive forecaster':'naive','gpt-5.4-2026-03-05':'5.4',
         'gpt-5.4-mini-2026-03-17':'5.4mini','gpt-5.5-2026-04-23':'5.5',
         'gpt-5.5-2026-04-23-high-web-search':'5.5web',
         'gpt-5.6-sol-standard-medium-web-search-code-execution':'5.6webcode',
         'gpt-5.6-sol-standard-high-web-search':'5.6web',
         'gemma-4-31b-it':'gemma','gemini-3.5-flash':'gemini35',
         'gemini-3.8-flash-high-web-search':'gemini38web'}
    for n,k in (('6-astra','6A'),('6-sol','6S'),('6-luna','6L'),('6.1-sol','6.1S')):
        lut['gpt-'+n+'-standard-high-web-search']=k
    return lut.get(s)

def key(r):
    return r['source'],str(r['id']),json.dumps(r['direction'] or [],separators=(',',':')),str(r['resolution_date'])[:10]

def prob(p):
    return isinstance(p,(int,float)) and not isinstance(p,bool) and math.isfinite(p) and 0<=p<=1

def native(r):
    return r is not None and r.get('imputed') is False and prob(r.get('forecast'))

def lookup(rows):
    found={};dups=set()
    for r in rows:
        if r['source'] not in ALLOWED:continue
        k=key(r)
        if k in found:dups.add(k)
        found[k]=r
    for k in dups:del found[k]
    return found,dups

def load(archive):
    dates={d for _,ds,models,cs in PANELS for d in ds}
    table=collections.defaultdict(dict); manifest={}; newer_names=[]
    target_strings=('gemma-4-31b','gemini-3.5-flash','gemini-3.8-flash','gpt-5.4-','gpt-5.5-','gpt-5.6-sol','gpt-6-','gpt-6.1-sol','naive-forecaster')
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            name=Path(member.name).name;low=name.lower(); dt=name[:10]
            if 'gpt-6.2' in low and member.isfile():newer_names.append(member.name)
            if not member.isfile() or dt not in dates or not low.endswith('.json') or not any(x in low for x in target_strings):continue
            naive='naive-forecaster' in low
            suffix=('-zero-shot-with-freeze-values.json',) if dt in O else ('-zero-shot.json','_zero_shot.json')
            if not naive and not low.endswith(suffix):continue
            raw=tf.extractfile(member).read();d=json.loads(raw);model=classify(d)
            if not model or d.get('leaderboard_eligible') is not True or not any(model in models and dt in ds for _,ds,models,_ in PANELS):continue
            assert d['forecast_due_date']==dt and not (model in table[dt]),(dt,model)
            if not naive and d.get('forecast_variant_key'):
                expected='zero-shot-with-freeze-values' if dt in O else 'zero-shot'
                assert d['forecast_variant_key']==expected,(dt,d['forecast_variant_key'])
            forecasts,dups=lookup(d['forecasts'])
            table[dt][model]=(forecasts,dups)
            manifest[dt+'/'+model]={'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'model':d['model'],
                'model_run_slug':d.get('model_run_slug'),'row_metadata':sorted(d['forecasts'][0].keys()),
                'eligible_source_duplicates':len(dups)}
    return table,manifest,newer_names

def loss(p,y):
    p=np.asarray(p,float); y=np.asarray(y,float)[:,None]; q=np.clip(p,1e-6,1-1e-6)
    return np.stack(((p-y)**2,-np.log(np.where(y==1,q,1-q))),axis=2)

def descript(rows,models):
    a=loss([x[3] for x in rows],[x[2] for x in rows])
    return {m:{'brier':v[0],'logloss':v[1]} for m,v in zip(models,a.mean(axis=0).tolist())}

def summary(rows,models,contrasts):
    if not rows:return {'n':0}
    rounds=sorted({r[0] for r in rows})
    dates=[[r for r in rows if r[0]==d] for d in rounds]
    date_stats=[loss([x[3] for x in rs],[x[2] for x in rs]) for rs in dates]
    n=np.array([len(d) for d in dates])
    out={'n':int(n.sum()),'unique_targets':len(set(r[1] for r in rows)),
         'unique_underlying_ids':len(set(r[1][:2] for r in rows)),
         'rounds':rounds,'by_round':{d:{'n':len(rs),'scores':descript(rs,models)} for d,rs in zip(rounds,dates)},
         'scores':descript(rows,models),'contrasts':{},'by_source':{}}
    for s in sorted(ALLOWED):
        sub=[r for r in rows if r[1][0]==s]
        if sub:out['by_source'][s]={'n':len(sub),'scores':descript(sub,models)}
    summ=np.array([d.sum(axis=0) for d in date_stats])
    weights=n[:,None,None]
    gains=summ.sum(axis=0)/sum(n)
    if len(rounds)>1:
        ix=np.array(list(itertools.product(range(len(n)),repeat=len(n)))) if len(n)<=5 else np.random.default_rng(8824).integers(0,len(n),(16384,len(n)))
        boot=summ[ix].sum(axis=1)/n[ix].sum(axis=1)[:,None,None]
    for a,b in contrasts:
        aa,bb=models.index(a),models.index(b)
        delta=gains[bb]-gains[aa]
        d=(summ/weights)[:,bb,:]-(summ/weights)[:,aa,:]
        entry={'gain_brier':float(delta[0]),'gain_log':float(delta[1]),'positive_rounds':[int(np.sum(d[:,i]>0)) for i in (0,1)]}
        if len(rounds)>1:
            bd=boot[:,bb,:]-boot[:,aa,:]
            entry['round_bootstrap_95_brier_log']=np.quantile(bd,[.025,.975],axis=0).T.tolist()
            signs=np.array(list(itertools.product((-1,1),repeat=len(rounds))))
            entry['p_one_sided_round_sign']=float(np.mean((signs*d[:,0]).sum(axis=1)>=d[:,0].sum()-1e-14))
        out['contrasts'][a+'_vs_'+b]=entry
    return out

def main(archive,inputs,dest,res_extra):
    if filehash(ROOT/'preregister.json')!=DESIGN:raise ValueError('Change to locked design')
    table,manifest,newer=load(archive)
    stats={}; resolutions={}; rs_sha={}
    for dt in sorted(table):
        path=inputs/(dt+'_resolution_set.json')
        if not path.is_file():path=res_extra/path.name
        raw=path.read_bytes();d=json.loads(raw); assert d['forecast_due_date']==dt
        ix,dups=lookup(d['resolutions'])
        resolutions[dt]={k:r for k,r in ix.items() if r['resolved'] is True and type(r['resolved_to']) in (int,float) and r['resolved_to'] in (0,1) and dt<k[-1]<=CUT}
        rs_sha[dt]=hashlib.sha256(raw).hexdigest()
    for panel,ds,modelnames,contrasts in PANELS:
        rounds=[d for d in ds if all(m in table[d] for m in modelnames)]
        ct=collections.Counter();coverage=collections.Counter();rows=[]
        for dt in rounds:
            for k,truth in resolutions[dt].items():
                ct['eligible_resolutions']+=1; y=int(truth['resolved_to'])
                forecast=[table[dt][m][0].get(k) for m in modelnames]
                for m,r in zip(modelnames,forecast):
                    if native(r):coverage[m]+=1
                if any(not native(r) for r in forecast):
                    ct['not_all_native']+=1;continue
                if any(str(r['forecast_due_date'])[:10]!=dt or r.get('resolved') is True and r.get('resolved_to')!=y for r in forecast):
                    ct['inconsistent_due_date_or_label']+=1;continue
                rows.append((dt,k,y,[r['forecast'] for r in forecast]));ct['scored']+=1
        first=[]; seen=set()
        for r in rows:
            if r[1] not in seen:first.append(r);seen.add(r[1])
        stats[panel]={'models':modelnames,'missing_required_models_by_round':{dt:[m for m in modelnames if m not in table[dt]] for dt in ds if dt not in rounds},
            'coverage_counts':dict(coverage),'exclusions':dict(ct),'main':summary(rows,modelnames,contrasts),
            'unique_target_sensitivity':summary(first,modelnames,contrasts)}
    pvals=[]
    for panel in ('early','middle'):
        for name,rec in stats[panel]['main'].get('contrasts',{}).items():
            if 'p_one_sided_round_sign' in rec:pvals.append((rec['p_one_sided_round_sign'],panel,name))
    carry=0
    for i,(p,pan,n) in enumerate(sorted(pvals)):
        carry=max(carry,min(1,p*(8-i)))
        data=stats[pan]['main']; rec=data['contrasts'][n]
        rec['p_holm_eight_planned']=carry
        rec['archival_screen']=bool(len(data['rounds'])>=5 and carry<.05 and rec['round_bootstrap_95_brier_log'][0][0]>0 and rec['gain_log']>=0)
    out={'design_sha256':DESIGN,'script_sha256':filehash(Path(__file__)),'as_of':CUT,'archive_sha256':filehash(archive),'source_url':CONTENT,
        'resolution_repository':'https://github.com/forecastingresearch/forecastbench-datasets','resolution_revision':PIN,'resolution_sha256':rs_sha,
        'forecast_files':manifest,'gpt_6_2_files_in_entire_archive':newer,
        'score_gain':'Positive = OpenAI candidate has lower loss than the named comparison on the matched cases. No official leaderboard rank.',
        'caution':'A new archive contrast on historical records, not a blind rep from a new bot or new sample for the overall mission. Small numbers of round clusters and recurring series matter.',
        'data_license':'ForecastBench, Forecasting Research Institute, CC BY-SA 4.0; derived comparison CC BY-SA 4.0.','panels':stats}
    dest.write_text(json.dumps(out,allow_nan=False,indent=2)+'\n')
    for panel,r in stats.items():
        d=r['main'];print(panel,d.get('n'),d.get('rounds'),r['missing_required_models_by_round'])
        print(d.get('scores'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('resolutions',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--extra-resolutions',type=Path,default=ROOT);a=p.parse_args();main(a.archive,a.resolutions,a.output,a.extra_resolutions)
