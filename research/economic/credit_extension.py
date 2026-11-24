"""Preregistered secondary cohort covering accessible as-of credit vintages."""
import argparse
import hashlib
import itertools
import json
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date,timedelta
from urllib.parse import urlencode

import numpy as np
from fetch_inputs import ROOT, DATA, SERIES, fetch
from paired_economic import DESIGN, METHODS, WIDTHS, adjust, csv_records, make_features, outcome, scale, scores, target, vintage

EXT='0245645fe9642076ee3b247b4abec38125f7b2760714145bf8ad0940e9ce77aa'
DATES=[date(2024,9,2)+timedelta(days=14*i) for i in range(49)]

def fetch_all():
    DATA.mkdir(exist_ok=True)
    items=[]
    for o in DATES:
        d=o-timedelta(days=2)
        qs=urlencode({'id':SERIES,'cosd':(d-timedelta(days=810)).isoformat(),'vintage_date':d.isoformat()})
        items.append((vintage(o),'https://alfred.stlouisfed.org/graph/alfredgraph.csv?'+qs,SERIES+'_'+d.strftime('%Y%m%d')))
    with ThreadPoolExecutor(max_workers=7) as p: manifest=list(p.map(fetch,items))
    (ROOT/'ext_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def estimate_contrast(by_index,k):
    diff={i:statistics.mean(r['score'][k]-r['score']['A'] for r in rs) for i,rs in by_index.items() if len(rs)==2}
    blocks=[(diff[i]+diff[i+1])/2 for i in range(20,46,2) if i in diff and i+1 in diff]
    result={'paired_dates':len(diff),'paired_questions':2*len(diff),'four_week_blocks':len(blocks)}
    if not blocks: return result
    a=np.asarray(blocks); mean=a.mean()
    count=sum(np.mean(a*np.asarray(signs))>=mean-1e-14 for signs in itertools.product((-1,1),repeat=len(a)))
    rng=np.random.default_rng(77411)
    ci=np.quantile(a[rng.integers(0,len(a),(8192,len(a)))].mean(axis=1),[.025,.975])
    halves=[[v for i,v in diff.items() if (i<=32)==b] for b in (True,False)]
    return {**result,'mean_log_gain':statistics.mean(diff.values()),'mean_complete_blocks':float(mean),
            'ci95_blocks':list(map(float,ci)),'p_sign':count/2**len(a),
            'wins_ties_losses':[sum(v>1e-12 for v in diff.values()),sum(abs(v)<=1e-12 for v in diff.values()),sum(v< -1e-12 for v in diff.values())],
            'two_halves':[statistics.mean(z) if z else None for z in halves]}

def run():
    if hashlib.sha256((ROOT/'extension.json').read_bytes()).hexdigest()!=EXT or hashlib.sha256((ROOT/'design.json').read_bytes()).hexdigest()!=DESIGN:
        raise ValueError('Design changed')
    obs={}; missing=[]
    for ent in json.loads((ROOT/'ext_manifest.json').read_text()):
        if 'sha256' not in ent: continue
        p=DATA/ent['file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=ent['sha256']: raise ValueError('vintage changed')
        v=date.fromisoformat(p.stem); k=SERIES+'_'+v.strftime('%Y%m%d')
        obs[p.name]={date.fromisoformat(r['observation_date']):float(r[k]) for r in csv_records(p) if r[k] not in ('','.')}
    forecasts=[]
    for i,o in enumerate(DATES[:47]):
        for h in (5,10):
            try:
                a,g,last=make_features('credit',o,h,obs[vintage(o)]); assert last<o
                forecasts.append((i,o,h,a,g,last))
            except (ValueError,KeyError) as e: missing.append([i,str(o),h,'feature',str(e)])
    rows=[]
    for i,o,h,a,g,last in forecasts:
        try:
            y=outcome('credit',o,h,DATES[i+2],obs); assert np.isfinite(y)
            rows.append(dict(i=i,origin=str(o),h=h,end=str(target(o,h)),last_input=str(last),label=y,A_quantiles=list(a),G_quantiles=list(g),
                             A=scores('credit',a,y),G=scores('credit',g,y),Cs={w:scores('credit',scale(a,w),y) for w in WIDTHS}))
        except (ValueError,KeyError) as e: missing.append([i,str(o),h,'label',str(e)])
    # No label after May 26 is available to fit the calibrator.
    # Its longest development target ends June 6, before June 9's first OOS forecast.
    dev=[r for r in rows if r['i']<20]; n=len(set(r['i'] for r in dev))
    assert all(r['end']<str(DATES[20]) for r in dev)
    # Use the last delayed vintage for numerical reconstruction only. If a training label
    # vintage postdates the first holdout, refetch it at the forecast vintage; if not yet
    # reported, drop that row from fitting, not from the later OOS uncertainty analysis.
    visible=obs.get(vintage(DATES[20]),{})
    fit=[]; late_labels=[]
    for r in dev:
        if DATES[r['i']+2]-timedelta(days=2)<=DATES[20]-timedelta(days=2):
            fit.append(r);continue
        end=date.fromisoformat(r['end'])
        if end in visible:
            a=np.array(r['A_quantiles']);g=np.array(r['G_quantiles']);y=visible[end]
            fit.append({**r,'label':y,'A':scores('credit',a,y),'G':scores('credit',g,y),'Cs':{w:scores('credit',scale(a,w),y) for w in WIDTHS}})
        else: late_labels.append([r['origin'],r['h']])
    n=len(set(r['i'] for r in fit))
    w=max(WIDTHS,key=lambda w:statistics.mean(r['Cs'][w] for r in fit)) if n>=18 else 1.
    routes={}
    for h in (5,10):
        sel=[r for r in fit if r['h']==h]
        routes[h]='G' if len(sel)>=18 and statistics.mean(r['G']-r['A'] for r in sel)>.02 else 'A'
    by_index=defaultdict(list); pair=[]
    for r in rows:
        if r['i']<20:continue
        rr={k:v for k,v in r.items() if k not in ('Cs','A','G')}|{'score':{'A':r['A'],'G':r['G'],'F':r[routes[r['h']]],'C':r['Cs'][w]}}
        by_index[r['i']].append(rr);pair.append(rr)
    stats={m:estimate_contrast(by_index,m) for m in METHODS}
    for name,value in adjust({m:v['p_sign'] for m,v in stats.items() if 'p_sign' in v}).items():
        v=stats[name];v['p_holm']=value
        v['screen']=bool(v['four_week_blocks']>=12 and value<.05 and v['ci95_blocks'][0]>0 and all(t is not None and t>0 for t in v['two_halves']))
    summary={'design':EXT,'source_sha256':hashlib.sha256((ROOT/'ext_manifest.json').read_bytes()).hexdigest(),
             'input_vintages':len(obs),'development_dates':n,'development_rows':len(fit),'fit_vintage_not_yet_released':late_labels,
             'C_width':w,'F_routes':routes,'omissions':missing,'stats':stats,'status':'supplementary availability cohort; exploratory only'}
    (ROOT/'credit_extension_rows.json').write_text(json.dumps(pair,separators=(',',':'),allow_nan=False)+'\n')
    (ROOT/'credit_extension_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='omissions'}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fetch',action='store_true');args=parser.parse_args()
    if args.fetch:fetch_all()
    run()
