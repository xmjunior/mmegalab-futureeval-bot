"""Independent public economic forecast study. No credentials and no tournament data."""
import csv
import hashlib
import io
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from fetch_inputs import DATA, ROOT, SERIES, origins

PROBS = np.array([.1, .25, .5, .75, .9])
WIDTHS = (1., .75, 1.25, 1.5, 2.)
METHODS = ("C", "F", "G")
DESIGN = "ea10115001704fd47c085a5d01d5f1c1dd375bddd03d30a2f0049cc82c53f5ab"

def csv_records(path):
    return csv.DictReader(io.StringIO(path.read_bytes().decode('utf-8-sig')))

def load_sources():
    if hashlib.sha256((ROOT/'design.json').read_bytes()).hexdigest() != DESIGN:
        raise ValueError('The frozen design has changed')
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    data={}
    for entry in manifest:
        if 'sha256' not in entry:
            continue
        name=entry['file']
        f=DATA/name
        if hashlib.sha256(f.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Source hash mismatch: '+name)
        if name=='vix.csv':
            vix={}
            for r in csv_records(f):
                if r.get('CLOSE') and r.get('HIGH'):
                    d=datetime.strptime(r['DATE'],'%m/%d/%Y').date()
                    vix[d]=(float(r['CLOSE']),float(r['HIGH']))
            data[name]=vix
        else:
            vintage=date.fromisoformat(f.stem)
            col=SERIES+'_'+vintage.strftime('%Y%m%d')
            data[name]={date.fromisoformat(r['observation_date']):float(r[col]) for r in csv_records(f)
                        if r[col] not in ('','.')}
    return data

def weekdays(start,end):
    return sum((start+timedelta(days=i)).weekday()<5 for i in range((end-start).days+1))

def target(day,h):
    end=day
    while weekdays(day,end)<h:
        end+=timedelta(days=1)
    return end

def vintage(origin):
    return str(origin-timedelta(days=2))+'.csv'

def make_features(ser,origin,h,observations):
    asof=origin-timedelta(days=2)
    dates=sorted(d for d in observations if d<=asof)[-505:]
    if len(dates)<160 or weekdays(dates[-1]+timedelta(days=1),asof)>2:
        raise ValueError('Insufficient or stale strictly pre-origin observations')
    lag=weekdays(dates[-1]+timedelta(days=1),origin-timedelta(days=1))
    if lag>15: raise ValueError('stale features')
    if ser=='credit':
        values=np.asarray([observations[d] for d in dates])
        length=h+lag
        hist=values[length:]-values[:-length]
        center=values[-1]
        daily=np.diff(values)
        factor=np.clip(np.sqrt(np.std(daily[-30:],ddof=1)/np.std(daily[-252:],ddof=1)),.7,2.)
        frozen=np.quantile(center+factor*(hist-np.median(hist)),PROBS)
        frozen=center+1.25*(frozen-center)
        ecdf_samples=center+hist
    else:
        values=np.asarray([observations[d][0] for d in dates])
        highs=np.asarray([observations[d][1] for d in dates])
        left=np.arange(len(values)-h-lag)
        hi=np.max(highs[left[:,None]+np.arange(1+lag,h+lag+1)],axis=1)
        ecdf_samples=hi/values[left]*values[-1]
        nearest=np.argsort(abs(np.log(values[left]/values[-1])))[:max(80,len(left)//3)]
        frozen=np.quantile(ecdf_samples[nearest],PROBS)
    # Equal mass for the short and full empirical distributions.
    grid=np.linspace(0,1,1001)
    mixed=np.concatenate((np.quantile(ecdf_samples[-126:],grid),np.quantile(ecdf_samples,grid)))
    g=np.quantile(mixed,PROBS)
    return frozen,g,dates[-1]

def masses(ser,quantiles):
    lo,hi,n=(0,25,50) if ser=='credit' else (0,120,120)
    edges=np.linspace(lo,hi,n+1)
    q=np.asarray(quantiles,dtype=float).copy()
    if not np.all(np.isfinite(q)) or np.any(np.diff(q)<-1e-10):
        raise ValueError('Invalid quantiles')
    for i in range(1,len(q)):
        q[i]=max(q[i],q[i-1]+1e-7)
    res=np.interp(edges,q,PROBS)
    left=edges<q[0]; right=edges>q[-1]
    left_scale=(q[1]-q[0])*PROBS[0]/(PROBS[1]-PROBS[0])
    right_scale=(q[-1]-q[-2])*(1-PROBS[-1])/(PROBS[-1]-PROBS[-2])
    res[left]=PROBS[0]*np.exp(np.maximum(-745,(edges[left]-q[0])/left_scale))
    res[right]=1-(1-PROBS[-1])*np.exp(np.maximum(-745,-(edges[right]-q[-1])/right_scale))
    weights=.99*np.diff(np.concatenate(([0.],res,[1.])))+.01/(n+2)
    if np.min(weights)<=0 or abs(weights.sum()-1)>1e-8: raise ValueError('Invalid probability mass')
    return edges,weights

def outcome(ser,origin,h,next_origin,obs):
    end=target(origin,h)
    if ser=='credit':
        later=obs[vintage(next_origin)]
        dates=[d for d in later if d<=end]
        if not dates or weekdays(max(dates)+timedelta(days=1),end)>1 or max(dates)<origin:
            raise ValueError('Missing OAS resolution bar')
        return later[max(dates)]
    dates=[d for d in obs['vix.csv'] if origin<=d<=end]
    if not dates or len(dates)<h-2:
        raise ValueError('Missing VIX target bar')
    return max(obs['vix.csv'][d][1] for d in dates)

def scores(ser,quantiles,y):
    edges,m=masses(ser,quantiles)
    # Separate endpoints: exact upper bound goes to final internal bin.
    i=0 if y<edges[0] else len(m)-1 if y>edges[-1] else min(len(edges)-1,int(np.searchsorted(edges,y,side='right')))
    return float(np.log(m[i]))

def scale(q,w):
    return q[2]+w*(q-q[2])

def quarter(origin):
    return origin[:4]+'Q'+str((int(origin[5:7])-1)//3+1)

def adjust(raw):
    names=sorted(raw,key=raw.get);n=len(names);running=0;out={}
    for i,k in enumerate(names):
        running=max(running,min(1,(n-i)*raw[k])); out[k]=running
    return out

def summarize(records,model):
    monthly=defaultdict(list)
    for x in records: monthly[x['origin']].append(x['score'][model]-x['score']['A'])
    # Each paired date requires both horizons. Then weight its two targets equally.
    dates={o:statistics.mean(d) for o,d in monthly.items() if len(d)==2}
    byq=defaultdict(list)
    for o,d in dates.items(): byq[quarter(o)].append(d)
    quarters=np.array([np.mean(g) for _,g in sorted(byq.items()) if len(g)==3])
    if not len(quarters) or not dates:
        return {'origins':len(dates),'full_quarters':0,'status':'insufficient paired dates'}
    avg=quarters.mean()
    if len(quarters)>19: raise ValueError('enumeration bound')
    count=sum(np.mean(np.array(signs)*quarters)>=avg-1e-14 for signs in itertools.product((-1,1), repeat=len(quarters)))
    rng=np.random.default_rng(77411)
    sims=quarters[rng.integers(0,len(quarters),(8192,len(quarters)))].mean(axis=1)
    ci=list(map(float,np.quantile(sims,[.025,.975])))
    halves=[[v for k,v in dates.items() if (int(k[:4])<=2022)==b] for b in (True,False)]
    return {'origins':len(dates),'full_quarters':len(quarters),'paired_questions':2*len(dates),
            'mean_log_gain':float(np.mean(list(dates.values()))),'mean_full_quarters':float(avg),
            'quarter_bootstrap_95':ci if len(quarters)>=2 else None,
            'p_quarter_sign_permutation':count/2**len(quarters),
            'win_tie_loss_origins':[sum(v>1e-12 for v in dates.values()),sum(abs(v)<=1e-12 for v in dates.values()),sum(v< -1e-12 for v in dates.values())],
            'two_year_halves':[float(np.mean(z)) if z else None for z in halves]}

def main():
    obs=load_sources(); dates=list(origins()); forecasts=[]; omitted=[]
    # Freeze label-free features for all arms, then resolve.
    for o,next_o in zip(dates,dates[1:]):
        for s in ('credit','vix'):
            for h in (5,10):
                key={'series':s,'origin':o.isoformat(),'days':h,'target_date':target(o,h).isoformat()}
                try:
                    data=obs[vintage(o)] if s=='credit' else obs['vix.csv']
                    a,g,latest=make_features(s,o,h,data)
                    assert latest < o
                    forecasts.append((key,o,next_o,h,a,g,latest))
                except (KeyError,ValueError) as e:
                    omitted.append({**key,'stage':'forecast','reason':str(e)})
    rows=[]
    for key,o,no,h,a,g,latest in forecasts:
        s=key['series']
        try:
            y=outcome(s,o,h,no,obs); assert np.isfinite(y)
            rows.append({**key,'last_input':str(latest),'label':float(y),'A_quantiles':a.tolist(),'G_quantiles':g.tolist(),
                         'A':scores(s,a,y),'G':scores(s,g,y),'Cs':{w:scores(s,scale(a,w),y) for w in WIDTHS}})
        except (KeyError, ValueError) as e:
            omitted.append({**key,'stage':'resolution','reason':str(e)})
    fitted={}; summary={}; out=[]
    for s in ('credit','vix'):
        subset=[r for r in rows if r['series']==s]
        dev=[r for r in subset if r['origin']<'2021']; test=[r for r in subset if r['origin']>='2021']
        n=len({r['origin'] for r in dev}); width=1.
        if n>=18: width=max(WIDTHS,key=lambda w:np.mean([r['Cs'][w] for r in dev]))
        route={}
        for h in (5,10):
            de=[r for r in dev if r['days']==h]
            route[h]='G' if len(de)>=18 and np.mean([r['G']-r['A'] for r in de])>.02 else 'A'
        fitted[s]={'development_months':n,'development_questions':len(dev),'C_width':width,'F_route':route}
        for r in test:
            sc=dict(A=r['A'],C=r['Cs'][width],G=r['G'],F=r[route[r['days']]])
            out.append({k:v for k,v in r.items() if k not in ('A','G','Cs')}|{'score':sc})
        data=[r for r in out if r['series']==s]
        summary[s]={k:summarize(data,k) for k in METHODS}
        methods={k:v['p_quarter_sign_permutation'] for k,v in summary[s].items() if 'p_quarter_sign_permutation' in v}
        for k,v in adjust(methods).items():
            stat=summary[s][k]; stat['p_holm']=v
            stat['meets_archival_screen']=bool(stat['full_quarters']>=12 and v<.05 and stat['quarter_bootstrap_95'][0]>0 and all(x is not None and x>0 for x in stat['two_year_halves']))
    (ROOT/'paired_rows.json').write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n')
    report={'design_sha256':DESIGN,'primary':'credit','shadow_only':'vix','controls':fitted,'summary':summary,
            'expected_cases_all_splits':(len(dates)-1)*4,'forecasted':len(forecasts),'resolved_paired_all_splits':len(rows),
            'omitted':omitted,'notes':['No tournament questions, peer scores or community predictions. These are independent economic analogues.', 'Vintage ALFRED input and following-month vintage label; exchange archive VIX reconstructed retrospectively.', 'Frozen numerical credit comparator; no hosted model or live calibration gate is simulated. A positive archival result cannot authorize a production promotion.']}
    (ROOT/'paired_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('controls','summary','forecasted','resolved_paired_all_splits')}))
if __name__=='__main__': main()
