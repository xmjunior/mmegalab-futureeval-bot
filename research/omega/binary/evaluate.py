"""Evaluate the fixed OMEGA methods. No providers, research APIs or network calls."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
MODELS=("haiku","mini","deepseek")
PRIMARY=("venn_mean","correlation_pool","hedge","disagreement")
EPS=1e-6


def logits(p):
    q=np.clip(p,EPS,1-EPS)
    return np.log(q/(1-q))


def logistic(z):
    return 1/(1+np.exp(-z))


def pava_value(xs, ones, counts, x, candidate):
    """At score x, augment the tied score bin, then pool adjacent violators."""
    at=int(np.searchsorted(xs,x))
    if at < len(xs) and xs[at] == x:
        yes=ones.copy()
        weight=counts.copy()
        yes[at]+=candidate
        weight[at]+=1
    else:
        yes=np.insert(ones,at,candidate)
        weight=np.insert(counts,at,1)
    blocks=[]
    for i,(s,n) in enumerate(zip(yes,weight)):
        blocks.append([i,i,float(s),float(n)])
        while len(blocks)>1:
            a,b=blocks[-2:]
            if a[2]*b[3] <= b[2]*a[3]:
                break
            blocks[-2:]=[[a[0],b[1],a[2]+b[2],a[3]+b[3]]]
    return next(b[2]/b[3] for b in blocks if b[0]<=at<=b[1])


def venn(score,y):
    xs,inverse,counts=np.unique(score,return_inverse=True,return_counts=True)
    ones=np.bincount(inverse,weights=y)
    @lru_cache(None)
    def predict(p):
        lo=pava_value(xs,ones,counts,p,0)
        hi=pava_value(xs,ones,counts,p,1)
        if not 0 <= lo <= hi <= 1:
            raise ValueError("isotonic interval violated")
        return hi/(1-lo+hi)
    return predict


def bucket(row):
    return (row[2],(date.fromisoformat(row[4])-date.fromisoformat(row[1])).days<=30)


def train_hedge(rows):
    """One common update per round/bucket, so large rounds cannot dominate."""
    byround=defaultdict(list)
    for r in rows:
        byround[(r[1],None)].append(r)
        byround[(r[1],bucket(r))].append(r)
    weights,counts={},Counter()
    for (_,key),block in sorted(byround.items(),key=lambda kv:(kv[0][0],str(kv[0][1]))):
        w=weights.get(key,np.ones(3)/3)
        p=np.array([r[6:9] for r in block])
        y=np.array([r[9] for r in block])
        updated=w*np.exp(-2*np.mean((p-y[:,None])**2,axis=0))
        weights[key]=0.95*updated/updated.sum()+0.05/3
        counts[key]+=1
    def get(key):
        k=counts[key]
        return (k*weights.get(key,weights[None])+5*weights[None])/(k+5)
    return get


def components(train):
    probs=np.array([r[6:9] for r in train])
    y=np.array([r[9] for r in train])
    prior_loss=np.mean((probs-y[:,None])**2,axis=0)
    best=min(range(len(MODELS)),key=lambda i:(prior_loss[i],MODELS[i]))
    corr=np.corrcoef((probs-y[:,None]).T)
    rho=float(np.clip(0.8*np.mean(corr[np.triu_indices(3,k=1)]),0,1))
    alpha=min(1.8,math.sqrt(3/(1+2*rho)))
    return best,alpha,train_hedge(train),{"best_prior_model":MODELS[best],"residual_correlation":corr.tolist(),"shrunk_mean_rho":rho,"log_odds_multiplier":alpha}


def primitive(rows, best, alpha, hedge):
    p=np.array([r[6:9] for r in rows],dtype=float)
    m=p.mean(axis=1)
    logmean=logits(p).mean(axis=1)
    w=np.array([hedge(bucket(r)) for r in rows])
    return {**dict(zip(MODELS,p.T)), "coin": m*0+0.5,"best_prior": p[:,best],
            "mean":m,"median":np.median(p,axis=1),"fixed_extreme":logistic(1.5*logmean),
            "correlation_pool":logistic(alpha*logmean),"hedge":np.sum(p*w,axis=1),
            "hedge_top1":p[np.arange(len(p)), np.argmax(w,axis=1)],
            "disagreement":np.where(np.ptp(p,axis=1)>=0.30,0.5*m+0.25,m)}


def predict(train,valid,test):
    best,alpha,hedge,fit=components(train)
    validation=primitive(valid,best,alpha,hedge)
    result=primitive(test,best,alpha,hedge)
    y=np.array([r[9] for r in valid])
    for key, arm in (("mean","venn_mean"),("correlation_pool","venn_corr")):
        calibrator=venn(validation[key],y)
        result[arm]=np.array([calibrator(p) for p in result[key]])
    return result,fit


def losses(p,y):
    pc=np.clip(p,EPS,1-EPS)
    return (p-y)**2,-np.log(np.where(y==1,pc,1-pc))


def summarize(p,y):
    bs,ll=losses(p,y)
    index=np.minimum(9,(p*10).astype(int))
    bins=[]
    for k in sorted(set(index.tolist())):
        mask=index==k
        bins.append((int(mask.sum()),float(p[mask].mean()),float(y[mask].mean())))
    reliability=sum(n*(m-f)**2 for n,m,f in bins)/len(y)
    resolution=sum(n*(f-y.mean())**2 for n,m,f in bins)/len(y)
    return {"n":len(y),"brier":float(bs.mean()),"log_loss":float(ll.mean()),
            "log_loss_p95":float(np.quantile(ll,.95)),"ece10":sum(n*abs(m-f) for n,m,f in bins)/len(y),
            "binned_reliability":reliability,"binned_resolution":resolution}


def boot_interval(gains, keys, circular=False):
    groups=sorted(set(keys))
    ix={k:i for i,k in enumerate(groups)}
    blocks=np.array([ix[k] for k in keys])
    totals=np.bincount(blocks,weights=gains,minlength=len(groups))
    counts=np.bincount(blocks,minlength=len(groups))
    rng=np.random.default_rng(331125)
    m=len(groups)
    if circular:
        starts=rng.integers(0,m,(5000,(m+1)//2))
        samples=np.stack((starts,(starts+1)%m),axis=-1).reshape(5000,-1)[:,:m]
    else:
        samples=rng.integers(0,m,(5000,m))
    vals=totals[samples].sum(axis=1)/counts[samples].sum(axis=1)
    return np.quantile(vals,[.025,.975]).tolist()


def round_sign_p(gains,dates):
    values=[sum(g for g,d in zip(gains,dates) if d==rd) for rd in sorted(set(dates))]
    wins=sum(x>1e-12 for x in values)
    n=sum(abs(x)>1e-12 for x in values)
    return sum(math.comb(n,k) for k in range(wins,n+1))/(2**n)


def family_bucket(family):
    # Whole normalized geographic/series families belong to a single fold.
    return int(hashlib.sha256(family.encode()).hexdigest(),16)%10


def block_sign_p(gains,dates):
    # Paired adjacent non-overlapping two-round blocks. One-sided exact
    # randomization is descriptive given cross-block residual dependence.
    rounds=sorted(set(dates))
    blocks=[sum(float(g) for g,d in zip(gains,dates) if d in rounds[i:i+2]) for i in range(0,len(rounds),2)]
    observed=sum(blocks)
    return sum(sum(a*b for a,b in zip(s,blocks))>=observed-1e-12
               for s in itertools.product([-1,1],repeat=len(blocks)))/(2**len(blocks))


def evaluate(rows,manifest):
    train,valid,test=([r for r in rows if r[0]==part] for part in ("train","valid","test"))
    p,fit=predict(train,valid,test)
    y=np.array([r[9] for r in test])
    dates=[r[1] for r in test]
    sources=[r[2] for r in test]
    # Data-source blocks additionally preserve common shocks across underlying
    # ticker series and opaque question identifiers within a provider.
    scores={arm:summarize(prob,y) for arm,prob in p.items()}
    controls={"haiku":1,"mini":1,"deepseek":1,"hedge_top1":1,"best_prior":1,"coin":0}
    for arm,s in scores.items():
        s["archived_model_forecasts_per_case"]=controls.get(arm,3)
    contrasts={}
    for arm in PRIMARY:
        gain=losses(p["mean"],y)[0]-losses(p[arm],y)[0]
        contrasts[arm]={"gain_mean_brier":float(gain.mean()),"brier_skill_vs_mean":float(gain.mean()/scores["mean"]["brier"]),
              "ci95_two_round_blocks":boot_interval(gain,dates,True),
              "ci95_whole_source_blocks":boot_interval(gain,sources),
              "ci95_same_target_date_blocks":boot_interval(gain,[r[4] for r in test]),
              "ci95_family_blocks":boot_interval(gain,[r[10] for r in test]),
              "p_round_sign":round_sign_p(gain,dates),
              "p_adjacent_block_sign":block_sign_p(gain,dates)}
        c=contrasts[arm]
        c["ci95_conservative_envelope"]=[min(c["ci95_family_blocks"][0],c["ci95_two_round_blocks"][0]),
                                          max(c["ci95_family_blocks"][1],c["ci95_two_round_blocks"][1])]
        c["brier_skill_vs_best_prior"]=1-scores[arm]["brier"]/scores["best_prior"]["brier"]
    running=0
    for i,(arm,c) in enumerate(sorted(contrasts.items(),key=lambda v:v[1]["p_round_sign"])):
        running=max(running,min(1,(len(PRIMARY)-i)*c["p_round_sign"]))
        c["holm_p"]=running
        c["screen_pass"]=bool(c["brier_skill_vs_mean"]>=.02
                 and c["brier_skill_vs_best_prior"]>=.02
                 and scores[arm]["log_loss"]<min(scores["mean"]["log_loss"],scores["best_prior"]["log_loss"])
                 and scores[arm]["log_loss_p95"]<=scores["mean"]["log_loss_p95"]
                 and c["ci95_conservative_envelope"][0]>0
                 and running<.05)
    grouped={}
    for source in sorted(set(sources)):
        mask=np.array([s==source for s in sources])
        grouped[source]={arm:summarize(p[arm][mask],y[mask]) for arm in ["mean","best_prior",*PRIMARY]}
    split_rows={}
    for data in (train,valid,test):
        info=manifest["coverage"]
        periods=sorted(d for d in info if ("train" if d < "2026-01-04" else "valid" if d < "2026-03-01" else "test")==data[0][0])
        common=len(data)
        possible=sum(info[d].get("eligible_first_issue",0) for d in periods)
        split_rows[data[0][0]]={"rounds":periods,"native_common":common,"eligible_first_issue":possible,"coverage":common/possible,
                                "families":len({(r[2],r[3]) for r in data})}
    train_f=[r for r in train if family_bucket(r[10])<5]
    valid_f=[r for r in valid if 5<=family_bucket(r[10])<7]
    test_f=[r for r in test if family_bucket(r[10])>=7]
    pf,fit_f=predict(train_f,valid_f,test_f)
    yf=np.array([r[9] for r in test_f])
    families={part:sorted(set(r[10] for r in data)) for part,data in
              (("train",train_f),("valid",valid_f),("test",test_f))}
    disjoint={"n":[len(train_f),len(valid_f),len(test_f)],"family_counts":{k:len(v) for k,v in families.items()},
              "train_fit":fit_f,"scores":{a:summarize(pp,yf) for a,pp in pf.items()}}
    if any(set(families[a])&set(families[b]) for a,b in itertools.combinations(families,2)):
        raise ValueError("same family entered multiple folds")
    horizons={}
    for span in ("short","long"):
        mask=np.array([bucket(r)[1]==(span=="short") for r in test])
        horizons[span]={"n_fit":sum(bucket(r)[1]==(span=="short") for r in train),
                        "n_calibration":sum(bucket(r)[1]==(span=="short") for r in valid),
                        "n_test":int(mask.sum()),"test_scores":{}}
        if mask.any():
            horizons[span]["test_scores"]={a:summarize(pp[mask],y[mask]) for a,pp in p.items()}
    known={(r[2],r[3]) for r in train+valid}
    novel=np.array([(r[2],r[3]) not in known for r in test])
    sensitivity={"n":int(novel.sum()),"unseen_series_test_scores":{}}
    if novel.any():
        sensitivity["unseen_series_test_scores"]={arm:summarize(prob[novel],y[novel]) for arm,prob in p.items()}
    return {"family_disjoint_sensitivity":disjoint,"horizons":horizons,"novel_underlying_sensitivity":sensitivity,"status":"historical_only","input_rows_sha256":manifest["rows_jsonl_sha256"],
            "cost_usd_new_inference":0,"original_provider_costs":"unavailable",
            "train_fit":fit,"splits":split_rows,"test_scores":scores,"primary_contrasts":contrasts,
            "test_by_source":grouped,"promotion_allowed":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,default=HERE)
    p.add_argument("--out",type=Path,default=HERE/"results.json")
    args=p.parse_args()
    manifest=json.loads((args.inputs/"inputs.json").read_bytes())
    raw=gzip.decompress((args.inputs/"rows.jsonl.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest()!=manifest["rows_jsonl_sha256"]:
        raise ValueError("derived rows changed")
    rows=[json.loads(line) for line in raw.splitlines()]
    results=evaluate(rows,manifest)
    args.out.write_text(json.dumps(results,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print({s:v["native_common"] for s,v in results["splits"].items()})


if __name__=="__main__":
    main()
