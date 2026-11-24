"""Replay U.S. unemployment vintages. Only the leakage diagnostic sees later revisions."""
import calendar
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "unrate_realtime.csv"
P = np.array([.1, .25, .5, .75, .9])
PGRID = (np.arange(99) + .5) / 99
NORMAL_CDF = np.vectorize(NormalDist().cdf, otypes=[float])
NORMAL_Q = np.vectorize(NormalDist().inv_cdf, otypes=[float])
HORIZONS = (1, 3, 6)
SEED = 20261124
LATEST = date(2025, 12, 31)
FIRST_VALID = date(2007, 1, 1)
FIRST_TEST = date(2016, 1, 1)


def month_shift(d, k):
    y, m0 = divmod(d.year * 12 + d.month - 1 + k, 12)
    return date(y, m0 + 1, 1)


def archive():
    observations = defaultdict(list)
    with DATA.open() as f:
        for r in csv.DictReader(f):
            if not r["UNRATE"]:
                continue
            obs = date.fromisoformat(r["period_start_date"])
            start = date.fromisoformat(r["realtime_start_date"])
            end = date.fromisoformat(r["realtime_end_date"]) if r["realtime_end_date"] else date.max
            if start < obs or end < start:
                raise ValueError("Invalid vintage")
            observations[obs].append((start, end, float(r["UNRATE"])))
    for versions in observations.values():
        versions.sort()
        if any(old[1] >= new[0] for old, new in zip(versions, versions[1:])):
            raise ValueError("Overlapping release intervals")
    return observations


def snapshot(rows, asof):
    return {obs: value for obs, versions in rows.items() if obs <= asof
            for start, end, value in versions if start <= asof <= end}


def split(origin, known):
    if origin < FIRST_VALID:
        return "train" if known < FIRST_VALID else "purged"
    if origin < FIRST_TEST:
        return "valid" if known < FIRST_TEST else "purged"
    return "test"


def cohort(rows):
    revised = snapshot(rows, LATEST)
    cases = []
    for year in range(1990, 2025):
        for month in (3, 6, 9, 12):
            origin = date(year, month, calendar.monthrange(year, month)[1])
            available = snapshot(rows, origin)
            last = max(available)
            dates = [month_shift(last, i) for i in range(-119, 1)]
            if any(d not in available for d in dates):
                continue
            for h in HORIZONS:
                target = month_shift(last, h)
                if target not in rows:
                    continue
                published, _, actual = rows[target][0]
                if published <= origin or published > LATEST:
                    raise ValueError("Unavailable holdout")
                cases.append((origin, last, h, target, published, actual, np.array([available[d] for d in dates]),
                              np.array([revised[d] for d in dates]), split(origin, published)))
    return cases


def strict_quantiles(q):
    q = np.asarray(q, dtype=float).copy()
    for i in range(1, len(q)):
        q[i] = max(q[i], q[i-1] + .025)
    return q


class Distribution:
    """Truncated Gaussian or five quantiles joined by linear CDF and exponential tails."""
    def __init__(self, center, scale, knots=None, mixture=0.0):
        self.center, self.scale = float(center), max(float(scale), .05)
        self.q = strict_quantiles(knots) if knots is not None else None
        self.mix = mixture
        if self.q is not None:
            self.slopes = np.diff(P) / np.diff(self.q)
            self.left = P[0] / self.slopes[0]
            self.right = (1 - P[-1]) / self.slopes[-1]
        self.low = self.raw_cdf(0.)
        self.mass = self.raw_cdf(100.) - self.low
        if not self.mass > 0:
            raise ValueError("Lost physical support")

    def raw_cdf(self, x):
        x = np.asarray(x)
        if self.q is None:
            return NORMAL_CDF((x - self.center) / self.scale)
        q = self.q
        cdf = np.interp(x, q, P)
        cdf = np.where(x < q[0], P[0] * np.exp(np.minimum(0, (x - q[0]) / self.left)), cdf)
        cdf = np.where(x > q[-1], 1 - (1-P[-1])*np.exp(np.minimum(0, (q[-1]-x)/self.right)), cdf)
        if self.mix:
            cauchy = .5 + np.arctan((x - self.center) / self.scale) / math.pi
            cdf = (1-self.mix)*cdf + self.mix*cauchy
        return cdf

    def cdf(self, x):
        return np.clip((self.raw_cdf(x) - self.low) / self.mass, 0, 1)

    def log_density(self, x):
        if not 0 <= x <= 100:
            return -math.inf
        if self.q is None:
            z = (x - self.center) / self.scale
            logpdf = -.5*z*z - math.log(self.scale*math.sqrt(2*math.pi))
        elif x < self.q[0]:
            logpdf = math.log(P[0]/self.left) + (x-self.q[0])/self.left
        elif x >= self.q[-1]:
            logpdf = math.log((1-P[-1])/self.right) + (self.q[-1]-x)/self.right
        else:
            logpdf = math.log(self.slopes[np.searchsorted(self.q, x, side="right")-1])
        if self.mix:
            z = (x - self.center) / self.scale
            tail = -math.log(math.pi*self.scale*(1+z*z))
            logpdf = np.logaddexp(math.log1p(-self.mix)+logpdf, math.log(self.mix)+tail)
        return float(logpdf - math.log(self.mass))

    def quantile(self, p):
        p = np.asarray(p)
        rawp = self.low + p*self.mass
        if self.q is None:
            return self.center + self.scale*NORMAL_Q(rawp)
        if not self.mix:
            xs = np.interp(rawp, P, self.q)
            below, above = rawp < P[0], rawp > P[-1]
            xs[below] = self.q[0] + self.left*np.log(rawp[below]/P[0])
            xs[above] = self.q[-1] - self.right*np.log((1-rawp[above])/(1-P[-1]))
            return xs
        lo, hi = np.zeros_like(p), np.full_like(p, 100.)
        for _ in range(40):
            mid = (lo+hi)/2
            low_mask = self.raw_cdf(mid) < rawp
            lo, hi = np.where(low_mask, mid, lo), np.where(low_mask, hi, mid)
        return (lo+hi)/2


def model_knots(hist, h):
    center = hist[-1]
    diff = np.diff(hist)
    move = hist[h:] - hist[:-h]
    shift = move - np.median(move)
    empirical = np.quantile(center + shift, P)
    longvol = np.std(diff, ddof=1)
    ratio = (np.std(diff[-30:], ddof=1)/longvol)**.5 if longvol else 1.
    champion = center + np.clip(ratio, .7, 2.) * (empirical-center)
    return empirical, champion


def arms(hist, h, residual):
    center = hist[-1]
    emp, champ = model_knots(hist, h)
    sd = np.std(np.diff(hist[-61:]), ddof=1) * math.sqrt(h)
    changes, weights = [], []
    for n in (24, 60, 120):
        data = hist[-n:]
        moves = data[h:] - data[:-h]
        changes.extend(moves)
        weights.extend(np.full(len(moves), 1./(3*len(moves))))
    order = np.argsort(changes)
    x, w = np.array(changes)[order], np.array(weights)[order]
    med = np.interp(.5, np.cumsum(w)-.5*w, x)
    tri = center + np.interp(P, np.cumsum(w)-.5*w, x)-med
    mu = np.mean(hist)
    a, b = hist[:-1]-mu, hist[1:]-mu
    phi = np.clip((a@b+1)/(a@a+1), 0, 1.)
    ar_center = mu+phi**h*(center-mu)
    errors = hist[h:] - (mu+phi**h*(hist[:-h]-mu))
    ar_scale = np.std(errors, ddof=1)
    corrected = strict_quantiles(champ)
    width = corrected[3]-corrected[1]
    corrected[[0,4]] += np.array([-1.,1.])*residual*width
    return {
        "naive_rw": Distribution(center, sd),
        "empirical": Distribution(center, sd, emp),
        "champion_rule": Distribution(center, sd, champ),
        "three_scales": Distribution(center, sd, tri),
        "ridge_ar": Distribution(ar_center, ar_scale),
        "cqr_tails": Distribution(center, max(width, .05), corrected, .03),
    }


def calibration(cases):
    residuals = defaultdict(list)
    for origin,last,h,target,known,y,hist,_,group in cases:
        if group != "train":
            continue
        _, q = model_knots(hist,h)
        lower, q25, _, q75, upper = strict_quantiles(q)
        residuals[h].append(max(lower-y,y-upper)/(q75-q25))
    return {h: max(0, sorted(s)[math.ceil(.8*(len(s)+1))-1]) for h,s in residuals.items()}


def metrics(dist, y, threshold):
    x = dist.quantile(np.r_[PGRID, P])
    if not np.all(np.isfinite(x)) or np.any(np.diff(x[:99]) < 0) or np.min(x)<0 or np.max(x)>100:
        raise ValueError("Invalid quantile function")
    q10,q25,_,q75,q90 = x[-5:]
    e = y-x[:99]
    out = dict(crps_q99=float(2*np.mean(np.where(e>=0,PGRID*e,(PGRID-1)*e))),
               nll=-dist.log_density(y), brier=float((1-dist.cdf(threshold) - (y>threshold))**2),
               cover80=float(q10 <= y <= q90), cover50=float(q25 <= y <= q75),
               width80=q90-q10, width50=q75-q25, pit=float(dist.cdf(y)))
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError("Invalid score")
    return out


def summary(rows):
    values = {k: round(float(np.mean([r[k] for r in rows])), 7)
              for k in ("crps_q99","nll","brier","cover80","cover50","width80","width50")}
    pit = np.sort([r["pit"] for r in rows])
    n=len(rows)
    values.update(n=n, origins=len({r["origin"] for r in rows}),
                  pit_hist=np.histogram(pit, np.linspace(0,1,11))[0].tolist(),
                  pit_ks_descriptive=round(max(np.max(np.arange(1,n+1)/n - pit),
                                               np.max(pit-np.arange(n)/n)),7))
    return values


def write_results():
    cases = cohort(archive())
    expansion = calibration(cases)
    scored = []
    pairs = []
    affected = 0
    example = None
    for origin,last,h,target,known,y,hist,revised,group in cases:
        if group == "purged":
            continue
        bundle = arms(hist,h,expansion[h])
        bundle["revised_diagnostic"] = arms(revised,h,0.)["champion_rule"]
        if group == "test" and np.any(hist != revised):
            affected += 1
            if not example:
                j = int(np.flatnonzero(hist != revised)[-1])
                example = dict(origin=str(origin), obs=str(month_shift(last,j-119)), historic=hist[j],revised=revised[j])
        pair = dict(origin=str(origin), last_known_obs=str(last), horizon_months=h, target_month=str(target),
                    first_release=str(known), actual=y, anchor=hist[-1], threshold=round(hist[-1]+.2, 2))
        for name,d in bundle.items():
            if group == "train" and name == "cqr_tails":
                continue
            measurement = metrics(d,y,pair["threshold"])
            scored.append(dict(origin=str(origin),h=h,group=group,model=name,**measurement))
            pair[name+"_crps_q99"] = round(measurement["crps_q99"], 7)
        if group == "test":
            pairs.append(pair)
    with (HERE/"test_pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pairs[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)
    test = [r for r in scored if r["group"]=="test"]
    arms_order = sorted({r["model"] for r in test})
    sums = {g:{m:summary(rr) for m in arms_order if (rr:=[r for r in scored if r["group"]==g and r["model"]==m])}
            for g in ("train","valid","test")}
    grouped = {str(h):{m:summary([r for r in test if r["model"]==m and r["h"]==h]) for m in arms_order}
               for h in HORIZONS}
    periods = {span:{m:summary([r for r in test if r["model"]==m and a <= r["origin"][:4] <= b])
                     for m in arms_order} for span,a,b in
               (("pre_2020","2016","2019"),("shock_period","2020","2021"),("post_2021","2022","2024"))}
    origins = sorted({r["origin"] for r in test})
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0,len(origins), size=(2000,math.ceil(len(origins)/4)))
    idx = ((starts[:,:,None]+np.arange(4))%len(origins)).reshape(2000,-1)[:,:len(origins)]
    by = {m:np.array([np.mean([r["crps_q99"] for r in test if r["origin"]==d and r["model"]==m])
                      for d in origins]) for m in arms_order}
    ci = {}
    for base in ("naive_rw","champion_rule"):
        ci[base] = {}
        for m in arms_order:
            if m in (base,"revised_diagnostic"):
                continue
            delta = by[base]-by[m]
            ci[base][m] = dict(gain=round(float(delta.mean()),7), interval=np.quantile(delta[idx].mean(axis=1),[.025,.975]).round(7).tolist())
    st = sums["test"]
    cqr = st["cqr_tails"]
    result = dict(data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256((HERE/"protocol.json").read_bytes()).hexdigest(),
                  cohort={k:sum(c[-1]==k for c in cases) for k in ("train","valid","test","purged")},
                  first_origin=str(cases[0][0]), last_origin=str(cases[-1][0]), cqr_expansion_by_horizon=expansion,
                  cqr_calibration_count={h:sum(c[-1]=="train" and c[2]==h for c in cases) for h in HORIZONS},
                  scores=sums, test_by_horizon=grouped, test_by_period=periods, bootstrap_gain_crps=ci,
                  e5_proxy_gate=(.74<=cqr["cover80"]<=.86 and all(cqr["crps_q99"] < st[m]["crps_q99"] and
                                  cqr["nll"]<st[m]["nll"] for m in ("naive_rw","empirical"))),
                  e2_full_llm_gate="unmeasured_without_prospective_llm_arm",
                  e5_full_llm_gate="unmeasured_without_archived_llm_quantile_distributions",
                  affected_test_histories=affected, revision_example=example,
                  revision_effect={k:round(st["champion_rule"][k]-st["revised_diagnostic"][k],7) for k in ("crps_q99","nll","brier")})
    (HERE/"results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


if __name__ == "__main__":
    result = write_results()
    print(json.dumps(dict(cohort=result["cohort"], test=result["scores"]["test"],
                          revision_effect=result["revision_effect"], gate=result["e5_proxy_gate"]), indent=2))
