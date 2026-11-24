"""Optional numerical parity probe against frozen parent branch's market_prices.py."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import paired_economic as study

p=argparse.ArgumentParser();p.add_argument('reference',type=Path);args=p.parse_args()
body=args.reference.read_bytes()
if hashlib.sha256(body).hexdigest()!='8884b7db667b072d238c62950fb98b2dca042ee67293ccd6cdf656c0b565b8cb':
    raise ValueError('Unexpected frozen reference bytes')
spec=importlib.util.spec_from_file_location('market_prices',args.reference)
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
vix={dt.datetime.strptime(r['DATE'],'%m/%d/%Y').date():(float(r['CLOSE']),float(r['HIGH'])) for r in study.csv_records(study.DATA/'vix.csv') if r.get('CLOSE') and r.get('HIGH')}
col='BAMLH0A0HYM2_20241123'
credit={dt.date.fromisoformat(r['observation_date']):float(r[col]) for r in study.csv_records(study.DATA/'2024-11-23.csv') if r[col] not in ('','.')}
ref._vix=lambda:(vix,'comparison snapshot')
ref._fred=lambda *a:(credit,'comparison snapshot')
o=dt.date(2024,11,25)
spot=dt.datetime.combine(o,dt.time(12),dt.timezone.utc)
errors=[]
for series,pid,width,values in [('credit',46010,1.25,credit),('vix',46015,1.,vix)]:
    for h in (5,10):
        end=dt.datetime.combine(study.target(o,h),dt.time(21),dt.timezone.utc)
        expected=list(ref.model(pid,spot-dt.timedelta(seconds=1),spot,end).quantiles(width).values())
        computed=study.make_features(series,o,h,values)[0]
        errors.append(float(np.max(np.abs(computed-expected))))
assert max(errors)<1e-12, errors
print('4 economic forecast targets, 20 quantiles, maximum absolute error:',max(errors))
