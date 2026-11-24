"""Export anonymous paired losses and calibration diagnostic from local study output."""
import csv
import json

import numpy as np

from paired_economic import ROOT

def main():
    diagnostics={}
    cases=[]
    for cohort,filename in [('base','paired_rows.json'),('credit_extension','credit_extension_rows.json')]:
        rows=json.loads((ROOT/filename).read_text())
        for row in rows:
            s=row.get('series','credit')
            h=row['days'] if 'days' in row else row['h']
            scores=row['score']
            cases.append({'cohort':cohort,'family':s,'origin':row['origin'],'business_days':h,**{f'delta_{m}':scores[m]-scores['A'] for m in ('C','F','G')}})
            for m in ('A','G'):
                q=row[m+'_quantiles']; y=row['label']
                bucket=diagnostics.setdefault((cohort,s,m),[])
                bucket.append([q[0]<=y<=q[-1],q[1]<=y<=q[-2],q[-1]-q[0]])
    with (ROOT/'paired_gains.csv').open('w',newline='') as dest:
        w=csv.DictWriter(dest,fieldnames=['cohort','family','origin','business_days','delta_C','delta_F','delta_G'])
        w.writeheader();w.writerows(cases)
    summary={str('/'.join(k)):{'questions':len(rows),'nominal_80_observed':float(np.mean([r[0] for r in rows])),
                            'nominal_50_observed':float(np.mean([r[1] for r in rows])),
                            'mean_width_80':float(np.mean([r[2] for r in rows]))} for k,rows in diagnostics.items()}
    (ROOT/'calibration_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__': main()
