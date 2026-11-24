import gzip
import json
import unittest
from pathlib import Path

import numpy as np

import derive
import evaluate as ev


class BinaryTests(unittest.TestCase):
    def test_venn_matches_independent_minmax_for_both_labels_and_ties(self):
        scores=np.array([.1,.1,.2,.45,.7,.8])
        y=np.array([1.,0.,0.,1.,0.,1.])
        xs,inv,count=np.unique(scores,return_inverse=True,return_counts=True)
        yes=np.bincount(inv,weights=y)
        for value in (0,.1,.3,.7,1.):
            estimates=[]
            for label in (0,1):
                s=np.append(scores,value)
                labels=np.append(y,label)
                # Isotonic min-max representation with ties as one score.
                expected=max(min(labels[(s>=lo)&(s<=hi)].mean() for hi in set(s[s>=value]))
                             for lo in set(s[s<=value]))
                self.assertAlmostEqual(ev.pava_value(xs,yes,count,value,label),expected)
                estimates.append(expected)
            p=ev.venn(scores,y)(value)
            self.assertAlmostEqual(p,estimates[1]/(1-estimates[0]+estimates[1]))

    def test_final_predictions_ignore_every_holdout_label(self):
        train=[["train",f"2025-11-{i:02}","fred",str(i),"2025-12-31","[]",.1+i*.02,.2+i*.01,.8-i*.015,i%2] for i in range(1,22)]
        validation=[["valid","2026-01-04",*r[2:4],"2026-02-14",*r[5:]] for r in train[:10]]
        test=[["test","2026-03-01",*r[2:4],"2026-05-08",*r[5:]] for r in train[:8]]
        a,_=ev.predict(train,validation,test)
        altered=[[*r[:-1],1-r[-1]] for r in test]
        b,_=ev.predict(train,validation,altered)
        for arm in a:
            np.testing.assert_array_equal(a[arm],b[arm])

    def test_hedge_ignores_order_within_round(self):
        rows=[["train","2025-11-23","fred",str(i),"2025-12-20","[]",.2,.4,.8,i%2] for i in range(12)]
        w1=ev.train_hedge(rows)(ev.bucket(rows[0]))
        w2=ev.train_hedge(list(reversed(rows)))(ev.bucket(rows[0]))
        np.testing.assert_almost_equal(w1,w2)
        self.assertAlmostEqual(float(sum(w1)),1)

    def test_real_rows_unique_temporally_eligible_and_manifest_locked(self):
        path=Path(__file__).resolve().parent
        manifest=json.loads((path/"inputs.json").read_bytes())
        raw=gzip.decompress((path/"rows.jsonl.gz").read_bytes())
        self.assertEqual(derive.digest(raw),manifest["rows_jsonl_sha256"])
        self.assertEqual(manifest["columns"][-1],"dependence_family")
        self.assertEqual(len(manifest["columns"]),11)
        events=set()
        for line in raw.splitlines():
            r=json.loads(line)
            self.assertEqual(len(manifest["columns"]),len(r))
            split,cutoff,_,snapshot=next(p for p in derive.PHASE if r[1]<p[1])
            self.assertEqual(split,r[0])
            self.assertLess(r[1],r[4])
            self.assertLessEqual(r[4],snapshot[:10])
            self.assertLess(r[4],cutoff)
            event=tuple(r[2:5])
            self.assertNotIn(event,events)
            self.assertIn(r[2],derive.SOURCES)
            self.assertTrue(all(0<=x<=1 for x in r[6:9]))
            events.add(event)
        self.assertEqual(len(manifest["forecast_sha256"]),45)
        self.assertEqual(sum(h is None for h in manifest["resolution_sha256"].values()),2)


if __name__=="__main__":
    unittest.main()
