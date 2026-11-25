"""Contract checks for the nine closed HY questions. All network calls are blocked."""
import copy
from datetime import timedelta
import json
import math
from pathlib import Path
import socket
import unittest
from unittest.mock import Mock, patch


class MarketAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(patch.object(socket.socket, "connect", side_effect=AssertionError("network disabled")))
        import market_prices
        import market_pulse
        cls.prices = market_prices
        cls.market = market_pulse
        fixture = json.loads(Path(__file__).with_name("fixtures").joinpath("market_discrete_closed.json").read_text())
        cls.rows = {int(g): [{**r, "scaling": fixture["scales"][str(r["inbound_outcome_count"])]}
                             for r in rows] for g, rows in fixture["groups"].items()}

    def post(self, rows):
        return {"id": 46010, "projects": {"default_project": {"id": self.market.PROJECT},
                "tournament": [{"id": self.market.PROJECT, "slug": "market-pulse-26q4"}]},
                "group_of_questions": {"questions": rows}}

    def test_nine_resolved_domains_have_centered_outcomes_and_official_grid(self):
        for group, rows in self.rows.items():
            for r in rows:
                with self.subTest(group=group, question=r["id"]):
                    self.assertEqual(r["status"], "resolved")
                    size = self.market.discrete_geometry(r)
                    s = r["scaling"]
                    result = float(r["resolution"])
                    left = math.floor((result - s["range_min"]) / (s["range_max"] - s["range_min"]) * (size - 1))
                    self.assertAlmostEqual(s["continuous_range"][left] + .005, result)
                    self.assertEqual(left+1, int((result-s["range_min"])/(s["range_max"]-s["range_min"])*(size-1)+1-1e-10))
        self.assertEqual({len(r["scaling"]["continuous_range"]) for rows in self.rows.values() for r in rows},
                         {177, 201, 252})

    def test_group_keeps_sdk_type_and_real_size(self):
        account = self.market.Account.__new__(self.market.Account)
        account.get = Mock(return_value=self.post(self.rows[46010]))
        questions = account.group(46010)
        self.assertEqual(set(questions), {46212, 46213, 46709})
        self.assertEqual([q.cdf_size for q in questions.values()], [201, 201, 252])
        self.assertTrue(all(isinstance(q, self.market.DiscreteQuestion) for q in questions.values()))
        self.assertEqual(set(account.group(46010, question_ids={46709})), {46709})
        self.assertIsNone(self.market.parse_market_question({"id": 46014}, self.rows[46010][0]))

    def test_original_q4_cdf_and_realized_mass_match_at_all_widths(self):
        data = {r['question_id']:r for r in self.prices.HISTORY}
        account = self.market.Account.__new__(self.market.Account)
        account.get = Mock(return_value=self.post(self.rows[46010]))
        for qid,q in account.group(46010).items():
            row=data[qid]
            est=self.prices.model(46010,row['spot']-timedelta(seconds=1),row['spot'],row['resolve'])
            numeric=self.market.NumericQuestion.from_metaculus_api_json(q.api_json)
            sets=[dict(zip(self.prices.LEVELS,est.naive_quantiles))]+[est.quantiles(w) for w in self.prices.WIDTHS]
            for quantiles in sets:
                values=self.market.numeric_cdf(q,quantiles)
                self.assertEqual(values,self.market.numeric_cdf(numeric,quantiles))
                index=round((row['outcome']-q.nominal_lower_bound)/.01)
                self.assertEqual(values[index+1]-values[index],self.prices._mass(quantiles,row['low'],row['high'],row['outcome'],row['bins']))
        stats=self.prices.benchmark([r for r in self.prices.HISTORY if r['post_id']==46010])
        self.assertFalse(stats['failed'])
        self.assertIn(1.25,stats['allowed_widths']['46010'])
        self.assertGreater(next(r for r in stats['cases'] if r['question_id']==46709)['widths'][1.25],0)
        self.assertIn(1.25,self.market.cross_quarter_menu()['46010'])

    def test_changed_geometry_blocks_before_sdk_or_post(self):
        for key,value in [('zero_point',0),('nominal_min',1.76),('inbound_outcome_count',251),('open_upper_bound',False)]:
            row=copy.deepcopy(self.rows[46010][0])
            row['scaling'][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.market.parse_market_question(self.post([row]),row)
        row=copy.deepcopy(self.rows[46010][0])
        row['scaling']['continuous_range'][70]+=.005
        with self.assertRaises(ValueError):
            self.market.parse_market_question(self.post([row]),row)

    def test_submission_contract_uses_target_size_and_checks_readback(self):
        row=copy.deepcopy(self.rows[46010][-1])
        account=self.market.Account.__new__(self.market.Account)
        account.session=Mock()
        account.get=Mock(return_value=self.post([row]))
        question=account.group(46010)[row['id']]
        values=self.market.numeric_cdf(question,{.1:2.5,.5:3.,.9:3.6})
        with self.assertRaises(ValueError):
            account.submit_once(question,values[:-1])
        account.session.post.assert_not_called()
        history={'forecast_values':values,'start_time':1770000000.,'end_time':None}
        row['my_forecasts']={'history':[history], 'latest':history}
        account.submit_once(question,values)
        account.session.post.assert_called_once_with(self.market.BASE+'/questions/forecast/',timeout=40,
            json=[{'question':row['id'],'source':'api','continuous_cdf':values}])
        history['forecast_values']=[v*.99 for v in values]
        with self.assertRaisesRegex(RuntimeError,'readback differs'):
            account.submit_once(question,values)


if __name__ == '__main__':
    unittest.main()
