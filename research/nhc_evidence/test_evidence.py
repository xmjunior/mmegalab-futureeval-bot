import unittest
from datetime import datetime, timezone
import gzip

from evidence import dossier, parse_bulletin, Unsupported
from evaluate import truth_at, stats

URL = "https://www.nhc.noaa.gov/archive/2024/al02/al022024.discus.001.shtml"
BODY = b"""<PRE>
Tropical Depression Two Discussion Number  1
NWS National Hurricane Center Miami FL       AL022024
1100 PM AST Sun Jun 30 2024
INIT  01/0300Z 13.3N  56.0W  100 KT  115 MPH
 24H  02/0000Z 15.3N  62.0W  110 KT  125 MPH
</PRE>"""
CUT = datetime(2024,7,1,4,tzinfo=timezone.utc)


class ResearchTests(unittest.TestCase):
    def test_time_month_rollover_and_publisher_receipt(self):
        p = dossier("Wind criterion at " + URL, CUT, lambda url, size: BODY)
        self.assertEqual(p["initial_wind_kt"], 100)
        self.assertEqual(p["official_24h_wind_kt"], 110)
        self.assertEqual(p["target_utc"], "2024-07-02T00:00:00+00:00")
        self.assertNotIn("verified_wind_kt", p)

    def test_lookahead_and_stale_dossier_abstain(self):
        for t in (CUT.replace(hour=3), CUT.replace(day=2)):
            self.assertIsNone(dossier("wind " + URL, t, lambda *a:BODY))

    def test_only_exact_cited_source_can_trigger_network(self):
        for text in ["wind on https://evil.example/" + URL, "wind "+ URL + "?redirect=x",
                     "wind "+ URL.replace(".gov/", ".gov.evil/"), "pressure "+URL]:
            self.assertIsNone(dossier(text, CUT, lambda *a:self.fail("unsafe query fetched")))

    def test_identity_undated_corrections_and_failures(self):
        for bad in [BODY.replace(b"022024",b"052024"), BODY.replace(b"Number  1",b"Number  2"),
                    BODY.replace(b"<PRE>",b"<PRE> CORRECTED\n"), BODY.replace(b"AST",b"XXX")]:
            self.assertRaises(Unsupported, parse_bulletin, URL, bad)
        def unavailable(*args):
            raise TimeoutError()
        self.assertIsNone(dossier("wind "+URL, CUT, unavailable))

    def test_exact_official_verifier_and_ambiguous_revisions(self):
        line = b"AL, 02, 2024070200, , BEST, 0, 105N, 625W, 95, 995\n"
        date = CUT.replace(day=2, hour=0)
        self.assertEqual(truth_at(gzip.compress(line+line), "AL022024", date), 95)
        self.assertRaises(Unsupported, truth_at, gzip.compress(line+line.replace(b", 95,",b", 90,")), "AL022024", date)

    def test_paired_missingness_ties_sign_and_insufficient_groups(self):
        wins = [{"initial_wind_kt":50,"official_24h_wind_kt":55,"verified_wind_kt":55}]*8
        self.assertEqual(stats(wins, True)["sign_p_one_sided"], 1/256)
        self.assertEqual(stats(wins, False)["scientific_gate"], "not_passed")
        tied=dict(wins[0],official_24h_wind_kt=50)
        self.assertEqual(stats([{},tied], True)["ties"], 1)
