import unittest
import numpy as np
from volatility_window import session_volatility_ratio


class WindowTests(unittest.TestCase):
    def test_30_months_cannot_supply_30_session_returns(self):
        monthly = np.linspace(3., 9., 120)
        self.assertIsNone(session_volatility_ratio(monthly, observation_unit="month"))

    def test_daily_requires_30_complete_session_changes(self):
        changes = np.r_[np.tile([-1., 1.], 60), np.tile([-3., 3.], 15)]
        daily = np.r_[0., np.cumsum(changes)]
        expected = np.sqrt(np.std(changes[-30:], ddof=1) / np.std(changes, ddof=1))
        self.assertAlmostEqual(session_volatility_ratio(daily, observation_unit="trading_session"), expected)
        self.assertIsNone(session_volatility_ratio(daily[:30], observation_unit="trading_session"))
        self.assertAlmostEqual(session_volatility_ratio(daily[:31], observation_unit="trading_session"), 1.)

    def test_invalid_frequency_or_data(self):
        with self.assertRaises(ValueError):
            session_volatility_ratio([1, 2, 3], observation_unit="unknown")
        with self.assertRaises(ValueError):
            session_volatility_ratio([1, np.nan, 3], observation_unit="trading_session")
