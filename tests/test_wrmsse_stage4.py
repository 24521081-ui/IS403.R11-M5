"""Regressions for the final forecast's history boundary and metric conventions."""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from scipy import sparse

from src.wrmsse_stage4 import business, encode


class Stage4Test(unittest.TestCase):
    def test_future_sales_do_not_change_earlier_features(self):
        days = np.array([1914, 1915, 1941])
        keys = pd.DataFrame({"series_id": [0, 1, 2], "day": days})
        series = pd.DataFrame({
            "item_id": ["i1", "i2", "i3"], "dept_id": ["d"] * 3, "cat_id": ["c"] * 3,
            "store_id": ["CA_1", "TX_1", "WI_1"], "state_id": ["CA", "TX", "WI"],
        })
        calendar = pd.DataFrame({
            "day": days, "date": ["2016-04-25", "2016-04-26", "2016-05-22"],
            "wday": [3, 4, 2], "month": [4, 4, 5], "year": [2016] * 3,
        })
        encoder = Mock()
        encoder.transform.return_value = sparse.csr_matrix((3, 3072), dtype=np.float32)
        history = np.ones((3, 1913), dtype=np.uint16)
        expected = encode(keys, series, calendar, history, encoder).toarray()
        history[0, 1886:] = 100
        history[1, 1887:] = 100
        history[2, 1912] = 100
        updated = encode(keys, series, calendar, history, encoder).toarray()
        np.testing.assert_array_equal(expected[:2], updated[:2])
        self.assertEqual(updated[2, 3075], 100)
        with self.assertRaises(AssertionError):
            encode(keys, series, calendar, np.pad(history, ((0, 0), (0, 1))), encoder)

    def test_signed_bias_and_zero_denominator(self):
        metrics = business([2, 3], [1, 2])
        self.assertEqual(metrics["mae"], 1)
        self.assertEqual(metrics["forecast_bias_percent"], -40)
        self.assertEqual(metrics["underforecast_percent"], 40)
        self.assertIsNone(business([0], [1])["wape_percent"])
