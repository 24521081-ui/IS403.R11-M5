import unittest

import numpy as np
import pandas as pd

from src.wrmsse_demand_groups import BINS, adi_bin, join_labels


class DemandGroupTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({"item_id": ["A", "B", "A"], "store_id": ["X"] * 3,
                                  "forecast": [1.25, 0., 1.]})
        self.labels = pd.DataFrame({"item_id": ["B", "A"], "store_id": ["X", "X"],
                                    "ADI": [4., 1.1], "demand_group": ["intermittent", "frequent"]})

    def test_shuffled_labels_preserve_observation_order_and_forecasts(self):
        enriched = join_labels(self.rows, self.labels)
        pd.testing.assert_frame_equal(enriched[self.rows.columns], self.rows, check_exact=True)
        self.assertEqual(enriched.ADI.tolist(), [1.1, 4., 1.1])

    def test_duplicate_and_missing_keys_fail(self):
        with self.assertRaises(pd.errors.MergeError):
            join_labels(self.rows, pd.concat([self.labels, self.labels.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "Missing"):
            join_labels(self.rows, self.labels.iloc[:1])

    def test_bins_at_boundaries_and_nonfinite_values(self):
        self.assertEqual(adi_bin(pd.Series([1., 1.32, 2., 4., 8., np.inf, np.nan])).tolist(), BINS)
