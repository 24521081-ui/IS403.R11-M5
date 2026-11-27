import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.wrmsse_business_v2 import ABC_LABELS, MAE_TOL, VOLUME_LABELS, WIN_LABELS, abc_segments, pareto_series, sales_segments, wins
from src.wrmsse_stage5 import components, summarize
from src.wrmsse_integrity import verify_historical_files


class BusinessV2Tests(unittest.TestCase):
    def test_new_archive_replaces_prior_keep_disposition_for_historical_bytes(self):
        checksum = sha256(b"previous report").hexdigest()
        name = "report.md"
        cleanup = {name: {"sha256": checksum, "operation": "keep"}}
        revision = {"archived": {name: {"sha256": checksum, "operation": "revise",
                                       "archive_member": "objects/" + checksum}}}
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / name).write_text("revised report", encoding="utf-8")
            with patch("src.wrmsse_integrity.ROOT", root), \
                    patch("src.wrmsse_integrity.inventory", return_value=cleanup), \
                    patch("src.wrmsse_integrity.business_revision", return_value=revision):
                self.assertEqual(verify_historical_files(root, {name: checksum})["indexed_external"], 1)

    def test_pareto_uses_minimal_prefix_and_breaks_equal_errors_by_id(self):
        data = pd.DataFrame({"series_id": [3, 2, 1, 4], "item_id": ["A"] * 4, "store_id": ["S"] * 4,
                             "rf_absolute_error": [2, 2, 6, 0]})
        sorted_, cut = pareto_series(data)
        self.assertEqual(sorted_.series_id.tolist(), [1, 2, 3, 4])
        self.assertEqual(cut["rank"].tolist(), [1, 2, 3])
        self.assertEqual(sorted_.cumulative_error_pct.iloc[-1], 100)

    def test_abc_uses_history_weights_with_zero_in_class_c(self):
        data = pd.DataFrame({"series_id": [4, 1, 3, 2], "weight": [0, .8, .05, .15]})
        ordered = abc_segments(data)
        self.assertEqual(ordered.series_id.tolist(), [1, 2, 3, 4])
        self.assertEqual(ordered.abc.tolist(), [*ABC_LABELS, "C"])
        crossing = abc_segments(pd.DataFrame({"series_id": [1, 2, 3, 4], "weight": [.74, .10, .12, .04]}))
        self.assertEqual(crossing.abc.tolist(), ["A", "A", "B", "C"])
        self.assertAlmostEqual(ordered.historical_revenue_share.sum(), 1)

    def test_volume_tercile_equality_enters_lower_group(self):
        groups, boundaries = sales_segments([0, 2, 2, 2, 8])
        np.testing.assert_array_equal(boundaries, [2, 2])
        self.assertEqual(groups.tolist(), [VOLUME_LABELS[0], *[VOLUME_LABELS[1]] * 3, VOLUME_LABELS[3]])

    def test_mae_tolerance_and_zero_denominator(self):
        bottom = pd.DataFrame({"group": ["All"] * 3, "rf_mae": [1, 1 + MAE_TOL / 2, 2],
                               "sn_mae": [2, 1, 1]})
        scores = wins(bottom, ["group"]).set_index("outcome")
        self.assertEqual(scores.loc[WIN_LABELS].n_series.tolist(), [1, 1, 1])
        rows = pd.DataFrame({"group": [0, 0, 1, 1], "actual_sales": [0, 0, 2, 3],
                             "random_forest_forecast": [1, 1, 1, 4], "seasonal_naive_forecast": [0, 1, 1, 2]})
        by_group = summarize(components(rows), ["group"]).set_index("group")
        self.assertTrue(pd.isna(by_group.at[0, "rf_wape_pct"]) and pd.isna(by_group.at[0, "rf_bias_pct"]))
        self.assertEqual(by_group.at[1, "rf_wape_pct"], 40)
        self.assertEqual(by_group.at[1, "rf_bias_pct"], 0)
