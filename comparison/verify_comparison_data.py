"""Automated verification test suite for comparison charts and summary tables."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "comparison"


def test_selected_four_charts_exist_and_non_empty():
    expected = [
        "01_tong_quan_wrmsse_validation_test.png",
        "02_wrmsse_theo_12_cap_m5.png",
        "03_bang_tong_hop_da_tieu_chi.png",
        "04_do_lech_theo_nhom_nhu_cau_va_nganh_hang.png",
    ]
    for name in expected:
        p = COMP / "charts" / name
        assert p.is_file() and p.stat().st_size > 50_000, f"Missing or tiny chart: {p}"


def test_summary_table_matches_published_sources():
    s = pd.read_csv(COMP / "summary_all_models.csv").set_index("model")
    assert list(s.index) == [
        "Seasonal Naive",
        "ES_bu",
        "Croston-SBA",
        "Random Forest",
        "LightGBM Two-Part",
        "LightGBM Tweedie",
    ]
    assert np.allclose(s["actual_sales"], 1_231_764.0)
    # Cross-check exact bias formula = 100 * (forecast/actual - 1)
    np.testing.assert_allclose(
        s["bias_pct"],
        100.0 * (s["forecast_sales"] / s["actual_sales"] - 1.0),
        atol=1e-9,
    )
    # Exact published Test WRMSSE checks
    np.testing.assert_allclose(s.loc["LightGBM Tweedie", "wrmsse_test"], 0.6234560168189278, atol=1e-12)
    np.testing.assert_allclose(s.loc["LightGBM Two-Part", "wrmsse_test"], 0.6284865669601195, atol=1e-12)
    np.testing.assert_allclose(s.loc["Random Forest", "wrmsse_test"], 0.7039124143420078, atol=1e-12)
    np.testing.assert_allclose(s.loc["ES_bu", "wrmsse_test"], 0.770677, atol=1e-6)
    np.testing.assert_allclose(s.loc["Seasonal Naive", "wrmsse_test"], 0.8697007023310609, atol=1e-6)


def test_daily_and_store_totals_reconcile_to_summary():
    s = pd.read_csv(COMP / "summary_all_models.csv").set_index("model")
    daily = pd.read_csv(COMP / "daily_all_models.csv")
    store = pd.read_csv(COMP / "store_bias_all_models.csv")

    assert len(daily) == 28
    assert daily["actual"].sum() == 1_231_764
    for m in [
        "LightGBM Tweedie",
        "LightGBM Two-Part",
        "Random Forest",
        "ES_bu",
        "Seasonal Naive",
        "Croston-SBA",
    ]:
        np.testing.assert_allclose(daily[m].sum(), s.loc[m, "forecast_sales"], rtol=1e-6)
        st = store[store["model"] == m]
        assert len(st) == 10
        assert st["actual"].sum() == 1_231_764
        np.testing.assert_allclose(st["forecast"].sum(), s.loc[m, "forecast_sales"], rtol=1e-6)


def test_category_metrics_match_c7_events_and_tweedie():
    cat = pd.read_csv(COMP / "category_all_models.csv")
    c7e = pd.read_csv(ROOT / "pipelines/lightgbm_twopart/reports/c7_events/test_group_scores.csv")
    c7e_cat = c7e[c7e.dimension == "cat_id"].set_index("group")
    sub = cat[cat.model == "LightGBM Two-Part"].set_index("cat_id")
    for c in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        np.testing.assert_allclose(sub.loc[c, "weighted_RMSSE"], c7e_cat.loc[c, "revenue_weighted_RMSSE"], atol=1e-12)
        np.testing.assert_allclose(sub.loc[c, "bias_pct"], c7e_cat.loc[c, "bias_pct"], atol=1e-12)
