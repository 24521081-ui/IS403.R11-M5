"""Business report calculations from the sealed forecasts and pre-origin history."""

import numpy as np
import pandas as pd

from .project_paths import local_path
from .wrmsse_stage4 import read, verify_files
from .wrmsse_stage5 import (
    DETAIL, FORECAST, REPORTS, ROOT, SELECTED, STAGE4, TABLES,
    components, csv, ratio, summarize, verify_inputs,
)

PREFIX = "business_v2_"
MANIFEST = REPORTS / "manifest_business_v2.json"
NOTEBOOK = ROOT / "notebooks/09_business_insights_and_visualization.ipynb"
ABC_LABELS = ["A", "B", "C"]
VOLUME_LABELS = ["0 · Chưa bán", "1 · Thấp", "2 · Trung bình", "3 · Cao"]
WIN_LABELS = ["Random Forest thấp hơn", "Tương đương", "Seasonal Naive thấp hơn"]
MAE_TOL = 1e-9


def table(name):
    return csv(TABLES / (name + ".csv"))


def sales_segments(previous):
    values = np.asarray(previous, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError("Invalid historical units")
    thresholds = np.quantile(values[values > 0], [1 / 3, 2 / 3], method="linear")
    groups = np.where(values == 0, 0, np.searchsorted(thresholds, values, side="left") + 1)
    return np.asarray(VOLUME_LABELS)[groups], thresholds


def abc_segments(keys):
    """Assign the threshold-crossing row to the preceding class. Zero weights go to C."""
    out = keys[["series_id", "weight"]].sort_values(["weight", "series_id"], ascending=[False, True]).copy()
    weight = out.weight.to_numpy()
    if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("Invalid historical revenue weights")
    share = weight / weight.sum()
    before = np.r_[0., np.cumsum(share)[:-1]]
    out["historical_revenue_share"] = share
    out["revenue_cumulative_before"] = before
    out["abc"] = np.select([(weight > 0) & (before < .8 - 1e-12),
                            (weight > 0) & (before < .95 - 1e-12)], ["A", "B"], default="C")
    return out


def pareto_series(bottom):
    ranked = bottom[["series_id", "item_id", "store_id", "rf_absolute_error"]].sort_values(
        ["rf_absolute_error", "series_id"], ascending=[False, True]
    ).reset_index(drop=True)
    error = ranked.rf_absolute_error.to_numpy()
    if not np.isfinite(error).all() or (error < 0).any() or error.sum() <= 0:
        raise ValueError("Positive absolute error total required")
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["series_pct"] = ranked["rank"] * 100 / len(ranked)
    ranked["error_share_pct"] = 100 * error / error.sum()
    ranked["cumulative_error_pct"] = 100 * np.cumsum(error) / error.sum()
    ranked.loc[ranked.index[-1], "cumulative_error_pct"] = 100.
    targets = np.array([50, 80, 90])
    positions = np.searchsorted(ranked.cumulative_error_pct, targets)
    cut = ranked.iloc[positions][["rank", "series_pct", "cumulative_error_pct"]].reset_index(drop=True)
    cut.insert(0, "target_pct", targets)
    return ranked, cut


def wins(frame, slices):
    diff = frame.rf_mae - frame.sn_mae
    status = np.select([diff < -MAE_TOL, diff > MAE_TOL], [WIN_LABELS[0], WIN_LABELS[2]], default=WIN_LABELS[1])
    rows = []
    data = frame.assign(outcome=status)
    for dim in slices:
        for label, group in data.groupby(dim, sort=True, observed=True):
            counts = group.outcome.value_counts()
            rows.extend((dim, str(label), outcome, int(counts.get(outcome, 0)), len(group))
                        for outcome in WIN_LABELS)
    result = pd.DataFrame(rows, columns=["dimension", "group", "outcome", "n_series", "group_series"])
    result["share_pct"] = ratio(result.n_series, result.group_series)
    return result


def representatives(bottom, rows):
    eligible = bottom[bottom.actual_sales >= bottom.loc[bottom.actual_sales > 0, "actual_sales"].median()]
    choices = (
        ("Lượng bán Test cao nhất", bottom, ["actual_sales"], [False]),
        ("Nhiều ngày không bán, có ngày bán", bottom[bottom.actual_sales > 0], ["zero_actual_days", "actual_sales"], [False, False]),
        ("WAPE thấp trong nhóm lượng bán ≥ trung vị dương", eligible, ["rf_wape_pct"], [True]),
        ("Thiếu nhiều nhất", bottom, ["rf_error"], [True]),
        ("Thừa nhiều nhất", bottom, ["rf_error"], [False]),
        ("Đóng góp WRMSSE lớn nhất", bottom, ["rf_contribution"], [False]),
    )
    found = []
    for label, pool, keys, desc in choices:
        selected = pool[~pool.series_id.isin([row["series_id"] for row in found])].sort_values(
            [*keys, "series_id"], ascending=[*desc, True]
        ).iloc[0]
        found.append({"criterion": label, **selected[[
            "series_id", "item_id", "store_id", "actual_sales", "zero_actual_days", "rf_error",
            "rf_wape_pct", "sn_wape_pct", "rf_bias_pct", "sn_bias_pct", "rf_contribution",
        ]].to_dict()})
    chosen = pd.DataFrame(found)
    times = rows.loc[rows.series_id.isin(chosen.series_id),
                     ["series_id", "day", "date", "actual_sales", *FORECAST.values()]]
    return chosen, times


def error_curves(rows):
    arrays = [np.sort(rows[f"{model}_error"].to_numpy()) for model in FORECAST]
    bound = max(np.abs(values[[0, -1]]).max() for values in arrays)
    step = np.geomspace(1e-4, bound, 480)
    grid = np.unique(np.r_[-step, 0, step, *[values[[0, -1]] for values in arrays]])
    curve = pd.DataFrame({"error_threshold": grid})
    stats = []
    for model, values in zip(FORECAST, arrays):
        curve[model + "_cdf_pct"] = 100 * np.searchsorted(values, grid, side="right") / len(values)
        stats.append((model, len(values), np.count_nonzero(values == 0), *np.quantile(values, [.01, .5, .99]),
                      *np.quantile(np.abs(values), [.5, .95, .99])))
    return curve, pd.DataFrame(stats, columns=[
        "model", "observations", "exact_zero_error", "signed_q01", "signed_median", "signed_q99",
        "absolute_median", "absolute_q95", "absolute_q99",
    ])


def scoped(rows, assignments, key):
    measured = summarize(rows.merge(assignments[["series_id", key]], on="series_id", validate="many_to_one"), [key])
    totals = assignments.groupby(key, sort=True, observed=True).agg(
        n_series=("series_id", "size"), historical_units=("historical_units", "sum"),
        historical_revenue_share=("historical_revenue_share", "sum"), rf_contribution=("rf_contribution", "sum"),
        sn_contribution=("sn_contribution", "sum"),
    )
    return measured.merge(totals, on=key, validate="one_to_one")


def build():
    verify_inputs()
    selection = read(SELECTED)
    metrics = read(STAGE4 / "test_metrics.json")
    verify_files(ROOT, read(REPORTS / "stage5_inputs.json")["table_sha256"])
    frames = {name: table(name) for name in (
        "model_comparison", "wrmsse_12_levels", "daily_sales", "weekly_analysis", "weekday_analysis",
        "state_analysis", "store_analysis", "store_department", "category_analysis", "department_analysis",
        "feature_importance", "experiment_results",
    )}
    rows = pd.read_parquet(DETAIL)
    start, end = metrics["test_days"]
    n = metrics["bottom_series"]
    assert len(rows) == metrics["observations"] == n * (end - start + 1)
    np.testing.assert_array_equal(rows.series_id, np.repeat(np.arange(n), end - start + 1))
    np.testing.assert_array_equal(rows.day, np.tile(np.arange(start, end + 1), n))
    keys = ["series_id", "item_id", "store_id", "state_id", "cat_id", "dept_id"]
    rows = components(rows)
    rows["zero_actual"] = rows.actual_sales == 0
    bottom = summarize(rows, keys)
    bottom["zero_actual_days"] = rows.groupby("series_id").zero_actual.sum().reindex(bottom.series_id).to_numpy()
    official = pd.read_parquet(STAGE4 / "test_wrmsse_by_series.parquet")
    official = official[official.level == 12]
    for name, model in (("rf", "random_forest"), ("sn", "seasonal_naive")):
        series = official[official.model == model]
        joined = pd.DataFrame(series.key.tolist(), columns=["item_id", "store_id"], index=series.index)
        joined = joined.join(series[["weight", "rmsse", "contribution"]].add_prefix(name + "_"))
        bottom = bottom.merge(joined, on=["item_id", "store_id"], validate="one_to_one")
    assert len(bottom) == n
    np.testing.assert_allclose(bottom.rf_weight, bottom.sn_weight, atol=1e-15, rtol=0)
    np.testing.assert_allclose(bottom.rf_weight.sum(), 1 / len(frames["wrmsse_12_levels"]), atol=1e-12, rtol=0)
    np.testing.assert_allclose(bottom.rf_contribution.sum(),
                               frames["wrmsse_12_levels"].set_index("level").at[12, "rf_contribution"], atol=1e-12, rtol=0)
    pre = read(STAGE4 / "preflight.json")
    processed = local_path(pre["processed"])
    manifest = read(processed / "manifest.json")
    assert manifest["validation"] == metrics["revenue_weight_days"]
    verify_files(processed, {name: manifest["files"][name] for name in ("validation_sales.npy", "series.parquet")})
    membership = pd.read_parquet(processed / "series.parquet", columns=["series_id", "item_id", "store_id"])
    history = np.load(processed / "validation_sales.npy", mmap_mode="r", allow_pickle=False)
    assert history.shape == (n, 28)
    previous = membership.assign(historical_units=history.sum(axis=1, dtype=np.float64)).merge(
        bottom[["series_id", "item_id", "store_id"]], validate="one_to_one"
    )
    assert len(previous) == n
    bottom = bottom.merge(previous, on=["series_id", "item_id", "store_id"], validate="one_to_one")
    bottom["volume"], thresholds = sales_segments(bottom.historical_units)
    assigned = abc_segments(bottom.rename(columns={"rf_weight": "weight"}))
    bottom = bottom.merge(assigned.drop(columns="weight"), on="series_id", validate="one_to_one")
    for frame in (rows, bottom):
        for name, model in (("rf", "random_forest"), ("sn", "seasonal_naive")):
            m = metrics["models"][model]
            checks = [frame.actual_sales.sum(), frame[FORECAST[name]].sum(),
                      100 * frame[f"{name}_absolute_error"].sum() / frame.actual_sales.sum(),
                      100 * frame[f"{name}_error"].sum() / frame.actual_sales.sum()]
            np.testing.assert_allclose(checks, [m[k] for k in ("actual_sales", "forecast_sales", "wape_percent",
                                                              "forecast_bias_percent")], rtol=1e-12)
    daily = frames["daily_sales"].copy()
    for column in ["actual_sales", *FORECAST.values(), "rf_error", "sn_error"]:
        daily[column + "_cumulative"] = daily[column].cumsum()
    dates = pd.to_datetime(rows.date, format="%Y-%m-%d")
    rows["snap_active"] = False
    for state, active in selection["snap_rule"]["days_of_month_on"].items():
        rows.loc[rows.state_id.eq(state) & dates.dt.day.isin(active), "snap_active"] = True
    snap = []
    for scope, data in (("Toàn bộ", rows), ("FOODS", rows[rows.cat_id == "FOODS"])):
        days = data.groupby(["state_id", "snap_active"]).day.nunique().rename("n_days")
        roll = summarize(data, ["state_id", "snap_active"]).join(days, on=["state_id", "snap_active"])
        roll["actual_per_day"] = roll.actual_sales / roll.n_days
        roll["scope"] = scope
        snap.append(roll)
    stores = frames["store_analysis"].copy()
    median_sales, median_wape = stores.actual_sales.median(), stores.rf_wape_pct.median()
    stores["quadrant"] = np.select(
        [(stores.actual_sales >= median_sales) & (stores.rf_wape_pct >= median_wape),
         (stores.actual_sales >= median_sales), (stores.rf_wape_pct >= median_wape)],
        ["Cao / cao", "Cao / thấp", "Thấp / cao"], default="Thấp / thấp",
    )
    stores = stores.assign(median_actual=median_sales, median_wape=median_wape)
    rank, cut = pareto_series(bottom)
    highest = bottom.sort_values(["rf_contribution", "series_id"], ascending=[False, True]).head(15).copy()
    highest["share_of_level12_pct"] = 100 * highest.rf_contribution / bottom.rf_contribution.sum()
    extrema = pd.concat([
        bottom[bottom.rf_error.lt(0)].sort_values(["rf_error", "series_id"]).head(15).assign(direction="Thiếu"),
        bottom[bottom.rf_error.gt(0)].sort_values(["rf_error", "series_id"], ascending=[False, True]).head(15).assign(direction="Thừa"),
    ], ignore_index=True)
    examples, examples_daily = representatives(bottom, rows)
    ecdf, quantiles = error_curves(rows)
    bottom["all_series"] = "Toàn bộ"
    no_sale = summarize(rows, ["zero_actual"])
    for name, column in FORECAST.items():
        no_sale[name + "_mean_forecast"] = no_sale[column] / no_sale.observations
        no_sale[name + "_mean_error"] = no_sale[name + "_error"] / no_sale.observations
    importance = frames["feature_importance"].copy()
    feature = importance.feature
    importance["feature_group"] = np.select(
        [feature.str.startswith("lag_"), feature.str.startswith("rolling_"), feature.eq("snap_state"),
         feature.isin(["wday", "month", "year"])], ["Lag", "Rolling Mean", "SNAP", "Thời gian"], default="Định danh",
    )
    scores = frames["experiment_results"].set_index("experiment").loc[["FE-01", "FE-04"], ["validation_wrmsse"]].reset_index()
    result = {
        "daily": daily, "store_priority": stores, "series": bottom, "pareto": rank,
        "pareto_thresholds": cut, "top_contribution": highest, "volume_segments": scoped(rows, bottom, "volume"),
        "abc": scoped(rows, bottom, "abc"), "exceptions": extrema, "representatives": examples,
        "representatives_daily": examples_daily, "error_ecdf": ecdf, "error_quantiles": quantiles,
        "series_wins": wins(bottom, ["all_series", "state_id", "cat_id", "abc"]),
        "zero_actual": no_sale, "snap": pd.concat(snap, ignore_index=True), "importance": importance,
        "snap_validation": scores, "velocity_cutoffs": pd.DataFrame({"quantile": [1/3, 2/3], "historical_units": thresholds}),
    }
    return result, frames, metrics
