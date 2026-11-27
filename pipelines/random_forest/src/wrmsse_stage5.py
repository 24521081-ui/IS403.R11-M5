"""Read-only inputs and business aggregates for the official M5 Test forecasts."""

import ast

import numpy as np
import pandas as pd

from .project_paths import local_path
from .wrmsse import M5_COUNTS
from .wrmsse_config import CONFIG, CONTRACT_ID, ROOT, require_stage_permission
from .wrmsse_integrity import is_compact, verify_historical_files, verify_raw
from .wrmsse_stage2_data import sha, write_new
from .wrmsse_stage4 import PARAMS, ROWS, read, verify_files

RESULTS = local_path("results/wrmsse_v1")
STAGE2 = RESULTS / "stage2_20261123T202104Z"
STAGE3 = RESULTS / "stage3_20261124T082126Z"
EXTENDED = RESULTS / "stage3_all_features"
STAGE4 = RESULTS / "stage4_20261124T104618287922Z"
SELECTED = STAGE3 / "selection_v2/selected_model.json"
REPORTS = local_path("reports")
TABLES = REPORTS / "tables"
DETAIL = STAGE4 / "test_actual_and_forecast.parquet"
FORECAST = {"rf": "random_forest_forecast", "sn": "seasonal_naive_forecast"}
LABELS = {"rf": "Random Forest", "sn": "Seasonal Naive", "actual": "Thực tế"}
LEVELS = (
    "Toàn hệ thống", "Bang", "Cửa hàng", "Ngành hàng", "Bộ phận", "Bang × ngành hàng",
    "Bang × bộ phận", "Cửa hàng × ngành hàng", "Cửa hàng × bộ phận", "Sản phẩm",
    "Sản phẩm × bang", "Sản phẩm × cửa hàng",
)
WEEKDAYS = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật")
DETAIL_COLUMNS = (
    "series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "day", "date",
    "actual_sales", *FORECAST.values(), "rf_error", "rf_absolute_error", "sn_error", "sn_absolute_error",
)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in directory.rglob("*") if p.is_file()}


def check_gates():
    for action in ("training", "tuning", "test_evaluation"):
        try:
            require_stage_permission(action)
        except RuntimeError:
            continue
        raise RuntimeError(f"Unexpected open gate: {action}")
    assert not read(ROOT / "configs/stage3.json")["allow_tuning"]


def verify_inputs():
    """Verify the sealed inputs and preserve the provenance recorded before reporting."""
    check_gates()
    pre = read(STAGE4 / "preflight.json")
    audit = read(STAGE4 / "postflight_verification.json")
    manifest4 = read(STAGE4 / "manifest_completed.json")
    assert manifest4["status"] == "completed" and audit["status"] == "passed"
    assert manifest4["contract_id"] == CONTRACT_ID and manifest4["fit_count"] == 1
    verify_files(ROOT, manifest4["linked_sha256"])
    verify_historical_files(ROOT, pre["input_manifests_sha256"])
    verify_historical_files(ROOT, read(SELECTED.parent / "manifest.json")["input_sha256"])
    manifests = (
        STAGE2 / "manifest_completed.json", STAGE3 / "manifest_completed.json",
        SELECTED.parent / "manifest.json", EXTENDED / "manifest.json",
        local_path(pre["processed"]) / "manifest.json", local_path(pre["features"]) / "manifest.json",
        STAGE4 / "manifest_completed.json",
    )
    checked = {}
    for path in manifests:
        files = read(path)["files"]
        if path.parent == STAGE4:
            verify_files(path.parent, files)
        else:
            verify_historical_files(path.parent, files)
        checked[rel(path)] = {"sha256": sha(path), "files_verified": len(files)}
    evidence = (
        "preparation/preparation_summary.json", "preparation/data_context_audit.json",
        "preparation/final_receipt.json", "preparation/source_manifest.json",
        "preparation/stage2_completion.json", "preparation/stage3_snap_audit.json",
        "docs/WRMSSE_PREPARATION.md", "docs/STAGE2_REPORT.md", "docs/STAGE3_REPORT.md",
        "docs/STAGE3_SNAP_REVIEW.md", "docs/STAGE4_REPORT.md", "configs/experiment.json",
        "src/wrmsse.py", "src/wrmsse_data.py",
    )
    history = read(ROOT / evidence[0])
    assert history["status"] == "prepared_metric_and_contexts_verified" and history["unit_tests"]["status"] == "passed"
    contexts = read(ROOT / evidence[1])
    assert not contexts["test_actuals_loaded"] and set(contexts["contexts"]) == {"validation", "test"}
    assert read(ROOT / evidence[2])["status"] == "passed"
    assert "OK" in local_path(history["unit_tests"]["log"]).read_text(encoding="utf-8").splitlines()[-1]
    assert sha(ROOT / evidence[7]) == read(ROOT / "preparation/stage2_completion.json")["report_sha256"]
    historical_code = read(STAGE3 / "manifest_completed.json")["files"]
    for name in evidence[-2:]:
        assert sha(ROOT / name) == historical_code["code_snapshot/" + name]
    config = read(STAGE4 / "final_model_config.json")
    assert config["candidate"] == "RF-15 + FE-04" and config["selected_model_sha256"] == sha(SELECTED)
    assert all(config["params"][key] == value for key, value in PARAMS.items())
    assert config["logical_features"] == 15 and config["encoded_features"] == 3082
    protected = read(STAGE4 / "protected_before.json")
    if snapshot(ROOT / "data/raw") != protected["raw"]:
        verify_raw()
    evidence_hashes = {p: sha(ROOT / p) for p in evidence}
    if is_compact():
        original_hashes = read(REPORTS / "stage5_inputs.json")["evidence_sha256"]
        assert set(evidence_hashes) == set(original_hashes)
        verify_historical_files(ROOT, original_hashes)
        evidence_hashes = original_hashes
    return {
        "contract_id": CONTRACT_ID, "manifests": checked, "evidence_sha256": evidence_hashes,
        "protected_inventory_counts": {key: len(value) for key, value in protected.items()},
        "raw_access": "Metadata inventories for isolation. Business analysis reads the locked Stage 4 export.",
    }


def components(frame):
    """Build additive sufficient statistics for item-store-day metrics."""
    out = frame.copy()
    for key, column in FORECAST.items():
        error = out[column].astype("float64") - out.actual_sales
        out[f"{key}_error"] = error
        out[f"{key}_absolute_error"] = error.abs()
        out[f"{key}_squared_error"] = error ** 2
    return out


def ratio(numerator, denominator):
    return 100 * numerator / denominator.where(denominator != 0)


def summarize(frame, keys, grain="item-store-day"):
    totals = frame.groupby(keys, sort=True, observed=True)
    columns = ["actual_sales", *FORECAST.values()] + [
        f"{model}_{metric}" for model in FORECAST for metric in ("error", "absolute_error", "squared_error")
    ]
    out = totals[columns].sum().join(totals.size().rename("observations")).reset_index()
    out["metric_grain"] = grain
    out["actual_share_pct"] = ratio(out.actual_sales, pd.Series(frame.actual_sales.sum(), index=out.index))
    for key in FORECAST:
        out[f"{key}_net_absolute_error"] = out[f"{key}_error"].abs()
        out[f"{key}_mae"] = out[f"{key}_absolute_error"] / out.observations
        out[f"{key}_rmse"] = np.sqrt(out[f"{key}_squared_error"] / out.observations)
        out[f"{key}_wape_pct"] = ratio(out[f"{key}_absolute_error"], out.actual_sales)
        out[f"{key}_bias_pct"] = ratio(out[f"{key}_error"], out.actual_sales)
    for metric in ("mae", "rmse", "wape_pct"):
        out[f"{metric}_improvement_pct"] = ratio(out[f"sn_{metric}"] - out[f"rf_{metric}"], out[f"sn_{metric}"])
    return out


def hierarchy():
    levels = csv(STAGE4 / "test_wrmsse_by_level.csv")
    series = pd.read_parquet(STAGE4 / "test_wrmsse_by_series.parquet")
    metrics = read(STAGE4 / "test_metrics.json")
    for model, result in metrics["models"].items():
        by_level = levels.loc[levels.model == model].set_index("level")
        by_series = series.loc[series.model == model].groupby("level")
        np.testing.assert_array_equal(by_level.n_series, M5_COUNTS)
        np.testing.assert_array_equal(by_series.size(), M5_COUNTS)
        np.testing.assert_allclose(by_level.weight, 1 / 12, atol=1e-12, rtol=0)
        np.testing.assert_allclose(by_series.contribution.sum(), by_level.contribution, atol=1e-12, rtol=0)
        np.testing.assert_allclose(by_level.contribution.sum(), result["wrmsse"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(by_level.wrmsse_within_level, 12 * by_level.contribution, atol=1e-12, rtol=0)
    left, right = [levels[levels.model == model].drop(columns="model").set_index("level")
                   for model in ("random_forest", "seasonal_naive")]
    np.testing.assert_allclose(left.weight, right.weight, atol=1e-12, rtol=0)
    table = left[["n_series", "weight"]].join(
        left[["wrmsse_within_level", "contribution"]].add_prefix("rf_")
    ).join(right[["wrmsse_within_level", "contribution"]].add_prefix("sn_")).reset_index()
    table.insert(1, "hierarchy", LEVELS)
    table["wrmsse_improvement_pct"] = ratio(
        table.sn_wrmsse_within_level - table.rf_wrmsse_within_level, table.sn_wrmsse_within_level
    )
    assert (table.rf_wrmsse_within_level < table.sn_wrmsse_within_level).all()
    return table, series


def attach_series_stats(groups, series, level, keys):
    out = groups
    for model, prefix in (("random_forest", "rf"), ("seasonal_naive", "sn")):
        rows = series.loc[(series.model == model) & (series.level == level)]
        identity = pd.DataFrame(rows.key.tolist(), columns=keys, index=rows.index)
        stats = identity.join(rows[["weight", "scale", "scale_status", "rmsse", "contribution"]].add_prefix(prefix + "_"))
        out = out.merge(stats, on=keys, validate="one_to_one", how="left")
    assert out.rf_weight.notna().all() and out.sn_weight.notna().all()
    return out


def experiments():
    original = csv(STAGE3 / "all_results.csv")
    fe = csv(STAGE4 / "stage3_fe01_fe10.csv").set_index("experiment")
    rf = original.rename(columns={"name": "experiment", "wrmsse": "validation_wrmsse",
                                  "n_raw_features": "logical", "n_features": "encoded",
                                  "peak_rss_mib": "peak_worker_rss_mib"})
    rf["n_jobs"] = rf.params.map(lambda value: ast.literal_eval(value)["n_jobs"])
    rf["feature_set"] = rf.version
    rf["eligible_for_selection"] = rf.selection_eligible
    rf["selection_status"] = np.where(rf.selection_eligible, "eligible", "exploratory")
    rf["artifact_path"] = rf.artifact_dir
    rf = rf.set_index("experiment")
    common = fe.index.intersection(rf.index)
    shared = fe.columns.intersection(rf.columns)
    rf.loc[common, shared] = fe.loc[common, shared]
    additional = fe.loc[["FE-09", "FE-10"]].assign(group="features", reuse=False)
    columns = ["group", "feature_set", "logical", "encoded", "validation_wrmsse", "fit_seconds",
               "predict_seconds", "peak_worker_rss_mib", "n_jobs", "reuse",
               "eligible_for_selection", "selection_status", "artifact_path"]
    table = pd.concat([rf[columns], additional[columns]]).reset_index()
    assert len(table) == table.experiment.nunique() == 36
    for group, winner in {"initial": "RF-04", "ablation": "ABL_rf_v1", "advanced": "RF-15", "features": "FE-04"}.items():
        eligible = table[(table.group == group) & table.eligible_for_selection]
        assert eligible.loc[eligible.validation_wrmsse.idxmin(), "experiment"] == winner
    assert table.reuse.sum() == 4
    return table


def build_tables():
    metrics = read(STAGE4 / "test_metrics.json")
    selected = read(SELECTED)
    start, end = metrics["test_days"]
    frame = pd.read_parquet(DETAIL)
    assert len(frame) == ROWS and not frame.isna().any().any()
    np.testing.assert_array_equal(frame.series_id, np.repeat(np.arange(metrics["bottom_series"]), 28))
    np.testing.assert_array_equal(frame.day, np.tile(np.arange(start, end + 1), metrics["bottom_series"]))
    date = pd.to_datetime(frame.date, format="%Y-%m-%d")
    dates = pd.Timestamp(metrics["test_dates"][0]) + pd.to_timedelta(frame.day - start, unit="D")
    np.testing.assert_array_equal(date, dates)
    frame = components(frame)
    frame["scope"] = "Toàn hệ thống"
    frame["week"] = (frame.day - start) // 7 + 1
    frame["weekday_no"] = date.dt.dayofweek + 1
    frame["weekday"] = date.dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    frame["snap_active"] = False
    for state, active in selected["snap_rule"]["days_of_month_on"].items():
        frame.loc[(frame.state_id == state) & date.dt.day.isin(active), "snap_active"] = True
    assert set(frame.state_id) == {"CA", "TX", "WI"}
    levels, series_scores = hierarchy()
    system = summarize(frame, ["scope"])
    model_comparison = csv(STAGE4 / "test_model_comparison.csv")
    for model, key in zip(("random_forest", "seasonal_naive"), FORECAST):
        official = metrics["models"][model]
        for metric, column in {
            "actual_sales": "actual_sales", "forecast_sales": FORECAST[key], "sales_difference": f"{key}_error",
            "mae": f"{key}_mae", "rmse": f"{key}_rmse", "wape_percent": f"{key}_wape_pct",
            "forecast_bias_percent": f"{key}_bias_pct",
        }.items():
            np.testing.assert_allclose(system.at[0, column], official[metric], rtol=1e-12)
        for metric, value in official.items():
            np.testing.assert_allclose(model_comparison.set_index("model").at[model, metric], value, rtol=1e-12)
    model_comparison["metric_grain"] = "MAE/RMSE/WAPE/Bias: item-store-day. WRMSSE: 12 cấp M5."
    daily = summarize(frame, ["date", "day", "week", "weekday_no", "weekday"])
    daily["actual_change"] = daily.actual_sales.diff()
    daily["actual_change_pct"] = ratio(daily.actual_change, daily.actual_sales.shift())
    daily["rf_abs_aggregate_error"] = daily.rf_error.abs()
    daily["sn_abs_aggregate_error"] = daily.sn_error.abs()
    for key, column in FORECAST.items():
        daily[f"{key}_change"] = daily[column].astype("float64").diff()
    calendar_summary = daily.groupby(["weekday_no", "weekday"], as_index=False).agg(
        mean_actual=("actual_sales", "mean"), mean_rf=("random_forest_forecast", "mean"),
        mean_sn=("seasonal_naive_forecast", "mean"),
    )
    time_alignment = pd.DataFrame({
        "model": ["random_forest", "seasonal_naive"],
        "daily_total_pearson": [daily.actual_sales.corr(daily[column]) for column in FORECAST.values()],
        "direction_matches": [(np.sign(daily.actual_change[1:]) == np.sign(daily[f"{key}_change"][1:])).sum()
                              for key in FORECAST],
        "day_to_day_comparisons": len(daily) - 1,
    })
    group = ["series_id", "item_id", "store_id", "state_id", "cat_id", "dept_id"]
    bottom = attach_series_stats(summarize(frame, group), series_scores, 12, ["item_id", "store_id"])
    highest_sales = bottom.sort_values(["actual_sales", "series_id"], ascending=[False, True])
    under = bottom[bottom.rf_error < 0].sort_values(["rf_error", "series_id"]).head(20)
    over = bottom[bottom.rf_error > 0].sort_values(["rf_error", "series_id"], ascending=[False, True]).head(20)
    contributions = bottom.sort_values(["rf_contribution", "series_id"], ascending=[False, True]).head(20)
    reps = []
    for criterion, pool in (
        ("Lượng bán lớn nhất", highest_sales), ("Thiếu dự báo nhiều nhất", under),
        ("Thừa dự báo nhiều nhất", over), ("Đóng góp WRMSSE lớn nhất", contributions),
    ):
        free = pool.loc[~pool.series_id.isin([row["series_id"] for row in reps])]
        reps.append({"selection": criterion, **free.iloc[0][["series_id", "item_id", "store_id", "actual_sales",
                                                            "rf_error", "rf_wape_pct", "rf_contribution"]].to_dict()})
    representatives = pd.DataFrame(reps)
    representatives_daily = frame.loc[frame.series_id.isin(representatives.series_id), list(DETAIL_COLUMNS)].merge(
        representatives[["selection", "series_id"]], on="series_id", validate="many_to_one"
    )
    stores = attach_series_stats(summarize(frame, ["store_id", "state_id"]), series_scores, 3, ["store_id"])
    store_day = components(frame.groupby(["store_id", "date"], as_index=False)[
        ["actual_sales", *FORECAST.values()]].sum())
    store_day_metrics = summarize(store_day, ["store_id"], grain="store-day")
    stores = stores.merge(store_day_metrics[["store_id", "rf_mae", "rf_rmse", "rf_wape_pct",
                                             "sn_mae", "sn_rmse", "sn_wape_pct"]].rename(
        columns={col: f"store_day_{col}" for col in ("rf_mae", "rf_rmse", "rf_wape_pct", "sn_mae", "sn_rmse", "sn_wape_pct")}),
        on="store_id", validate="one_to_one")
    calendar_weeks = daily.groupby("week").date.agg(week_start="min", week_end="max").reset_index()
    importance = csv(STAGE4 / "feature_importance.csv").sort_values("importance", ascending=False)
    np.testing.assert_allclose(importance.importance.sum(), 1, rtol=0, atol=1e-12)
    assert len(importance) == 15
    importance["importance_pct"] = 100 * importance.importance
    return {
        "system": system, "model_comparison": model_comparison, "daily_sales": daily,
        "state_analysis": attach_series_stats(summarize(frame, ["state_id"]), series_scores, 2, ["state_id"]),
        "store_analysis": stores, "category_analysis": attach_series_stats(
            summarize(frame, ["cat_id"]), series_scores, 4, ["cat_id"]),
        "department_analysis": attach_series_stats(summarize(frame, ["dept_id", "cat_id"]), series_scores, 5, ["dept_id"]),
        "weekly_analysis": summarize(frame, ["week"]).merge(calendar_weeks, on="week", validate="one_to_one"),
        "weekday_analysis": summarize(frame, ["weekday_no", "weekday"]).merge(
            calendar_summary, on=["weekday_no", "weekday"], validate="one_to_one"),
        "store_department": summarize(frame, ["store_id", "dept_id"]),
        "snap_analysis": summarize(frame, ["snap_active"]),
        "snap_by_state": summarize(frame, ["state_id", "snap_active"]),
        "daily_alignment": time_alignment, "wrmsse_12_levels": levels,
        "top_underforecast": under, "top_overforecast": over, "top_wrmsse_contribution": contributions,
        "top_items_sales": summarize(frame, ["item_id", "cat_id", "dept_id"]).sort_values(
            ["actual_sales", "item_id"], ascending=[False, True]).head(20),
        "top_series_sales": highest_sales.head(20), "representative_series": representatives,
        "representative_daily": representatives_daily, "feature_importance": importance,
        "experiment_results": experiments(),
        "validation_test": pd.DataFrame([
            ("Validation", selected["wrmsse"], selected["seasonal_naive_wrmsse"], "d_1886–d_1913",
             "d_1–d_1885", "d_1858–d_1885"),
            ("Test", metrics["models"]["random_forest"]["wrmsse"], metrics["models"]["seasonal_naive"]["wrmsse"],
             "d_1914–d_1941", "d_1–d_1913", "d_1886–d_1913"),
        ], columns=["window", "random_forest", "seasonal_naive", "target_days", "scale_days", "weight_days"]),
    }


def prepare_tables():
    receipt = verify_inputs()
    tables = build_tables()
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)
    write_new(REPORTS / "stage5_inputs.json", receipt | {
        "table_sha256": {rel(TABLES / f"{name}.csv"): sha(TABLES / f"{name}.csv") for name in tables},
    })
    return tables
