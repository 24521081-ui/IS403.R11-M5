"""Fit and evaluate the locked M5 configuration in disjoint run-scoped gates."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow
import scipy
from scipy import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from .project_paths import local_path
from .wrmsse import M5_COUNTS, WRMSSEEvaluator
from .wrmsse_config import (
    CONFIG, CONTRACT_ID, _TEST_SCOPE, _TRAINING_SCOPE, fold, require_stage_permission,
)
from .wrmsse_data import load_actuals, load_weight_prices
from .wrmsse_stage2_data import BASE, CATS, load_matrix, numeric_block, sample_training, sha, write_new
from .wrmsse_workflows import seasonal_naive

SPEC = fold("test")
VALIDATION = fold("validation")
BOTTOM = M5_COUNTS[-1]
ROWS = BOTTOM * SPEC.horizon
LOGICAL = CATS + BASE + ["snap_state"]
PARAMS = dict(n_estimators=150, max_depth=20, max_features=0.5, max_samples=0.5,
              min_samples_leaf=5, bootstrap=True, random_state=42, n_jobs=8)
SNAP = {"CA": list(range(1, 11)), "TX": [1, 3, 5, 6, 7, 9, 11, 12, 13, 15],
        "WI": [2, 3, 5, 6, 8, 9, 11, 12, 14, 15]}
LOCKED = (
    "final_model.joblib", "final_model_config.json", "final_training_sample.parquet",
    "final_training_sample_manifest.json", "final_training_strata.parquet",
    "final_feature_metadata.json", "test_feature_matrix.npz", "test_feature_keys.parquet",
    "test_predictions.parquet", "seasonal_naive_predictions.parquet", "fit_receipt.json",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_files(root, files):
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")


@contextmanager
def permission(run, action):
    root = local_path(CONFIG["outputs"])
    if run.parent != root or not run.name.startswith("stage4_"):
        raise ValueError("Stage 4 requires a dedicated run directory")
    if action not in ("training", "test_evaluation"):
        raise ValueError("Unsupported Stage 4 action")
    if _TRAINING_SCOPE.get() or _TEST_SCOPE.get() or any(
        CONFIG.get("allow_" + name) for name in ("training", "tuning", "test_evaluation")
    ):
        raise RuntimeError("Stage 4 requires closed default gates and disjoint permissions")
    preflight = read(run / "preflight.json")
    if (preflight["status"], preflight["contract_id"], preflight["run"]) != (
        "passed", CONTRACT_ID, run.name
    ):
        raise RuntimeError("Stage 4 requires matching preflight evidence")
    evidence = read(local_path(preflight["test_evidence"]))
    if evidence["status"] != "passed" or evidence["contract_id"] != CONTRACT_ID:
        raise RuntimeError("Stage 4 requires passing unit tests")
    verify_files(root.parents[1], evidence["code_sha256"])
    verify_files(root.parents[1], preflight["input_manifests_sha256"])
    if action == "training":
        if (run / "forecast_lock_manifest.json").exists() or (run / "test_access.json").exists():
            raise RuntimeError("Forecast or Test has already been locked for this run")
        receipt = run / "fit_started.json"
        record = {"at": now(), "test_gate": "closed", "params": PARAMS}
        scope = _TRAINING_SCOPE
    else:
        lock_path = run / "forecast_lock_manifest.json"
        lock = read(lock_path)
        if (lock["run"] != run.name or lock["contract_id"] != CONTRACT_ID or lock["status"] != "locked"
                or lock["preflight_sha256"] != sha(run / "preflight.json")
                or set(lock["files"]) != set(LOCKED) or lock["test_actual_accessed"]):
            raise RuntimeError("Test access requires both committed forecast artifacts")
        verify_files(run, lock["files"])
        receipt = run / "test_access.json"
        record = {"at": now(), "training_gate": "closed", "forecast_lock_sha256": sha(lock_path)}
        scope = _TEST_SCOPE
    # Exclusive receipt prevents refitting or reopening Test within the same run.
    write_new(receipt, record)
    token = scope.set(True)
    try:
        yield
    finally:
        scope.reset(token)


def load_inputs(preflight):
    directory = local_path(preflight["processed"])
    series = pd.read_parquet(directory / "series.parquet")
    calendar = pd.read_parquet(
        directory / "calendar.parquet", columns=["day", "date", "wday", "month", "year"],
        filters=[("day", "<=", SPEC.origin + SPEC.horizon)],
    )
    train = np.load(directory / "train_sales.npy", allow_pickle=False)
    validation = np.load(directory / "validation_sales.npy", allow_pickle=False)
    assert train.shape == (BOTTOM, VALIDATION.origin) and train.dtype == np.uint16
    assert validation.shape == (BOTTOM, SPEC.horizon)
    assert np.isfinite(validation).all() and (validation >= 0).all()
    np.testing.assert_array_equal(validation, validation.astype(np.uint16))
    history = np.concatenate((train, validation.astype(np.uint16)), axis=1)
    np.testing.assert_array_equal(series.series_id, np.arange(BOTTOM))
    assert not series[CATS].isna().any().any() and not series.duplicated(["item_id", "store_id"]).any()
    assert series.groupby("store_id").state_id.nunique().eq(1).all()
    np.testing.assert_array_equal(series.store_id.str.split("_").str[0], series.state_id)
    np.testing.assert_array_equal(calendar.day, np.arange(1, SPEC.origin + SPEC.horizon + 1))
    np.testing.assert_array_equal(calendar.date, pd.date_range("2011-01-29", periods=len(calendar)).strftime("%Y-%m-%d"))
    return series, calendar, history


def training_sample(run, preflight, series, calendar, history):
    directory = local_path(preflight["processed"])
    original = pd.read_parquet(directory / "train_keys.parquet")
    recreated, train_strata = sample_training(series, calendar)
    pd.testing.assert_frame_equal(original, recreated)
    pd.testing.assert_frame_equal(train_strata, pd.read_parquet(directory / "strata.parquet"))
    eligible_days = VALIDATION.origin - 57 + 1
    size = round(len(original) * SPEC.horizon / eligible_days)
    extra, extra_strata = sample_training(
        series, calendar, n=size, start=VALIDATION.origin + 1, end=SPEC.origin,
    )
    again, again_strata = sample_training(
        series, calendar, n=size, start=VALIDATION.origin + 1, end=SPEC.origin,
    )
    pd.testing.assert_frame_equal(extra, again)
    pd.testing.assert_frame_equal(extra_strata, again_strata)
    sample = pd.concat([original.assign(split="train"), extra.assign(split="validation")],
                       ignore_index=True)
    assert not sample.duplicated(["series_id", "day"]).any()
    assert sample.day.between(57, SPEC.origin).all() and extra.day.between(1886, 1913).all()
    sample["actual_sales"] = history[sample.series_id, sample.day - 1]
    sample.to_parquet(run / "final_training_sample.parquet", index=False)
    pd.concat([train_strata.assign(split="train"), extra_strata.assign(split="validation")],
              ignore_index=True).to_parquet(run / "final_training_strata.parquet", index=False)
    write_new(run / "final_training_sample_manifest.json", {
        "seed": 42, "train_rows": len(original), "validation_rows": len(extra), "rows": len(sample),
        "eligible_train_days": eligible_days, "validation_days": [1886, 1913],
        "target_extra_before_rounding": len(original) * SPEC.horizon / eligible_days,
        "rounding": "Python round to nearest integer", "order": "Original Train then Validation",
        "sampling": read(directory / "manifest.json")["sampling"], "with_replacement": False,
        "original_keys_sha256": sha(directory / "train_keys.parquet"), "reproduced_exactly": True,
        "files": {name: sha(run / name) for name in (
            "final_training_sample.parquet", "final_training_strata.parquet")},
    })
    return sample


def forecast_grid(series, calendar):
    days = np.arange(SPEC.origin + 1, SPEC.origin + SPEC.horizon + 1, dtype=np.int32)
    grid = series.loc[series.index.repeat(SPEC.horizon), ["series_id", *CATS]].reset_index(drop=True)
    grid["day"] = np.tile(days, len(series))
    grid["date"] = calendar.set_index("day").loc[grid.day, "date"].to_numpy()
    assert len(grid) == ROWS and not grid.duplicated(["series_id", "day"]).any()
    assert grid.date.iloc[0] == "2016-04-25" and grid.date.iloc[-1] == "2016-05-22"
    return grid


def encode(keys, series, calendar, history, encoder):
    ordered = keys.series_id.to_numpy()
    days = keys.day.to_numpy()
    assert (ordered[1:] >= ordered[:-1]).all()
    assert (days - 28).max() <= history.shape[1] == SPEC.origin
    assert not keys[["series_id", "day"]].duplicated().any()
    numeric = np.empty((len(keys), len(BASE) + 1), dtype=np.float32)
    dates = calendar.set_index("day").loc[days]
    numeric[:, :3] = dates[BASE[:3]].to_numpy()
    states = series.state_id.to_numpy()[ordered]
    day_of_month = pd.to_datetime(dates.date, format="%Y-%m-%d").dt.day.to_numpy()
    assert set(states) == set(SNAP)
    numeric[:, -1] = 0
    for state, active_days in SNAP.items():
        mask = states == state
        numeric[mask, -1] = np.isin(day_of_month[mask], active_days)
    for start in range(0, len(series), 256):
        first, end = ordered.searchsorted([start, start + 256])
        if first < end:
            source_days = days[first:end]
            cutoff = int(source_days.max()) - 28
            values = numeric_block(history[start:start + 256, :cutoff], ordered[first:end] - start, source_days)
            numeric[first:end, 3:-1] = values[BASE[3:]].to_numpy()
    assert np.isfinite(numeric).all()
    matrix = sparse.hstack(
        [encoder.transform(series.iloc[ordered][CATS]), sparse.csr_matrix(numeric)],
        format="csr", dtype=np.float32,
    )
    assert matrix.shape == (len(keys), 3082) and np.isfinite(matrix.data).all()
    return matrix


def matrix_equal(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype == np.float32
    assert (left != right).nnz == 0


def fit_forecasts(run, preflight):
    series, calendar, history = load_inputs(preflight)
    sample = training_sample(run, preflight, series, calendar, history)
    original = sample.loc[sample.split == "train"]
    extra = sample.loc[sample.split == "validation"]
    vocabulary = [sorted(series[column].unique()) for column in CATS]
    training_ids = series.iloc[sample.series_id.unique()][CATS]
    assert all(set(training_ids[column]) == set(values) for column, values in zip(CATS, vocabulary))
    encoder = OneHotEncoder(categories=vocabulary, handle_unknown="error", sparse_output=True,
                            dtype=np.float32).fit(training_ids)
    encoded = encoder.get_feature_names_out(CATS).tolist() + BASE + ["snap_state"]
    mapping = [name for name, values in zip(CATS, vocabulary) for _ in values] + BASE + ["snap_state"]
    features = preflight["features"]
    feature_manifest = read(local_path(features) / "manifest.json")
    assert encoded == feature_manifest["specs"]["rf_v1"]["feature_names"] + ["snap_state"]
    assert LOGICAL == feature_manifest["specs"]["rf_v1"]["raw_features"] + ["snap_state"]
    train_matrix = encode(original, series, calendar, history, encoder)
    matrix_equal(train_matrix, load_matrix(features, "rf_v1", "train", ("snap",)))
    validation_matrix = load_matrix(features, "rf_v1", "validation", ("snap",))
    positions = extra.series_id.to_numpy() * SPEC.horizon + extra.day.to_numpy() - 1886
    extra_matrix = encode(extra, series, calendar, history, encoder)
    matrix_equal(extra_matrix, validation_matrix[positions])
    del validation_matrix
    fit_matrix = sparse.vstack([train_matrix, extra_matrix], format="csr", dtype=np.float32)
    del train_matrix, extra_matrix
    np.testing.assert_array_equal(original.actual_sales, np.load(local_path(features) / "y_train.npy"))
    grid = forecast_grid(series, calendar)
    test_matrix = encode(grid, series, calendar, history, encoder)
    sparse.save_npz(run / "test_feature_matrix.npz", test_matrix)
    grid.to_parquet(run / "test_feature_keys.parquet", index=False)
    write_new(run / "final_feature_metadata.json", {
        "logical_features": LOGICAL, "logical_count": len(LOGICAL), "encoded_count": len(encoded),
        "feature_names": encoded, "encoded_to_logical": mapping, "format": "CSR float32",
        "vocabulary": dict(zip(CATS, vocabulary)), "encoder": "OneHotEncoder", "unknown": "error",
        "source_feature_manifest_sha256": sha(local_path(features) / "manifest.json"),
        "matrix_rows": {"train": len(sample), "test": len(grid)}, "origin": SPEC.origin,
        "history_cutoff_formula": "min(target_day - 28, d_1913)",
        "test_sales_feature_cutoff_days": [int(grid.day.min() - 28), int(grid.day.max() - 28)],
        "snap_rule": SNAP, "snap_source": "State and target calendar day of month",
        "snap_rule_history_end": 1885, "snap_audit_sha256": preflight["snap_audit_sha256"],
        "original_train_and_selected_validation_matrices_exactly_match_stage3": True,
    })
    model = RandomForestRegressor(**preflight["params"])
    assert not model.warm_start and not hasattr(model, "estimators_")
    with permission(run, "training"):
        require_stage_permission("training")
        started = perf_counter()
        print(f"Fitting RF-15 + FE-04 on {len(sample):,} rows", flush=True)
        model.fit(fit_matrix, sample.actual_sales.to_numpy(dtype=np.float32))
        fit_seconds = perf_counter() - started
    del fit_matrix
    assert model.get_params() | PARAMS == model.get_params() and len(model.estimators_) == 150
    joblib.dump(model, run / "final_model.joblib")
    write_new(run / "final_model_config.json", {
        "model": "RandomForestRegressor", "candidate": "RF-15 + FE-04",
        "selected_model": preflight["selected_model"], "selected_model_sha256": preflight["selected_model_sha256"],
        "params": model.get_params(), "train_scope": [1, 1913], "train_rows": len(sample),
        "logical_features": len(LOGICAL), "encoded_features": model.n_features_in_,
        "forecast_origin": SPEC.origin, "horizon": SPEC.horizon, "fit_count": 1,
    })
    started = perf_counter()
    predictions = model.predict(test_matrix)
    predict_seconds = perf_counter() - started
    started = perf_counter()
    baseline = seasonal_naive(history, horizon=SPEC.horizon, period=7).ravel()
    baseline_seconds = perf_counter() - started
    assert predictions.shape == baseline.shape == (ROWS,) and np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    grid.assign(predicted_sales=predictions).to_parquet(run / "test_predictions.parquet", index=False)
    grid.assign(predicted_sales=baseline).to_parquet(run / "seasonal_naive_predictions.parquet", index=False)
    importance = pd.Series(model.feature_importances_, index=mapping).groupby(level=0).sum().reindex(LOGICAL)
    np.testing.assert_allclose(importance.sum(), 1.0, rtol=0, atol=1e-12)
    importance.rename_axis("feature").rename("importance").to_csv(run / "feature_importance.csv")
    write_new(run / "fit_receipt.json", {
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "seasonal_naive_predict_seconds": baseline_seconds, "train_rows": len(sample),
        "logical_features": len(LOGICAL), "encoded_features": len(encoded), "n_jobs": model.n_jobs,
        "seasonal_naive_last_history_days": [1907, 1913],
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "joblib": joblib.__version__, "pyarrow": pyarrow.__version__, "psutil": psutil.__version__},
    })


def load_forecasts(run, grid, history, index):
    matrices = {}
    for name, filename in (("random_forest", "test_predictions.parquet"),
                           ("seasonal_naive", "seasonal_naive_predictions.parquet")):
        frame = pd.read_parquet(run / filename)
        pd.testing.assert_frame_equal(frame.drop(columns="predicted_sales"), grid)
        assert np.isfinite(frame.predicted_sales).all() and len(frame) == ROWS
        matrices[name] = pd.DataFrame(
            frame.predicted_sales.to_numpy().reshape(BOTTOM, SPEC.horizon),
            index=index, columns=SPEC.forecast_days,
        )
    np.testing.assert_array_equal(matrices["seasonal_naive"], seasonal_naive(history, SPEC.horizon))
    return matrices


def business(actual, forecast):
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    error = forecast - actual
    total = float(actual.sum())
    difference = float(error.sum())
    bias = 100 * difference / total if total else None
    return {
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_percent": 100 * float(np.abs(error).sum()) / total if total else None,
        "forecast_bias_percent": bias, "actual_sales": total, "forecast_sales": float(forecast.sum()),
        "sales_difference": difference,
        "overforecast_percent": max(bias, 0.0) if bias is not None else None,
        "underforecast_percent": max(-bias, 0.0) if bias is not None else None,
    }


def evaluate(run, preflight):
    series, calendar, history = load_inputs(preflight)
    grid = forecast_grid(series, calendar)
    pd.testing.assert_frame_equal(pd.read_parquet(run / "test_feature_keys.parquet"), grid)
    index = pd.MultiIndex.from_frame(series[["item_id", "store_id"]])
    forecasts = load_forecasts(run, grid, history, index)
    historical_frame = pd.DataFrame(history, index=index, columns=SPEC.history_days)
    metadata = series.set_index(index)
    prices = load_weight_prices(metadata, historical_frame, "test")
    evaluator = WRMSSEEvaluator.fit(metadata, historical_frame, prices, SPEC, strict_m5=True)
    del historical_frame, prices
    with permission(run, "test_evaluation"):
        print("Forecast lock verified. Loading Test actual through the evaluation gate.", flush=True)
        source = local_path(CONFIG["test_actual_source"])
        if sha(source) != preflight["expected_test_actual_sha256"]:
            raise ValueError("Test actual checksum mismatch")
        actual = load_actuals("test", index)
    np.testing.assert_equal(actual.shape, (BOTTOM, SPEC.horizon))
    assert actual.index.equals(index) and actual.columns.tolist() == SPEC.forecast_days
    scores, metrics, levels, details = {}, {}, [], []
    for name, forecast in forecasts.items():
        score = evaluator.score(actual, forecast)
        scores[name] = score.wrmsse
        metrics[name] = {"wrmsse": score.wrmsse, **business(actual, forecast)}
        np.testing.assert_array_equal(score.per_level.n_series, M5_COUNTS)
        np.testing.assert_allclose(score.per_level.weight, 1 / 12, rtol=0, atol=1e-12)
        for frame in (score.per_level, score.per_series):
            np.testing.assert_allclose(frame.weight.sum(), 1, rtol=0, atol=1e-12)
            np.testing.assert_allclose(frame.contribution.sum(), score.wrmsse, rtol=0, atol=1e-12)
        levels.append(score.per_level.reset_index().assign(model=name))
        details.append(score.per_series.assign(model=name))
    difference = scores["seasonal_naive"] - scores["random_forest"]
    comparison = pd.DataFrame(metrics).T.rename_axis("model")
    comparison.to_csv(run / "test_model_comparison.csv")
    pd.concat(levels, ignore_index=True).to_csv(run / "test_wrmsse_by_level.csv", index=False)
    pd.concat(details, ignore_index=True).to_parquet(run / "test_wrmsse_by_series.parquet", index=False)
    grid.assign(
        actual_sales=actual.to_numpy().ravel(),
        random_forest_forecast=forecasts["random_forest"].to_numpy().ravel(),
        seasonal_naive_forecast=forecasts["seasonal_naive"].to_numpy().ravel(),
    ).to_parquet(run / "test_actual_and_forecast.parquet", index=False)
    write_new(run / "test_metrics.json", {
        "contract_id": CONTRACT_ID, "primary_metric": "WRMSSE", "origin": SPEC.origin,
        "test_days": [1914, 1941], "scale_history": [1, 1913], "revenue_weight_days": [1886, 1913],
        "revenue_weight_basis": "historical units × sell_price", "test_dates": ["2016-04-25", "2016-05-22"],
        "bottom_series": BOTTOM, "observations": ROWS, "levels": 12, "hierarchical_series": sum(M5_COUNTS),
        "models": metrics, "wrmsse_reduction": difference, "absolute_wrmsse_difference": abs(difference),
        "improvement_percent": 100 * difference / scores["seasonal_naive"] if scores["seasonal_naive"] else None,
        "zero_actual_sales_policy": "WAPE, Bias and excess/shortfall percentages are null if sum(actual) = 0",
        "test_actual_sha256": preflight["expected_test_actual_sha256"],
    })
