"""Audit SNAP availability and saved Stage 3 forecasts using Train and Validation only."""
from pathlib import Path
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project_paths import local_path
from src.wrmsse_config import CONFIG, CONTRACT_ID, fold
from src.wrmsse_data import normalize_calendar
from src.wrmsse_stage2_data import load_matrix, sha
from verify_stage3 import read, verify


RUN = ROOT / "results/wrmsse_v1/stage3_20261124T082126Z"
BASELINE_RUN = RUN.parent / "stage2_20261123T202104Z"
STATES = ("CA", "TX", "WI")
SNAP_COLUMNS = [f"snap_{state}" for state in STATES]


def audit_calendar(spec):
    calendar_path = local_path("data/raw/calendar.csv")
    # Materialize only the leading Train and Validation rows.
    columns = ["date", *SNAP_COLUMNS]
    calendar = normalize_calendar(
        pd.read_csv(calendar_path, usecols=columns, nrows=spec.origin + spec.horizon)
    )
    calendar["day"] = calendar["d"].str[2:].astype(int)
    np.testing.assert_array_equal(calendar["day"], np.arange(1, 1914))
    assert calendar[SNAP_COLUMNS].isin([0, 1]).all().all()
    calendar["month_day"] = pd.to_datetime(calendar["date"]).dt.day
    history = calendar.loc[calendar["day"] <= spec.origin]
    validation = calendar.loc[calendar["day"] > spec.origin]
    grouped = history.groupby("month_day")[SNAP_COLUMNS]
    assert grouped.ngroups == 31
    assert grouped.nunique().eq(1).all().all()
    rule = grouped.first()
    reconstructed = rule.reindex(validation["month_day"]).to_numpy()
    np.testing.assert_array_equal(reconstructed, validation[SNAP_COLUMNS].to_numpy())
    return calendar.set_index("day"), rule, {
        "last_train_date": history["date"].iloc[-1],
        "first_validation_date": validation["date"].iloc[0],
        "last_validation_date": validation["date"].iloc[-1],
        "train_days": len(history),
        "validation_state_days": reconstructed.size,
        "train_rule_state_days": 31 * len(STATES),
        "mismatches": 0,
        "snap_days_of_month": {
            state: rule.index[rule[f"snap_{state}"] == 1].tolist() for state in STATES
        },
        "calendar_sha256": sha(calendar_path),
    }


def audit_matrix(directory, series, keys, calendar, rule, history, split):
    ids = keys["series_id"].to_numpy(dtype=int)
    days = keys["day"].to_numpy(dtype=int)
    bands = (57, 1885, 600000) if split == "train" else (1886, 1913, 853720)
    assert len(keys) == bands[2] and keys["day"].between(*bands[:2]).all()
    assert not keys.duplicated(["series_id", "day"]).any()
    assert ids.min() == 0 and ids.max() == len(series) - 1
    states = series["state_id"].to_numpy()[ids]
    actual = sparse.load_npz(directory / "snap" / f"extra_{split}.npz")
    assert actual.shape == (len(keys), 1)
    actual = actual.toarray().ravel()
    calendar_rows = calendar.loc[days, SNAP_COLUMNS].to_numpy()
    reconstructed = rule.reindex(calendar.loc[days, "month_day"]).to_numpy()
    state_index = pd.Index(STATES).get_indexer(states)
    assert (state_index >= 0).all()
    row_axis = np.arange(len(keys))
    np.testing.assert_array_equal(actual, calendar_rows[row_axis, state_index])
    np.testing.assert_array_equal(actual, reconstructed[row_axis, state_index])

    # Independently reconstruct all three behavior columns for FE-07.
    block = sparse.load_npz(directory / "behavior" / f"extra_{split}.npz")
    assert block.shape == (len(keys), 3)
    cutoff = days - 28
    assert cutoff.min() >= 29 and cutoff.max() <= 1885
    expected = np.empty((len(keys), 3), dtype=np.float32)
    width = np.arange(56)
    for start in range(0, len(keys), 16384):
        end = min(start + 16384, len(keys))
        indices = cutoff[start:end, None] - 56 + width
        window = history[ids[start:end, None], np.maximum(0, indices)].astype(np.float64)
        window[indices < 0] = np.nan
        recent = window[:, 28:]
        expected[start:end] = np.column_stack((
            recent.std(axis=1), np.nanmean(window, axis=1),
            recent.mean(axis=1) - np.nanmean(window[:, :28], axis=1),
        ))
    np.testing.assert_allclose(block.toarray(), expected, rtol=2e-6, atol=2e-6)
    return {
        "rows": len(keys), "snap_calendar_mismatches": 0, "snap_train_rule_mismatches": 0,
        "behavior_values_compared": expected.size,
        "latest_sales_day": int(cutoff.max()),
        "snap_on_rows_by_state": {
            state: int(actual[states == state].sum()) for state in STATES
        },
    }


def main():
    verify(RUN.relative_to(ROOT).as_posix())
    spec = fold("validation")
    assert spec.origin == 1885 and spec.horizon == 28
    assert CONFIG["validation"] == [1886, 1913] and not CONFIG["allow_test_evaluation"]
    calendar, rule, calendar_checks = audit_calendar(spec)

    stage2 = read(BASELINE_RUN / "summary.json")
    dataset = local_path(stage2["processed"])
    feature_name = stage2["features"]
    features = local_path(feature_name)
    feature_manifest = read(features / "manifest.json")
    assert feature_manifest["specs"]["snap"]["extra_names"] == ["snap_state"]
    assert feature_manifest["specs"]["behavior"]["extra_names"] == [
        "rolling_std_28", "rolling_mean_56", "sales_trend_nonoverlap_28"
    ]
    series = pd.read_parquet(dataset / "series.parquet")
    assert set(series["state_id"]) == set(STATES)
    assert not series.duplicated(["item_id", "store_id"]).any()
    assert series["store_id"].str.split("_").str[0].eq(series["state_id"]).all()
    np.testing.assert_array_equal(series["series_id"], np.arange(30490))
    index = pd.MultiIndex.from_frame(series[["item_id", "store_id"]])
    history = np.load(dataset / "train_sales.npy", allow_pickle=False, mmap_mode="r")
    assert history.shape == (30490, 1885)
    train_keys = pd.read_parquet(dataset / "train_keys.parquet")
    validation_keys = pd.read_parquet(dataset / "validation_keys.parquet")
    np.testing.assert_array_equal(
        validation_keys["series_id"], np.repeat(np.arange(30490), 28)
    )
    np.testing.assert_array_equal(
        validation_keys["day"], np.tile(np.arange(1886, 1914), 30490)
    )

    checks = {
        split: audit_matrix(features, series, keys, calendar, rule, history, split)
        for split, keys in (("train", train_keys), ("validation", validation_keys))
    }
    actual = pd.DataFrame(
        np.load(dataset / "validation_sales.npy", allow_pickle=False), index=index,
        columns=spec.forecast_days,
    )
    preflight = read(RUN / "preflight.json")
    assert sha(RUN / "evaluator.joblib") == preflight["evaluator_sha256"]
    evaluator = joblib.load(RUN / "evaluator.joblib")
    assert evaluator.fold == spec
    assert len(evaluator.weights) == 42840
    baseline = history[:, -7:][:, np.arange(28) % 7]
    baseline_score = evaluator.score(
        actual, pd.DataFrame(baseline, index=index, columns=spec.forecast_days)
    ).wrmsse
    assert math.isclose(baseline_score, stage2["baseline"]["wrmsse"], abs_tol=1e-12)
    snap = sparse.load_npz(features / "snap/extra_validation.npz").toarray().ravel()
    candidates = {}
    for name, groups in (("FE-04", ["snap"]), ("FE-07", ["behavior", "snap"])):
        row = read(RUN / "features" / name / "record.json")
        artifact = local_path(row["artifact_dir"])
        assert artifact.is_relative_to(RUN)
        request = read(artifact / "request.json")
        assert request["groups"] == row["groups"] == groups
        assert request["params"] == row["params"] and request["version"] == row["version"] == "rf_v1"
        assert request["dataset"] == stage2["processed"] and request["features"] == feature_name
        matrix = load_matrix(feature_name, row["version"], "validation", groups)
        np.testing.assert_array_equal(matrix[:, -1].toarray().ravel(), snap)
        model = joblib.load(artifact / "model.joblib")
        assert matrix.shape == (853720, model.n_features_in_)
        assert model.n_features_in_ == row["n_features"]
        assert all(model.get_params()[key] == value for key, value in row["params"].items())
        prediction = np.load(artifact / "validation_predictions.npy", allow_pickle=False)
        predicted = np.concatenate([
            model.predict(matrix[start:start + 32768]) for start in range(0, matrix.shape[0], 32768)
        ]).reshape(30490, 28)
        np.testing.assert_allclose(predicted, prediction, rtol=1e-12, atol=1e-12)
        result = evaluator.score(
            actual, pd.DataFrame(prediction, index=index, columns=spec.forecast_days)
        )
        assert math.isclose(result.wrmsse, row["wrmsse"], abs_tol=1e-12)
        assert len(result.per_series) == 42840
        levels = pd.read_csv(artifact / "wrmsse_by_level.csv", index_col="level")
        pd.testing.assert_frame_equal(result.per_level, levels, check_exact=False, atol=1e-12, rtol=1e-12)
        candidates[name] = {
            "wrmsse": result.wrmsse, "levels": len(result.per_level),
            "aggregated_series": len(result.per_series),
            "model_prediction_max_absolute_delta": float(np.abs(predicted - prediction).max()),
            "model_sha256": sha(artifact / "model.joblib"),
            "predictions_sha256": sha(artifact / "validation_predictions.npy"),
            "manifest_sha256": sha(artifact / "manifest.json"),
        }
        del model, matrix, predicted
        print(name, candidates[name], flush=True)
    output = ROOT / "preparation/stage3_snap_audit.json"
    with output.open("x", encoding="utf-8") as file:
        json.dump({
            "status": "passed", "contract_id": CONTRACT_ID, "calendar": calendar_checks,
            "matrices": checks, "seasonal_naive_wrmsse": baseline_score, "candidates": candidates,
            "fit_executed": False, "test_accessed": False,
        }, file, indent=2)
    print(output)


if __name__ == "__main__":
    main()

