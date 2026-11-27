"""Verify completed Stage 3 results without fitting or opening Test."""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project_paths import local_path
from src.wrmsse_config import CONFIG, CONTRACT_ID, require_stage_permission
from src.wrmsse_integrity import is_compact, verify_historical_files

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(directory, manifest):
    return verify_historical_files(directory, manifest["files"])


def metadata(directory):
    return {
        path.relative_to(directory).as_posix(): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in directory.rglob("*") if path.is_file()
    }


def verify(run):
    run = local_path(run)
    summary = read(run / "summary.json")
    archived_selection = run / "selection_v2/selected_model_v1.json"
    selected = read(archived_selection if archived_selection.is_file() else run / "selected_model.json")
    preflight = read(run / "preflight.json")
    assert summary["status"] == "complete"
    assert selected == summary["selection"]
    assert preflight["contract_id"] == CONTRACT_ID
    manifest = read(run / "manifest_completed.json")
    artifact_counts = verify_files(run, manifest)
    evidence = read(local_path(preflight["evidence"]))
    assert evidence["status"] == "passed"
    verify_historical_files(run / "code_snapshot", evidence["code_sha256"])

    policy = read(run / "code_snapshot/configs/stage3.json")
    switch = read(run / "n_jobs8_resume.json")
    preserved = set(switch["checkpoints_preserved"])
    resumed = read(run / "code_snapshot_n_jobs8/test_evidence.json")
    assert resumed["status"] == "passed"
    verify_historical_files(run / "code_snapshot_n_jobs8", resumed["code_sha256"])

    def model_parameters(params):
        return {key: value for key, value in params.items() if key != "n_jobs"}
    assert not policy["allow_test_evaluation"] and not CONFIG["allow_test_evaluation"]
    for action in ("training", "tuning", "test_evaluation"):
        try:
            require_stage_permission(action)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Gate remains open: {action}")
    assert not read(ROOT / "configs/stage3.json")["allow_tuning"]
    for name, sha256 in preflight["raw_sha256"].items():
        assert digest(local_path(f"data/raw/{name}")) == sha256
    stage2 = read(run.parent / "stage2_20261123T202104Z/summary.json")
    for label, relative in (("feature", stage2["features"]), ("dataset", stage2["processed"])):
        directory = local_path(relative)
        assert digest(directory / "manifest.json") == preflight[f"{label}_manifest_sha256"]
        verify_files(directory, read(directory / "manifest.json"))

    official = {row["name"]: row["params"] for row in policy["configs"]}
    expected = {
        "initial": set(list(official)[:5]),
        "advanced": set(list(official)[5:]),
        "ablation": {f"ABL_rf_v{i}" for i in range(1, 5)},
        "features": set(policy["FE"]),
    }
    rows = [read(path) for path in sorted(run.glob("*/*/record.json"))]
    assert len(rows) == 34
    for group, names in expected.items():
        assert {row["name"] for row in rows if row["group"] == group} == names
    eligible = []
    scores = {}
    local_predictions = 0
    for row in rows:
        name = row["name"]
        assert row["contract_id"] == CONTRACT_ID
        assert row["purpose"] == "tuning"
        assert row["dataset_hash"] == preflight["dataset_manifest_sha256"]
        assert row["feature_hash"] == preflight["feature_manifest_sha256"]
        if name in official:
            assert model_parameters(row["params"]) == model_parameters(official[name]), name
        if row["group"] == "advanced":
            assert row["version"] == summary["base_feature"], name
        if row["group"] == "ablation":
            assert row["params"] == summary["ablation_hyperparameters"], name
        if row["group"] == "features":
            assert model_parameters(row["params"]) == model_parameters(summary["feature_tuning_hyperparameters"]), name
            assert row["version"] == summary["base_feature"], name
            assert row["requested_groups"] == policy["FE"][name], name
        if row["status"] != "success":
            assert not row["selection_eligible"]
            continue
        assert row["split"] == "validation" and row["origin"] == 1885
        assert row["train_rows"] == 600000 and row["validation_rows"] == 853720
        directory = local_path(row["artifact_dir"])
        assert directory.is_relative_to(run)
        model_path = (directory / "model.joblib").relative_to(run).as_posix()
        assert model_path in manifest["files"]
        fitted = read(directory / "result.json")
        assert row["params"] == fitted["params"]
        assert row["params"]["n_jobs"] in (4, 8)
        if name not in preserved and not row["reuse"]:
            assert row["params"]["n_jobs"] == switch["new_fit_n_jobs"]
            assert read(directory / "manifest.json")["n_jobs"] == switch["new_fit_n_jobs"]
        if name not in preserved and row["reuse"]:
            assert row["requested_n_jobs"] == switch["new_fit_n_jobs"]
        assert row["wrmsse"] == fitted["wrmsse"]
        if directory not in scores:
            prediction_path = directory / "validation_predictions.npy"
            verify_historical_files(directory, {prediction_path.name: fitted["prediction_sha256"]})
            if prediction_path.is_file():
                predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
                assert predictions.shape == (30490, 28) and np.isfinite(predictions).all()
                local_predictions += 1
            levels = pd.read_csv(directory / "wrmsse_by_level.csv")
            assert list(levels.level) == list(range(1, 13))
            assert levels.n_series.sum() == 42840
            assert math.isclose(levels.weight.sum(), 1.0, abs_tol=1e-12)
            score = float(levels.contribution.sum())
            series_path = directory / "wrmsse_by_series.parquet"
            assert series_path.relative_to(run).as_posix() in manifest["files"]
            if series_path.is_file():
                series = pd.read_parquet(series_path, columns=["contribution"])
                assert len(series) == 42840
                assert math.isclose(series.contribution.sum(), score, abs_tol=1e-12)
            scores[directory] = score
        assert math.isclose(scores[directory], row["wrmsse"], abs_tol=1e-12)
        is_eligible = not (set(row["requested_groups"]) & set(policy["selection_excluded_groups"]))
        assert row["selection_eligible"] == is_eligible
        if is_eligible:
            eligible.append(row)
    assert eligible and selected["selection_eligible"]
    assert selected["wrmsse"] == min(row["wrmsse"] for row in eligible)
    assert any(row["name"] == selected["name"] and row["artifact_dir"] == selected["artifact_dir"] for row in eligible)
    assert math.isclose(
        selected["improvement_vs_naive_percent"],
        100 * (1 - selected["wrmsse"] / stage2["baseline"]["wrmsse"]),
        abs_tol=1e-12,
    )
    tables = [pd.read_csv(run / "all_results.csv"), pd.read_parquet(run / "all_results.parquet")]
    for table in tables:
        assert len(table) == len(rows)
        indexed = table.set_index("name")
        for row in rows:
            assert indexed.at[row["name"], "status"] == row["status"]
            if row["status"] == "success":
                assert math.isclose(indexed.at[row["name"], "wrmsse"], row["wrmsse"], abs_tol=1e-12)
    assert not summary["test_actual_accessed"] and not summary["final_retrain"]
    result = {
        "status": "passed", "run": run.relative_to(ROOT).as_posix(),
        "tested_code_suite": evidence["tests"], "records": len(rows), "verified_fits": len(scores),
        "eligible_records": len(eligible), "selected_name": selected["name"],
        "selected_wrmsse": selected["wrmsse"], "test_locked": True,
    }
    output = run / "postflight_verification.json"
    if output.is_file():
        original = read(output)
        assert all(original[key] == value for key, value in result.items())
    else:
        if is_compact():
            raise FileNotFoundError(output)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result | {"selection_snapshot": "v1 historical", "manifest_checks": artifact_counts,
                              "locally_checked_prediction_arrays": local_predictions}))


if __name__ == "__main__":
    verify(sys.argv[1])

