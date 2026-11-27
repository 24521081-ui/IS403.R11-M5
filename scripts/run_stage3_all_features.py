"""Run only the two Stage 3 feature-union candidates under the existing gate."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "preparation")]
from src.project_paths import local_path
from src.wrmsse_config import CONFIG, CONTRACT_ID, fold, stage3_tuning_scope
from src.wrmsse_stage2_data import load_matrix, sha, write_new
from src.wrmsse_workflows import run_rf
from verify_stage3 import metadata, read, verify_files

STAGE3 = ROOT / "results/wrmsse_v1/stage3_20261124T082126Z"
STAGE2 = STAGE3.parent / "stage2_20261123T202104Z"
DESTINATION = STAGE3.parent / "stage3_all_features"
POLICY = ROOT / "configs/stage3.json"
PARAMS = dict(n_estimators=150, max_depth=20, max_features=0.5, max_samples=0.5,
              min_samples_leaf=5, bootstrap=True, random_state=42, n_jobs=8)
CANDIDATES = {
    "FE-09": ["prices", "snap", "behavior"],
    "FE-10": ["prices", "snap", "behavior", "events"],
}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def protect_test(policy):
    assert CONFIG["stage"] == "preparation" and CONFIG["allow_test_evaluation"] is False
    assert policy["allow_test_evaluation"] is False
    assert not read(ROOT / "configs/stage2.json")["training_enabled"]


def stage4_evidence():
    paths = []
    for folder in (ROOT / "configs", ROOT / "scripts", ROOT / "results", ROOT / "preparation"):
        for path in folder.rglob("*"):
            name = path.name.lower().replace("-", "").replace("_", "")
            if "stage4" in name or name.startswith((
                "finalretrain", "finaltraining", "testpredictions", "testevaluation", "testresult",
            )):
                paths.append(relative(path))
    summary = read(STAGE3 / "summary.json")
    return {"matching_paths": paths, "test_actual_accessed": summary["test_actual_accessed"],
            "final_retrain": summary["final_retrain"],
            "exploratory_only": bool(paths or summary["test_actual_accessed"] or summary["final_retrain"])}


def other_training():
    # Scope to Python launched from this project or these scripts by an absolute path.
    pids = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    matches = []
    for process in psutil.process_iter(["pid", "cwd", "cmdline", "name"], ad_value=None):
        info = process.info
        if info["pid"] in pids or "python" not in (info["name"] or "").lower():
            continue
        args = info["cmdline"] or []
        cwd = Path(info["cwd"]).resolve() if info["cwd"] else None
        if cwd and cwd.is_relative_to(ROOT) or any(str(ROOT).lower() in arg.lower() for arg in args):
            matches.append(info["pid"])
    return matches


def load_validation_inputs(processed):
    directory = local_path(processed)
    series = pd.read_parquet(directory / "series.parquet")
    return SimpleNamespace(
        history=np.load(directory / "train_sales.npy", mmap_mode="r", allow_pickle=False),
        actual=np.load(directory / "validation_sales.npy", mmap_mode="r", allow_pickle=False),
        train_keys=pd.read_parquet(directory / "train_keys.parquet"),
        validation_keys=pd.read_parquet(directory / "validation_keys.parquet"),
        series=series,
        index=pd.MultiIndex.from_frame(series[["item_id", "store_id"]]),
    )


def preflight(batch, evidence):
    policy = read(POLICY)
    protect_test(policy)
    assert policy["allow_tuning"] is False
    assert not other_training(), "Another project Python process is active"
    phase4 = stage4_evidence()
    tests = read(local_path(evidence))
    assert tests["status"] == "passed" and tests["contract_id"] == CONTRACT_ID
    for path, expected in tests["code_sha256"].items():
        assert sha(local_path(path)) == expected, path

    verify_files(STAGE3, read(STAGE3 / "manifest_completed.json"))
    selection_dir = STAGE3 / "selection_v2"
    verify_files(selection_dir, read(selection_dir / "manifest.json"))
    selected = read(selection_dir / "selected_model.json")
    assert selected["name"] == "FE-04" and selected["params"] == PARAMS
    rf15 = read(STAGE3 / "advanced/RF-15/record.json")
    assert rf15["params"] == PARAMS
    stage2 = read(STAGE2 / "summary.json")
    pre = read(STAGE3 / "preflight.json")
    features = local_path(stage2["features"])
    processed = local_path(stage2["processed"])
    for label, folder in (("feature", features), ("dataset", processed)):
        assert sha(folder / "manifest.json") == pre[f"{label}_manifest_sha256"]
        verify_files(folder, read(folder / "manifest.json"))
    for name, expected in pre["raw_sha256"].items():
        assert sha(local_path(f"data/raw/{name}")) == expected
    protected = {"raw": metadata(ROOT / "data/raw")}
    assert sha(STAGE3 / "evaluator.joblib") == pre["evaluator_sha256"]
    assert sha(local_path(selected["audit_path"])) == selected["audit_sha256"]
    snap_audit = read(local_path(selected["audit_path"]))
    assert snap_audit["status"] == "passed" and snap_audit["calendar"]["mismatches"] == 0

    ds = load_validation_inputs(stage2["processed"])
    assert ds.history.shape == (30490, 1885) and ds.actual.shape == (30490, 28)
    assert ds.train_keys.day.between(57, 1885).all() and not ds.train_keys.duplicated().any()
    assert len(ds.train_keys) == 600000
    np.testing.assert_array_equal(ds.validation_keys.day, np.tile(np.arange(1886, 1914), 30490))
    np.testing.assert_array_equal(ds.validation_keys.series_id, np.repeat(np.arange(30490), 28))
    assert (ds.validation_keys.day - 28).max() == 1885
    labels = np.load(features / "y_train.npy", allow_pickle=False)
    np.testing.assert_array_equal(labels, ds.history[ds.train_keys.series_id, ds.train_keys.day - 1])

    manifest = read(features / "manifest.json")
    details = manifest["details"]
    assert details["max_price_source_day_train"] <= 1857
    assert details["max_price_source_day_validation"] <= 1885
    assert len(details["price_train_medians"]) == 4 and np.isfinite(details["price_train_medians"]).all()
    # Compare SNAP for every series-day to the already-locked Train-only rule.
    dates = pd.read_parquet(processed / "calendar.parquet", columns=["day", "date"],
                            filters=[("day", "<=", 1913)]).set_index("day")
    states = ds.series.state_id.to_numpy()
    rule = selected["snap_rule"]["days_of_month_on"]
    assert selected["snap_rule"]["calendar_history_end"] == 1885
    for split, keys in (("train", ds.train_keys), ("validation", ds.validation_keys)):
        month_days = pd.to_datetime(dates.loc[keys.day, "date"]).dt.day.to_numpy()
        expected = np.zeros(len(keys), dtype=np.float32)
        for state, days in rule.items():
            mask = states[keys.series_id.to_numpy()] == state
            expected[mask] = np.isin(month_days[mask], days)
        actual = sparse.load_npz(features / "snap" / f"extra_{split}.npz").toarray().ravel()
        np.testing.assert_array_equal(actual, expected)

    # Existing Events code masks to rule-matched civil holidays and learns its
    # multi-hot vocabulary from Train, but consumes labels from the target date.
    # Unlike SNAP, no reconstruction from pre-origin inputs has been verified.
    event_policy = {
        "status": "exploratory",
        "reason": "Target-date event labels are read from M5; origin-time availability is not established.",
        "artifact_policy": details["events_policy"],
        "vocabulary": details["event_vocabulary"],
    }
    specs = manifest["specs"]
    for name, groups in CANDIDATES.items():
        columns = specs["rf_v4"]["feature_names"] + [
            col for group in groups for col in specs[group]["extra_names"]
        ]
        if len(set(columns)) != len(columns):
            raise ValueError(f"Duplicate encoded names in {name}")
        logical = specs["rf_v4"]["n_raw_features"] + sum(specs[g]["extra_logical"] for g in groups)
        for split, nrows in (("train", 600000), ("validation", 853720)):
            matrix = load_matrix(stage2["features"], "rf_v4", split, groups)
            assert sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32
            assert matrix.shape == (nrows, len(columns)) and np.isfinite(matrix.data).all()
            del matrix
        path = batch / (name + "_" + stamp())
        path.mkdir()
        write_new(path / "feature_metadata.json", {
            "name": name, "base": "rf_v4", "groups": groups, "n_raw_features": logical,
            "n_model_features": len(columns), "feature_names": columns, "duplicates": 0,
            "dtype": "float32", "format": "CSR", "price_train_medians": details["price_train_medians"],
            "events": event_policy if "events" in groups else None,
            "snap_rule": selected["snap_rule"], "feature_manifest_sha256": pre["feature_manifest_sha256"],
        })
        write_new(path / "request.json", {
            "name": name, "output": relative(path), "params": PARAMS, "version": "rf_v4", "groups": groups,
            "dataset": stage2["processed"], "features": stage2["features"], "evidence": evidence,
            "evaluator_sha256": pre["evaluator_sha256"], "contract_id": CONTRACT_ID,
            "selection_eligible": name == "FE-09" and not phase4["exploratory_only"],
            "test_accessed": False, "split": "validation",
        })
    receipts = {
        "stage4": phase4, "evidence": evidence, "tests": tests["tests"],
        "test_gate_locked": True, "protected_before": protected,
        "old_run_manifest_sha256": sha(STAGE3 / "manifest_completed.json"),
        "selection_v2_sha256": sha(selection_dir / "selected_model.json"),
        "selection_v2_manifest_sha256": sha(selection_dir / "manifest.json"),
        "evaluator_sha256": pre["evaluator_sha256"], "input_sha256": {
            relative(processed / "manifest.json"): pre["dataset_manifest_sha256"],
            relative(features / "manifest.json"): pre["feature_manifest_sha256"],
            **{f"data/raw/{name}": value for name, value in pre["raw_sha256"].items()},
        }, "event_review": event_policy,
    }
    write_new(batch / "preflight.json", receipts)
    print(f"Preflight passed: {tests['tests']} tests, exact Train/Validation, two verified CSR schemas.", flush=True)
    return policy


def worker(request_ref):
    path = local_path(request_ref)
    assert path.is_relative_to(DESTINATION)
    request = read(path)
    assert request["params"] == PARAMS and request["groups"] == CANDIDATES[request["name"]]
    assert request["contract_id"] == CONTRACT_ID and request["split"] == "validation"
    protect_test(read(POLICY))
    output = local_path(request["output"])
    assert output / "request.json" == path
    ds = load_validation_inputs(request["dataset"])
    assert sha(STAGE3 / "evaluator.joblib") == request["evaluator_sha256"]
    evaluator = joblib.load(STAGE3 / "evaluator.joblib")
    assert evaluator.fold == fold("validation") and len(evaluator.weights) == 42840
    with stage3_tuning_scope(request["evidence"]):
        run_rf(ds, request["features"], output, evaluator, PARAMS, "rf_v4",
               request["groups"], request["name"], "tuning",
               selection_eligible=request["selection_eligible"])


def supervise(path, limits):
    assert not other_training(), "Another project Python process is active"
    ram_available = psutil.virtual_memory().available / 2**20
    assert ram_available >= limits["min_available_mib"] + 256, "Insufficient RAM reserve"
    print(f"{path.name}: starting isolated fit with n_jobs=8", flush=True)
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1", "JOBLIB_TEMP_FOLDER": str(path / "tmp"),
        "MPLCONFIGDIR": str(path / "tmp"),
    }
    started = time.perf_counter()
    peak = 0.0
    min_available = ram_available
    samples = 0
    peak_machine_cpu = 0.0
    peak_worker_cpu = 0.0
    last_cpu = last_t = None
    stop_reason = None
    with (path / "worker.log").open("x", encoding="utf-8") as log, (
        path / "resources.jsonl"
    ).open("x", encoding="utf-8") as resources:
        process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", relative(path / "request.json")],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            while process.poll() is None:
                moment = time.perf_counter()
                try:
                    leader = psutil.Process(process.pid)
                    members = [leader, *leader.children(recursive=True)]
                    rss = cpu = 0
                    for member in members:
                        try:
                            rss += member.memory_info().rss
                            usage = member.cpu_times()
                            cpu += usage.user + usage.system
                        except psutil.NoSuchProcess:
                            continue
                    rss /= 2**20
                    available = psutil.virtual_memory().available / 2**20
                    machine_cpu = psutil.cpu_percent()
                    worker_cpu = (100 * max(0, cpu - last_cpu) / (moment - last_t)
                                  if last_cpu is not None else None)
                    last_cpu, last_t = cpu, moment
                    peak = max(peak, rss)
                    min_available = min(min_available, available)
                    peak_machine_cpu = max(peak_machine_cpu, machine_cpu)
                    peak_worker_cpu = max(peak_worker_cpu, worker_cpu or 0.0)
                    resources.write(json.dumps({
                        "time": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": moment - started,
                        "rss_tree_mib": rss, "available_mib": available, "machine_cpu_pct": machine_cpu,
                        "worker_cpu_pct_one_core": worker_cpu, "worker_cpu_seconds": cpu,
                    }) + "\n")
                    resources.flush()
                    samples += 1
                    if rss > limits["max_rss_mib"] or available < limits["min_available_mib"]:
                        stop_reason = "RAM limit or reserve crossed"
                        break
                except psutil.NoSuchProcess:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                leader = psutil.Process(process.pid)
                for member in reversed(leader.children(recursive=True)):
                    try:
                        member.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                process.wait(timeout=30)
    write_new(path / "resources_summary.json", {
        "returncode": process.returncode, "stop_reason": stop_reason,
        "sample_count": samples, "elapsed_seconds": time.perf_counter() - started,
        "peak_tree_rss_mib": peak, "min_available_mib": min_available,
        "peak_machine_cpu_pct": peak_machine_cpu, "peak_worker_cpu_pct_one_core": peak_worker_cpu,
    })
    status = "success" if process.returncode == 0 else "failed"
    if status == "success":
        row = read(path / "result.json")
        assert row["params"] == PARAMS and row["n_features"] == read(path / "feature_metadata.json")["n_model_features"]
        levels = pd.read_csv(path / "wrmsse_by_level.csv")
        assert list(levels.level) == list(range(1, 13)) and levels.n_series.sum() == 42840
        assert np.isclose(levels.contribution.sum(), row["wrmsse"], rtol=0, atol=1e-12)
        assert len(pd.read_parquet(path / "wrmsse_by_series.parquet", columns=["contribution"])) == 42840
        assert sha(path / "validation_predictions.npy") == row["prediction_sha256"]
        print(f"{row['name']} WRMSSE={row['wrmsse']:.12f}", flush=True)
    else:
        print((path / "worker.log").read_text(encoding="utf-8")[-5000:], flush=True)
    write_new(path / "manifest.json", {
        "status": status, "contract_id": CONTRACT_ID, "run_id": path.name, "n_jobs": 8,
        "files": {relative_file.relative_to(path).as_posix(): sha(relative_file)
                  for relative_file in path.rglob("*") if relative_file.is_file()},
    })
    return status


def main(evidence):
    if DESTINATION.exists():
        raise FileExistsError(f"Existing experiment artifacts: {DESTINATION}")
    DESTINATION.mkdir(parents=True)
    batch = DESTINATION
    policy_raw = POLICY.read_bytes()
    policy = preflight(batch, evidence)
    statuses = {}
    try:
        assert POLICY.read_bytes() == policy_raw
        enabled = policy | {"allow_tuning": True}
        POLICY.write_text(json.dumps(enabled, indent=2) + "\n", encoding="utf-8")
        for name in CANDIDATES:
            paths = list(batch.glob(name + "_*"))
            assert len(paths) == 1
            statuses[name] = supervise(paths[0], policy["resource_limits"])
    finally:
        POLICY.write_bytes(policy_raw)
    assert POLICY.read_bytes() == policy_raw
    write_new(batch / "execution_summary.json", {
        "status": "complete" if all(x == "success" for x in statuses.values()) else "incomplete",
        "runs": statuses, "test_accessed": False, "final_retrain": False, "selection_updated": False,
        "stage_gate_restored": True,
    })
    print(json.dumps(statuses), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main(sys.argv[1])

