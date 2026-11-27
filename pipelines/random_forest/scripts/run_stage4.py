"""Run one supervised final fit, lock the forecasts, and evaluate Test once."""

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preparation"))

import numpy as np
import pandas as pd
import psutil
from scipy import sparse

from scripts.run_stage3_all_features import other_training
from verify_stage3 import metadata
from src.project_paths import local_path
from src.wrmsse_config import CONFIG, CONTRACT_ID, require_stage_permission
from src.wrmsse_integrity import verify_historical_files
from src.wrmsse_stage2_data import sha, write_new
from src.wrmsse_stage4 import LOCKED, LOGICAL, PARAMS, SNAP, evaluate, fit_forecasts, now, read, verify_files

OUTPUT = local_path(CONFIG["outputs"])
STAGE2 = OUTPUT / "stage2_20261123T202104Z"
STAGE3 = OUTPUT / "stage3_20261124T082126Z"
EXTENDED = OUTPUT / "stage3_all_features"
SELECTION = STAGE3 / "selection_v2" / "selected_model.json"
POLICY = ROOT / "configs" / "stage3.json"
LIMITS = read(POLICY)["resource_limits"]


def relative(path):
    return path.relative_to(ROOT).as_posix()


def closed_gates():
    for action in ("training", "tuning", "test_evaluation"):
        try:
            require_stage_permission(action)
        except RuntimeError:
            continue
        raise RuntimeError(f"Unexpected open stage gate: {action}")
    policy = read(POLICY)
    assert not policy["allow_tuning"] and not policy["allow_test_evaluation"]


def protected_state():
    return {"raw": metadata(ROOT / "data/raw"), "stage2": metadata(STAGE2), "stage3": metadata(STAGE3),
            "stage3_all_features": metadata(EXTENDED)}


def check_feature_code():
    rel = "src/wrmsse_stage2_data.py"
    trees = [ast.parse((folder / rel).read_text(encoding="utf-8"))
             for folder in (ROOT, STAGE3 / "code_snapshot_n_jobs8")]
    for name in ("numeric_block", "load_matrix"):
        definitions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
                       for tree in trees]
        assert ast.dump(definitions[0]) == ast.dump(definitions[1]), name
    for rel in ("src/wrmsse.py", "src/wrmsse_data.py"):
        expected = read(STAGE3 / "manifest_completed.json")["files"]["code_snapshot/" + rel]
        assert sha(ROOT / rel) == expected, rel


def confirm_events(table, feature_manifest, files, processed):
    audit = read(EXTENDED / "events_origin_audit.json")
    assert audit["calendar_history_end"] == 1885 and audit["validation_days"] == [1886, 1913]
    assert audit["eligible_by_origin_availability"] and not audit["test_accessed"]
    assert audit["feature_manifest_sha256"] == files["feature_hash"]
    assert audit["vocabulary"] == feature_manifest["details"]["event_vocabulary"]
    for split, nrows in (("train", 600000), ("validation", 853720)):
        check = audit["checks"][split]
        assert check["rows"] == nrows and check["encoded_columns"] == 22
        assert check["mismatched_values"] == 0
    requests = {}
    for name in ("FE-05", "FE-10"):
        path = local_path(table.at[name, "artifact_path"])
        manifest = read(path / "manifest.json")
        verify_historical_files(path, manifest["files"])
        request = read(path / "request.json")
        result = read(path / "result.json")
        assert request["contract_id"] == result["contract_id"] == CONTRACT_ID
        assert request["dataset"] == processed and request["features"] == files["feature_dir"]
        assert "events" in request["groups"] and result["split"] == "validation" and result["origin"] == 1885
        assert result["train_rows"] == 600000 and result["validation_rows"] == 853720
        if name == "FE-05":
            assert request["feature_hash"] == files["feature_hash"]
            assert request["dataset_hash"] == files["dataset_hash"]
        else:
            info = read(path / "feature_metadata.json")
            assert info["feature_manifest_sha256"] == files["feature_hash"]
        requests[name] = relative(path / "request.json")
    return {"status": "verified_for_this_window", "applies_to": ["FE-05", "FE-10"],
            "same_origin": 1885, "requests": requests, "events_days_in_validation": audit["events_on_days_in_validation"],
            "audit_sha256": sha(EXTENDED / "events_origin_audit.json"),
            "event_block_sha256": {name: digest for name, digest in feature_manifest["files"].items()
                                   if name.startswith("events/")}}


def close_optimization(run, table, events, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table["historical_selection_status"] = table["selection_status"]
    table.loc["FE-05", "eligible_for_selection"] = True
    table.loc[["FE-05", "FE-10"], "selection_status"] = "events_verified_for_this_window"
    summary = read(STAGE3 / "summary.json")
    execution = read(EXTENDED / "execution_summary.json")
    assert summary["successful"] == 30 and summary["reused"] == 4 and summary["failed"] == 0
    assert execution["runs"] == {"FE-09": "success", "FE-10": "success"}
    assert not summary["test_actual_accessed"] and not execution["test_accessed"]
    assert table.loc[table.eligible_for_selection, "validation_wrmsse"].idxmin() == "FE-04"
    table.to_csv(run / "stage3_fe01_fe10.csv")
    table.to_parquet(run / "stage3_fe01_fe10.parquet")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(list(table.index), table.validation_wrmsse,
                  color=["#17675e" if name == "FE-04" else "#65809a" for name in table.index])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axhline(selected["seasonal_naive_wrmsse"], color="#c27428", ls="--", label="Seasonal Naive")
    ax.set(title="So sánh FE-01–FE-10 trên Validation d_1886–d_1913",
           ylabel="WRMSSE", ylim=(0, 1.02))
    ax.legend(loc="upper left")
    fig.savefig(run / "stage3_fe01_fe10.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_new(run / "stage3_optimization_lock.json", {
        "status": "closed", "initial_tuning": "RF-04", "base_features": "RF-v1",
        "advanced_tuning": "RF-15", "selected_features": "FE-04",
        "selected_model_sha256": sha(SELECTION), "wrmsse_validation": selected["wrmsse"],
        "seasonal_naive_wrmsse_validation": selected["seasonal_naive_wrmsse"],
        "experiments": 36, "successful_fits": 32, "reused": 4, "events": events,
        "files": {name: sha(run / name) for name in (
            "stage3_fe01_fe10.csv", "stage3_fe01_fe10.parquet", "stage3_fe01_fe10.png")},
    })


def preflight(run):
    closed_gates()
    assert not other_training(), "Another project Python process is active"
    ram = psutil.virtual_memory()
    disk_free = psutil.disk_usage(str(ROOT)).free
    cpu = psutil.cpu_count() or 0
    assert ram.available / 2**20 >= LIMITS["min_available_mib"] + 256
    assert cpu >= PARAMS["n_jobs"] and disk_free >= 2**30
    before = protected_state()
    write_new(run / "protected_before.json", before)
    stage2 = read(STAGE2 / "summary.json")
    old = read(STAGE3 / "preflight.json")
    processed, features = local_path(stage2["processed"]), local_path(stage2["features"])
    manifests = [
        STAGE2 / "manifest_completed.json", STAGE3 / "manifest_completed.json", SELECTION.parent / "manifest.json",
        EXTENDED / "manifest.json", processed / "manifest.json", features / "manifest.json",
    ]
    for path in manifests:
        verify_historical_files(path.parent, read(path)["files"])
    # The worker's checkpoint predates its final log writes. Verify the sealed manifest from the supervisor.
    stage2_receipt = ROOT / "preparation/stage2_completion.json"
    assert sha(manifests[0]) == read(stage2_receipt)["manifest_completed_sha256"]
    verify_historical_files(ROOT, read(SELECTION.parent / "manifest.json")["input_sha256"])
    selected = read(SELECTION)
    assert selected["selection_version"] == 2 and selected["name"] == "FE-04"
    assert selected["hyperparameter_run"] == "RF-15" and selected["version"] == "rf_v1"
    assert selected["groups"] == ["snap"] and selected["params"] == PARAMS
    assert selected["contract_id"] == CONTRACT_ID and selected["origin"] == 1885
    assert not selected["test_accessed"] and not selected["final_retrain_performed"]
    assert selected["n_raw_features"] == len(LOGICAL) == 15 and selected["n_features"] == 3082
    assert round(selected["wrmsse"], 10) == 0.6239308540
    assert round(selected["seasonal_naive_wrmsse"], 10) == 0.9227813345
    assert selected["snap_rule"]["days_of_month_on"] == SNAP
    assert selected["snap_rule"]["calendar_history_end"] == 1885
    assert read(STAGE3 / "advanced/RF-15/record.json")["params"] == PARAMS
    actual_file = local_path(CONFIG["test_actual_source"])
    raw_history_checksums = old["raw_sha256"]
    verify_files(ROOT / "data/raw", raw_history_checksums)
    # Test contents are locked. Preflight checks only the path and size from the clone receipt.
    source_receipt = ROOT / "preparation/source_manifest.json"
    source_entry = next(row for row in read(source_receipt) if Path(row["path"]).as_posix() == CONFIG["test_actual_source"])
    assert actual_file.stat().st_size == source_entry["bytes"]
    assert sha(STAGE3 / "evaluator.joblib") == old["evaluator_sha256"]
    check_feature_code()
    feature_manifest = read(features / "manifest.json")
    dataset_hash = sha(processed / "manifest.json")
    feature_hash = sha(features / "manifest.json")
    assert dataset_hash == old["dataset_manifest_sha256"] == selected["dataset_hash"]
    assert feature_hash == old["feature_manifest_sha256"] == selected["feature_hash"]
    assert dataset_hash == feature_manifest["dataset_manifest_sha256"]
    assert feature_manifest["dataset"] == stage2["processed"]
    assert feature_manifest["specs"]["rf_v1"]["raw_features"] + ["snap_state"] == LOGICAL
    audit_path = local_path(selected["audit_path"])
    assert sha(audit_path) == selected["audit_sha256"] and read(audit_path)["status"] == "passed"
    table = pd.read_csv(EXTENDED / "comparison_fe01_fe10.csv").set_index("experiment")
    assert table.index.tolist() == [f"FE-{n:02}" for n in range(1, 11)]
    assert round(table.at["FE-09", "validation_wrmsse"], 10) == 0.6321382731
    assert round(table.at["FE-10", "validation_wrmsse"], 10) == 0.6360743504
    events = confirm_events(table, feature_manifest, {
        "feature_hash": feature_hash, "dataset_hash": dataset_hash, "feature_dir": stage2["features"],
    }, stage2["processed"])
    close_optimization(run, table, events, selected)
    with (run / "unit_tests.log").open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-B", "scripts/run_preparation_checks.py"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    tests = json.loads((run / "unit_tests.log").read_text(encoding="utf-8").splitlines()[-1])
    assert tests["status"] == "passed" and tests["contract_id"] == CONTRACT_ID
    evidence = local_path(tests["log"]).with_name(Path(tests["log"]).stem.replace("unit_tests_", "unit_tests_summary_", 1) + ".json")
    assert read(evidence) == tests
    assert not other_training(), "Another project Python process is active"
    write_new(run / "preflight.json", {
        "status": "passed", "at": now(), "run": run.name, "contract_id": CONTRACT_ID,
        "params": selected["params"], "selected_model": relative(SELECTION),
        "selected_model_sha256": sha(SELECTION), "snap_audit_sha256": selected["audit_sha256"],
        "processed": stage2["processed"], "features": stage2["features"],
        "input_manifests_sha256": {relative(path): sha(path) for path in
                                   [*manifests, SELECTION, source_receipt, stage2_receipt, POLICY]},
        "test_evidence": relative(evidence), "unit_tests_passed": tests["tests"],
        "expected_test_actual_sha256": source_entry["sha256"], "test_actual_opened": False,
        "stage3_validation_evaluator_sha256": old["evaluator_sha256"],
        "evaluator_source_sha256": sha(ROOT / "src/wrmsse.py"),
        "historical_raw_sha256": raw_history_checksums, "limits": LIMITS,
        "available_mib": ram.available / 2**20, "total_ram_mib": ram.total / 2**20,
        "disk_free_bytes": disk_free, "logical_cpu": cpu, "physical_cpu": psutil.cpu_count(logical=False),
    })


def worker(run):
    pre = read(run / "preflight.json")
    closed_gates()
    fit_forecasts(run, pre)
    closed_gates()
    write_new(run / "forecast_lock_manifest.json", {
        "status": "locked", "at": now(), "run": run.name, "contract_id": CONTRACT_ID,
        "preflight_sha256": sha(run / "preflight.json"), "test_actual_accessed": False,
        "files": {name: sha(run / name) for name in LOCKED},
    })
    evaluate(run, pre)
    closed_gates()


def supervise(run):
    assert not other_training(), "Another project Python process is active"
    started = perf_counter()
    peak_rss = peak_tree = peak_cpu = peak_host_cpu = 0.0
    minimum = psutil.virtual_memory().available / 2**20
    assert minimum >= LIMITS["min_available_mib"] + 256
    stop_reason = None
    previous = None
    samples = 0
    with (run / "worker.log").open("x", encoding="utf-8") as log, (
        run / "resources.jsonl"
    ).open("x", encoding="utf-8") as resources:
        process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", relative(run)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            while process.poll() is None:
                try:
                    leader = psutil.Process(process.pid)
                    own = leader.memory_info().rss / 2**20
                    tree = cpu = 0.0
                    for member in [leader, *leader.children(recursive=True)]:
                        try:
                            tree += member.memory_info().rss / 2**20
                            times = member.cpu_times()
                            cpu += times.user + times.system
                        except psutil.NoSuchProcess:
                            continue
                    elapsed = perf_counter() - started
                    available = psutil.virtual_memory().available / 2**20
                    host_cpu = psutil.cpu_percent()
                    worker_cpu = max(0, 100 * (cpu - previous[0]) / (elapsed - previous[1])) if previous else 0.0
                    previous = (cpu, elapsed)
                    peak_rss, peak_tree = max(peak_rss, own), max(peak_tree, tree)
                    peak_cpu, peak_host_cpu = max(peak_cpu, worker_cpu), max(peak_host_cpu, host_cpu)
                    minimum = min(minimum, available)
                    resources.write(json.dumps({
                        "at": now(), "elapsed_seconds": elapsed, "rss_worker_mib": own, "rss_tree_mib": tree,
                        "available_mib": available, "machine_cpu_percent": host_cpu,
                        "worker_tree_cpu_percent_one_core": worker_cpu,
                    }) + "\n")
                    resources.flush()
                    samples += 1
                    if tree > LIMITS["max_rss_mib"] or available < LIMITS["min_available_mib"]:
                        stop_reason = "Stage 3 RAM limit or machine reserve crossed"
                        break
                except psutil.NoSuchProcess:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                for child in reversed(psutil.Process(process.pid).children(recursive=True)):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                process.wait(timeout=30)
    receipt = read(run / "fit_receipt.json") if (run / "fit_receipt.json").exists() else {}
    write_new(run / "resource_summary.json", receipt | {
        "returncode": process.returncode, "stop_reason": stop_reason, "sample_count": samples,
        "worker_elapsed_seconds": perf_counter() - started, "peak_worker_rss_mib": peak_rss,
        "peak_tree_rss_mib": peak_tree, "minimum_available_mib": minimum,
        "peak_worker_tree_cpu_percent_one_core": peak_cpu, "peak_machine_cpu_percent": peak_host_cpu,
        "logical_cpu": psutil.cpu_count(), "physical_cpu": psutil.cpu_count(logical=False),
        "resource_limits_mib": LIMITS, "fit_and_predict_sample_interval_seconds": 1,
    })
    if process.returncode or stop_reason:
        raise RuntimeError(f"Stage 4 worker stopped: {stop_reason or process.returncode}; see {run / 'worker.log'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(local_path(args.worker))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = local_path(f"{CONFIG['outputs']}/stage4_{stamp}")
    run.mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", str(run)) + "/wrmsse_stage4_mpl"
    print(f"Stage 4 run: {relative(run)}", flush=True)
    try:
        preflight(run)
        print("Preflight passed; supervised fit authorized.", flush=True)
        supervise(run)
        assert protected_state() == read(run / "protected_before.json")
        closed_gates()
        print(f"Fit and Test evaluation complete: {relative(run)}", flush=True)
    except Exception as error:
        write_new(run / "manifest_stopped.json", {
            "status": "stopped", "at": now(), "reason": str(error),
            "fit_started": (run / "fit_started.json").exists(),
            "test_gate_opened": (run / "test_access.json").exists(),
        })
        raise


if __name__ == "__main__":
    main()
