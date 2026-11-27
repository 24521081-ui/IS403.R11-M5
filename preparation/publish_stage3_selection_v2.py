"""Publish Stage 3 selection v2 from verified artifacts without rerunning experiments."""
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project_paths import local_path
from src.wrmsse_config import CONTRACT_ID, CONFIG, require_stage_permission
from src.wrmsse_stage2_data import sha
from verify_stage3 import read, verify_files


RUN = ROOT / "results/wrmsse_v1/stage3_20261124T082126Z"
AUDIT = ROOT / "preparation/stage3_snap_audit.json"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert read(RUN / "postflight_verification.json")["status"] == "passed"
    for action in ("training", "tuning", "test_evaluation"):
        try:
            require_stage_permission(action)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Gate open: {action}")
    assert not read(ROOT / "configs/stage3.json")["allow_tuning"]
    audit = read(AUDIT)
    assert audit["status"] == "passed" and audit["contract_id"] == CONTRACT_ID
    assert audit["calendar"]["validation_state_days"] == 84 and audit["calendar"]["mismatches"] == 0
    assert not audit["fit_executed"] and not audit["test_accessed"]
    verify_files(RUN, read(RUN / "manifest_completed.json"))
    old_path = RUN / "selected_model.json"
    if not old_path.is_file():
        old_path = RUN / "selection_v2/selected_model_v1.json"
    old = read(old_path)
    assert old["name"] == "FE-03"
    rows = []
    for path in sorted(RUN.glob("*/*/record.json")):
        row = read(path)
        assert row["contract_id"] == CONTRACT_ID and row["purpose"] == "tuning"
        assert row["split"] == "validation" and row["origin"] == 1885 and row["status"] == "success"
        groups = set(row["requested_groups"])
        admitted = row["selection_eligible"]
        reason = "Eligible in Stage 3 v1"
        if "events" in groups:
            assert not admitted
            reason = "Event availability not reviewed"
        elif "snap" in groups:
            evidence = audit["candidates"][row["name"]]
            path = local_path(row["artifact_dir"])
            assert evidence["wrmsse"] == row["wrmsse"]
            assert evidence["model_sha256"] == sha(path / "model.joblib")
            assert evidence["predictions_sha256"] == sha(path / "validation_predictions.npy")
            assert evidence["manifest_sha256"] == sha(path / "manifest.json")
            admitted = True
            reason = "Verified SNAP rule derived entirely from pre-origin calendar"
        rows.append((row, admitted, reason))
    eligible = [row for row, admitted, _ in rows if admitted]
    assert len(rows) == 34 and len(eligible) == 33
    winner = min(eligible, key=lambda row: row["wrmsse"])
    assert winner["name"] == "FE-04"
    archive = local_path(winner["artifact_dir"])
    baseline = audit["seasonal_naive_wrmsse"]
    result = winner | {
        "selection_version": 2,
        "selection_eligible": True,
        "selection_basis": "Minimum Validation WRMSSE of 33 eligible Stage 3 records after SNAP review",
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "excluded_candidates": ["FE-05"],
        "hyperparameter_run": "RF-15",
        "prior_selection": old_path.relative_to(ROOT).as_posix(),
        "prior_selection_sha256": sha(old_path),
        "validation_days": CONFIG["validation"],
        "snap_rule": {
            "type": "state_by_day_of_month",
            "calendar_history_end": 1885,
            "days_of_month_on": audit["calendar"]["snap_days_of_month"],
            "validation_state_days_confirmed": 84,
            "series_day_rows_confirmed": {
                name: check["rows"] for name, check in audit["matrices"].items()
            },
        },
        "seasonal_naive_wrmsse": baseline,
        "improvement_vs_naive_percent": 100 * (1 - winner["wrmsse"] / baseline),
        "wrmsse_reduction_vs_naive": baseline - winner["wrmsse"],
        "audit_path": AUDIT.relative_to(ROOT).as_posix(),
        "audit_sha256": sha(AUDIT),
        "test_accessed": False,
        "final_retrain_performed": False,
    }
    destination = RUN / "selection_v2"
    destination.mkdir()
    shutil.copyfile(old_path, destination / "selected_model_v1.json")
    shutil.copyfile(archive / "wrmsse_by_level.csv", destination / "selected_wrmsse_by_level.csv")
    write_json(destination / "selected_model.json", result)
    eligibility = [{
        "name": row["name"], "group": row["group"], "version": row["version"],
        "feature_groups": ",".join(row["requested_groups"]), "wrmsse": row["wrmsse"],
        "eligible_v1": row["selection_eligible"], "eligible_v2": admitted,
        "selected_v2": row["name"] == winner["name"], "reason": reason,
        "record": f"{row['group']}/{row['name']}/record.json",
    } for row, admitted, reason in rows]
    write_csv(destination / "candidate_eligibility.csv", eligibility)
    comparison = [
        {"model": "Seasonal Naive", "wrmsse": baseline},
        {"model": "FE-03 (v1)", "wrmsse": old["wrmsse"]},
        {"model": "FE-04 (v2)", "wrmsse": winner["wrmsse"]},
    ]
    write_csv(destination / "baseline_comparison.csv", comparison)
    os.environ["MPLCONFIGDIR"] = str(local_path("preparation/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.4))
    values = [row["wrmsse"] for row in comparison]
    bars = axis.bar([row["model"] for row in comparison], values, color=["#64748b", "#475569", "#0369a1"])
    axis.bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=4)
    axis.set(ylim=(0, 1.05), ylabel="WRMSSE (12 M5 levels)", title="Validation d_1886–d_1913")
    fig.tight_layout()
    fig.savefig(destination / "baseline_comparison.png", dpi=160)
    plt.close(fig)
    manifest = {
        "status": "complete", "contract_id": CONTRACT_ID, "selection_version": 2,
        "input_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p) for p in (
                RUN / "manifest_completed.json", old_path, AUDIT,
                archive / "manifest.json", archive / "model.joblib",
                archive / "validation_predictions.npy", Path(__file__),
            )
        },
        "files": {p.name: sha(p) for p in destination.iterdir() if p.is_file()},
    }
    verify_files(destination, manifest)
    write_json(destination / "manifest.json", manifest)
    assert sha(old_path) == sha(destination / "selected_model_v1.json")
    print(json.dumps({"selected": result["name"], "score": result["wrmsse"],
                      "improvement_pct": result["improvement_vs_naive_percent"],
                      "result_dir": destination.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()

