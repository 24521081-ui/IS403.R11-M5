"""Verify the compact workspace and optionally rehash the historical ZIP."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wrmsse_integrity import is_compact, verify_archive, verify_historical_files, verify_raw
from src.wrmsse_stage4 import read, verify_files
from src.wrmsse_stage5 import REPORTS, STAGE4, check_gates, verify_inputs


def verify(archive=None):
    check_gates()
    raw_count = verify_raw()
    evidence = verify_inputs()
    manifest4 = read(STAGE4 / "manifest_completed.json")
    verify_files(STAGE4, manifest4["files"])
    lock = read(STAGE4 / "forecast_lock_manifest.json")
    verify_files(STAGE4, lock["files"])
    history = verify_historical_files(ROOT, read(REPORTS / "manifest_stage5.json")["files"])
    compact = is_compact()
    if compact:
        if (ROOT / "legacy").exists():
            raise ValueError("Unexpected legacy directory in compact project")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        assert requirements == ["-r requirements-lock.txt"]
        assert (ROOT / "requirements-lock.txt").is_file()
        assert len(list((ROOT / "notebooks").glob("*.ipynb"))) == 10
    check_gates()
    result = {
        "status": "passed", "compact": compact, "raw_csv_verified": raw_count,
        "test_run_files_verified_locally": len(manifest4["files"]),
        "forecast_lock_files_verified_locally": len(lock["files"]),
        "historical_manifest_count": len(evidence["manifests"]),
        "stage5_manifest_disposition": history,
        "archive_objects_rehashed": verify_archive(archive) if archive else None,
    }
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional ZIP to verify separately, without restoring it")
    verify(parser.parse_args().archive)
