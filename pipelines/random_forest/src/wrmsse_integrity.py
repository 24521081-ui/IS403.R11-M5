"""Check compact artifacts and, separately, the external historical archive."""

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from zipfile import ZipFile

from .wrmsse_config import ROOT
from .wrmsse_stage2_data import sha

INVENTORY = ROOT / "docs/PROJECT_CLEANUP_INVENTORY.csv"


def is_compact():
    return INVENTORY.is_file()


@lru_cache(maxsize=1)
def inventory():
    if not is_compact():
        return {}
    with INVENTORY.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {row["path"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate paths in cleanup inventory")
    return indexed


def _checked_path(base, name):
    suffix = Path(name)
    path = (base / suffix).resolve()
    if suffix.is_absolute() or suffix.drive or not path.is_relative_to(base) or path == base:
        raise ValueError(f"Unsafe artifact path: {name}")
    return path


def verify_raw():
    """Compare all eight local raw CSVs to the recorded source checksums."""
    receipt = ROOT / "preparation/source_manifest.json"
    expected = {
        path.as_posix(): row["sha256"]
        for row in json.loads(receipt.read_text(encoding="utf-8-sig"))
        if (path := PureWindowsPath(row["path"])).parts[:2] == ("data", "raw")
    }
    directory = ROOT / "data/raw"
    if len(expected) != 8 or {p.relative_to(ROOT).as_posix() for p in directory.glob("*.csv")} != set(expected):
        raise ValueError("Raw CSV inventory differs from the import receipt")
    for name, checksum in expected.items():
        if sha(_checked_path(ROOT, name)) != checksum:
            raise ValueError(f"Raw checksum differs: {name}")
    return len(expected)


def business_revision():
    path = ROOT / "reports/manifest_business_v2.json"
    if not path.is_file():
        return {}
    revision = json.loads(path.read_text(encoding="utf-8"))
    if (revision["version"], revision["status"]) != ("business_v2", "completed"):
        raise ValueError("Unexpected business archive revision")
    return revision


def verify_historical_files(base, files):
    """Verify local bytes or an indexed archival disposition, with separate counts."""
    counts = dict(local=0, equivalent=0, indexed_external=0)
    archived = inventory()
    revised = business_revision().get("archived", {})
    for name, expected in files.items():
        path = _checked_path(base, name)
        if path.is_file() and sha(path) == expected:
            counts["local"] += 1
            continue
        relative = path.relative_to(ROOT).as_posix()
        entry = archived.get(relative, {})
        if entry.get("sha256") != expected or entry.get("operation") not in {"remove", "revise"}:
            entry = revised.get(relative, {})
        operation = entry.get("operation")
        if (entry.get("sha256") != expected or operation not in {"remove", "revise"}
                or (operation == "remove" and path.exists())):
            raise ValueError(f"Checksum mismatch or missing artifact: {relative}")
        equivalent = entry.get("equivalent", "")
        if equivalent.startswith("project:"):
            duplicate = _checked_path(ROOT, equivalent.removeprefix("project:"))
            if not duplicate.is_file() or sha(duplicate) != expected:
                raise ValueError(f"Checksum mismatch for equivalent artifact: {relative}")
            counts["equivalent"] += 1
        elif entry["archive_member"] == "objects/" + expected:
            counts["indexed_external"] += 1
        else:
            raise ValueError(f"No preserved bytes indexed for sealed artifact: {relative}")
    return counts


def verify_archive(archive):
    """Read and hash each archived object without restoring anything."""
    if not is_compact():
        raise FileNotFoundError(INVENTORY)
    expected = {row["archive_member"]: row for row in inventory().values() if row["archive_member"]}
    with ZipFile(archive) as stream:
        names = stream.namelist()
        index_name = "cleanup_inventory_final.csv" if "cleanup_inventory_final.csv" in names else "cleanup_inventory.csv"
        indices = {"cleanup_inventory.csv", index_name}
        if len(names) != len(set(names)) or set(names) != set(expected) | indices:
            raise ValueError("External archive member index differs from the cleanup inventory")
        if hashlib.sha256(stream.read(index_name)).hexdigest() != sha(INVENTORY):
            raise ValueError("External archive inventory checksum differs")
        for name, row in expected.items():
            if stream.getinfo(name).file_size != int(row["bytes"]):
                raise ValueError(f"Archived size differs: {name}")
            with stream.open(name) as content:
                if hashlib.file_digest(content, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError(f"Archived checksum differs: {name}")
    return len(expected)
