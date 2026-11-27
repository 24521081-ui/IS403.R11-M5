"""Prepare, archive and publish the read-only business report version 2."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import gettempdir
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "wrmsse_business_v2_mpl"))

import pandas as pd
from PIL import Image

from src.wrmsse_business_v2 import MANIFEST, NOTEBOOK, PREFIX, build
from src.wrmsse_business_v2_figures import SPECS, draw, figure_path
from src.wrmsse_business_v2_report import VERSION, catalogue, create_notebook, describe_notebook, report
from src.wrmsse_integrity import verify_historical_files
from src.wrmsse_stage4 import now, read, verify_files
from src.wrmsse_stage5 import REPORTS, TABLES, STAGE4, check_gates, csv, rel, sha, verify_inputs

FIGURES = REPORTS / "figures"
BUSINESS_DOC = ROOT / "docs/STAGE5_BUSINESS_REPORT.md"
CATALOG = REPORTS / "FIGURE_CATALOG.md"
HANDOFF = REPORTS / "STAGE5_HANDOFF.md"
STAGE = Path(gettempdir()) / "wrmsse_business_v2_stage"
ARCHIVE = ROOT.with_name("PTDLKD_RandomForest_WRMSSE_business_v1.zip")
INDEX = STAGE / "archive_index.json"
HISTORICAL = REPORTS / "manifest_stage5.json"


def verify_archive(path, archived, expected_sha=None):
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError("Business v1 ZIP checksum differs")
    objects = {row["archive_member"]: row for row in archived.values()}
    with ZipFile(path) as archive:
        assert len(archive.namelist()) == len(objects) + 1
        assert set(archive.namelist()) == {*objects, "archive_index.json"}
        assert json.loads(archive.read("archive_index.json")) == archived
        for member, row in objects.items():
            assert archive.getinfo(member).file_size == row["bytes"]
            with archive.open(member) as source:
                assert hashlib.file_digest(source, "sha256").hexdigest() == row["sha256"]


def archive_previous():
    check_gates()
    assert not MANIFEST.exists() and not INDEX.exists()
    verify_historical_files(ROOT, read(HISTORICAL)["files"])
    originals = csv(REPORTS / "figure_catalog.csv")
    targets = {FIGURES / name: "remove" for name in originals.file.iloc[12:]}
    for path in (NOTEBOOK, BUSINESS_DOC, CATALOG, HANDOFF, ROOT / "scripts/run_stage5.py", ROOT / "src/wrmsse_integrity.py"):
        targets[path] = "revise"
    archived = {}
    for path, operation in targets.items():
        checksum = sha(path)
        archived[rel(path)] = {"sha256": checksum, "bytes": path.stat().st_size, "operation": operation,
                               "archive_member": "objects/" + checksum}
    temporary = ARCHIVE.with_suffix(".partial.zip")
    if ARCHIVE.exists():
        raise FileExistsError(ARCHIVE)
    with ZipFile(temporary, "x", compression=ZIP_DEFLATED, compresslevel=5) as archive:
        written = set()
        for name, row in archived.items():
            if row["archive_member"] not in written:
                archive.write(ROOT / name, row["archive_member"])
                written.add(row["archive_member"])
        archive.writestr("archive_index.json", json.dumps(archived, ensure_ascii=False, indent=2))
    verify_archive(temporary, archived)
    temporary.rename(ARCHIVE)
    with INDEX.open("x", encoding="utf-8") as out:
        json.dump({"archived": archived, "archive": {"filename": ARCHIVE.name, "sha256": sha(ARCHIVE),
                   "bytes": ARCHIVE.stat().st_size}}, out, ensure_ascii=False, indent=2)
    print(f"Archived and verified {len(archived)} prior files at {ARCHIVE}", flush=True)


def prepare():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    STAGE.mkdir(exist_ok=True)
    derived, base, metrics = build()
    for name, frame in derived.items():
        frame.to_csv(STAGE / (PREFIX + name + ".csv"), index=False)
    draw(STAGE, derived, base, metrics)
    print(f"Prepared {len(derived)} tables and {len(SPECS)} figures at {STAGE}", flush=True)


def verify_new_file_set(directory):
    for slug, _, _, _, _ in SPECS:
        path = directory / figure_path(slug)
        with Image.open(path) as image:
            assert min(image.info["dpi"]) > 299.9 and image.width > 2600
            image.verify()


def transfer(src, destination):
    with src.open("rb") as input_, destination.open("xb") as output:
        shutil.copyfileobj(input_, output)


def revised_handoff(archived):
    removed = {Path(name).name for name, row in archived.items() if row["operation"] == "remove"}
    lines = [line for line in HANDOFF.read_text(encoding="utf-8").splitlines()
             if not any(f"[{name}]" in line for name in removed)]
    lines.extend(["", "## Trực quan hóa báo cáo v2", "",
                  "Danh mục hiện hành có 17 nhóm hình kinh doanh mới hoặc thay thế. Bản v1 của hình và tài liệu "
                  "được đối chiếu bằng SHA-256. [Danh mục và bảng nguồn](FIGURE_CATALOG.md) · "
                  "[Manifest v2](manifest_business_v2.json)."])
    return "\n".join(lines) + "\n"


def check_links(*docs):
    count = 0
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        if doc.suffix == ".ipynb":
            text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"] if c["cell_type"] == "markdown")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            name = link.split("#", 1)[0]
            if not name or "://" in name:
                continue
            target = (doc.parent / name).resolve()
            assert target.is_relative_to(ROOT) and target.is_file(), (doc, link)
            count += 1
    return count


def sources():
    paths = [STAGE4 / name for name in ("manifest_completed.json", "forecast_lock_manifest.json",
             "test_actual_and_forecast.parquet", "test_metrics.json", "test_model_comparison.csv",
             "test_wrmsse_by_level.csv", "test_wrmsse_by_series.parquet", "feature_importance.csv")]
    pre = read(STAGE4 / "preflight.json")
    paths.extend(ROOT / pre["processed"] / name for name in ("manifest.json", "series.parquet", "validation_sales.npy"))
    paths.extend((HISTORICAL, REPORTS / "stage5_inputs.json", REPORTS / "figure_catalog.csv",
                 REPORTS / "RandomForest_Forecast_Actual.xlsx"))
    return {rel(path): sha(path) for path in paths}


def publish():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    receipt = read(INDEX)
    archived = receipt["archived"]
    verify_archive(ARCHIVE, archived, receipt["archive"]["sha256"])
    inputs = sources()
    derived, base, metrics = build()
    for name, frame in derived.items():
        path = STAGE / (PREFIX + name + ".csv")
        pd.testing.assert_frame_equal(csv(path), frame.reset_index(drop=True),
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    verify_new_file_set(STAGE)
    figures = [figure_path(spec[0]) for spec in SPECS]
    table_files = [PREFIX + name + ".csv" for name in derived]
    for file in figures:
        transfer(STAGE / file, FIGURES / file)
    for file in table_files:
        transfer(STAGE / file, TABLES / file)
    output = STAGE / NOTEBOOK.name
    notebook = create_notebook(derived, base, metrics, output)
    initial_catalogue = csv(REPORTS / "figure_catalog.csv")
    updates = {BUSINESS_DOC: report(derived, base, metrics), CATALOG: catalogue(initial_catalogue),
               HANDOFF: revised_handoff(archived)}
    for path in [*updates, NOTEBOOK]:
        assert sha(path) == archived[rel(path)]["sha256"]
    removals = []
    for name, row in archived.items():
        if row["operation"] == "remove":
            path = ROOT / name
            assert path.resolve() == path and path.parent == FIGURES and sha(path) == row["sha256"]
            removals.append(path)
    for path, content in updates.items():
        path.write_text(content, encoding="utf-8")
    shutil.copyfile(output, NOTEBOOK)
    for path in removals:
        path.unlink()
    files = [*updates, NOTEBOOK, *[FIGURES / name for name in figures],
             *[TABLES / name for name in table_files], Path(__file__).resolve(),
             *ROOT.glob("src/wrmsse_business_v2*.py"), ROOT / "tests/test_wrmsse_business_v2.py",
             ROOT / "scripts/run_stage5.py", ROOT / "src/wrmsse_integrity.py"]
    verify_files(ROOT, inputs)
    suite = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-t", ".",
                            "-p", "test_wrmsse*.py"], cwd=ROOT, check=True, capture_output=True, text=True)
    test_count = int(re.search(r"Ran (\d+) tests", suite.stderr)[1])
    manifest = {"version": VERSION, "status": "completed", "at": now(), "test_observations": metrics["observations"],
                "notebook": notebook, "unit_tests_passed": test_count, "figures": figures,
                "table_rows": {PREFIX + name + ".csv": len(frame) for name, frame in derived.items()},
                "source_sha256": inputs, **receipt, "files": {rel(path): sha(path) for path in files},
                "training": False, "tuning": False, "test_evaluation": False, "visual_reviewed_png": len(SPECS)}
    with MANIFEST.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    verify(archive=True)


def verify(archive=False):
    check_gates()
    manifest = read(MANIFEST)
    assert manifest["version"] == VERSION and manifest["status"] == "completed"
    verify_files(ROOT, manifest["source_sha256"])
    verify_files(ROOT, manifest["files"])
    verify_inputs()
    assert describe_notebook(NOTEBOOK) == manifest["notebook"]
    verify_new_file_set(FIGURES)
    assert set(manifest["figures"]) == {figure_path(spec[0]) for spec in SPECS}
    for path, rows in manifest["table_rows"].items():
        assert len(csv(TABLES / path)) == rows
    if archive:
        verify_archive(ARCHIVE, manifest["archived"], manifest["archive"]["sha256"])
    links = check_links(NOTEBOOK, BUSINESS_DOC, CATALOG, HANDOFF)
    check_gates()
    print(f"Business v2 verified: {len(manifest['figures'])} figures, {len(manifest['table_rows'])} tables, {links} links, "
          f"{manifest['unit_tests_passed']} passing tests.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--archive-v1", action="store_true")
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--verify-archive", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.archive_v1:
        archive_previous()
    elif args.publish:
        publish()
    else:
        verify(args.verify_archive)
