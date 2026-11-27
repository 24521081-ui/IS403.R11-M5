"""Publish or verify Stage 5. Entry points read the sealed Stage 4 result only."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import gettempdir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "wrmsse_stage5_mpl"))

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

from src.wrmsse_stage4 import now, read, verify_files
from src.wrmsse_integrity import business_revision, verify_historical_files
from src.wrmsse_stage5 import REPORTS, STAGE4, TABLES, build_tables, check_gates, csv, prepare_tables, rel, sha, verify_inputs
from src.wrmsse_stage5_docs import BUSINESS_REPORT, HANDOFF, write_docs, write_handoff
from src.wrmsse_stage5_excel import WORKBOOK, create_workbook, verify_workbook
from src.wrmsse_stage5_figures import FIGURES, create_figures
from src.wrmsse_stage5_notebooks import build_notebooks, cells, presentation_source
from src.wrmsse_stage2_data import write_new

MANIFEST = REPORTS / "manifest_stage5.json"
POSTFLIGHT = REPORTS / "stage5_verification.json"
BASELINE = REPORTS / "stage5_inputs.json"
CASE_LOG = REPORTS / "stage5_unit_tests.log"
CATALOG = REPORTS / "FIGURE_CATALOG.md"
LOCK = ROOT / "requirements-lock.txt"


def tests():
    with CASE_LOG.open("x", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover",
             "-s", "tests", "-t", ".", "-p", "test_wrmsse*.py"],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True,
        )
    return unit_count()


def unit_count():
    content = CASE_LOG.read_text(encoding="utf-8")
    count, = re.findall(r"Ran (\d+) tests? in ", content)
    assert content.rstrip().endswith("OK")
    return int(count)


def check_links(path, content, future=()):
    assert "\ufffd" not in content, path
    links = re.findall(r"\]\(([^)]+)\)", content)
    for link in links:
        destination = (path.parent / link.split("#")[0]).resolve()
        assert destination.is_relative_to(ROOT) and (destination.is_file() or destination in future), (path, link)
    return len(links)


def verify_notebooks(future=()):
    verified = {}
    revision = business_revision()
    for name, title, sections in cells():
        path = ROOT / "notebooks" / (name + ".ipynb")
        if revision and name == "09_business_insights_and_visualization":
            from src.wrmsse_business_v2_report import describe_notebook

            verified[rel(path)] = describe_notebook(path)
            assert verified[rel(path)] == revision["notebook"]
            continue
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        intended = presentation_source(name, title, sections)
        assert [(item.cell_type, item.source) for item in notebook.cells] == [
            (item.cell_type, item.source) for item in intended.cells
        ]
        assert notebook.metadata["wrmsse_mode"] == "read_only_artifacts"
        code = [item for item in notebook.cells if item.cell_type == "code"]
        assert [item.execution_count for item in code] == list(range(1, len(code) + 1))
        output = [entry for item in code for entry in item.outputs]
        assert output and all(entry.output_type != "error" for entry in output)
        fig_count = sum("image/png" in entry.get("data", {}) for entry in output)
        assert fig_count
        links = sum(check_links(path, cell.source, future) for cell in notebook.cells if cell.cell_type == "markdown")
        verified[rel(path)] = {"code_cells": len(code), "image_outputs": fig_count, "links": links}
    return verified


def verify_analysis():
    evidence = read(BASELINE)
    inputs = verify_inputs()
    assert all(inputs[key] == evidence[key] for key in inputs)
    verify_files(ROOT, evidence["table_sha256"])
    tables = build_tables()
    for name, expected in tables.items():
        stored = csv(TABLES / (name + ".csv"))
        pd.testing.assert_frame_equal(stored, expected.reset_index(drop=True), check_dtype=False,
                                      rtol=1e-12, atol=1e-12)
    day = tables["daily_sales"]
    for model, col in (("rf", "random_forest_forecast"), ("sn", "seasonal_naive_forecast")):
        np.testing.assert_allclose(day[f"{model}_change"].iloc[1:], np.diff(day[col].to_numpy(dtype=float)))
    assert day["day"].tolist() == list(range(1914, 1942))
    assert int(day.observations.sum()) == read(STAGE4 / "test_metrics.json")["observations"]
    assert sorted(tables["store_analysis"].store_id) == ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    assert len(tables["department_analysis"]) == 7 and len(tables["representative_daily"]) == 4 * 28
    stage4_level = csv(STAGE4 / "test_wrmsse_by_level.csv")
    for field in ("weight", "contribution", "wrmsse_within_level"):
        actual = tables["wrmsse_12_levels"]
        for model, prefix in (("random_forest", "rf"), ("seasonal_naive", "sn")):
            column = field if field == "weight" else f"{prefix}_{field}"
            np.testing.assert_allclose(actual[column], stage4_level.loc[stage4_level.model == model, field], atol=1e-12, rtol=0)
    return evidence, {name: len(rows) for name, rows in tables.items()}


def verify_figures():
    catalog = csv(REPORTS / "figure_catalog.csv", encoding="utf-8-sig")
    revision = business_revision()
    moved = {name for name, row in revision.get("archived", {}).items()
             if name.startswith("reports/figures/") and row["operation"] == "remove"}
    original = set(catalog.file)
    assert len(original) == 24 and {Path(name).name for name in moved} <= original
    active = (original - {Path(name).name for name in moved}) | set(revision.get("figures", ()))
    assert {path.name for path in FIGURES.glob("*.png")} == active
    if moved:
        sealed = read(MANIFEST)["files"]
        verify_historical_files(ROOT, {name: sealed[name] for name in moved})
    for row in catalog.to_dict("records"):
        image_path = FIGURES / row["file"]
        if rel(image_path) in moved:
            continue
        with Image.open(image_path) as image:
            assert min(image.info["dpi"]) >= 200 and image.width > 1200 and image.height > 600
            image.verify()
        for source in row["nguồn_dữ_liệu"].split(" | "):
            assert (ROOT / source).is_file()
            if source.endswith(".png"):
                assert sha(ROOT / source) == sha(image_path)
    return catalog.file.tolist()


def publish():
    for path in (MANIFEST, POSTFLIGHT, BUSINESS_REPORT, HANDOFF):
        if path.exists():
            raise FileExistsError(path)
    evidence, table_counts = verify_analysis()
    figures = verify_figures()
    notarized = verify_notebooks((BUSINESS_REPORT,))
    print("Analysis, figures, and ten notebooks verified. Reading full XLSX detail.", flush=True)
    counts = verify_workbook()
    test_count = tests()
    write_docs()
    files = sorted([
        *[path for path in REPORTS.rglob("*") if path.is_file()],
        *[ROOT / name for name in notarized], *ROOT.glob("src/wrmsse_stage5*.py"),
        Path(__file__).resolve(), LOCK, BUSINESS_REPORT, ROOT / "README.md",
    ])
    checks = [
        ("Unit test evaluator và pipeline", f"{test_count} đạt"),
        ("Toàn lưới Forecast_Detail", f"{counts['Forecast_Detail']:,} dòng khớp Parquet theo từng trường".replace(",", ".")),
        ("WRMSSE và trọng số", "12 cấp khớp evaluator, contribution cộng đúng WRMSSE"),
        ("Notebook", f"{len(notarized)} tệp chạy Run All và có bảng/hình thật"),
        ("Hình", f"{len(figures)} PNG với nguồn đã định danh và từ 200 DPI"),
        ("Manifest Giai đoạn 1–4", "Bằng chứng và các manifest niêm phong đã đối chiếu"),
        ("Dữ liệu raw, dự án nguồn và legacy", "Danh sách tệp, kích thước và mốc sửa đổi khớp snapshot"),
        ("Training, tuning, Test", "Gate đóng trong toàn bộ Giai đoạn 5"),
    ]
    write_handoff(counts, [*files, POSTFLIGHT], checks)
    documents = [ROOT / "README.md", BUSINESS_REPORT, CATALOG, HANDOFF]
    links = sum(check_links(path, path.read_text(encoding="utf-8"), (MANIFEST, POSTFLIGHT)) for path in documents)
    record = {
        "status": "passed", "at": now(), "candidate": "RF-15 + FE-04",
        "official_test_days": [1914, 1941], "manifest_inputs_verified": len(evidence["manifests"]),
        "unit_tests_passed": test_count, "workbook_sheets": counts, "workbook_full_detail_checked": True,
        "workbook_values_match_analytical_csv_and_stage4_parquet": True,
        "tables_recomputed": table_counts, "figure_files": figures, "notebooks_executed_read_only": notarized,
        "utf8_markdown_links_verified": links, "historical_unchanged": True, "protected_inventory_counts": evidence["protected_inventory_counts"],
        "fit": False, "tuning": False, "raw_test_loaded": False,
    }
    write_new(POSTFLIGHT, record)
    manifest = {"status": "completed", "at": now(), "stage": 5,
                "stage4_run": STAGE4.name, "contract_id": evidence["contract_id"],
                "input_manifests": evidence["manifests"], "workbook": rel(WORKBOOK),
                "files": {rel(path): sha(path) for path in sorted([*files, HANDOFF, POSTFLIGHT])},
                "updated": ["README.md"]}
    write_new(MANIFEST, manifest)
    verify_files(ROOT, manifest["files"])
    check_gates()
    print(json.dumps(record, ensure_ascii=False), flush=True)


def verify():
    manifest = read(MANIFEST)
    assert manifest["status"] == "completed" and manifest["stage4_run"] == STAGE4.name
    dispositions = verify_historical_files(ROOT, manifest["files"])
    record = read(POSTFLIGHT)
    assert record["status"] == "passed" and not record["fit"] and not record["tuning"]
    assert verify_analysis()[1] == record["tables_recomputed"]
    assert verify_figures() == record["figure_files"]
    assert verify_workbook() == record["workbook_sheets"]
    notebooks = verify_notebooks()
    original = record["notebooks_executed_read_only"]
    assert notebooks.keys() == original.keys()
    revision = business_revision()
    revised_notebook = "notebooks/09_business_insights_and_visualization.ipynb"
    for path, data in notebooks.items():
        expected = revision["notebook"] if revision and path == revised_notebook else original[path]
        assert data == expected
    if revision:
        from scripts.run_business_v2 import verify as verify_business

        verify_business()
    assert unit_count() == record["unit_tests_passed"]
    for path in (ROOT / "README.md", BUSINESS_REPORT, CATALOG, HANDOFF):
        check_links(path, path.read_text(encoding="utf-8"))
    check_gates()
    print(f"Stage 5 manifest: {dispositions}; {record['workbook_sheets']['Forecast_Detail']:,} Excel detail rows verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Read-only verification of the published Stage 5 final result")
    mode.add_argument("--publish", action="store_true", help="Verify and seal existing generated workbook, tables, plots and notebooks")
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.publish:
        publish()
    else:
        if REPORTS.exists() or BUSINESS_REPORT.exists():
            raise FileExistsError("Stage 5 requires a fresh reports directory or --verify for existing outputs.")
        prepare_tables()
        create_workbook()
        create_figures()
        build_notebooks()
        publish()


if __name__ == "__main__":
    main()
