"""Verify the current analytical edition independently of the historical report receipts."""

import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

from .wrmsse_business_v2 import MANIFEST, NOTEBOOK
from .wrmsse_demand_explorer import series_index, series_png
from .wrmsse_demand_groups import ENRICHED, build
from .wrmsse_demand_workbook import WORKBOOK, verify_extension
from .wrmsse_integrity import verify_raw
from .wrmsse_stage4 import read, verify_files
from .wrmsse_stage5 import REPORTS, ROOT, SELECTED, STAGE4, TABLES, check_gates, csv, rel, sha

RECEIPT = REPORTS / "manifest_demand_groups.json"
TEXT_TYPES = {".csv", ".ipynb", ".json", ".md", ".py", ".txt", ".jsonl"}


def protected_inventory():
    paths = [p for name in ("configs", "data", "preparation", "results") for p in (ROOT / name).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    return {rel(p): sha(p) for p in sorted(paths)}


def inventory_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def historical_bytes(content, path, expected):
    if hashlib.sha256(content).hexdigest() == expected:
        return content
    if path.suffix in TEXT_TYPES:
        lf = content.replace(b"\r\n", b"\n")
        for version in (lf, lf.replace(b"\n", b"\r\n")):
            if hashlib.sha256(version).hexdigest() == expected:
                return version
    raise ValueError(f"Prior edition checksum mismatch: {path}")


def verify_time_windows(detail, tables):
    dates = pd.to_datetime(detail.date)
    periods = [("daily", "date", detail.date),
               ("weekly", "week", (dates - pd.Timestamp("2016-04-25")).dt.days // 7 + 1),
               ("weekday", "weekday_no", dates.dt.day_of_week + 1)]
    cols = ["rf_mae", "rf_rmse", "rf_wape_pct", "rf_bias_pct"]
    for name, key, period in periods:
        published = tables[name].set_index(["demand_group", key], verify_integrity=True)
        observations = detail.groupby([detail.demand_group, period], observed=True)
        assert observations.ngroups == len(published)
        for identity, sub in observations:
            actual = sub.actual_sales.to_numpy()
            error = sub.random_forest_forecast.to_numpy() - actual
            units = actual.sum()
            values = [np.abs(error).mean(), np.sqrt(np.mean(error ** 2)),
                      100 * np.abs(error).sum() / units if units else np.nan,
                      100 * error.sum() / units if units else np.nan]
            np.testing.assert_allclose(published.loc[identity, cols].to_numpy(dtype=float), values, rtol=1e-13)
    daily = tables["daily"].sort_values("date").groupby("demand_group")
    full = tables["metrics"].query("split == 'Test'").set_index("demand_group")
    for col in ("actual_sales", "random_forest_forecast", "rf_error"):
        np.testing.assert_allclose(daily.last()[col + "_cumulative"], full[col], rtol=1e-13)


def verify():
    check_gates()
    record = read(RECEIPT)
    assert record["version"] == "business_v2_demand_groups"
    verify_files(ROOT, record["files"])
    verify_files(ROOT, record["inputs"])
    protected = protected_inventory()
    assert len(protected) == record["protected_files"]
    assert inventory_digest(protected) == record["protected_before_sha256"] == record["protected_after_sha256"]
    assert verify_raw() == 8
    final = read(STAGE4 / "manifest_completed.json")
    lock = read(STAGE4 / "forecast_lock_manifest.json")
    verify_files(STAGE4, final["files"])
    verify_files(STAGE4, lock["files"])
    config = read(STAGE4 / "final_model_config.json")
    assert config["candidate"] == "RF-15 + FE-04" and config["logical_features"] == 15
    assert config["selected_model_sha256"] == sha(SELECTED) and final["fit_count"] == 1
    model_results = read(STAGE4 / "test_metrics.json")["models"]
    np.testing.assert_allclose([read(SELECTED)["wrmsse"], model_results["random_forest"]["wrmsse"],
                                model_results["seasonal_naive"]["wrmsse"]],
                               [.6239308540, .7039124143, .8697007023], atol=5e-11, rtol=0)
    # Preserve the prior report in Git, including the exact XLSX bytes. Compare
    # each changed path with its own previous digest and the latest sealed entry.
    parent = read(MANIFEST)
    previous = {**parent["source_sha256"], **parent["files"]}
    revision = record["previous_git_commit"]
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Invalid prior Git revision")
    with TemporaryDirectory(prefix="m5_demand_verify_") as directory:
        old_book = Path(directory) / WORKBOOK.name
        for name, expected in {**previous, **record["prior_files"]}.items():
            if name in record["prior_files"]:
                if name in previous:
                    assert previous[name] == expected
                content = subprocess.check_output(["git", "cat-file", "blob", revision + ":" + name], cwd=ROOT)
                valid = historical_bytes(content, ROOT / name, expected)
                if name == rel(WORKBOOK):
                    old_book.write_bytes(valid)
            elif sha(ROOT / name) != expected:
                historical_bytes((ROOT / name).read_bytes(), ROOT / name, expected)
        detail, tables, audit = build()
        verify_time_windows(detail, tables)
        assert audit == record["label_audit"]
        pd.testing.assert_frame_equal(pd.read_parquet(ENRICHED), detail, check_exact=True)
        for name, expected in tables.items():
            stored = csv(TABLES / f"demand_group_{name}.csv", keep_default_na=False, na_values=[""])
            pd.testing.assert_frame_equal(stored, expected.reset_index(drop=True), check_dtype=False, check_exact=True)
        verify_extension(old_book, WORKBOOK, detail, tables)
    png_dir = REPORTS / "figures"
    catalog = csv(REPORTS / "figure_catalog.csv")
    assert catalog.file.is_unique and set(catalog.file) == {p.name for p in png_dir.glob("*.png")}
    for row in catalog.to_dict("records"):
        for name in row["nguồn_dữ_liệu"].split(" | "):
            assert (ROOT / name).is_file(), name
    for name in record["figures_new"] + record["figures_revised"]:
        with Image.open(png_dir / name) as im:
            assert im.width > 2600 and min(im.info["dpi"]) > 299.9
            im.verify()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    images = []
    explorer_images = []
    for cell in code:
        assert all(o.output_type != "error" for o in cell.outputs)
        target = explorer_images if "series-explorer" in cell.metadata.get("tags", []) else images
        target.extend(hashlib.sha256(base64.b64decode(out.data["image/png"])).hexdigest()
                      for out in cell.outputs if "image/png" in out.get("data", {}))
    expected_images = parent["figures"] + record["figures_new"]
    assert sorted(images) == sorted(sha(png_dir / name) for name in expected_images)
    index = series_index()
    assert len(index) == 30490 and index.index.is_unique
    assert index.loc[index.actual_sales.eq(0), ["rf_wape_pct", "rf_bias_pct"]].isna().all().all()
    png, _ = series_png(index, "intermittent", "FOODS_2_389", "TX_3")
    assert explorer_images == [hashlib.sha256(png).hexdigest()]
    for doc in [ROOT / "README.md", ROOT / "docs/STAGE5_BUSINESS_REPORT.md", REPORTS / "FIGURE_CATALOG.md",
                REPORTS / "STAGE5_HANDOFF.md", NOTEBOOK]:
        content = doc.read_text(encoding="utf-8") if doc != NOTEBOOK else "\n".join(
            c.source for c in notebook.cells if c.cell_type == "markdown")
        for link in re.findall(r"\]\(([^)]+)\)", content):
            target = link.split("#")[0]
            if target and "://" not in target:
                assert (doc.parent / target).is_file(), (doc, target)
    check_gates()
    return {"status": "passed", "protected_files": len(protected), "raw_csv": 8,
            "stage4_files": len(final["files"]), "forecast_lock_files": len(lock["files"]),
            "group_tables": len(tables), "test_rows_compared": len(detail), "excel_rows_compared": len(detail),
            "existing_sheets": 23, "new_sheets": 9, "notebook_code_cells": len(code),
            "embedded_figures": len(images) + len(explorer_images)}
