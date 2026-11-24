"""Chấm WRMSSE 12 cấp chuẩn M5 cho các mô hình thống kê đối chứng (Validation & Test)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from score_c7_wrmsse import score  # noqa: E402
from src.benchmarks.config import (  # noqa: E402
    DATA_DIR,
    METRICS_DIR,
    ORIGIN_DAY_TEST,
    ORIGIN_DAY_VAL,
    PREDICTIONS_DIR,
)

SPLIT_ORIGINS = {
    "validation": ORIGIN_DAY_VAL,
    "test": ORIGIN_DAY_TEST,
}


def load_reference_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nạp dữ liệu M5 gốc và bảng giá cho cửa sổ doanh thu cả Validation và Test."""
    wide = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    revenue_days = [f"d_{d}" for d in range(ORIGIN_DAY_VAL - 27, ORIGIN_DAY_TEST + 1)]
    weeks = set(calendar.loc[calendar["d"].isin(revenue_days), "wm_yr_wk"])
    prices = pd.concat(
        [
            part.loc[part["wm_yr_wk"].isin(weeks)]
            for part in pd.read_csv(data_dir / "sell_prices.csv", chunksize=250000)
        ],
        ignore_index=True,
    )
    return wide, calendar, prices


def export_split_by_model(
    forecasts_csv: Path,
    output_dir: Path,
    models: list[str] | None = None,
) -> list[Path]:
    """Tách file forecasts.csv nhiều mô hình thành từng CSV chuẩn 4 cột cho mỗi mô hình."""
    df = pd.read_csv(forecasts_csv, float_precision="round_trip")
    target_models = models or sorted(df["model"].unique())
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for model_name in target_models:
        sub = df.loc[df["model"] == model_name, ["item_id", "store_id", "target_day", "y_pred"]]
        if sub.empty:
            continue
        out_path = output_dir / f"{model_name}.csv"
        sub.to_csv(out_path, index=False)
        exported.append(out_path)
    return exported


def score_benchmarks_wrmsse(
    predictions_dir: Path = PREDICTIONS_DIR / "all_stores",
    metrics_dir: Path = METRICS_DIR / "all_stores",
    data_dir: Path = DATA_DIR,
    models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chấm WRMSSE 12 cấp cho Validation & Test từ forecasts.csv và cập nhật bảng chỉ số."""
    wide, calendar, prices = load_reference_data(data_dir)
    summary_rows: list[dict] = []
    level_rows: list[dict] = []

    for split, origin in SPLIT_ORIGINS.items():
        forecast_path = predictions_dir / split / "forecasts.csv"
        if not forecast_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp dự báo: {forecast_path}")

        df_split = pd.read_csv(forecast_path, float_precision="round_trip")
        target_models = models or sorted(df_split["model"].unique())

        for model_name in target_models:
            sub = df_split.loc[
                df_split["model"] == model_name,
                ["item_id", "store_id", "target_day", "y_pred"],
            ].copy()
            if sub.empty:
                raise ValueError(f"Không tìm thấy dự báo cho mô hình {model_name} ở kỳ {split}")

            internal = sub.assign(origin_day=f"d_{origin}", split=split)
            levels, details = score(origin, wide, calendar, prices, internal, return_details=True)
            wrmsse = float(sum(row["contribution_WRMSSE"] for row in levels))

            summary_rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "origin": origin,
                    "start_day": f"d_{origin + 1}",
                    "end_day": f"d_{origin + 28}",
                    "n_series": len(details),
                    "n_levels": len(levels),
                    "rmsse_mean": round(float(details["RMSSE"].mean()), 4),
                    "wrmsse": round(wrmsse, 6),
                }
            )
            for row in levels:
                level_rows.append({"split": split, "model": model_name, **row})

    summary_df = pd.DataFrame(summary_rows)
    levels_df = pd.DataFrame(level_rows)[
        [
            "split",
            "model",
            "origin",
            "level",
            "group_by",
            "n_series",
            "weight_sum",
            "weighted_RMSSE",
            "contribution_WRMSSE",
        ]
    ]

    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(metrics_dir / "wrmsse_summary.csv", index=False)
    levels_df.to_csv(metrics_dir / "wrmsse_by_level.csv", index=False)

    overall_path = metrics_dir / "summary_overall.csv"
    if overall_path.is_file():
        overall = pd.read_csv(overall_path)
        if "wrmsse" in overall.columns:
            overall = overall.drop(columns=["wrmsse"])
        overall = overall.merge(
            summary_df[["split", "model", "wrmsse"]],
            on=["split", "model"],
            how="left",
        )
        overall.to_csv(overall_path, index=False)

    return summary_df, levels_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR / "all_stores")
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR / "all_stores")
    parser.add_argument("--models", nargs="+", default=None, help="Danh sách mô hình cần chấm")
    parser.add_argument(
        "--export-test-by-model",
        action="store_true",
        help="Tách thêm CSV từng mô hình ở kỳ test để dùng với score_model_wrmsse.py",
    )
    args = parser.parse_args(argv)

    if args.export_test_by_model:
        exported = export_split_by_model(
            args.predictions_dir / "test" / "forecasts.csv",
            args.predictions_dir / "test" / "by_model",
            models=args.models,
        )
        for path in exported:
            print(f"Đã xuất dự báo test: {path}")

    summary_df, _ = score_benchmarks_wrmsse(
        predictions_dir=args.predictions_dir,
        metrics_dir=args.metrics_dir,
        data_dir=args.data_dir,
        models=args.models,
    )
    print("\n=== WRMSSE 12 CẤP CÁC MÔ HÌNH BENCHMARK ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
