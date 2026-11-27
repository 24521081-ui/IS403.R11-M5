"""Tổng hợp bảng so sánh các mô hình với đủ MAE, RMSE, RMSSE và WRMSSE 12 cấp."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_c7_wrmsse import META, score  # noqa: E402


KEYS = ["item_id", "store_id", "target_day"]


def load_m5_reference(data_dir: Path, origin: int = 1913):
    """Nạp dữ liệu M5 gốc và bảng giá trong cửa sổ doanh thu 28 ngày trước origin."""
    wide = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    rev_days = [f"d_{d}" for d in range(origin - 27, origin + 1)]
    weeks = set(calendar.loc[calendar.d.isin(rev_days), "wm_yr_wk"])
    prices = pd.concat(
        [
            part.loc[part.wm_yr_wk.isin(weeks)]
            for part in pd.read_csv(data_dir / "sell_prices.csv", chunksize=250000)
        ],
        ignore_index=True,
    )
    return wide, calendar, prices


def compute_level12_metrics(
    forecasts: pd.DataFrame, wide: pd.DataFrame, origin: int = 1913
) -> tuple[float, float]:
    """Tính MAE và RMSE trung bình trên 30.490 chuỗi cấp 12 từ forecast và actual."""
    days = [f"d_{d}" for d in range(origin + 1, origin + 29)]
    matrix = forecasts.pivot(
        index=["item_id", "store_id"], columns="target_day", values="y_pred"
    )
    index = pd.MultiIndex.from_frame(wide[["item_id", "store_id"]])
    pred = matrix.reindex(index=index, columns=days).to_numpy(dtype=float)
    actual = wide[days].to_numpy(dtype=float)
    err = actual - pred
    mae_mean = float(np.abs(err).mean(axis=1).mean())
    rmse_mean = float(np.sqrt(np.square(err).mean(axis=1)).mean())
    return mae_mean, rmse_mean


def score_benchmark_forecasts(
    benchmark_csv: Path,
    wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
) -> list[dict]:
    """Chấm WRMSSE 12 cấp cho từng mô hình trong file forecasts.csv của benchmarks."""
    df_all = pd.read_csv(benchmark_csv)
    rows = []
    for model_name, group in df_all.groupby("model", sort=True):
        internal = group[KEYS + ["y_pred"]].assign(
            origin_day="d_1913", split="test"
        )
        levels, details = score(
            1913, wide, calendar, prices, internal, return_details=True
        )
        wrmsse = float(sum(r["contribution_WRMSSE"] for r in levels))
        rmsse_mean = float(details["RMSSE"].mean())

        matched = benchmark_metrics[
            (benchmark_metrics["split"] == "test")
            & (benchmark_metrics["model"] == model_name)
        ]
        if not matched.empty:
            mae_mean = float(matched.iloc[0]["mae_mean"])
            rmse_mean = float(matched.iloc[0]["rmse_mean"])
        else:
            mae_mean, rmse_mean = compute_level12_metrics(internal, wide, 1913)

        rows.append(
            {
                "model": model_name,
                "split": "test",
                "mae_mean": round(mae_mean, 4),
                "rmse_mean": round(rmse_mean, 4),
                "rmsse_mean": round(rmsse_mean, 4),
                "wrmsse": round(wrmsse, 6),
                "series_count": len(details),
                "source": "benchmarks",
            }
        )
    return rows


def collect_evidence_models(
    evidence_dir: Path, wide: pd.DataFrame
) -> list[dict]:
    """Đọc các mô hình đã chấm bằng scripts/score_model_wrmsse.py trong evidence_dir."""
    if not evidence_dir.is_dir():
        return []
    rows = []
    for summary_path in sorted(evidence_dir.glob("*/diem_chung.csv")):
        model_dir = summary_path.parent
        summary = pd.read_csv(summary_path).iloc[0]
        details_path = model_dir / "rmsse_theo_cap_san_pham.csv"
        forecast_path = model_dir / "forecast_test.csv"

        rmsse_mean = (
            float(pd.read_csv(details_path)["RMSSE"].mean())
            if details_path.is_file()
            else float("nan")
        )
        if forecast_path.is_file():
            forecasts = pd.read_csv(forecast_path)
            mae_mean, rmse_mean = compute_level12_metrics(forecasts, wide, 1913)
        else:
            mae_mean, rmse_mean = float("nan"), float("nan")

        rows.append(
            {
                "model": str(summary["model"]),
                "split": str(summary["split"]),
                "mae_mean": round(mae_mean, 4),
                "rmse_mean": round(rmse_mean, 4),
                "rmsse_mean": round(rmsse_mean, 4),
                "wrmsse": round(float(summary["WRMSSE"]), 6),
                "series_count": int(summary["n_series"]),
                "source": "score_model_wrmsse",
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1] / "data/raw")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / ".local_evidence/wrmsse_6_models",
    )
    parser.add_argument(
        "--benchmark-forecasts",
        type=Path,
        default=ROOT / "results/benchmarks/predictions/all_stores/test/forecasts.csv",
    )
    parser.add_argument(
        "--benchmark-metrics",
        type=Path,
        default=ROOT / "results/benchmarks/metrics/all_stores/summary_overall.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/final_comparison.csv",
    )
    args = parser.parse_args(argv)

    wide, calendar, prices = load_m5_reference(args.data_dir, origin=1913)
    benchmark_metrics = (
        pd.read_csv(args.benchmark_metrics)
        if args.benchmark_metrics.is_file()
        else pd.DataFrame(columns=["split", "model", "mae_mean", "rmse_mean"])
    )

    all_rows = collect_evidence_models(args.evidence_dir, wide)
    existing_models = {r["model"] for r in all_rows}

    if args.benchmark_forecasts.is_file():
        bench_rows = score_benchmark_forecasts(
            args.benchmark_forecasts, wide, calendar, prices, benchmark_metrics
        )
        for row in bench_rows:
            if row["model"] not in existing_models:
                all_rows.append(row)

    if not all_rows:
        raise RuntimeError("Không tìm thấy dữ liệu mô hình nào để tổng hợp.")

    result_df = (
        pd.DataFrame(all_rows)
        .sort_values("wrmsse", ascending=True)
        .reset_index(drop=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(result_df.to_string(index=False))
    print(f"\nĐã lưu bảng so sánh tại: {args.output}")


if __name__ == "__main__":
    main()
