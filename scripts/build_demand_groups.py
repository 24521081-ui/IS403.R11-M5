"""Phân nhóm 30.490 chuỗi M5 (item_id, store_id) theo ADI tại các mốc origin."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META_COLS = ["item_id", "store_id", "state_id", "cat_id", "dept_id"]


def compute_revenue_window(
    wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    origin: int,
    window: int = 28,
) -> np.ndarray:
    """Tính doanh thu thực tế của từng chuỗi trong `window` ngày kết thúc tại `origin`."""
    revenue = np.zeros(len(wide), dtype=np.float64)
    week_map = calendar.set_index("d")["wm_yr_wk"]
    for day in range(origin - window + 1, origin + 1):
        day_col = f"d_{day}"
        week = week_map[day_col]
        day_prices = (
            wide[["store_id", "item_id"]]
            .merge(
                prices.loc[prices["wm_yr_wk"].eq(week), ["store_id", "item_id", "sell_price"]],
                on=["store_id", "item_id"],
                how="left",
                validate="one_to_one",
            )["sell_price"]
            .to_numpy(dtype=np.float64)
        )
        units = wide[day_col].to_numpy(dtype=np.float64)
        revenue += units * np.nan_to_num(day_prices, nan=0.0)
    return revenue


def build_groups_for_origin(
    wide: pd.DataFrame,
    origin: int,
    adi_threshold: float = 1.32,
    expected_rows: int | None = 30490,
) -> pd.DataFrame:
    """Tính first_positive_day, ADI, CV2 và gán nhãn frequent/intermittent đến `origin`."""
    day_cols = [f"d_{d}" for d in range(1, origin + 1)]
    sales = wide[day_cols].to_numpy(dtype=np.float64)
    n_series, n_days = sales.shape

    pos_mask = sales > 0
    has_pos = pos_mask.any(axis=1)
    first_idx = np.argmax(pos_mask, axis=1)

    first_positive_day = np.empty(n_series, dtype=object)
    active_days = np.zeros(n_series, dtype=np.int64)
    positive_days = np.zeros(n_series, dtype=np.int64)
    zero_rate = np.ones(n_series, dtype=np.float64)
    adi = np.full(n_series, np.nan, dtype=np.float64)
    cv2 = np.full(n_series, np.nan, dtype=np.float64)
    cold_start = (~has_pos).astype(np.int64)
    insufficient_history = np.zeros(n_series, dtype=np.int64)
    demand_group = np.empty(n_series, dtype=object)

    for i in range(n_series):
        if not has_pos[i]:
            first_positive_day[i] = ""
            insufficient_history[i] = 1
            demand_group[i] = "intermittent"
            continue

        start = int(first_idx[i])
        active = sales[i, start:]
        t_i = n_days - start
        pos_vals = active[active > 0]
        n_pos = int(len(pos_vals))

        first_positive_day[i] = f"d_{start + 1}"
        active_days[i] = t_i
        positive_days[i] = n_pos
        zero_rate[i] = float((t_i - n_pos) / t_i)
        adi_val = float(t_i / n_pos)
        adi[i] = adi_val

        if n_pos < 2:
            insufficient_history[i] = 1
            demand_group[i] = "intermittent"
        else:
            mean_pos = float(pos_vals.mean())
            std_pos = float(pos_vals.std(ddof=0))
            cv2[i] = float((std_pos / mean_pos) ** 2) if mean_pos > 0 else np.nan
            demand_group[i] = "frequent" if adi_val <= adi_threshold else "intermittent"

    out = wide[META_COLS].copy().reset_index(drop=True)
    out["first_positive_day"] = first_positive_day
    out["active_days"] = active_days
    out["positive_days"] = positive_days
    out["zero_rate"] = zero_rate
    out["ADI"] = adi
    out["CV2"] = cv2
    out["demand_group"] = demand_group
    out["cold_start"] = cold_start
    out["insufficient_history"] = insufficient_history

    if expected_rows is not None:
        assert len(out) == expected_rows, f"Kỳ vọng {expected_rows} chuỗi, nhận {len(out)}"
    assert not out.duplicated(["item_id", "store_id"]).any(), "Trùng cặp (item_id, store_id)"
    assert set(out["demand_group"]).issubset({"frequent", "intermittent"})
    return out


def summarize_groups(
    groups_df: pd.DataFrame,
    revenue: np.ndarray,
    origin: int,
) -> pd.DataFrame:
    """Tạo bảng tóm tắt số chuỗi và tỷ trọng doanh thu 28 ngày trước origin theo nhóm."""
    df = groups_df.copy()
    df["revenue_28d"] = revenue
    total_series = len(df)
    total_rev = float(df["revenue_28d"].sum())

    rows = []
    for grp in ["frequent", "intermittent"]:
        sub = df[df["demand_group"] == grp]
        rev = float(sub["revenue_28d"].sum())
        rows.append(
            {
                "origin": origin,
                "demand_group": grp,
                "n_series": int(len(sub)),
                "series_share_pct": 100.0 * len(sub) / total_series,
                "revenue_28d": rev,
                "revenue_share_pct": 100.0 * rev / total_rev if total_rev > 0 else 0.0,
                "cold_start_count": int(sub["cold_start"].sum()),
                "insufficient_history_count": int(sub["insufficient_history"].sum()),
                "median_ADI": float(sub["ADI"].median()) if sub["ADI"].notna().any() else np.nan,
                "median_zero_rate": float(sub["zero_rate"].median()),
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origins",
        nargs="+",
        type=int,
        default=[1885, 1913],
        help="Danh sách origin cần dán nhãn (mặc định: 1885 1913)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/raw",
        help="Thư mục chứa sales_train_evaluation.csv, calendar.csv, sell_prices.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/demand_groups",
        help="Thư mục lưu demand_groups_origin_<origin>.csv",
    )
    parser.add_argument(
        "--adi-threshold",
        type=float,
        default=1.32,
        help="Ngưỡng ADI phân tách frequent vs intermittent (mặc định: 1.32)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")

    needed_days = set()
    for origin in args.origins:
        for d in range(origin - 27, origin + 1):
            needed_days.add(f"d_{d}")
    needed_weeks = set(calendar.loc[calendar["d"].isin(needed_days), "wm_yr_wk"])
    prices = pd.concat(
        [
            part.loc[part["wm_yr_wk"].isin(needed_weeks)]
            for part in pd.read_csv(data_dir / "sell_prices.csv", chunksize=250_000)
        ],
        ignore_index=True,
    )

    summaries = []
    for origin in args.origins:
        groups_df = build_groups_for_origin(wide, origin=origin, adi_threshold=args.adi_threshold)
        out_path = out_dir / f"demand_groups_origin_{origin}.csv"
        groups_df.to_csv(out_path, index=False)

        revenue = compute_revenue_window(wide, calendar, prices, origin=origin, window=28)
        summary_df = summarize_groups(groups_df, revenue=revenue, origin=origin)
        summaries.append(summary_df)
        print(f"Đã ghi: {out_path}")
        print(summary_df.to_string(index=False))
        print("-" * 80)

    all_summary = pd.concat(summaries, ignore_index=True)
    summary_path = out_dir / "demand_groups_summary.csv"
    all_summary.to_csv(summary_path, index=False)
    print(f"Đã ghi bảng tổng hợp: {summary_path}")


if __name__ == "__main__":
    main()
