"""
Chấm WRMSSE 12 cấp cho Seasonal_Naive từ dự báo đã lưu.
Không train, không dự báo lại và không ghi đè file forecasts.csv.

Yêu cầu:
- Đặt file này trong scripts/
- Có scripts/score_c7_wrmsse.py lấy từ nhánh origin/main
- Có dữ liệu M5 trong data/raw/
- Có forecasts.csv của 10 cửa hàng trong results/benchmarks/predictions/

Ví dụ:
    python scripts/score_seasonal_naive_wrmsse.py --split validation
    python scripts/score_seasonal_naive_wrmsse.py --split test
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_c7_wrmsse import META, score


ROOT = Path(__file__).resolve().parents[1]
STORES = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3",
]
KEYS = ["item_id", "store_id", "target_day"]
MODEL = "Seasonal_Naive"
SCORER_SOURCE_COMMIT = "185f46caa969738f6647fb4aed1a6157cc02368e"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_forecasts(split: str, origin: int, forecast_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    expected_days = {f"d_{day}" for day in range(origin + 1, origin + 29)}
    tables: list[pd.DataFrame] = []
    paths: list[Path] = []

    for store in STORES:
        path = forecast_root / store.lower() / split / "forecasts.csv"
        require(path.is_file(), f"Không tìm thấy file dự báo: {path}")

        table = pd.read_csv(path, float_precision="round_trip")
        required = {
            "model", "split", "origin_day", "item_id", "store_id",
            "target_day", "horizon", "y_pred",
        }
        require(required.issubset(table.columns), f"Thiếu cột trong {path}")

        table = table.loc[table["model"].eq(MODEL)].copy()
        require(len(table) == 3049 * 28, f"{store}: cần 85.372 dòng Seasonal_Naive, nhận {len(table):,}")
        require(table["store_id"].eq(store).all(), f"{store}: có dòng thuộc cửa hàng khác")
        require(table["split"].eq(split).all(), f"{store}: sai split")
        # CSV dự báo lưu origin_day dạng số (1885/1913), còn scorer WRMSSE
        # chung yêu cầu chuỗi "d_1885"/"d_1913". Chấp nhận cả hai dạng,
        # sau đó chuẩn hóa trong bộ nhớ, không sửa CSV dự báo gốc.
        origin_as_text = table["origin_day"].astype(str).str.strip()
        origin_as_number = pd.to_numeric(table["origin_day"], errors="coerce")
        origin_ok = origin_as_text.eq(f"d_{origin}") | origin_as_number.eq(origin)
        require(
            origin_ok.all(),
            f"{store}: sai origin_day (nhận được: {sorted(origin_as_text.unique().tolist())})",
        )
        table["origin_day"] = f"d_{origin}"
        require(set(table["target_day"]) == expected_days, f"{store}: sai tập ngày dự báo")
        day_numbers = table["target_day"].str[2:].astype(int)
        require(table["horizon"].astype(int).eq(day_numbers - origin).all(), f"{store}: sai horizon")
        require(not table.duplicated(KEYS).any(), f"{store}: trùng khóa dự báo")
        table["y_pred"] = pd.to_numeric(table["y_pred"], errors="raise")
        require(np.isfinite(table["y_pred"]).all(), f"{store}: dự báo có NaN/Inf")
        require(table["y_pred"].ge(0).all(), f"{store}: dự báo âm")

        tables.append(table[
            ["model", "split", "origin_day", "item_id", "store_id",
             "target_day", "horizon", "y_pred"]
        ])
        paths.append(path)

    forecasts = pd.concat(tables, ignore_index=True)
    require(len(forecasts) == 30490 * 28, "Tổng dự báo phải đủ 853.720 dòng")
    require(not forecasts.duplicated(KEYS).any(), "Trùng khóa item_id-store_id-target_day sau khi ghép")
    require(forecasts[["item_id", "store_id", "target_day", "y_pred"]].notna().all().all(),
            "Dự báo có trường bị trống")

    pair_counts = forecasts.groupby(["item_id", "store_id"]).size()
    require(len(pair_counts) == 30490 and pair_counts.eq(28).all(),
            "Phải có đúng 30.490 cặp sản phẩm-cửa hàng, mỗi cặp đủ 28 ngày")
    return forecasts, paths


def load_m5_data(data_dir: Path, origin: int):
    sales_path = data_dir / "sales_train_evaluation.csv"
    calendar_path = data_dir / "calendar.csv"
    prices_path = data_dir / "sell_prices.csv"

    for path in [sales_path, calendar_path, prices_path]:
        require(path.is_file(), f"Thiếu dữ liệu M5: {path}")

    days_needed = [f"d_{day}" for day in range(1, origin + 29)]
    sales = pd.read_csv(
        sales_path,
        usecols=META + days_needed,
        dtype={day: np.int16 for day in days_needed},
    )
    require(len(sales) == 30490, "sales_train_evaluation.csv phải có 30.490 chuỗi")
    require(not sales.duplicated(["item_id", "store_id"]).any(), "Dữ liệu M5 bị trùng cặp sản phẩm-cửa hàng")
    require(set(sales["store_id"]) == set(STORES), "Dữ liệu M5 không đủ đúng 10 cửa hàng")
    require(sales.groupby("store_id").size().eq(3049).all(), "Mỗi cửa hàng phải có 3.049 chuỗi")

    calendar = pd.read_csv(calendar_path)
    require({"d", "wm_yr_wk"}.issubset(calendar.columns), "calendar.csv thiếu d hoặc wm_yr_wk")
    require(calendar["d"].is_unique, "calendar.csv có mã ngày trùng")

    revenue_days = [f"d_{day}" for day in range(origin - 27, origin + 1)]
    require(set(revenue_days).issubset(set(calendar["d"])), "calendar.csv thiếu ngày tính trọng số doanh thu")
    revenue_weeks = set(
        calendar.loc[calendar["d"].isin(revenue_days), "wm_yr_wk"]
    )

    price_columns = ["item_id", "store_id", "wm_yr_wk", "sell_price"]
    price_parts = []
    for chunk in pd.read_csv(prices_path, chunksize=250_000):
        selected = chunk.loc[chunk["wm_yr_wk"].isin(revenue_weeks), price_columns]
        if not selected.empty:
            price_parts.append(selected)

    require(bool(price_parts), "Không tìm thấy giá bán trong các tuần tính doanh thu")
    prices = pd.concat(price_parts, ignore_index=True)
    require(not prices.duplicated(["item_id", "store_id", "wm_yr_wk"]).any(),
            "sell_prices.csv bị trùng khóa sản phẩm-cửa hàng-tuần")
    require(np.isfinite(prices["sell_price"]).all() and prices["sell_price"].ge(0).all(),
            "sell_price có giá trị không hợp lệ")

    return sales, calendar, prices, [sales_path, calendar_path, prices_path], revenue_days


def score_one_split(split: str, args: argparse.Namespace) -> None:
    origin = 1885 if split == "validation" else 1913
    start_day, end_day = origin + 1, origin + 28

    forecast_root = args.forecast_dir
    if not forecast_root.is_absolute():
        forecast_root = ROOT / forecast_root

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    output_root = args.output_dir
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_dir = output_root / split

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Thư mục đã có kết quả: {output_dir}. "
            "Hãy chọn --output-dir mới để không ghi đè."
        )

    forecasts, forecast_paths = load_forecasts(split, origin, forecast_root)
    wide, calendar, prices, raw_paths, revenue_days = load_m5_data(data_dir, origin)

    # Đối chiếu độ phủ với đúng 30.490 cặp trong dữ liệu M5.
    pairs = wide[["item_id", "store_id"]]
    counts = forecasts.groupby(["item_id", "store_id"]).size().rename("n").reset_index()
    coverage = pairs.merge(
        counts, on=["item_id", "store_id"], how="outer",
        validate="one_to_one", indicator=True,
    )
    require(coverage["_merge"].eq("both").all() and coverage["n"].eq(28).all(),
            "Dự báo không khớp chính xác 30.490 cặp trong dữ liệu M5")

    levels, details = score(
        origin, wide, calendar, prices, forecasts, return_details=True
    )
    levels_df = pd.DataFrame(levels)
    require(len(levels_df) == 12, "Bộ chấm không trả về đủ 12 cấp")
    require(int(levels_df["n_series"].sum()) == 42840, "Tổng số chuỗi sau phân cấp phải là 42.840")
    np.testing.assert_allclose(levels_df["weight_sum"], 1.0, rtol=0, atol=1e-12)

    wrmsse = float(levels_df["weighted_RMSSE"].mean())
    contribution_sum = float(levels_df["contribution_WRMSSE"].sum())
    np.testing.assert_allclose(wrmsse, contribution_sum, rtol=0, atol=1e-12)

    output_dir.mkdir(parents=True, exist_ok=True)
    levels_df.insert(0, "model", MODEL)
    levels_df.insert(1, "split", split)
    details.insert(0, "model", MODEL)
    details.insert(1, "split", split)

    summary = pd.DataFrame([{
        "model": MODEL,
        "split": split,
        "origin": origin,
        "start_day": f"d_{start_day}",
        "end_day": f"d_{end_day}",
        "n_stores": 10,
        "n_series": 30490,
        "n_rows": len(forecasts),
        "n_levels": len(levels_df),
        "revenue_start_day": revenue_days[0],
        "revenue_end_day": revenue_days[-1],
        "WRMSSE": wrmsse,
        "scorer": "score_c7_wrmsse.score",
        "scorer_source_commit": SCORER_SOURCE_COMMIT,
    }])

    summary.to_csv(output_dir / "diem_chung.csv", index=False)
    levels_df.to_csv(output_dir / "diem_theo_12_cap.csv", index=False)
    details.to_csv(output_dir / "rmsse_va_trong_so_cap_san_pham_cua_hang.csv", index=False)

    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in [*forecast_paths, *raw_paths]}
    manifest = {
        "status": "complete",
        "model": MODEL,
        "split": split,
        "origin": origin,
        "forecast_days": [start_day, end_day],
        "revenue_days": [origin - 27, origin],
        "n_stores": 10,
        "n_series": 30490,
        "n_forecast_rows": len(forecasts),
        "n_hierarchy_levels": 12,
        "WRMSSE": wrmsse,
        "scorer_source": "24521081-ui/IS403.R11-M5, branch main",
        "scorer_source_commit": SCORER_SOURCE_COMMIT,
        "inputs_sha256": hashes,
        "no_training_or_inference": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 66)
    print(f"HOÀN TẤT WRMSSE — {MODEL} — {split.upper()}")
    print("=" * 66)
    print(summary.to_string(index=False))
    print("\nĐiểm theo 12 cấp:")
    print(levels_df[["level", "group_by", "n_series", "weighted_RMSSE"]].to_string(index=False))
    print(f"\nĐã lưu kết quả tại: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=["validation", "test", "both"], default="validation",
        help="Chọn kỳ chấm. Nên chấm validation trước; test sau khi chốt cấu hình.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw"),
        help="Thư mục chứa 3 file dữ liệu M5.",
    )
    parser.add_argument(
        "--forecast-dir", type=Path, default=Path("results/benchmarks/predictions"),
        help="Thư mục chứa dự báo của 10 cửa hàng.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/benchmarks/metrics/seasonal_naive_wrmsse"),
        help="Thư mục xuất kết quả; mỗi split có thư mục con riêng.",
    )
    args = parser.parse_args()

    splits = ["validation", "test"] if args.split == "both" else [args.split]
    for split in splits:
        score_one_split(split, args)


if __name__ == "__main__":
    main()
