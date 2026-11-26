"""Analyze saved Seasonal Naive WRMSSE results by category and demand frequency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "results/benchmarks/metrics/seasonal_naive_wrmsse"
DEFAULT_LABELS = ROOT / "reports/demand_groups/demand_groups_origin_1885.csv"
CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]
FREQUENCIES = ["frequent", "intermittent"]
EXPECTED_WRMSSE = {"validation": 0.922781334536, "test": 0.869700702331}
GROUP_LABELS = {
    "FOODS": "Thực phẩm",
    "HOBBIES": "Sở thích",
    "HOUSEHOLD": "Đồ gia dụng",
    "frequent": "Bán thường xuyên",
    "intermittent": "Bán gián đoạn",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    group_iter = frame.groupby(keys, sort=False, observed=True, dropna=False)
    for key, part in group_iter:
        key = key if isinstance(key, tuple) else (key,)
        revenue_share = float(part["revenue_weight"].sum())
        actual = float(part["actual_units"].sum())
        predicted = float(part["predicted_units"].sum())
        rows.append({
            **dict(zip(keys, key)),
            "n_series": int(len(part)),
            "revenue_share": revenue_share,
            "revenue_share_pct": revenue_share * 100,
            "weighted_rmsse": float(np.average(part["RMSSE"], weights=part["revenue_weight"])),
            "unweighted_mean_rmsse": float(part["RMSSE"].mean()),
            "actual_units": actual,
            "predicted_units": predicted,
            "unit_difference": predicted - actual,
            "forecast_bias_pct": (predicted - actual) / actual * 100 if actual else np.nan,
        })
    return pd.DataFrame(rows)


def write_bar_chart(frame, labels, value_col, title, ylabel, path, percent=False):
    values = frame[value_col].to_numpy(float)
    colors = ["#277da1", "#e76f51", "#43aa8b", "#f4a261", "#577590", "#90be6d"]
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    bars = ax.bar(labels, values, color=colors[:len(values)], width=0.66)
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    span = max(float(np.nanmax(np.abs(values))) if len(values) else 0, 1e-6)
    ax.set_ylim(min(0, float(np.nanmin(values)) - span * 0.18), float(np.nanmax(values)) + span * 0.24)
    for bar, value in zip(bars, values):
        text = f"{value:+.1f}%" if percent else f"{value:.3f}"
        y = bar.get_height()
        ax.annotate(text, (bar.get_x() + bar.get_width()/2, y), xytext=(0, 5 if y >= 0 else -16),
                    textcoords="offset points", ha="center", va="bottom" if y >= 0 else "top", fontsize=10)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def analyze(metrics_dir: Path, labels_path: Path, output_dir: Path) -> dict:
    require(labels_path.is_file(), f"Thiếu bảng nhãn chung: {labels_path}")
    require(not output_dir.exists() or not any(output_dir.iterdir()),
            f"Thư mục đầu ra không rỗng, từ chối ghi đè: {output_dir}")
    labels = pd.read_csv(labels_path)
    required_labels = {"item_id", "store_id", "cat_id", "demand_group"}
    require(required_labels.issubset(labels.columns), "Bảng nhãn thiếu cột bắt buộc")
    require(len(labels) == 30490, f"Cần 30.490 nhãn, nhận {len(labels):,}")
    require(not labels.duplicated(["item_id", "store_id"]).any(), "Bảng nhãn trùng item_id × store_id")
    require(labels["demand_group"].isin(FREQUENCIES).all(), "Có nhãn tần suất không hợp lệ")
    require(labels["cat_id"].isin(CATEGORIES).all(), "Có nhãn ngành hàng không hợp lệ")

    output_dir.mkdir(parents=True, exist_ok=True)
    checks = {"n_labels": len(labels), "label_duplicates": 0, "splits": {}}
    all_frames = []
    for split in ["validation", "test"]:
        split_dir = metrics_dir / split
        detail_path = split_dir / "rmsse_va_trong_so_cap_san_pham_cua_hang.csv"
        levels_path = split_dir / "diem_theo_12_cap.csv"
        summary_path = split_dir / "diem_chung.csv"
        require(all(p.is_file() for p in [detail_path, levels_path, summary_path]),
                f"Thiếu kết quả Seasonal Naive đã lưu cho {split}")
        details = pd.read_csv(detail_path)
        levels = pd.read_csv(levels_path)
        summary = pd.read_csv(summary_path).iloc[0]
        require(len(details) == 30490, f"{split}: cần đúng 30.490 dòng RMSSE")
        require(not details.duplicated(["item_id", "store_id"]).any(), f"{split}: trùng cặp sản phẩm-cửa hàng")
        require(details["split"].eq(split).all(), f"{split}: cột split không khớp")
        require(np.isfinite(details[["RMSSE", "revenue_weight", "actual_units", "predicted_units"]]).all().all(),
                f"{split}: chỉ số có NaN hoặc vô hạn")
        require(details["revenue_weight"].ge(0).all(), f"{split}: trọng số âm")
        require(np.isclose(details["revenue_weight"].sum(), 1, rtol=0, atol=1e-12),
                f"{split}: tổng trọng số cấp 12 khác 1")
        require(len(levels) == 12 and levels["level"].nunique() == 12,
                f"{split}: bảng cấp không có đủ 12 cấp")
        require(np.isclose(levels["weight_sum"].to_numpy(float), 1, rtol=0, atol=1e-12).all(),
                f"{split}: có cấp với tổng trọng số khác 1")

        joined = details.merge(labels, on=["item_id", "store_id"], how="outer",
                               validate="one_to_one", indicator=True)
        require(joined["_merge"].eq("both").all(), f"{split}: thiếu nhãn hoặc thiếu chuỗi RMSSE")
        joined = joined.drop(columns="_merge")
        require(len(joined) == 30490 and not joined[["cat_id", "demand_group"]].isna().any().any(),
                f"{split}: còn chuỗi thiếu nhãn")

        bottom_score = float(np.dot(details["revenue_weight"], details["RMSSE"]))
        level12 = float(levels.loc[levels["level"].eq(12), "weighted_RMSSE"].iloc[0])
        require(np.isclose(bottom_score, level12, rtol=0, atol=1e-12),
                f"{split}: tổng RMSSE có trọng số không khớp cấp 12 ({bottom_score} != {level12})")
        expected = EXPECTED_WRMSSE[split]
        require(np.isclose(float(summary["WRMSSE"]), expected, rtol=0, atol=5e-13),
                f"{split}: WRMSSE lưu không khớp giá trị tham chiếu")
        require(np.isclose(float(summary["WRMSSE"]), float(levels["weighted_RMSSE"].mean()), rtol=0, atol=1e-12),
                f"{split}: WRMSSE không khớp trung bình 12 cấp")

        category = aggregate(joined, ["cat_id"])
        frequency = aggregate(joined, ["demand_group"])
        cross = aggregate(joined, ["cat_id", "demand_group"])
        for table in [category, frequency, cross]:
            require(int(table["n_series"].sum()) == 30490, f"{split}: phân nhóm làm mất hoặc lặp chuỗi")
            require(np.isclose(table["revenue_share"].sum(), 1, atol=1e-12),
                    f"{split}: tổng tỷ trọng doanh thu phân nhóm khác 1")
        category.to_csv(output_dir / f"rmsse_theo_nganh_hang_{split}.csv", index=False)
        frequency.to_csv(output_dir / f"rmsse_theo_tan_suat_{split}.csv", index=False)
        cross.to_csv(output_dir / f"giao_nganh_hang_tan_suat_{split}.csv", index=False)
        pd.concat([
            category.assign(group_type="category", group=category["cat_id"]),
            frequency.assign(group_type="frequency", group=frequency["demand_group"]),
            cross.assign(group_type="category_frequency", group=cross["cat_id"] + " / " + cross["demand_group"]),
        ], ignore_index=True, sort=False).to_csv(output_dir / f"sai_lech_tong_luong_{split}.csv", index=False)

        # Translate categories only for chart labels; CSVs retain canonical keys.
        cat_labels = [GROUP_LABELS[x] for x in category["cat_id"]]
        freq_labels = [GROUP_LABELS[x] for x in frequency["demand_group"]]
        cross_labels = [f"{GROUP_LABELS[c]}\n{GROUP_LABELS[f]}" for c, f in zip(cross.cat_id, cross.demand_group)]
        write_bar_chart(category, cat_labels, "weighted_rmsse", f"Seasonal Naïve | RMSSE theo ngành hàng — {split.title()}", "RMSSE có trọng số doanh thu", output_dir / f"rmsse_nganh_hang_{split}.png")
        write_bar_chart(frequency, freq_labels, "weighted_rmsse", f"Seasonal Naïve | RMSSE theo tần suất — {split.title()}", "RMSSE có trọng số doanh thu", output_dir / f"rmsse_tan_suat_{split}.png")
        write_bar_chart(cross, cross_labels, "forecast_bias_pct", f"Sai lệch tổng lượng theo 6 nhóm — {split.title()}", "Dự báo dư (+) / thiếu (−), %", output_dir / f"sai_lech_tong_luong_{split}.png", percent=True)
        write_bar_chart(cross, cross_labels, "weighted_rmsse", f"Seasonal Naïve | 6 nhóm kết hợp — {split.title()}", "RMSSE có trọng số doanh thu", output_dir / f"rmsse_6_nhom_{split}.png")

        checks["splits"][split] = {
            "n_series": len(joined),
            "n_merged_groups": int(joined["demand_group"].notna().sum()),
            "weight_sum_bottom_level": float(details["revenue_weight"].sum()),
            "bottom_level_weighted_rmsse": bottom_score,
            "level_12_weighted_rmsse": level12,
            "wrmsse_12_level": float(summary["WRMSSE"]),
            "reference_wrmsse": expected,
            "wrmsse_matches_reference": True,
        }
        all_frames.append(joined)

    (output_dir / "kiem_tra.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRICS / "group_analysis")
    args = parser.parse_args()
    for name in ["metrics_dir", "labels", "output_dir"]:
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    results = analyze(args.metrics_dir, args.labels, args.output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
