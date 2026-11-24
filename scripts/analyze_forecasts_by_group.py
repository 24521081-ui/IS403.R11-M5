"""Phân tích kết quả dự báo cấp 12 theo hai nhóm nhu cầu (frequent vs intermittent)."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from score_c7_wrmsse import score

ROOT = Path(__file__).resolve().parents[1]
KEY = ["item_id", "store_id"]


def load_level12_from_evidence(evidence_dir: Path) -> dict[str, pd.DataFrame]:
    """Đọc các bảng rmsse_theo_cap_san_pham.csv có sẵn trong .local_evidence/wrmsse_6_models."""
    tables: dict[str, pd.DataFrame] = {}
    if not evidence_dir.is_dir():
        return tables
    for model_dir in sorted(p for p in evidence_dir.iterdir() if p.is_dir()):
        csv_path = model_dir / "rmsse_theo_cap_san_pham.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        model_name = str(df["model"].iloc[0]) if "model" in df.columns and len(df) else model_dir.name
        tables[model_name] = df
    return tables


def load_level12_from_forecast_specs(
    specs: list[str],
    origin: int,
    wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Chấm các file dự báo chi tiết (item_id, store_id, target_day, y_pred) thành bảng cấp 12."""
    tables: dict[str, pd.DataFrame] = {}
    split = "validation" if origin == 1885 else "test"
    days = [f"d_{d}" for d in range(origin + 1, origin + 29)]
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Định dạng --forecast phải là TenMoHinh=duong_dan_hoac_glob, nhận: {spec}")
        model_name, pattern = spec.split("=", 1)
        cand = Path(pattern)
        full_pat = str(cand if cand.is_absolute() else ROOT / cand)
        paths = sorted(Path(p).resolve() for p in glob.glob(full_pat))
        if not paths:
            raise FileNotFoundError(f"Không tìm thấy file dự báo cho {model_name}: {pattern}")
        parts = [pd.read_csv(p, float_precision="round_trip") for p in paths]
        forecasts = pd.concat(parts, ignore_index=True)
        forecasts = forecasts[KEY + ["target_day", "y_pred"]].copy()
        forecasts["origin_day"] = f"d_{origin}"
        forecasts["split"] = split
        assert set(forecasts["target_day"]) == set(days), f"{model_name} không khớp kỳ {days[0]}..{days[-1]}"
        _, details = score(origin, wide, calendar, prices, forecasts, return_details=True)
        details["model"] = model_name
        tables[model_name] = details
    return tables


def evaluate_by_group(
    model_tables: dict[str, pd.DataFrame],
    groups_df: pd.DataFrame,
    ref_model: str | None = None,
    expected_rows: int | None = 30490,
) -> pd.DataFrame:
    """Tính RMSSE có trọng số, sai lệch tổng lượng (%) và tỷ lệ thắng theo từng nhóm."""
    if not model_tables:
        raise ValueError("Không có bảng kết quả mô hình nào để phân tích.")

    if ref_model is None:
        if "C6" in model_tables:
            ref_model = "C6"
        elif "ES_bu" in model_tables:
            ref_model = "ES_bu"
        else:
            ref_model = next(iter(model_tables.keys()))

    ref_df = None
    if ref_model in model_tables:
        ref_df = model_tables[ref_model][KEY + ["RMSSE"]].rename(columns={"RMSSE": "RMSSE_ref"})

    group_meta = groups_df[KEY + ["demand_group", "ADI"]]
    rows = []

    for model_name, df in model_tables.items():
        merged = df.merge(group_meta, on=KEY, how="inner", validate="one_to_one")
        if expected_rows is not None:
            assert len(merged) == expected_rows, f"{model_name}: kỳ vọng {expected_rows} dòng sau ghép nhóm, nhận {len(merged)}"
        if ref_df is not None:
            merged = merged.merge(ref_df, on=KEY, how="left", validate="one_to_one")

        for grp, label in [("frequent", "Thường xuyên"), ("intermittent", "Gián đoạn")]:
            sub = merged[merged["demand_group"] == grp]
            w_sum = float(sub["revenue_weight"].sum())
            w_rmsse = float(np.dot(sub["revenue_weight"], sub["RMSSE"]) / w_sum) if w_sum > 0 else np.nan
            act_tot = float(sub["actual_units"].sum())
            pred_tot = float(sub["predicted_units"].sum())
            bias_pct = 100.0 * (pred_tot / act_tot - 1.0) if act_tot > 0 else np.nan

            if ref_df is not None and model_name != ref_model:
                win_rate_pct = 100.0 * float((sub["RMSSE"] < sub["RMSSE_ref"]).mean())
            elif model_name == ref_model:
                win_rate_pct = np.nan
            else:
                win_rate_pct = np.nan

            rows.append(
                {
                    "model": model_name,
                    "demand_group": grp,
                    "nhom": label,
                    "so_chuoi": int(len(sub)),
                    "ty_trong_chuoi_pct": 100.0 * len(sub) / len(merged),
                    "ty_trong_doanh_thu_pct": 100.0 * w_sum,
                    "weighted_RMSSE": w_rmsse,
                    "actual_total": act_tot,
                    "forecast_total": pred_tot,
                    "bias_pct": bias_pct,
                    "ref_model": ref_model,
                    "win_rate_vs_ref_pct": win_rate_pct,
                }
            )

    res = pd.DataFrame(rows)
    ref_scores = (
        res.loc[res["model"] == ref_model, ["demand_group", "weighted_RMSSE"]]
        .rename(columns={"weighted_RMSSE": "ref_weighted_RMSSE"})
    )
    res = res.merge(ref_scores, on="demand_group", how="left")
    res["diff_vs_ref"] = res["weighted_RMSSE"] - res["ref_weighted_RMSSE"]
    return res


def plot_figures(
    groups_df: pd.DataFrame,
    results_df: pd.DataFrame,
    fig_dir: Path,
    split: str,
) -> None:
    """Vẽ các biểu đồ phân bố nhóm, RMSSE có trọng số và sai lệch theo nhóm."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    # 1. Phân bố số chuỗi & tỷ trọng doanh thu theo nhóm
    grp_summary = (
        results_df[["demand_group", "nhom", "so_chuoi", "ty_trong_chuoi_pct", "ty_trong_doanh_thu_pct"]]
        .drop_duplicates("demand_group")
        .sort_values("demand_group")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = grp_summary["nhom"].tolist()
    colors = ["#2563eb", "#f97316"]
    axes[0].bar(labels, grp_summary["ty_trong_chuoi_pct"], color=colors)
    axes[0].set_title("Tỷ trọng số chuỗi (%)")
    axes[0].set_ylim(0, 100)
    for idx, (val, cnt) in enumerate(zip(grp_summary["ty_trong_chuoi_pct"], grp_summary["so_chuoi"])):
        axes[0].text(idx, val + 2, f"{val:.1f}%\n({cnt:,})", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, grp_summary["ty_trong_doanh_thu_pct"], color=colors)
    axes[1].set_title("Tỷ trọng doanh thu 28 ngày (%)")
    axes[1].set_ylim(0, 100)
    for idx, val in enumerate(grp_summary["ty_trong_doanh_thu_pct"]):
        axes[1].text(idx, val + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Phân bố hai nhóm nhu cầu ({split})", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "demand_group_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Weighted RMSSE theo nhóm giữa các mô hình
    models = list(dict.fromkeys(results_df["model"]))
    x = np.arange(len(models))
    width = 0.36
    freq = results_df[results_df["demand_group"] == "frequent"].set_index("model").loc[models]
    inter = results_df[results_df["demand_group"] == "intermittent"].set_index("model").loc[models]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.8), 4.6))
    b1 = ax.bar(x - width / 2, freq["weighted_RMSSE"], width, label="Thường xuyên (ADI <= 1.32)", color="#2563eb")
    b2 = ax.bar(x + width / 2, inter["weighted_RMSSE"], width, label="Gián đoạn (ADI > 1.32)", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("RMSSE có trọng số doanh thu")
    ax.set_title(f"So sánh RMSSE có trọng số theo nhóm nhu cầu ({split})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "wrmsse_by_demand_group.png", dpi=150)
    plt.close(fig)

    # 3. Sai lệch tổng lượng (%) theo nhóm
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.8), 4.6))
    b1 = ax.bar(x - width / 2, freq["bias_pct"], width, label="Thường xuyên (ADI <= 1.32)", color="#2563eb")
    b2 = ax.bar(x + width / 2, inter["bias_pct"], width, label="Gián đoạn (ADI > 1.32)", color="#f97316")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Sai lệch tổng lượng (%)")
    ax.set_title(f"Sai lệch tổng lượng dự báo theo nhóm nhu cầu ({split})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 0.3 if h >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:+.2f}%", ha="center", va=va, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "bias_by_demand_group.png", dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--origin", type=int, default=None, help="Mặc định: 1885 cho validation, 1913 cho test")
    parser.add_argument("--groups-csv", type=Path, default=None, help="Mặc định: reports/demand_groups/demand_groups_origin_<origin>.csv")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / ".local_evidence/wrmsse_6_models")
    parser.add_argument("--forecast", nargs="*", default=[], help="Thêm mô hình từ file dự báo: Ten=glob_csv")
    parser.add_argument("--ref-model", type=str, default=None, help="Mô hình chuẩn để tính win_rate và chênh lệch")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "reports/demand_groups/figures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    origin = args.origin or (1885 if args.split == "validation" else 1913)
    groups_path = args.groups_csv or (ROOT / f"reports/demand_groups/demand_groups_origin_{origin}.csv")
    if not groups_path.is_absolute():
        groups_path = ROOT / groups_path
    groups_df = pd.read_csv(groups_path)

    model_tables: dict[str, pd.DataFrame] = {}
    if args.split == "test" and not args.forecast:
        ev_dir = args.evidence_dir if args.evidence_dir.is_absolute() else ROOT / args.evidence_dir
        model_tables.update(load_level12_from_evidence(ev_dir))
    elif args.split == "test" and args.evidence_dir.exists():
        ev_dir = args.evidence_dir if args.evidence_dir.is_absolute() else ROOT / args.evidence_dir
        model_tables.update(load_level12_from_evidence(ev_dir))

    if args.forecast:
        data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
        wide = pd.read_csv(data_dir / "sales_train_evaluation.csv")
        calendar = pd.read_csv(data_dir / "calendar.csv")
        weeks = set(calendar.loc[calendar["d"].isin([f"d_{d}" for d in range(origin - 27, origin + 1)]), "wm_yr_wk"])
        prices = pd.concat(
            [
                part.loc[part["wm_yr_wk"].isin(weeks)]
                for part in pd.read_csv(data_dir / "sell_prices.csv", chunksize=250_000)
            ],
            ignore_index=True,
        )
        model_tables.update(load_level12_from_forecast_specs(args.forecast, origin, wide, calendar, prices))

    results_df = evaluate_by_group(model_tables, groups_df, ref_model=args.ref_model)
    out_csv = args.output_csv or (ROOT / f"reports/demand_groups/results_by_demand_group_{args.split}.csv")
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)

    fig_dir = args.fig_dir if args.fig_dir.is_absolute() else ROOT / args.fig_dir
    plot_figures(groups_df, results_df, fig_dir=fig_dir, split=args.split)

    print(f"Đã lưu kết quả: {out_csv}")
    print(results_df[["model", "nhom", "so_chuoi", "ty_trong_doanh_thu_pct", "weighted_RMSSE", "diff_vs_ref", "win_rate_vs_ref_pct", "bias_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
