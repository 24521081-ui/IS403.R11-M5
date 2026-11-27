"""Tạo bộ biểu đồ trực quan phân tích WRMSSE 12 cấp cho ES_bu và Croston (kèm Croston_SBA & đối chiếu)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.config import DATA_DIR, METRICS_DIR, PREDICTIONS_DIR, RESULTS_DIR  # noqa: E402
from src.benchmarks.evaluation.metrics import categorize_intermittency, compute_zero_ratio  # noqa: E402

OUT_DIR = RESULTS_DIR.parent / "figures" / "benchmarks_wrmsse"

COLORS = {
    "Actual": "#1F2937",
    "ES_bu": "#1D4ED8",
    "Croston": "#D97706",
    "Croston_SBA": "#059669",
    "Seasonal_Naive": "#6B7280",
    "LightGBM_Tweedie_20261124": "#CC6677",
}

LEVEL_NAMES_VI = [
    "1. Toàn hệ thống",
    "2. Bang",
    "3. Cửa hàng",
    "4. Ngành hàng",
    "5. Nhóm hàng",
    "6. Bang × ngành",
    "7. Bang × nhóm",
    "8. Cửa hàng × ngành",
    "9. Cửa hàng × nhóm",
    "10. Sản phẩm",
    "11. Bang × sản phẩm",
    "12. Cửa hàng × sản phẩm",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 220,
        }
    )


def save_fig(fig: plt.Figure, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_test_daily_series(
    wide: pd.DataFrame,
    calendar: pd.DataFrame,
    evidence_dir: Path,
    models: list[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Tính tổng lượng bán theo ngày và từng cửa hàng trên kỳ Test d_1914–d_1941."""
    test_days = [f"d_{d}" for d in range(1914, 1942)]
    cal_map = calendar.set_index("d")[["date", "weekday"]]

    actual_daily = wide[test_days].sum(axis=0).to_numpy(dtype=float)
    actual_by_store = wide.groupby("store_id")[test_days].sum().sum(axis=1)

    daily_df = pd.DataFrame(
        {
            "target_day": test_days,
            "date": pd.to_datetime([cal_map.loc[d, "date"] for d in test_days]),
            "weekday": [cal_map.loc[d, "weekday"] for d in test_days],
            "Actual": actual_daily,
        }
    )

    stats: dict[str, dict] = {}
    for model in models:
        m_dir = evidence_dir / model
        fc = pd.read_csv(m_dir / "forecast_test.csv", float_precision="round_trip")
        summary = pd.read_csv(m_dir / "diem_chung.csv").iloc[0]

        pred_daily = fc.groupby("target_day")["y_pred"].sum().reindex(test_days).to_numpy(dtype=float)
        daily_df[model] = pred_daily

        pred_by_store = fc.groupby("store_id")["y_pred"].sum().reindex(actual_by_store.index)
        bias_store = 100.0 * (pred_by_store - actual_by_store) / actual_by_store

        total_actual = float(actual_daily.sum())
        total_pred = float(pred_daily.sum())
        stats[model] = {
            "wrmsse": float(summary["WRMSSE"]),
            "total_actual": total_actual,
            "total_pred": total_pred,
            "bias_pct": 100.0 * (total_pred - total_actual) / total_actual,
            "bias_by_store": bias_store,
        }

    return daily_df, stats


def plot_actual_vs_forecast_test(daily_df: pd.DataFrame, stats: dict[str, dict]) -> Path:
    """Biểu đồ 1: Thực tế và dự báo 28 ngày Test (d_1914–d_1941) của ES_bu và Croston."""
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(daily_df))

    for i, wd in enumerate(daily_df["weekday"]):
        if wd in ("Saturday", "Sunday"):
            ax.axvspan(i - 0.45, i + 0.45, color="#F3F4F6", alpha=0.85, zorder=0)

    ax.plot(
        x,
        daily_df["Actual"],
        label="Thực tế (Actual)",
        color=COLORS["Actual"],
        marker="o",
        markersize=4,
        linewidth=2.2,
        zorder=5,
    )
    ax.plot(
        x,
        daily_df["ES_bu"],
        label="ES_bu (Holt-Winters Bottom-Up)",
        color=COLORS["ES_bu"],
        marker="s",
        markersize=4,
        linewidth=2.0,
        zorder=4,
    )
    ax.plot(
        x,
        daily_df["Croston"],
        label="Croston (1972)",
        color=COLORS["Croston"],
        marker="^",
        markersize=4.5,
        linewidth=2.0,
        zorder=3,
    )
    ax.plot(
        x,
        daily_df["Croston_SBA"],
        label="Croston-SBA (khử chệch)",
        color=COLORS["Croston_SBA"],
        linestyle="--",
        linewidth=1.8,
        zorder=2,
    )

    ticks = x[::3]
    labels = [f"{daily_df.loc[i, 'target_day']}\n({daily_df.loc[i, 'date'].strftime('%d/%m')})" for i in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_ylabel("Tổng lượng bán theo ngày (đơn vị)")
    ax.set_xlabel("Ngày Test M5 (Vùng xám nhạt là Thứ Bảy & Chủ Nhật)")
    ax.set_title(
        "Thực tế và Dự báo 28 ngày Test (d_1914–d_1941): ES_bu vs Croston\n"
        "Tổng lượng bán toàn hệ thống · 10 cửa hàng · 30.490 chuỗi",
        fontweight="bold",
        loc="left",
    )

    info_lines = [
        f"ES_bu:       WRMSSE = {stats['ES_bu']['wrmsse']:.4f} | Sai lệch tổng: {stats['ES_bu']['bias_pct']:+.2f}%",
        f"Croston:     WRMSSE = {stats['Croston']['wrmsse']:.4f} | Sai lệch tổng: {stats['Croston']['bias_pct']:+.2f}%",
        f"Croston_SBA: WRMSSE = {stats['Croston_SBA']['wrmsse']:.4f} | Sai lệch tổng: {stats['Croston_SBA']['bias_pct']:+.2f}%",
    ]
    ax.text(
        0.99,
        0.97,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#D1D5DB", alpha=0.95),
    )

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", ncol=2)
    y_min = min(daily_df[["Actual", "ES_bu", "Croston", "Croston_SBA"]].min())
    y_max = max(daily_df[["Actual", "ES_bu", "Croston", "Croston_SBA"]].max())
    ax.set_ylim(y_min * 0.94, y_max * 1.12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_fig(fig, "01_thuc_te_vs_du_bao_test_28_ngay.png")


def plot_validation_vs_test_wrmsse(wrmsse_summary: pd.DataFrame) -> Path:
    """Biểu đồ 2: So sánh WRMSSE 12 cấp giữa Validation và Test cho các mô hình benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    order = ["ES_bu", "Seasonal_Naive", "Croston_SBA", "Croston"]
    display = ["ES_bu", "Seasonal\nNaive", "Croston\nSBA", "Croston"]

    for ax, split, title in zip(
        axes,
        ["validation", "test"],
        ["Validation (d_1886–d_1913 · origin 1885)", "Test (d_1914–d_1941 · origin 1913)"],
    ):
        sub = wrmsse_summary[wrmsse_summary["split"] == split].set_index("model").loc[order]
        bar_colors = [COLORS[m] for m in order]
        bars = ax.bar(display, sub["wrmsse"], color=bar_colors, width=0.55, edgecolor="#1F2937", linewidth=0.5)
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.annotate(
                f"{row['wrmsse']:.4f}\n(RMSSE TB: {row['rmsse_mean']:.3f})",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(wrmsse_summary["wrmsse"]) * 1.25)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("WRMSSE 12 cấp (thấp hơn là tốt hơn)")
    fig.suptitle(
        "So sánh WRMSSE 12 cấp trên hai kỳ Validation và Test — Mô hình Thống kê",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, "02_wrmsse_validation_vs_test.png")


def plot_12_levels_test(evidence_dir: Path) -> Path:
    """Biểu đồ 3: Điểm WRMSSE trên từng cấp trong 12 cấp M5 (Test d_1914–d_1941)."""
    models = [
        ("ES_bu", "ES_bu (WRMSSE = 0.7707)", "s", -0.18),
        ("Croston", "Croston (WRMSSE = 1.0566)", "^", 0.0),
        ("Croston_SBA", "Croston-SBA (WRMSSE = 1.0918)", "o", 0.18),
    ]
    lgb_path = evidence_dir / "LightGBM_Tweedie_20261124" / "diem_theo_cap.csv"
    if lgb_path.is_file():
        models.insert(0, ("LightGBM_Tweedie_20261124", "LightGBM_Tweedie (0.6235 · tham chiếu)", "D", -0.28))

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    y_base = np.arange(12)

    for model, label, marker, offset in models:
        df_lvl = pd.read_csv(evidence_dir / model / "diem_theo_cap.csv").sort_values("level")
        ax.scatter(
            df_lvl["weighted_RMSSE"],
            y_base + offset,
            label=label,
            color=COLORS[model],
            marker=marker,
            s=65,
            zorder=4,
        )
        for y_val, x_val in zip(y_base + offset, df_lvl["weighted_RMSSE"]):
            ax.hlines(y_val, 0, x_val, color=COLORS[model], alpha=0.22, linewidth=1.2)

    ax.set_yticks(y_base, LEVEL_NAMES_VI)
    ax.invert_yaxis()
    ax.set_xlabel("Điểm cấp có trọng số doanh thu (weighted_RMSSE, chưa chia 12) · thấp hơn tốt hơn")
    ax.set_title(
        "Phân tích sai số qua 12 cấp phân cấp M5 (Test d_1914–d_1941)\n"
        "Croston tăng vọt ở các cấp tổng hợp (1–9) do dự báo phẳng không nắm bắt mùa vụ tuần",
        fontweight="bold",
        loc="left",
        pad=52,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save_fig(fig, "03_diem_12_cap_es_bu_vs_croston_test.png")


def plot_store_bias_test(stats: dict[str, dict]) -> Path:
    """Biểu đồ 4: Sai lệch tổng lượng Test (%) theo từng cửa hàng."""
    stores = stats["ES_bu"]["bias_by_store"].sort_values().index.tolist()
    y = np.arange(len(stores))

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axvline(0, color="#111827", linewidth=1.1, linestyle="-")

    for model, label, marker, dy in [
        ("ES_bu", "ES_bu", "s", -0.15),
        ("Croston", "Croston", "^", 0.0),
        ("Croston_SBA", "Croston-SBA", "o", 0.15),
    ]:
        vals = stats[model]["bias_by_store"].loc[stores].to_numpy()
        ax.scatter(vals, y + dy, label=label, color=COLORS[model], marker=marker, s=70, zorder=4)
        for val, yi in zip(vals, y + dy):
            ax.hlines(yi, 0, val, color=COLORS[model], alpha=0.35, linewidth=1.5)

    ax.set_yticks(y, stores)
    ax.invert_yaxis()
    all_vals = np.concatenate([stats[m]["bias_by_store"].to_numpy() for m in ("ES_bu", "Croston", "Croston_SBA")])
    ax.set_xlim(min(all_vals.min() - 1.0, -12.5), max(all_vals.max() + 1.2, 5.5))
    ax.set_xlabel("Sai lệch tổng lượng Test (%) = 100 × (Dự báo − Thực tế) / Thực tế")
    ax.set_title(
        "Sai lệch tổng lượng Test theo từng cửa hàng (d_1914–d_1941)\n"
        "Mỗi cửa hàng gồm 3.049 chuỗi · thiếu (−), dư (+)",
        fontweight="bold",
        loc="left",
        pad=26,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save_fig(fig, "04_sai_lech_test_theo_cua_hang.png")


def plot_category_and_intermittency(wide: pd.DataFrame, evidence_dir: Path) -> Path:
    """Biểu đồ 5: Phân tích theo Ngành hàng (có trọng số doanh thu) và Mức độ gián đoạn."""
    train_mat = wide[[f"d_{d}" for d in range(1, 1914)]].to_numpy(dtype=float)
    zero_ratios = compute_zero_ratio(train_mat, from_first_sale=True)
    inter_groups = categorize_intermittency(zero_ratios)

    meta = wide[["item_id", "store_id", "cat_id"]].copy()
    meta["intermittency_group"] = inter_groups

    models = ["ES_bu", "Croston", "Croston_SBA"]
    cat_records, inter_records = [], []

    for model in models:
        det = pd.read_csv(evidence_dir / model / "rmsse_theo_cap_san_pham.csv")
        merged = det.merge(meta, on=["item_id", "store_id"], how="left", validate="one_to_one")

        for cat_id, grp in merged.groupby("cat_id"):
            w = grp["revenue_train_28"].to_numpy()
            w_rmsse = float(np.average(grp["RMSSE"], weights=w)) if w.sum() > 0 else float(grp["RMSSE"].mean())
            cat_records.append({"model": model, "cat_id": cat_id, "weighted_rmsse": w_rmsse})

        for ig, grp in merged.groupby("intermittency_group"):
            inter_records.append(
                {
                    "model": model,
                    "intermittency_group": ig,
                    "rmsse_mean": float(grp["RMSSE"].mean()),
                    "bias_pct": 100.0 * (grp["predicted_units"].sum() - grp["actual_units"].sum()) / grp["actual_units"].sum(),
                }
            )

    cat_df = pd.DataFrame(cat_records)
    inter_df = pd.DataFrame(inter_records)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # Left: Category weighted RMSSE
    cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    x_cat = np.arange(len(cats))
    width = 0.25
    for idx, model in enumerate(models):
        vals = cat_df[cat_df["model"] == model].set_index("cat_id").loc[cats, "weighted_rmsse"]
        bars = axes[0].bar(x_cat + (idx - 1) * width, vals, width=width, label=model, color=COLORS[model])
        for bar, v in zip(bars, vals):
            axes[0].annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
    axes[0].set_xticks(x_cat, cats)
    axes[0].set_ylabel("RMSSE có trọng số doanh thu (Cấp 12)")
    axes[0].set_title("Theo Ngành hàng (Test)", fontweight="bold")
    axes[0].set_ylim(0, cat_df["weighted_rmsse"].max() * 1.18)
    axes[0].grid(axis="y", alpha=0.25)

    # Right: Intermittency group unweighted RMSSE
    groups = sorted(inter_df["intermittency_group"].unique())
    short_labels = ["Thường xuyên\n(<30% số 0)", "Trung bình\n(30–50% số 0)", "Gián đoạn\n(50–70% số 0)", "Rất gián đoạn\n(≥70% số 0)"]
    x_int = np.arange(len(groups))
    for idx, model in enumerate(models):
        vals = inter_df[inter_df["model"] == model].set_index("intermittency_group").loc[groups, "rmsse_mean"]
        bars = axes[1].bar(x_int + (idx - 1) * width, vals, width=width, label=model, color=COLORS[model])
        for bar, v in zip(bars, vals):
            axes[1].annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.6)
    axes[1].set_xticks(x_int, short_labels)
    axes[1].set_ylabel("RMSSE trung bình (Cấp 12)")
    axes[1].set_title("Theo Mức độ gián đoạn (Test)", fontweight="bold")
    axes[1].set_ylim(0, inter_df["rmsse_mean"].max() * 1.18)
    axes[1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False)
    fig.suptitle(
        "Hiệu năng ES_bu vs Croston theo Ngành hàng và Mức độ gián đoạn (Test d_1914–d_1941)",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return save_fig(fig, "05_phan_tich_nganh_hang_va_gian_doan.png")


def plot_overall_ranking(final_csv: Path) -> Path:
    """Biểu đồ 6: Xếp hạng tổng thể Test: WRMSSE 12 cấp vs RMSSE trung bình cấp 12."""
    df = pd.read_csv(final_csv).sort_values("wrmsse", ascending=True)
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    bar_colors = [COLORS.get(m, "#4477AA") for m in df["model"]]

    # WRMSSE 12 levels
    bars1 = axes[0].barh(y, df["wrmsse"], color=bar_colors, height=0.55)
    for bar, v in zip(bars1, df["wrmsse"]):
        axes[0].annotate(f"{v:.4f}", (v, bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9.5, fontweight="bold")
    axes[0].set_yticks(y, df["model"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, df["wrmsse"].max() * 1.22)
    axes[0].set_xlabel("WRMSSE 12 cấp (tiêu chí chính · thấp hơn tốt hơn)")
    axes[0].set_title("Xếp hạng theo WRMSSE 12 cấp", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.25)

    # Unweighted RMSSE level 12
    bars2 = axes[1].barh(y, df["rmsse_mean"], color=bar_colors, height=0.55, alpha=0.85)
    for bar, v in zip(bars2, df["rmsse_mean"]):
        axes[1].annotate(f"{v:.4f}", (v, bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9.5)
    axes[1].set_xlim(0, df["rmsse_mean"].max() * 1.22)
    axes[1].set_xlabel("RMSSE trung bình cấp 12 (không trọng số · đối chiếu)")
    axes[1].set_title("Đối chiếu RMSSE trung bình không trọng số", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Đối chiếu Xếp hạng WRMSSE 12 cấp vs RMSSE không trọng số (Test d_1914–d_1941)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_fig(fig, "06_xep_hang_wrmsse_toan_du_an.png")


def write_readme(saved_paths: list[Path]) -> Path:
    content = """# Bộ biểu đồ trực quan WRMSSE cho ES_bu và Croston (M5 Forecasting)

Thư mục này chứa bộ biểu đồ phân tích thực nghiệm cho các mô hình thống kê đối chứng (**ES_bu**, **Croston**, **Croston-SBA**) theo tiêu chuẩn **WRMSSE 12 cấp M5**, được tạo bởi `scripts/benchmarks/plot_wrmsse_benchmarks.py`.

| STT | Tệp biểu đồ | Mục đích & Nội dung chính |
|---:|---|---|
| 1 | `01_thuc_te_vs_du_bao_test_28_ngay.png` | So sánh chuỗi thời gian tổng lượng bán **Thực tế vs Dự báo 28 ngày Test** (`d_1914–d_1941`) của `ES_bu`, `Croston`, `Croston-SBA` kèm WRMSSE và % sai lệch tổng lượng. |
| 2 | `02_wrmsse_validation_vs_test.png` | So sánh **WRMSSE 12 cấp** giữa hai kỳ **Validation** (`d_1886–d_1913`) và **Test** (`d_1914–d_1941`) của 4 mô hình thống kê. |
| 3 | `03_diem_12_cap_es_bu_vs_croston_test.png` | Phân rã điểm **weighted_RMSSE trên đủ 12 cấp phân cấp M5** (kỳ Test), lý giải vì sao `Croston` yếu ở các cấp tổng hợp (1–9) còn `ES_bu` ổn định hơn. |
| 4 | `04_sai_lech_test_theo_cua_hang.png` | Sai lệch tổng lượng Test (%) theo từng cửa hàng trong **10 cửa hàng Walmart** (`CA_1`–`WI_3`). |
| 5 | `05_phan_tich_nganh_hang_va_gian_doan.png` | Phân tích hiệu năng `ES_bu` vs `Croston` / `Croston-SBA` theo **3 ngành hàng** (`FOODS`, `HOBBIES`, `HOUSEHOLD`) và **4 nhóm mức độ bán gián đoạn**. |
| 6 | `06_xep_hang_wrmsse_toan_du_an.png` | Bảng xếp hạng đối chiếu giữa **WRMSSE 12 cấp** (tiêu chí chính) và **RMSSE trung bình cấp 12** (không trọng số) trên toàn dự án. |
"""
    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def main() -> None:
    setup_style()
    local_evidence = REPO_ROOT / ".local_evidence" / "wrmsse_6_models"
    shared_evidence = REPO_ROOT.parents[1] / ".local_evidence" / "wrmsse_6_models"
    evidence_dir = local_evidence if local_evidence.exists() else shared_evidence
    wide = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    wrmsse_summary = pd.read_csv(METRICS_DIR / "all_stores" / "wrmsse_summary.csv")
    final_csv = RESULTS_DIR.parent / "final_comparison.csv"

    daily_df, stats = load_test_daily_series(
        wide, calendar, evidence_dir, ["ES_bu", "Croston", "Croston_SBA"]
    )

    saved = [
        plot_actual_vs_forecast_test(daily_df, stats),
        plot_validation_vs_test_wrmsse(wrmsse_summary),
        plot_12_levels_test(evidence_dir),
        plot_store_bias_test(stats),
        plot_category_and_intermittency(wide, evidence_dir),
        plot_overall_ranking(final_csv),
    ]
    readme_path = write_readme(saved)

    for p in saved:
        print(f"Đã tạo: {p}")
    print(f"Đã tạo: {readme_path}")


if __name__ == "__main__":
    main()
