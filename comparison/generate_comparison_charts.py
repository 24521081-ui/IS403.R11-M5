"""Generate multi-criteria comparison charts across all M5 forecasting methods."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "comparison" / "charts"
SUMMARY_CSV = ROOT / "comparison" / "summary_all_models.csv"
DAILY_CSV = ROOT / "comparison" / "daily_all_models.csv"
STORE_CSV = ROOT / "comparison" / "store_bias_all_models.csv"
CATEGORY_CSV = ROOT / "comparison" / "category_all_models.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

MODEL_ORDER = [
    "Seasonal Naive",
    "ES_bu",
    "Croston-SBA",
    "Random Forest",
    "LightGBM Two-Part",
    "LightGBM Tweedie",
]

MODEL_COLORS = {
    "Seasonal Naive": "#64748B",
    "ES_bu": "#7C3AED",
    "Croston-SBA": "#DC2626",
    "Random Forest": "#D97706",
    "LightGBM Two-Part": "#1D4ED8",
    "LightGBM Tweedie": "#0F766E",
}


def compute_tweedie_validation_wrmsse() -> float:
    """Compute exact 12-level WRMSSE on d_1886-d_1913 from the 10 local Tweedie validation CSVs."""
    val_dirs = sorted((ROOT / "pipelines/lightgbm_tweedie/results/validation").glob("validation_full_*_20261124T*"))
    by_store = {}
    for d in val_dirs:
        f = d / "lightgbm_tweedie.csv"
        if f.is_file():
            # e.g. validation_full_CA_1_20261124T...
            parts = d.name.split("_")
            store = f"{parts[2]}_{parts[3]}"
            by_store[store] = f
    if len(by_store) != 10:
        return float("nan")
    sys.path.insert(0, str(ROOT / "pipelines/lightgbm_twopart/scripts"))
    from score_c7_wrmsse import score

    data = ROOT / "data/raw"
    wide = pd.read_csv(data / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data / "calendar.csv")
    weeks = set(calendar.loc[calendar.d.isin([f"d_{d}" for d in range(1858, 1886)]), "wm_yr_wk"])
    prices = pd.concat(
        [p.loc[p.wm_yr_wk.isin(weeks)] for p in pd.read_csv(data / "sell_prices.csv", chunksize=250_000)],
        ignore_index=True,
    )
    forecasts = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in by_store.values()], ignore_index=True)
    rows = score(1885, wide, calendar, prices, forecasts)
    return float(sum(r["contribution_WRMSSE"] for r in rows))


def load_all_metrics():
    bm = pd.read_csv(ROOT / "pipelines/benchmarks/results/benchmarks/metrics/all_stores/wrmsse_summary.csv")
    bm_levels = pd.read_csv(ROOT / "pipelines/benchmarks/results/benchmarks/metrics/all_stores/wrmsse_by_level.csv")

    tp_scores = pd.read_csv(ROOT / "pipelines/lightgbm_twopart/reports/c7_events/scores.csv")
    tp_levels = pd.read_csv(ROOT / "pipelines/lightgbm_twopart/reports/c7_events/test_levels.csv")
    tp_groups = pd.read_csv(ROOT / "pipelines/lightgbm_twopart/reports/c7_events/test_group_scores.csv")
    tp_daily = pd.read_csv(ROOT / "pipelines/lightgbm_twopart/reports/c7_events/test_daily.csv")
    tp_store = pd.read_csv(ROOT / "pipelines/lightgbm_twopart/reports/sai_lech_test_C7_EVENTS_theo_cua_hang.csv")

    tw_overall = pd.read_csv(ROOT / ".local_evidence/wrmsse_6_models/LightGBM_Tweedie_20261124/diem_chung.csv")
    tw_levels = pd.read_csv(ROOT / ".local_evidence/wrmsse_6_models/LightGBM_Tweedie_20261124/diem_theo_cap.csv")

    rf_comp = pd.read_csv(ROOT / "pipelines/random_forest/results/wrmsse_v1/stage4_20261124T104618287922Z/test_model_comparison.csv")
    rf_levels = pd.read_csv(ROOT / "pipelines/random_forest/results/wrmsse_v1/stage4_20261124T104618287922Z/test_wrmsse_by_level.csv")
    rf_series = pd.read_parquet(ROOT / "pipelines/random_forest/results/wrmsse_v1/stage4_20261124T104618287922Z/test_wrmsse_by_series.parquet")
    rf_val_test = pd.read_csv(ROOT / "pipelines/random_forest/reports/tables/validation_test.csv")
    rf_groups = pd.read_csv(ROOT / "pipelines/random_forest/reports/tables/demand_group_metrics.csv")
    rf_daily = pd.read_csv(ROOT / "pipelines/random_forest/reports/tables/daily_sales.csv")
    rf_store = pd.read_csv(ROOT / "pipelines/random_forest/reports/tables/store_analysis.csv")
    rf_cat = pd.read_csv(ROOT / "pipelines/random_forest/reports/tables/category_analysis.csv")

    dg_6 = pd.read_csv(ROOT / "pipelines/random_forest/reports/demand_groups/results_by_demand_group_test.csv")

    tw_val_wrmsse = compute_tweedie_validation_wrmsse()

    def get_bm(m, s):
        row = bm[(bm.model == m) & (bm.split == s)]
        return float(row.wrmsse.iloc[0]) if len(row) else np.nan

    def get_bm_total_bias(m):
        sub = dg_6[dg_6.model == m]
        act = sub.actual_total.sum()
        pred = sub.forecast_total.sum()
        return act, pred, 100.0 * (pred / act - 1.0)

    tw_act, tw_pred, tw_bias = get_bm_total_bias("LightGBM_Tweedie_20261124")
    es_act, es_pred, es_bias = get_bm_total_bias("ES_bu")
    cs_act, cs_pred, cs_bias = get_bm_total_bias("Croston_SBA")

    rf_row = rf_comp[rf_comp.model == "random_forest"].iloc[0]
    sn_row = rf_comp[rf_comp.model == "seasonal_naive"].iloc[0]
    rf_val = float(rf_val_test.loc[rf_val_test.window.str.lower() == "validation", "random_forest"].iloc[0])

    c7e_val = tp_scores[(tp_scores.configuration == "C7_EVENTS") & (tp_scores.split == "validation")].iloc[0]
    c7e_tst = tp_scores[(tp_scores.configuration == "C7_EVENTS") & (tp_scores.split == "test")].iloc[0]

    def bm_lvl(m, lvl):
        return float(bm_levels[(bm_levels.model == m) & (bm_levels.split == "test") & (bm_levels.level == lvl)].weighted_RMSSE.iloc[0])

    def rf_lvl(m, lvl):
        return float(rf_levels[(rf_levels.model == m) & (rf_levels.level == lvl)].wrmsse_within_level.iloc[0])

    def dg_val(m, grp, col):
        return float(dg_6[(dg_6.model == m) & (dg_6.demand_group == grp)][col].iloc[0])

    rf_dg_test = rf_groups[rf_groups.split == "Test"]

    rows = [
        {
            "model": "LightGBM Tweedie",
            "group": "ML",
            "wrmsse_val": tw_val_wrmsse,
            "val_note": "local_10_stores_20261124",
            "wrmsse_test": float(tw_overall.WRMSSE.iloc[0]),
            "l1_test": float(tw_levels[tw_levels.level == 1].weighted_RMSSE.iloc[0]),
            "l12_test": float(tw_levels[tw_levels.level == 12].weighted_RMSSE.iloc[0]),
            "frequent_l12": dg_val("LightGBM_Tweedie_20261124", "frequent", "weighted_RMSSE"),
            "intermittent_l12": dg_val("LightGBM_Tweedie_20261124", "intermittent", "weighted_RMSSE"),
            "frequent_bias_pct": dg_val("LightGBM_Tweedie_20261124", "frequent", "bias_pct"),
            "intermittent_bias_pct": dg_val("LightGBM_Tweedie_20261124", "intermittent", "bias_pct"),
            "demand_group_origin": 1913,
            "actual_sales": tw_act,
            "forecast_sales": tw_pred,
            "bias_pct": tw_bias,
        },
        {
            "model": "LightGBM Two-Part",
            "group": "ML",
            "wrmsse_val": float(c7e_val.WRMSSE),
            "val_note": "published",
            "wrmsse_test": float(c7e_tst.WRMSSE),
            "l1_test": float(tp_levels[tp_levels.level == 1].weighted_RMSSE.iloc[0]),
            "l12_test": float(tp_levels[tp_levels.level == 12].weighted_RMSSE.iloc[0]),
            "frequent_l12": float(tp_groups[(tp_groups.dimension == "demand_group") & (tp_groups.group == "frequent")].revenue_weighted_RMSSE.iloc[0]),
            "intermittent_l12": float(tp_groups[(tp_groups.dimension == "demand_group") & (tp_groups.group == "intermittent")].revenue_weighted_RMSSE.iloc[0]),
            "frequent_bias_pct": float(tp_groups[(tp_groups.dimension == "demand_group") & (tp_groups.group == "frequent")].bias_pct.iloc[0]),
            "intermittent_bias_pct": float(tp_groups[(tp_groups.dimension == "demand_group") & (tp_groups.group == "intermittent")].bias_pct.iloc[0]),
            "demand_group_origin": 1885,
            "actual_sales": float(c7e_tst.actual),
            "forecast_sales": float(c7e_tst.predicted),
            "bias_pct": float(c7e_tst.bias_pct),
        },
        {
            "model": "Random Forest",
            "group": "ML",
            "wrmsse_val": rf_val,
            "val_note": "published",
            "wrmsse_test": float(rf_row.wrmsse),
            "l1_test": rf_lvl("random_forest", 1),
            "l12_test": rf_lvl("random_forest", 12),
            "frequent_l12": float(rf_dg_test[rf_dg_test.demand_group == "frequent"].rf_conditional_weighted_rmsse.iloc[0]),
            "intermittent_l12": float(rf_dg_test[rf_dg_test.demand_group == "intermittent"].rf_conditional_weighted_rmsse.iloc[0]),
            "frequent_bias_pct": float(rf_dg_test[rf_dg_test.demand_group == "frequent"].rf_bias_pct.iloc[0]),
            "intermittent_bias_pct": float(rf_dg_test[rf_dg_test.demand_group == "intermittent"].rf_bias_pct.iloc[0]),
            "demand_group_origin": 1913,
            "actual_sales": float(rf_row.actual_sales),
            "forecast_sales": float(rf_row.forecast_sales),
            "bias_pct": float(rf_row.forecast_bias_percent),
        },
        {
            "model": "ES_bu",
            "group": "Benchmark",
            "wrmsse_val": get_bm("ES_bu", "validation"),
            "val_note": "published",
            "wrmsse_test": get_bm("ES_bu", "test"),
            "l1_test": bm_lvl("ES_bu", 1),
            "l12_test": bm_lvl("ES_bu", 12),
            "frequent_l12": dg_val("ES_bu", "frequent", "weighted_RMSSE"),
            "intermittent_l12": dg_val("ES_bu", "intermittent", "weighted_RMSSE"),
            "frequent_bias_pct": dg_val("ES_bu", "frequent", "bias_pct"),
            "intermittent_bias_pct": dg_val("ES_bu", "intermittent", "bias_pct"),
            "demand_group_origin": 1913,
            "actual_sales": es_act,
            "forecast_sales": es_pred,
            "bias_pct": es_bias,
        },
        {
            "model": "Seasonal Naive",
            "group": "Benchmark",
            "wrmsse_val": get_bm("Seasonal_Naive", "validation"),
            "val_note": "published",
            "wrmsse_test": float(sn_row.wrmsse),
            "l1_test": rf_lvl("seasonal_naive", 1),
            "l12_test": rf_lvl("seasonal_naive", 12),
            "frequent_l12": float(rf_dg_test[rf_dg_test.demand_group == "frequent"].sn_conditional_weighted_rmsse.iloc[0]),
            "intermittent_l12": float(rf_dg_test[rf_dg_test.demand_group == "intermittent"].sn_conditional_weighted_rmsse.iloc[0]),
            "frequent_bias_pct": float(rf_dg_test[rf_dg_test.demand_group == "frequent"].sn_bias_pct.iloc[0]),
            "intermittent_bias_pct": float(rf_dg_test[rf_dg_test.demand_group == "intermittent"].sn_bias_pct.iloc[0]),
            "demand_group_origin": 1913,
            "actual_sales": float(sn_row.actual_sales),
            "forecast_sales": float(sn_row.forecast_sales),
            "bias_pct": float(sn_row.forecast_bias_percent),
        },
        {
            "model": "Croston-SBA",
            "group": "Benchmark",
            "wrmsse_val": get_bm("Croston_SBA", "validation"),
            "val_note": "published",
            "wrmsse_test": get_bm("Croston_SBA", "test"),
            "l1_test": bm_lvl("Croston_SBA", 1),
            "l12_test": bm_lvl("Croston_SBA", 12),
            "frequent_l12": dg_val("Croston_SBA", "frequent", "weighted_RMSSE"),
            "intermittent_l12": dg_val("Croston_SBA", "intermittent", "weighted_RMSSE"),
            "frequent_bias_pct": dg_val("Croston_SBA", "frequent", "bias_pct"),
            "intermittent_bias_pct": dg_val("Croston_SBA", "intermittent", "bias_pct"),
            "demand_group_origin": 1913,
            "actual_sales": cs_act,
            "forecast_sales": cs_pred,
            "bias_pct": cs_bias,
        },
    ]
    summary = pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER].reset_index()

    # 12-level table
    lvl_series = {
        "Seasonal Naive": rf_levels[rf_levels.model == "seasonal_naive"].set_index("level")["wrmsse_within_level"],
        "ES_bu": bm_levels[(bm_levels.model == "ES_bu") & (bm_levels.split == "test")].set_index("level")["weighted_RMSSE"],
        "Croston-SBA": bm_levels[(bm_levels.model == "Croston_SBA") & (bm_levels.split == "test")].set_index("level")["weighted_RMSSE"],
        "Random Forest": rf_levels[rf_levels.model == "random_forest"].set_index("level")["wrmsse_within_level"],
        "LightGBM Two-Part": tp_levels.set_index("level")["weighted_RMSSE"],
        "LightGBM Tweedie": tw_levels.set_index("level")["weighted_RMSSE"],
    }
    levels_df = pd.DataFrame(lvl_series)

    # Daily, Store, Category tables across 6 representative models
    local_map = {
        "LightGBM Tweedie": "LightGBM_Tweedie_20261124",
        "ES_bu": "ES_bu",
        "Croston-SBA": "Croston_SBA",
    }

    # Daily
    daily_df = pd.DataFrame({
        "target_day": [f"d_{d}" for d in rf_daily["day"]],
        "date": rf_daily["date"],
        "weekday": rf_daily["weekday"],
        "actual": rf_daily["actual_sales"],
        "Random Forest": rf_daily["random_forest_forecast"],
        "Seasonal Naive": rf_daily["seasonal_naive_forecast"],
    })
    daily_df = daily_df.merge(tp_daily[["target_day", "predicted"]].rename(columns={"predicted": "LightGBM Two-Part"}), on="target_day")
    for label, folder in local_map.items():
        ft = pd.read_csv(ROOT / ".local_evidence/wrmsse_6_models" / folder / "forecast_test.csv", usecols=["target_day", "y_pred"])
        d_sum = ft.groupby("target_day", as_index=False)["y_pred"].sum().rename(columns={"y_pred": label})
        daily_df = daily_df.merge(d_sum, on="target_day")

    # Store bias (%)
    store_rows = []
    for _, r in rf_store.iterrows():
        store_rows.append({"store_id": r["store_id"], "model": "Random Forest", "actual": r["actual_sales"], "forecast": r["random_forest_forecast"], "bias_pct": r["rf_bias_pct"]})
        store_rows.append({"store_id": r["store_id"], "model": "Seasonal Naive", "actual": r["actual_sales"], "forecast": r["seasonal_naive_forecast"], "bias_pct": r["sn_bias_pct"]})
    for _, r in tp_store.iterrows():
        store_rows.append({"store_id": r["store_id"], "model": "LightGBM Two-Part", "actual": r["actual"], "forecast": r["predicted"], "bias_pct": r["bias_pct"]})
    for label, folder in local_map.items():
        rms = pd.read_csv(ROOT / ".local_evidence/wrmsse_6_models" / folder / "rmsse_theo_cap_san_pham.csv")
        st_agg = rms.groupby("store_id", as_index=False)[["actual_units", "predicted_units"]].sum()
        for _, r in st_agg.iterrows():
            store_rows.append({
                "store_id": r["store_id"],
                "model": label,
                "actual": r["actual_units"],
                "forecast": r["predicted_units"],
                "bias_pct": 100.0 * (r["predicted_units"] / r["actual_units"] - 1.0),
            })
    store_df = pd.DataFrame(store_rows)

    # Category (FOODS, HOBBIES, HOUSEHOLD) L12 weighted_RMSSE & bias_pct
    cat_rows = []
    for _, r in tp_groups[tp_groups.dimension == "cat_id"].iterrows():
        cat_rows.append({
            "cat_id": r["group"],
            "model": "LightGBM Two-Part",
            "weighted_RMSSE": r["revenue_weighted_RMSSE"],
            "bias_pct": r["bias_pct"],
        })
    rf_l12 = rf_series[rf_series.level == 12].copy()
    rf_l12["cat_id"] = rf_l12["key"].apply(
        lambda k: str(k[0]).split("_")[0] if isinstance(k, (list, tuple, np.ndarray)) else str(k).strip("[]'\" ").split("_")[0]
    )
    rf_bias_cat = rf_cat.set_index("cat_id")
    for m_key, m_label, bias_col in [
        ("random_forest", "Random Forest", "rf_bias_pct"),
        ("seasonal_naive", "Seasonal Naive", "sn_bias_pct"),
    ]:
        sub = rf_l12[rf_l12.model == m_key]
        for cat, g in sub.groupby("cat_id"):
            w = g["weight"].values
            r_vals = g["rmsse"].values
            cat_rows.append({
                "cat_id": cat,
                "model": m_label,
                "weighted_RMSSE": float(np.sum(w * r_vals) / np.sum(w)),
                "bias_pct": float(rf_bias_cat.loc[cat, bias_col]),
            })
    for label, folder in local_map.items():
        rms = pd.read_csv(ROOT / ".local_evidence/wrmsse_6_models" / folder / "rmsse_theo_cap_san_pham.csv")
        rms["cat_id"] = rms["item_id"].str.split("_").str[0]
        for cat, g in rms.groupby("cat_id"):
            w = g["revenue_weight"].values
            r_vals = g["RMSSE"].values
            act = g["actual_units"].sum()
            pred = g["predicted_units"].sum()
            cat_rows.append({
                "cat_id": cat,
                "model": label,
                "weighted_RMSSE": float(np.sum(w * r_vals) / np.sum(w)),
                "bias_pct": float(100.0 * (pred / act - 1.0)),
            })
    cat_df = pd.DataFrame(cat_rows)

    return summary, levels_df, daily_df, store_df, cat_df


def plot_chart_1(summary: pd.DataFrame):
    df = summary.set_index("model").loc[MODEL_ORDER].reset_index()
    y = np.arange(len(df))
    h = 0.36

    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    ax.barh(y - h / 2, df["wrmsse_test"], height=h,
            color=[MODEL_COLORS[m] for m in df["model"]],
            label="Test (d_1914–d_1941)", edgecolor="white", linewidth=0.8)
    ax.barh(y + h / 2, df["wrmsse_val"].fillna(0), height=h,
            color=[MODEL_COLORS[m] for m in df["model"]], alpha=0.38,
            hatch="///", edgecolor="#475569", linewidth=0.6,
            label="Validation (d_1886–d_1913)")

    sn_test = float(df.loc[df.model == "Seasonal Naive", "wrmsse_test"].iloc[0])
    ax.axvline(sn_test, color="#475569", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(2.5, color="#94A3B8", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(sn_test + 0.015, 3.8, f"Mốc Seasonal Naive\n({sn_test:.4f})",
            color="#334155", fontsize=9, fontweight="bold", va="center", zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))

    label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6)
    for i, row in df.iterrows():
        diff_pct = 100.0 * (row["wrmsse_test"] / sn_test - 1.0)
        suffix = f" ({diff_pct:+.1f}%)" if row["model"] != "Seasonal Naive" else " (baseline)"
        ax.text(row["wrmsse_test"] + 0.012, i - h / 2, f"{row['wrmsse_test']:.4f}{suffix}",
                va="center", fontsize=9, fontweight="bold", color="#0F172A", zorder=4, bbox=label_bbox)
        if pd.notna(row["wrmsse_val"]):
            val_lbl = f"{row['wrmsse_val']:.4f}"
            if row["val_note"] != "published":
                val_lbl += " *"
            ax.text(row["wrmsse_val"] + 0.012, i + h / 2, val_lbl,
                    va="center", fontsize=8.5, color="#475569", zorder=4, bbox=label_bbox)

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.38)
    ax.set_xlabel("WRMSSE (càng thấp càng tốt)", fontsize=10.5, fontweight="bold")
    ax.set_title("So sánh WRMSSE tổng thể giữa các phương pháp (Validation vs Test — 12 cấp M5)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    fig.text(0.01, 0.01, "* LightGBM Tweedie Validation (0.7946) được chấm bổ sung từ 10 file validation cục bộ ngày 24/11 (bias +9.96%).",
             fontsize=8.2, color="#475569", style="italic")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT_DIR / "01_tong_quan_wrmsse_validation_test.png", dpi=300)
    plt.close(fig)


def plot_chart_2(levels_df: pd.DataFrame):
    level_labels = [
        "L1\nTotal", "L2\nState", "L3\nStore", "L4\nCat", "L5\nDept",
        "L6\nSt+Cat", "L7\nSt+Dept", "L8\nStr+Cat", "L9\nStr+Dept",
        "L10\nItem", "L11\nSt+Item", "L12\nStr+Item"
    ]
    x = np.arange(1, 13)
    markers = {
        "LightGBM Tweedie": "o",
        "LightGBM Two-Part": "s",
        "Random Forest": "^",
        "ES_bu": "D",
        "Seasonal Naive": "X",
        "Croston-SBA": "v",
    }

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for col in levels_df.columns:
        lw = 2.6 if col != "Seasonal Naive" else 2.0
        ls = "-" if col not in ("Seasonal Naive", "Croston-SBA") else "--"
        ax.plot(x, levels_df[col].values, marker=markers[col], markersize=6.5,
                linewidth=lw, linestyle=ls, color=MODEL_COLORS[col], label=col)

    for col, dy1, dy12 in [
        ("LightGBM Tweedie", -0.015, 0.006),
        ("LightGBM Two-Part", 0.012, -0.018),
        ("Random Forest", 0.0, -0.003),
        ("ES_bu", 0.0, 0.032),
        ("Seasonal Naive", 0.0, 0.0),
        ("Croston-SBA", 0.0, 0.020),
    ]:
        v1 = levels_df[col].iloc[0]
        v12 = levels_df[col].iloc[-1]
        ax.text(1 - 0.12, v1 + dy1, f"{v1:.3f}", ha="right", va="center",
                fontsize=8, fontweight="bold", color=MODEL_COLORS[col])
        lbl12 = f"{v12:.3f}" if col not in ("Random Forest", "Croston-SBA") else (
            f"RF {v12:.3f}" if col == "Random Forest" else f"SBA {v12:.3f}"
        )
        ax.text(12 + 0.12, v12 + dy12, lbl12, ha="left", va="center",
                fontsize=8.2, fontweight="bold", color=MODEL_COLORS[col])

    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, fontsize=9)
    ax.set_xlim(0.25, 12.95)
    ax.set_ylim(0.35, 1.38)
    ax.set_ylabel("Weighted RMSSE trong cấp (càng thấp càng tốt)", fontsize=10.5, fontweight="bold")
    ax.set_title("Hiệu năng theo 12 cấp phân cấp M5 trên tập Test (d_1914–d_1941)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=3, frameon=True, facecolor="white", framealpha=0.95, fontsize=9.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_wrmsse_theo_12_cap_m5.png", dpi=300)
    plt.close(fig)


def plot_chart_3(summary: pd.DataFrame):
    df = summary.set_index("model").loc[MODEL_ORDER].reset_index()
    x = np.arange(len(df))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.5), gridspec_kw={"width_ratios": [1.15, 1.0]})

    b1 = ax1.bar(x - w / 2, df["frequent_l12"], width=w, color="#2563EB", label="Frequent (~9,0% số chuỗi | ~36,3% sản lượng)")
    b2 = ax1.bar(x + w / 2, df["intermittent_l12"], width=w, color="#F59E0B", label="Intermittent (~91,0% số chuỗi | ~63,7% sản lượng)")
    ax1.axvline(2.5, color="#94A3B8", linestyle=":", linewidth=1.2)
    for bar in list(b1) + list(b2):
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0F172A")

    labels = [m + (" *" if "Two-Part" in m else "") for m in df["model"]]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8, fontweight="bold", rotation=18, ha="right")
    ax1.set_ylim(0, 1.28)
    ax1.set_ylabel("Weighted RMSSE cấp 12 (càng thấp càng tốt)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Sai số theo tần suất bán (Frequent vs Intermittent)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.grid(axis="y", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

    b3 = ax2.bar(x - w / 2, df["frequent_bias_pct"], width=w, color="#2563EB", alpha=0.85, label="Frequent Bias %")
    b4 = ax2.bar(x + w / 2, df["intermittent_bias_pct"], width=w, color="#F59E0B", alpha=0.85, label="Intermittent Bias %")
    ax2.axhline(0, color="#0F172A", linewidth=1.0)
    ax2.axvline(2.5, color="#94A3B8", linestyle=":", linewidth=1.2)
    for bar in list(b3) + list(b4):
        val = bar.get_height()
        va = "bottom" if val >= 0 else "top"
        dy = 0.4 if val >= 0 else -0.45
        ax2.text(bar.get_x() + bar.get_width() / 2, val + dy, f"{val:+.1f}%",
                 ha="center", va=va, fontsize=8, fontweight="bold", color="#0F172A")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5, fontweight="bold", rotation=18, ha="right")
    ax2.set_ylim(-13, 9)
    ax2.set_ylabel("Forecast Bias % (dương = thừa, âm = thiếu)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Độ lệch dự báo theo nhóm tần suất bán", fontsize=11.5, fontweight="bold", pad=10)
    ax2.grid(axis="y", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True)

    fig.suptitle("Phân tích hiệu năng theo nhóm nhu cầu trên tập Test (d_1914–d_1941)",
                 fontsize=13.5, fontweight="bold", y=0.98)
    fig.text(0.01, 0.01, "* Ghi chú đối chiếu: 6 mô hình dùng phân nhóm origin 1913 (2.754 Frequent / 27.736 Intermittent); LightGBM Two-Part dùng bảng công bố trên phân nhóm origin 1885 (2.749 / 27.741).",
             fontsize=8.2, color="#475569", style="italic")
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(OUT_DIR / "03_hieu_qua_theo_nhom_nhu_cau.png", dpi=300)
    plt.close(fig)


def plot_chart_4(summary: pd.DataFrame):
    df = summary.set_index("model").loc[MODEL_ORDER].reset_index()
    y = np.arange(len(df))
    actual = float(df["actual_sales"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax1.barh(y, df["forecast_sales"] / 1e6, height=0.56,
             color=[MODEL_COLORS[m] for m in df["model"]], alpha=0.9)
    ax1.axvline(actual / 1e6, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Thực tế: {actual:,.0f}".replace(",", "."))
    ax1.axhline(2.5, color="#94A3B8", linestyle=":", linewidth=1.2)
    for i, row in df.iterrows():
        val = row["forecast_sales"]
        ax1.text(val / 1e6 + 0.012, i, f"{val:,.0f}".replace(",", "."),
                 va="center", fontsize=8.8, fontweight="bold", color="#0F172A")

    ax1.set_yticks(y)
    ax1.set_yticklabels(df["model"], fontsize=9.5, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0.95, 1.42)
    ax1.set_xlabel("Tổng lượng bán dự báo 28 ngày Test (triệu đơn vị)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Tổng lượng bán dự báo so với thực tế", fontsize=11.5, fontweight="bold", pad=10)
    ax1.grid(axis="x", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", frameon=True)

    colors_bias = ["#0F766E" if b >= 0 else "#2563EB" if abs(b) < 5 else "#D97706" for b in df["bias_pct"]]
    ax2.barh(y, df["bias_pct"], height=0.56, color=colors_bias, alpha=0.9)
    ax2.axvline(0, color="#0F172A", linewidth=1.1)
    ax2.axhline(2.5, color="#94A3B8", linestyle=":", linewidth=1.2)
    for i, row in df.iterrows():
        b = row["bias_pct"]
        ha = "left" if b >= 0 else "right"
        dx = 0.22 if b >= 0 else -0.22
        ax2.text(b + dx, i, f"{b:+.2f}%", va="center", ha=ha, fontsize=9, fontweight="bold", color="#0F172A")

    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(-9.5, 5.5)
    ax2.set_xlabel("Forecast Bias % (dương = dự báo thừa, âm = dự báo thiếu)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Độ lệch tổng lượng bán (Bias %)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.grid(axis="x", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Đối chiếu tổng lượng bán dự báo & mức độ thiên lệch trên 28 ngày Test (d_1914–d_1941)",
                 fontsize=13.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "04_do_lech_tong_luong_du_bao_test.png", dpi=300)
    plt.close(fig)


def plot_chart_5(summary: pd.DataFrame):
    df = summary.set_index("model").loc[MODEL_ORDER].reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.0, 1.35]})

    label_xy = {
        "LightGBM Tweedie": (0.41, 0.875),
        "LightGBM Two-Part": (0.44, 0.785),
        "Random Forest": (0.56, 0.905),
        "ES_bu": (0.72, 0.925),
        "Seasonal Naive": (0.78, 1.105),
        "Croston-SBA": (1.06, 0.825),
    }
    for _, row in df.iterrows():
        m = row["model"]
        ax1.scatter(row["l1_test"], row["l12_test"], s=130, color=MODEL_COLORS[m],
                    edgecolors="white", linewidth=1.5, zorder=4)
        tx, ty = label_xy[m]
        ax1.annotate(m, xy=(row["l1_test"], row["l12_test"]), xytext=(tx, ty),
                     fontsize=8.5, fontweight="bold", color=MODEL_COLORS[m],
                     arrowprops=dict(arrowstyle="-", color=MODEL_COLORS[m], lw=0.8, alpha=0.7))

    ax1.set_xlabel("Level 1 (Total) Weighted RMSSE", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Level 12 (Item × Store) Weighted RMSSE", fontsize=10, fontweight="bold")
    ax1.set_title("A. Đánh đổi cấp Tổng hợp (L1) vs Chi tiết (L12)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xlim(0.37, 1.30)
    ax1.set_ylim(0.76, 1.15)
    ax1.grid(True, alpha=0.55)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(0.86, 0.785, "★ Góc dưới-trái = tối ưu nhất", fontsize=8.5, fontweight="bold", color="#0F766E")

    metrics_cols = ["wrmsse_test", "l1_test", "l12_test", "frequent_l12", "intermittent_l12"]
    col_labels = ["WRMSSE\nTest", "Cấp 1\n(Total)", "Cấp 12\n(SKU×Store)", "Frequent\n(L12)", "Intermittent\n(L12)", "|Bias %|"]
    raw_mat = df[metrics_cols].copy()
    raw_mat["abs_bias"] = df["bias_pct"].abs()
    raw_vals = raw_mat.values
    norm_vals = (raw_vals - raw_vals.min(axis=0)) / (raw_vals.max(axis=0) - raw_vals.min(axis=0) + 1e-12)

    im = ax2.imshow(norm_vals, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(col_labels)))
    ax2.set_xticklabels(col_labels, fontsize=9, fontweight="bold")
    ax2.set_yticks(np.arange(len(df)))
    ax2.set_yticklabels(df["model"], fontsize=9.5, fontweight="bold")

    for i in range(raw_vals.shape[0]):
        for j in range(raw_vals.shape[1]):
            val = raw_vals[i, j]
            txt = f"{val:.2f}%" if j == 5 else f"{val:.4f}"
            txt_color = "white" if norm_vals[i, j] > 0.78 else "#0F172A"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=8.5, fontweight="bold", color=txt_color)

    ax2.set_title("B. Bảng điểm đa tiêu chí (Xanh = tốt nhất, Đỏ = kém nhất)", fontsize=11.5, fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.03)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Tốt nhất", "Trung bình", "Kém nhất"])

    fig.suptitle("Tổng hợp so sánh đa tiêu chí các phương pháp dự báo M5 (tập Test)",
                 fontsize=13.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "05_bang_tong_hop_da_tieu_chi.png", dpi=300)
    plt.close(fig)


def plot_chart_6(daily_df: pd.DataFrame):
    x = np.arange(len(daily_df))
    short_dates = [d[5:] for d in daily_df["date"]]  # MM-DD
    wd_map = {
        "Thứ Hai": "T2", "Thứ Ba": "T3", "Thứ Tư": "T4",
        "Thứ Năm": "T5", "Thứ Sáu": "T6", "Thứ Bảy": "T7", "Chủ nhật": "CN", "Chủ Nhật": "CN",
    }

    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    for i, wd in enumerate(daily_df["weekday"]):
        if wd in ("Thứ Bảy", "Chủ nhật", "Chủ Nhật"):
            ax.axvspan(i - 0.45, i + 0.45, color="#F1F5F9", zorder=0)

    ax.plot(x, daily_df["actual"] / 1000.0, color="#0F172A", linewidth=2.8, marker="o",
            markersize=5, label="Thực tế (Actual)", zorder=5)

    for col, ls in [
        ("LightGBM Tweedie", "-"),
        ("LightGBM Two-Part", "-"),
        ("Random Forest", "-"),
        ("ES_bu", "--"),
        ("Seasonal Naive", ":"),
        ("Croston-SBA", "-."),
    ]:
        ax.plot(x, daily_df[col] / 1000.0, color=MODEL_COLORS[col], linewidth=1.9,
                linestyle=ls, alpha=0.9, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n({wd_map.get(w, w)})" for d, w in zip(short_dates, daily_df["weekday"])],
                       fontsize=8.2)
    ax.set_ylim(33, 66.5)
    ax.set_ylabel("Tổng lượng bán toàn hệ thống (nghìn đơn vị/ngày)", fontsize=10.5, fontweight="bold")
    ax.set_title("Diễn biến dự báo theo ngày trên 28 ngày Test (25/04–22/05/2016) — Bắt sóng cuối tuần",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=4, frameon=True, facecolor="white", framealpha=0.95, fontsize=8.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_dien_bien_du_bao_28_ngay_test.png", dpi=300)
    plt.close(fig)


def plot_chart_7(store_df: pd.DataFrame):
    stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    models = MODEL_ORDER
    piv = store_df.pivot(index="store_id", columns="model", values="bias_pct").loc[stores, models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A: Grouped lines/markers across 10 stores
    x = np.arange(len(stores))
    markers = ["X", "D", "v", "^", "s", "o"]
    ax1.axhline(0, color="#0F172A", linewidth=1.1)
    for m, mk in zip(models, markers):
        ax1.plot(x, piv[m].values, marker=mk, markersize=6.5, linewidth=2.0,
                 color=MODEL_COLORS[m], label=m)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stores, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Forecast Bias % (dương = thừa, âm = thiếu)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Biến động Forecast Bias % qua 10 cửa hàng", fontsize=11.5, fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.55)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower left", ncol=2, fontsize=8.2, frameon=True)

    # Panel B: Heatmap store x model Bias %
    vals = piv.values
    im = ax2.imshow(vals, cmap="RdBu", aspect="auto", vmin=-14, vmax=14)
    ax2.set_xticks(np.arange(len(models)))
    ax2.set_xticklabels(["S.Naive", "ES_bu", "Croston-SBA", "RF", "Two-Part", "Tweedie"],
                        fontsize=8.5, fontweight="bold", rotation=20, ha="right")
    ax2.set_yticks(np.arange(len(stores)))
    ax2.set_yticklabels(stores, fontsize=9.5, fontweight="bold")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            c = "white" if abs(v) > 8.5 else "#0F172A"
            ax2.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=8, fontweight="bold", color=c)
    ax2.set_title("B. Bản đồ nhiệt Bias % (Đỏ = thiếu, Xanh = thừa)", fontsize=11.5, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.03, label="Bias %")

    fig.suptitle("So sánh độ lệch dự báo theo 10 cửa hàng trên tập Test (d_1914–d_1941)",
                 fontsize=13.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "07_do_lech_theo_10_cua_hang.png", dpi=300)
    plt.close(fig)


def plot_chart_8(cat_df: pd.DataFrame):
    cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    models = MODEL_ORDER
    piv_r = cat_df.pivot(index="cat_id", columns="model", values="weighted_RMSSE").loc[cats, models]
    piv_b = cat_df.pivot(index="cat_id", columns="model", values="bias_pct").loc[cats, models]

    x = np.arange(len(cats))
    w = 0.13
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0))

    for idx, m in enumerate(models):
        offset = (idx - (len(models) - 1) / 2.0) * w
        bars = ax1.bar(x + offset, piv_r[m].values, width=w, color=MODEL_COLORS[m], label=m)
        for bar in bars:
            val = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0F172A", rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels(["FOODS\n(67,6% sản lượng)", "HOBBIES\n(9,4% sản lượng)", "HOUSEHOLD\n(23,0% sản lượng)"],
                        fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 1.28)
    ax1.set_ylabel("Weighted RMSSE cấp 12 (càng thấp càng tốt)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Weighted RMSSE cấp 12 theo ngành hàng", fontsize=11.5, fontweight="bold", pad=10)
    ax1.grid(axis="y", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.axhline(0, color="#0F172A", linewidth=1.0)
    for idx, m in enumerate(models):
        offset = (idx - (len(models) - 1) / 2.0) * w
        bars = ax2.bar(x + offset, piv_b[m].values, width=w, color=MODEL_COLORS[m], label=m)
        for bar in bars:
            val = bar.get_height()
            va = "bottom" if val >= 0 else "top"
            dy = 0.3 if val >= 0 else -0.35
            ax2.text(bar.get_x() + bar.get_width() / 2, val + dy, f"{val:+.1f}%",
                     ha="center", va=va, fontsize=7.2, fontweight="bold", color="#0F172A", rotation=90)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["FOODS", "HOBBIES", "HOUSEHOLD"], fontsize=9.5, fontweight="bold")
    ax2.set_ylim(-12.5, 9.0)
    ax2.set_ylabel("Forecast Bias % (dương = thừa, âm = thiếu)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Forecast Bias % theo ngành hàng", fontsize=11.5, fontweight="bold", pad=10)
    ax2.grid(axis="y", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=6,
               frameon=True, facecolor="white", fontsize=8.4)
    fig.suptitle("So sánh hiệu năng theo 3 ngành hàng trên tập Test (d_1914–d_1941)",
                 fontsize=13.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_DIR / "08_hieu_qua_theo_3_nganh_hang.png", dpi=300)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, levels_df, daily_df, store_df, cat_df = load_all_metrics()
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    daily_df.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")
    store_df.to_csv(STORE_CSV, index=False, encoding="utf-8-sig")
    cat_df.to_csv(CATEGORY_CSV, index=False, encoding="utf-8-sig")
    plot_chart_1(summary)
    plot_chart_2(levels_df)
    plot_chart_3(summary)
    plot_chart_4(summary)
    plot_chart_5(summary)
    plot_chart_6(daily_df)
    plot_chart_7(store_df)
    plot_chart_8(cat_df)
    print("Generated 8 charts in:", OUT_DIR)
    print(summary[["model", "wrmsse_val", "wrmsse_test", "l1_test", "l12_test", "bias_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
