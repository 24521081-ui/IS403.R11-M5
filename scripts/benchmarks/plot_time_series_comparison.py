"""
Script vẽ biểu đồ chuỗi thời gian so sánh Thực tế vs Dự báo của các mô hình đối chứng:
- ES_bu (Exponential Smoothing Bottom-Up)
- Croston & Croston-SBA (Hiệu chỉnh khử chệch)
- Seasonal Naive (Mốc cơ sở)
- Doanh số thực tế (Actual Sales)
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Thiết lập đường dẫn dự án
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw"
PRED_DIR = REPO_ROOT / "results" / "benchmarks" / "predictions" / "ca_1" / "validation"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình font và style cho biểu đồ học thuật
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 300,
})

def main():
    print("1. Đang nạp dữ liệu lịch calendar...")
    cal_df = pd.read_csv(DATA_DIR / "calendar.csv")
    
    # Mốc ngày: Lịch sử 14 ngày (d_1872 - d_1885) + Dự báo 28 ngày (d_1886 - d_1913)
    hist_days = [f"d_{i}" for i in range(1872, 1886)]
    val_days = [f"d_{i}" for i in range(1886, 1914)]
    all_days = hist_days + val_days

    cal_subset = cal_df[cal_df["d"].isin(all_days)].copy()
    cal_subset["date"] = pd.to_datetime(cal_subset["date"])
    day_to_date = dict(zip(cal_subset["d"], cal_subset["date"]))
    day_to_weekday = dict(zip(cal_subset["d"], cal_subset["weekday"]))
    dates_all = [day_to_date[d] for d in all_days]
    dates_val = [day_to_date[d] for d in val_days]
    origin_date = day_to_date["d_1885"]

    print("2. Đang đọc doanh số thực tế từ sales_train_evaluation.csv...")
    sales_df = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv", nrows=4000)
    ca1_df = sales_df[sales_df["store_id"] == "CA_1"].copy()

    items_info = [
        {
            "item_id": "FOODS_1_018",
            "title": "Trường hợp 1: FOODS_1_018 (Thực phẩm — Bán thường xuyên, Mùa vụ tuần rõ rệt)",
            "subtitle": "ES_bu (xanh dương) bám sát các đỉnh mua sắm cuối tuần; Croston/SBA đi ngang; Seasonal Naïve phóng đại cực đoan.",
        },
        {
            "item_id": "HOBBIES_1_021",
            "title": "Trường hợp 2: HOBBIES_1_021 (Đồ chơi — Nhu cầu gián đoạn, 71.4% số ngày bằng 0)",
            "subtitle": "Croston–SBA (xanh ngọc) duy trì mức kỳ vọng phẳng hợp lý (~0.72) và khử 5% chệch dương; Seasonal Naïve bị bẫy số 0.",
        }
    ]

    print("3. Đang đọc kết quả dự báo từ forecasts.csv...")
    pred_df = pd.read_csv(PRED_DIR / "forecasts.csv")

    # =========================================================================
    # HÌNH 1: SO SÁNH CHUỖI THỜI GIAN CẤP SẢN PHẨM (2 SUBPLOTS)
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.8), sharex=True)
    
    colors = {
        "Actual": "#111827",          # Đen xám đậm
        "ES_bu": "#1D4ED8",           # Xanh dương hoàng gia
        "Croston_SBA": "#059669",     # Xanh lục ngọc
        "Croston": "#D97706",         # Cam đất
        "Seasonal_Naive": "#9CA3AF",  # Xám nét đứt
    }

    for idx, item in enumerate(items_info):
        ax = axes[idx]
        item_id = item["item_id"]

        # Lấy lịch sử thực tế
        item_row = ca1_df[ca1_df["item_id"] == item_id].iloc[0]
        actual_hist = item_row[hist_days].values.astype(float)
        actual_val = item_row[val_days].values.astype(float)
        actual_full = np.concatenate([actual_hist, actual_val])

        # Lấy dự báo 28 ngày của 4 mô hình
        item_preds = pred_df[pred_df["item_id"] == item_id].copy()
        pred_dict = {}
        for m in ["Seasonal_Naive", "Croston", "Croston_SBA", "ES_bu"]:
            sub = item_preds[item_preds["model"] == m].sort_values("horizon")
            pred_dict[m] = sub["y_pred"].values

        # Tô bóng vùng cuối tuần (Thứ 7, CN)
        for d in val_days:
            if day_to_weekday[d] in ["Saturday", "Sunday"]:
                dt = day_to_date[d]
                ax.axvspan(dt - pd.Timedelta(hours=12), dt + pd.Timedelta(hours=12),
                          color="#F3F4F6", alpha=0.8, zorder=0)

        # Tô bóng vùng Lịch sử Train vs Cửa sổ Dự báo
        ax.axvspan(dates_all[0] - pd.Timedelta(hours=12), origin_date + pd.Timedelta(hours=12),
                   color="#F8FAFC", alpha=0.5, zorder=0)

        # Đường phân chia Lịch sử vs Dự báo
        ax.axvline(origin_date + pd.Timedelta(hours=12), color="#EF4444", linestyle="--", linewidth=1.4, alpha=0.9, zorder=2)

        # 1. Vẽ chuỗi Thực tế
        ax.plot(dates_all, actual_full, label="Thực tế (Actual)", color=colors["Actual"], linewidth=1.8, marker="o", markersize=3.5, zorder=5)

        # 2. Vẽ Seasonal Naive
        ax.plot(dates_val, pred_dict["Seasonal_Naive"], label="Seasonal Naïve (Mốc cơ sở)", color=colors["Seasonal_Naive"], linewidth=1.3, linestyle=":", zorder=3)

        # 3. Vẽ Croston
        ax.plot(dates_val, pred_dict["Croston"], label="Croston (1972)", color=colors["Croston"], linewidth=1.5, linestyle="--", alpha=0.85, zorder=4)

        # 4. Vẽ Croston-SBA
        ax.plot(dates_val, pred_dict["Croston_SBA"], label="Croston–SBA (2005, khử chệch)", color=colors["Croston_SBA"], linewidth=1.8, linestyle="-.", zorder=6)

        # 5. Vẽ ES_bu
        ax.plot(dates_val, pred_dict["ES_bu"], label="ES_bu (Mùa vụ tuần Holt-Winters)", color=colors["ES_bu"], linewidth=2.0, zorder=7)

        y_max = max(actual_full.max(), max(p.max() for p in pred_dict.values())) * 1.18
        ax.set_ylim(-0.5, y_max)
        
        # Tiêu đề và ghi chú phụ
        ax.set_title(item["title"], fontweight="bold", pad=4, loc="left", color="#1F2937")
        ax.text(0.01, 0.92, item["subtitle"], transform=ax.transAxes, fontsize=8.5, fontstyle="italic", color="#4B5563")

        ax.set_ylabel("Lượng bán (Đơn vị)")
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        
        # Legend đặt ở góc trên bên phải gọn gàng
        ax.legend(loc="upper right", framealpha=0.95, edgecolor="#D1D5DB", ncol=3, columnspacing=1.0)

    # Đặt nhãn ngày ở trục hoành
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n(%a)"))
    axes[1].set_xlabel("Mốc thời gian (Ngày / Thứ trong tuần) • [Vùng xám nhạt là 2 ngày cuối tuần Thứ Bảy & Chủ Nhật]")

    # Đặt text mốc phân cách ở trục trên
    axes[0].text(origin_date - pd.Timedelta(days=7), axes[0].get_ylim()[1] * 0.84, "Lịch sử Train (14 ngày)",
                 ha="center", fontsize=8.5, fontweight="bold", color="#64748B",
                 bbox=dict(boxstyle="square,pad=0.25", fc="#FFFFFF", ec="#CBD5E1", alpha=0.9))
    axes[0].text(origin_date + pd.Timedelta(days=14), axes[0].get_ylim()[1] * 0.84, "Cửa sổ dự báo Validation (28 ngày)",
                 ha="center", fontsize=8.5, fontweight="bold", color="#1D4ED8",
                 bbox=dict(boxstyle="square,pad=0.25", fc="#FFFFFF", ec="#93C5FD", alpha=0.9))

    plt.suptitle("SO SÁNH CHUỖI THỜI GIAN THỰC TẾ VÀ DỰ BÁO CỦA CÁC MÔ HÌNH ĐỐI CHỨNG (M5 FORECASTING)\nSiêu thị Walmart CA_1 • Cửa sổ đánh giá Validation d_1886 – d_1913",
                 fontsize=12, fontweight="bold", y=0.985, color="#111827")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    sku_fig_path = FIGURES_DIR / "time_series_actual_vs_forecast_sku.png"
    plt.savefig(sku_fig_path, dpi=300)
    plt.close()
    print(f"-> Đã xuất sắc cập nhật: {sku_fig_path}")

    # =========================================================================
    # HÌNH 2: TỔNG HỢP BOTTOM-UP CẤP TOÀN SIÊU THỊ CA_1 (LEVEL 3)
    # =========================================================================
    print("4. Đang xử lý tổng hợp Bottom-Up toàn siêu thị CA_1...")
    agg_df = pd.read_csv(REPO_ROOT / "results" / "benchmarks" / "predictions" / "ca_1" / "validation" / "es_bu_aggregated.csv", low_memory=False)
    store_es_bu = agg_df[agg_df["level"] == 3].sort_values("horizon")["y_pred"].values

    val_cols = [f"d_{i}" for i in range(1886, 1914)]
    hist_14_cols = [f"d_{i}" for i in range(1872, 1886)]
    actual_store_val = ca1_df[val_cols].sum(axis=0).values.astype(float)
    actual_store_hist = ca1_df[hist_14_cols].sum(axis=0).values.astype(float)
    actual_store_full = np.concatenate([actual_store_hist, actual_store_val])

    last_week_store = actual_store_hist[-7:]
    store_snaive = np.tile(last_week_store, 4)

    fig, ax = plt.subplots(figsize=(13.5, 5.8))

    for d in val_days:
        if day_to_weekday[d] in ["Saturday", "Sunday"]:
            dt = day_to_date[d]
            ax.axvspan(dt - pd.Timedelta(hours=12), dt + pd.Timedelta(hours=12),
                      color="#F3F4F6", alpha=0.8, zorder=0)

    ax.axvspan(dates_all[0] - pd.Timedelta(hours=12), origin_date + pd.Timedelta(hours=12),
               color="#F8FAFC", alpha=0.5, zorder=0)

    ax.axvline(origin_date + pd.Timedelta(hours=12), color="#EF4444", linestyle="--", linewidth=1.4, alpha=0.9, zorder=2)

    ax.plot(dates_all, actual_store_full, label="Tổng doanh số thực tế siêu thị CA_1 (Actual)", color="#111827", linewidth=2.0, marker="o", markersize=4, zorder=5)
    ax.plot(dates_val, store_snaive, label="Seasonal Naïve (Lặp lại tuần d_1879–d_1885)", color="#9CA3AF", linewidth=1.5, linestyle=":", zorder=3)
    ax.plot(dates_val, store_es_bu, label="ES_bu Bottom-Up (Cộng dồn từ 3.049 sản phẩm)", color="#1D4ED8", linewidth=2.2, marker="s", markersize=4, zorder=6)

    ax.set_title("HIỆU QUẢ CƠ CHẾ TỔNG HỢP PHÂN CẤP BOTTOM-UP (ES_bu) TRÊN TOÀN SIÊU THỊ CA_1\nTổng hợp nhất quán từ 3.049 chuỗi cơ sở lên Cấp 3 (Store Level) • Khắc phục độ trễ và bắt trọn chu kỳ mua sắm cuối tuần",
                 fontsize=11.5, fontweight="bold", pad=10, loc="left", color="#111827")
    ax.set_ylabel("Tổng sản lượng bán toàn siêu thị (Đơn vị)")
    ax.set_xlabel("Mốc thời gian (Ngày / Thứ trong tuần) • [Vùng xám nhạt là 2 ngày cuối tuần Thứ Bảy & Chủ Nhật]")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n(%a)"))
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#D1D5DB")

    plt.tight_layout()
    store_fig_path = FIGURES_DIR / "time_series_bottom_up_store_ca1.png"
    plt.savefig(store_fig_path, dpi=300)
    plt.close()
    print(f"-> Đã xuất sắc cập nhật: {store_fig_path}")

if __name__ == "__main__":
    main()
