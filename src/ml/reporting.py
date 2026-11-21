"""6. Đánh giá và xuất kết quả: chỉ số, biểu đồ và thông tin thực nghiệm."""

import json
import logging
import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

# Lưu cache trong project; dùng backend xuất ảnh không cần cửa sổ đồ họa.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .forecasting import validate_forecasts

LOGGER = logging.getLogger("m5")
RMSSE_MEAN_COLUMN = "RMSSE_mean_unweighted"


# 6.1. Tính các chỉ số đánh giá trên dự báo đã hoàn thành.
def mae_rmse(actual, predicted):
    """MAE là sai lệch tuyệt đối trung bình; RMSE phạt sai số lớn mạnh hơn."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(
        predicted, dtype=float
    )
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Hai mảng phải cùng kích thước và không rỗng.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Không tính sai số trên NaN hoặc vô hạn.")
    error = actual - predicted
    return float(np.abs(error).mean()), float(np.sqrt(np.square(error).mean()))


def rmsse_scale(train):
    """Mẫu số RMSSE theo M5: sai phân Train từ lần bán dương đầu tiên.

    Chỉ bỏ tiền tố 0 khi tính mẫu số chỉ số; dữ liệu Train vẫn giữ nguyên.
    Chuỗi hằng, toàn 0 hoặc chưa đủ hai điểm có mẫu số không xác định.
    """
    train = np.asarray(train, dtype=float)
    if train.ndim != 1 or not np.isfinite(train).all() or (train < 0).any():
        raise ValueError(
            "Lịch sử tính scale phải là chuỗi một chiều, hữu hạn, không âm."
        )
    positive = np.flatnonzero(train > 0)
    if not len(positive) or len(train) - positive[0] < 2:
        return np.nan
    denominator = np.square(np.diff(train[positive[0] :])).mean()
    return float(denominator) if denominator > 0 else np.nan


def evaluate_forecast(forecast, actual_wide, train_history, items, training_time=0):
    """Nối theo mã và ngày sau dự báo; không dựa vào thứ tự dòng ngẫu nhiên."""
    if forecast.empty or any(
        forecast[key].nunique() != 1
        for key in ["model", "split", "store_id", "origin_day"]
    ):
        raise ValueError("Mỗi lần đánh giá cần đúng một model/split/cửa hàng/origin.")
    if len(set(items)) != len(items) or set(forecast.item_id) != set(items):
        raise ValueError("Danh sách sản phẩm dự báo không khớp tập đánh giá.")
    origin = int(forecast.origin_day.iloc[0][2:])
    horizon = int(forecast.horizon.max())
    validate_forecasts(forecast, horizon)
    train_history = np.asarray(train_history)
    if train_history.shape != (len(items), origin):
        raise ValueError("Lịch sử tính scale phải có đúng số sản phẩm và origin ngày.")
    if (
        "store_id" in actual_wide
        and not actual_wide.store_id.eq(forecast.store_id.iloc[0]).all()
    ):
        raise ValueError("Doanh số thật không thuộc đúng cửa hàng dự báo.")
    if actual_wide.item_id.duplicated().any() or set(actual_wide.item_id) != set(items):
        raise ValueError("Tập doanh số thật có mã trùng, thiếu hoặc thừa.")
    actual = actual_wide.set_index("item_id", verify_integrity=True).loc[
        list(items), [f"d_{d}" for d in range(origin + 1, origin + horizon + 1)]
    ]
    prediction = forecast.pivot(
        index="item_id", columns="horizon", values="y_pred"
    ).loc[list(items), list(range(1, horizon + 1))]
    truth, predicted = actual.to_numpy(dtype=float), prediction.to_numpy(dtype=float)
    mae, rmse = mae_rmse(truth, predicted)
    per_item_mse = np.square(truth - predicted).mean(axis=1)
    scales = np.array([rmsse_scale(row) for row in train_history])
    valid = np.isfinite(scales)
    scores = np.full(len(items), np.nan)
    scores[valid] = np.sqrt(per_item_mse[valid] / scales[valid])
    # Không thay mẫu số 0 bằng epsilon: cách đó tạo chỉ số khó diễn giải.
    # Trung bình chỉ trên chuỗi xác định, đồng thời báo số chuỗi bị loại.
    result = {
        "model": forecast.model.iloc[0],
        "split": forecast.split.iloc[0],
        "store_id": forecast.store_id.iloc[0],
        "n_items": len(items),
        "MAE": mae,
        "RMSE": rmse,
        RMSSE_MEAN_COLUMN: float(scores[valid].mean()) if valid.any() else np.nan,
        "training_time_seconds": training_time,
        "rmsse_valid_items": int(valid.sum()),
        "rmsse_undefined_items": int((~valid).sum()),
    }
    per_item = pd.DataFrame({"item_id": list(items), "RMSSE": scores, "scale": scales})
    return result, per_item, truth, predicted


# 6.2. Vẽ và lưu biểu đồ phục vụ báo cáo.
def save_figures(
    directory,
    actual,
    predicted,
    baseline,
    model,
    features,
    items,
    demo=False,
    metrics=None,
):
    """Vẽ dữ liệu thật của lần chạy; demo luôn có nhãn giả lập trên hình."""
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "DEMO GIẢ LẬP — " if demo else ""
    days = np.arange(1, actual.shape[1] + 1)

    if metrics is not None:
        values = {row["model"]: row["RMSSE_mean_unweighted"] for row in metrics}
        plt.figure(figsize=(7, 4))
        plt.bar(
            ["LightGBM–Tweedie", "Seasonal Naive"],
            [values["lightgbm_tweedie"], values["seasonal_naive"]],
        )
        plt.title(prefix + "Chỉ số chính: RMSSE trung bình không trọng số")
        plt.ylabel("RMSSE_mean_unweighted (thấp hơn tốt hơn)")
        plt.tight_layout()
        plt.savefig(directory / "model_comparison_rmsse.png", dpi=220)
        plt.close()

    def finish(name, title, xlabel, ylabel):
        """Thống nhất độ phân giải và đóng figure để giải phóng bộ nhớ."""
        plt.title(prefix + title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(directory / name, dpi=220)
        plt.close()

    plt.figure(figsize=(10, 4))
    # Chọn sản phẩm đầu theo mã, không chọn theo kết quả đẹp nhất trên holdout.
    plt.plot(days, actual[0], label="Thực tế")
    plt.plot(days, predicted[0], label="LightGBM")
    plt.plot(days, baseline[0], label="Seasonal Naive", alpha=0.65)
    plt.legend()
    finish("actual_vs_prediction.png", str(items[0]), "Ngày dự báo", "Lượng bán")

    plt.figure(figsize=(10, 4))
    plt.plot(days, np.abs(actual - predicted).mean(axis=0), label="LightGBM")
    plt.plot(days, np.abs(actual - baseline).mean(axis=0), label="Seasonal Naive")
    plt.legend()
    finish("error_by_horizon.png", "MAE theo ngày dự báo", "Horizon", "MAE")

    plt.figure(figsize=(9, 5))
    importance = model.feature_importances_
    order = np.argsort(importance)
    plt.barh(np.asarray(features)[order], importance[order])
    finish(
        "feature_importance.png", "Độ quan trọng đặc trưng", "Tổng gain", "Đặc trưng"
    )

    plt.figure(figsize=(7, 4))
    plt.bar(
        ["LightGBM", "Seasonal Naive"],
        [np.abs(actual - predicted).mean(), np.abs(actual - baseline).mean()],
    )
    finish("model_comparison.png", "So sánh MAE", "Mô hình", "MAE")


def save_rmsse_comparisons(by_store, overall, directory, stores, models, labels):
    """Biểu đồ chính từ chỉ số đã tính, không tính lại hoặc thay đổi công thức."""
    metric = "RMSSE_mean_unweighted"
    colors = ["#2563a6", "#e39836"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    indexed = overall.set_index("model").loc[models]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([labels[m] for m in models], indexed[metric], color=colors)
    ax.bar_label(bars, fmt="%.6f", padding=5)
    ax.set_ylim(0, indexed[metric].max() * 1.2)
    ax.set_title(
        "RMSSE trung bình trên toàn bộ 10 cửa hàng\nValidation d_1886–d_1913 • 30.490 chuỗi/mô hình"
    )
    figures = [(fig, ax, "rmsse_mean_overall")]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    positions = np.arange(len(stores))
    for index, model in enumerate(models):
        frame = by_store.loc[by_store.model == model].set_index("store_id").loc[stores]
        bars = ax.bar(
            positions + (index - 0.5) * 0.38,
            frame[metric],
            width=0.38,
            color=colors[index],
            label=labels[model],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(positions, stores)
    ax.set_ylim(0, by_store[metric].max() * 1.2)
    ax.set_title(
        "RMSSE trung bình theo cửa hàng\nValidation d_1886–d_1913 • 3.049 chuỗi/cửa hàng"
    )
    ax.legend(ncol=2)
    figures.append((fig, ax, "rmsse_mean_by_store"))
    for fig, ax, name in figures:
        ax.set_ylabel(metric + " (thấp hơn tốt hơn)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.text(
            0.5,
            0.015,
            "Trung bình RMSSE từng chuỗi hợp lệ; không phải WRMSSE chính thức M5.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        for extension in ["png", "pdf"]:
            fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


# 6.3. Xuất thông tin truy vết và kết quả của lần chạy.
def save_json(data, path):
    """Lưu cấu hình và thông tin lần chạy để có thể đối chiếu về sau."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def evaluate_and_export(
    prediction,
    baseline,
    actual_wide,
    matrix,
    items,
    bundle,
    info,
    locations,
    context,
):
    """Chỉ nhận doanh số thật sau khi cả hai mô hình đã dự báo xong."""
    # 6.1. Tính sai số của hai mô hình trên cùng tập đánh giá.
    results = []

    for forecast, seconds in [
        (prediction, info["training_time_seconds"]),
        (baseline, 0),
    ]:
        result, per_item, truth, predicted = evaluate_forecast(
            forecast, actual_wide, matrix, items, seconds
        )
        results.append(result)

        per_item.to_csv(
            locations["metrics"] / f"{result['model']}_per_item.csv", index=False
        )

        if result["model"] == "lightgbm_tweedie":
            main_prediction = predicted
        else:
            naive_prediction = predicted

    # 6.2. Xuất bảng chỉ số và biểu đồ.
    pd.DataFrame(results).to_csv(locations["metrics"] / "summary.csv", index=False)
    LOGGER.info("Vẽ biểu đồ và lưu mô tả lần chạy")
    save_figures(
        locations["figures"],
        truth,
        main_prediction,
        naive_prediction,
        bundle["model"],
        bundle["features"],
        items,
        metrics=results,
    )

    # 6.3. Lưu cấu hình và thông tin truy vết của lần chạy.
    info = dict(info)
    info.update(
        {
            **context,
            "demo": False,
            "selected_items": items,
            "python_version": platform.python_version(),
            "libraries": {
                name: version(name)
                for name in ["numpy", "pandas", "lightgbm", "scikit-learn"]
            },
            "rmsse_policy": "M5 scale sau lần bán dương đầu tiên; loại chuỗi scale không xác định, báo số lượng",
            "aggregate_rmsse_column": "RMSSE_mean_unweighted",
            "official_wrmsse": False,
        }
    )
    save_json(info, locations["models"] / "metadata.json")
    LOGGER.info("Kết quả:\n%s", pd.DataFrame(results).to_string(index=False))
    LOGGER.info("Đã lưu bảng chỉ số: %s", locations["metrics"] / "summary.csv")

    return pd.DataFrame(results)
