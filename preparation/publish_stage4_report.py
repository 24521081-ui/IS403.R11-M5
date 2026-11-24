"""Verify persisted Stage 4 outputs and publish the Vietnamese report and figures."""

import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ROOT)) / "wrmsse_stage4_mpl"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

from run_stage4 import closed_gates, protected_state
from src.project_paths import local_path
from src.wrmsse import M5_COUNTS
from src.wrmsse_config import CONTRACT_ID, CONFIG
from src.wrmsse_stage2_data import sample_training, sha, write_new
from src.wrmsse_stage4 import (
    BOTTOM, LOCKED, LOGICAL, PARAMS, ROWS, SNAP, SPEC, business, load_inputs, now, read, verify_files,
)
from src.wrmsse_workflows import seasonal_naive

MODELS = ("random_forest", "seasonal_naive")
NAMES = ("Random Forest", "Seasonal Naive")
COLORS = ("#17675e", "#c27428")
WINDOW = "25/04/2016–22/05/2016"
LEVEL_LABELS = [
    "Toàn hệ thống", "Bang", "Cửa hàng", "Ngành hàng", "Bộ phận", "Bang × ngành hàng",
    "Bang × bộ phận", "Cửa hàng × ngành hàng", "Cửa hàng × bộ phận", "Sản phẩm",
    "Sản phẩm × bang", "Sản phẩm × cửa hàng",
]
CHARTS = {
    "test_wrmsse.png": "WRMSSE tổng thể trên Test",
    "test_wrmsse_by_level.png": "WRMSSE theo 12 cấp phân cấp",
    "test_daily_sales.png": "Tổng doanh số theo ngày Test",
    "validation_vs_test_wrmsse.png": "WRMSSE trên Validation và Test",
    "test_business_metrics.png": "MAE, RMSE, WAPE và Forecast Bias",
    "final_feature_importance.png": "Feature importance của mô hình cuối",
}


def close(left, right):
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


def verify_outputs(run):
    pre = read(run / "preflight.json")
    lock = read(run / "forecast_lock_manifest.json")
    access = read(run / "test_access.json")
    started = read(run / "fit_started.json")
    assert not (run / "manifest_stopped.json").exists()
    assert pre["at"] <= started["at"] < lock["at"] < access["at"]
    assert lock["status"] == "locked" and not lock["test_actual_accessed"]
    assert access["forecast_lock_sha256"] == sha(run / "forecast_lock_manifest.json")
    assert lock["preflight_sha256"] == sha(run / "preflight.json")
    assert started["test_gate"] == access["training_gate"] == "closed"
    assert len(list(run.glob("*started.json"))) == 1
    assert set(lock["files"]) == set(LOCKED)
    verify_files(run, lock["files"])
    verify_files(ROOT, pre["input_manifests_sha256"])
    verify_files(ROOT / "data/raw", pre["historical_raw_sha256"])
    evidence = read(local_path(pre["test_evidence"]))
    verify_files(ROOT, evidence["code_sha256"])
    assert protected_state() == read(run / "protected_before.json")
    for key in ("processed", "features"):
        path = local_path(pre[key])
        verify_files(path, read(path / "manifest.json")["files"])
    closed_gates()

    test = read(run / "test_metrics.json")
    comp = pd.read_csv(run / "test_model_comparison.csv").set_index("model")
    levels = pd.read_csv(run / "test_wrmsse_by_level.csv")
    details = pd.read_parquet(run / "test_wrmsse_by_series.parquet")
    actual = pd.read_parquet(run / "test_actual_and_forecast.parquet")
    keys = pd.read_parquet(run / "test_feature_keys.parquet")
    pd.testing.assert_frame_equal(actual[keys.columns], keys)
    assert len(actual) == ROWS and not actual.duplicated(["series_id", "day"]).any()
    np.testing.assert_array_equal(actual.series_id, np.repeat(np.arange(BOTTOM), SPEC.horizon))
    np.testing.assert_array_equal(actual.day, np.tile(np.arange(1914, 1942), BOTTOM))
    assert not actual.isna().any().any()
    assert test["test_actual_sha256"] == pre["expected_test_actual_sha256"]
    assert test["levels"] == 12 and test["hierarchical_series"] == 42840 and test["observations"] == ROWS
    for model, name in zip(MODELS, ("test_predictions.parquet", "seasonal_naive_predictions.parquet")):
        forecast = pd.read_parquet(run / name)
        pd.testing.assert_frame_equal(forecast[keys.columns], keys)
        np.testing.assert_array_equal(forecast.predicted_sales, actual[f"{model}_forecast"])
        assert np.isfinite(forecast.predicted_sales).all()
        for metric, value in business(actual.actual_sales, forecast.predicted_sales).items():
            close(value, test["models"][model][metric])
        for metric, value in test["models"][model].items():
            close(value, comp.at[model, metric])
        per_level = levels.loc[levels.model == model].set_index("level")
        per_series = details.loc[details.model == model]
        assert len(per_series) == 42840
        np.testing.assert_array_equal(per_level.n_series, M5_COUNTS)
        group = per_series.groupby("level", sort=True)
        np.testing.assert_array_equal(group.size(), M5_COUNTS)
        close(per_level.weight, np.full(12, 1 / 12))
        close(group.weight.sum(), per_level.weight)
        close(group.contribution.sum(), per_level.contribution)
        close(per_level.wrmsse_within_level, 12 * per_level.contribution)
        close(per_series.weight.sum(), 1)
        close(per_series.contribution.sum(), test["models"][model]["wrmsse"])
        positive = per_series.weight > 0
        assert (per_series.loc[positive, "scale"] > 0).all()
        close(per_series.loc[positive, "rmsse"],
              np.sqrt(per_series.loc[positive, "mse"] / per_series.loc[positive, "scale"]))
        close(per_series.loc[positive, "contribution"],
              per_series.loc[positive, "weight"] * per_series.loc[positive, "rmsse"])
        assert (per_series.loc[~positive, "contribution"] == 0).all()
    delta = test["models"]["seasonal_naive"]["wrmsse"] - test["models"]["random_forest"]["wrmsse"]
    close(delta, test["wrmsse_reduction"])
    close(abs(delta), test["absolute_wrmsse_difference"])
    close(100 * delta / test["models"]["seasonal_naive"]["wrmsse"], test["improvement_percent"])

    config = read(run / "final_model_config.json")
    feature = read(run / "final_feature_metadata.json")
    model = joblib.load(run / "final_model.joblib")
    assert model.get_params() == config["params"] and not model.warm_start
    assert all(model.get_params()[key] == value for key, value in PARAMS.items())
    assert config["fit_count"] == 1 and len(model.estimators_) == PARAMS["n_estimators"]
    assert feature["logical_features"] == LOGICAL and feature["encoded_count"] == 3082
    assert feature["history_cutoff_formula"] == "min(target_day - 28, d_1913)" and feature["snap_rule"] == SNAP
    assert feature["test_sales_feature_cutoff_days"] == [1886, 1913]
    matrix = sparse.load_npz(run / "test_feature_matrix.npz")
    assert sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32 and matrix.shape == (ROWS, 3082)
    assert np.isfinite(matrix.data).all()
    date = pd.to_datetime(keys.date, format="%Y-%m-%d")
    snap = np.zeros(ROWS, dtype=np.float32)
    for state, active in SNAP.items():
        snap[(keys.state_id == state) & date.dt.day.isin(active)] = 1
    np.testing.assert_array_equal(matrix[:, -1].toarray().ravel(), snap)
    close(model.predict(matrix), actual.random_forest_forecast)
    importance = pd.read_csv(run / "feature_importance.csv").set_index("feature").importance
    assert importance.index.tolist() == LOGICAL
    combined = pd.Series(model.feature_importances_, index=feature["encoded_to_logical"]).groupby(level=0).sum()
    close(importance, combined.reindex(LOGICAL))
    close(importance.sum(), 1)
    del model, forecast

    series, calendar, history = load_inputs(pre)
    for name in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        np.testing.assert_array_equal(keys[name], np.repeat(series[name].to_numpy(), SPEC.horizon))
    dates = calendar.set_index("day").loc[keys.day]
    np.testing.assert_array_equal(keys.date, dates.date)
    tail = matrix[:, -10:-1].toarray()
    np.testing.assert_array_equal(tail[:, :3], dates[["wday", "month", "year"]])
    row_ids = keys.series_id.to_numpy()
    target_days = keys.day.to_numpy()
    for column, lag in enumerate((28, 35, 42, 56), 3):
        np.testing.assert_array_equal(tail[:, column], history[row_ids, target_days - lag - 1])
    for start in range(0, ROWS, 32768):
        end = min(start + 32768, ROWS)
        sales_rows = row_ids[start:end, None]
        cutoff = target_days[start:end, None] - 28
        for column, days in ((7, 7), (8, 28)):
            source = cutoff - np.arange(1, days + 1)
            observed = history[sales_rows, source].mean(axis=1).astype(np.float32)
            np.testing.assert_array_equal(tail[start:end, column], observed)
    del matrix, tail
    np.testing.assert_array_equal(actual.seasonal_naive_forecast, seasonal_naive(history, SPEC.horizon).ravel())
    sample = pd.read_parquet(run / "final_training_sample.parquet")
    sampling = read(run / "final_training_sample_manifest.json")
    verify_files(run, sampling["files"])
    original = pd.read_parquet(local_path(pre["processed"]) / "train_keys.parquet")
    pd.testing.assert_frame_equal(sample.loc[sample.split == "train", original.columns], original)
    extra = sample.loc[sample.split == "validation"]
    assert extra.day.between(1886, 1913).all() and len(extra) == sampling["validation_rows"]
    reproduced, extra_strata = sample_training(
        series, calendar, n=sampling["validation_rows"], seed=sampling["seed"], start=1886, end=1913,
    )
    pd.testing.assert_frame_equal(extra[original.columns].reset_index(drop=True), reproduced)
    strata = pd.read_parquet(run / "final_training_strata.parquet")
    pd.testing.assert_frame_equal(strata.loc[strata.split == "validation", extra_strata.columns].reset_index(drop=True), extra_strata)
    assert len(sample) == config["train_rows"] and not sample.duplicated(["series_id", "day"]).any()
    np.testing.assert_array_equal(sample.actual_sales, history[sample.series_id, sample.day - 1])
    close(strata.sample_rows.sum(), len(sample))
    assert sampling["reproduced_exactly"]
    return test, levels, actual, config, importance, sampling


def finalize_resources(run):
    resource = read(run / "resource_summary.json")
    assert resource["returncode"] == 0 and resource["stop_reason"] is None
    log = [json.loads(line) for line in (run / "resources.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == resource["sample_count"]
    close(max(row["rss_tree_mib"] for row in log), resource["peak_tree_rss_mib"])
    identity = run / "worker_process_identity.jsonl"
    if identity.is_file() and "peak_venv_launcher_rss_mib" not in resource:
        # On Windows a venv redirector owns the monitored PID. The real Python worker is its only child.
        process = [json.loads(line) for line in identity.read_text(encoding="utf-8").splitlines()]
        assert process and all(not row["children"] for row in process)
        assert len({(row["pid"], row["launcher_pid"], row["exe"]) for row in process}) == 1
        assert resource["peak_worker_rss_mib"] < 20
        resource["peak_venv_launcher_rss_mib"] = resource["peak_worker_rss_mib"]
        resource["peak_worker_rss_mib"] = max(row["rss_tree_mib"] - row["rss_worker_mib"] for row in log)
        resource["worker_rss_basis"] = "Process tree RSS minus Windows venv launcher RSS. Child PID independently observed."
        resource["worker_process_confirmed_samples"] = len(process)
        resource["independently_sampled_worker_peak_mib"] = max(row["worker_rss_mib"] for row in process)
        tmp = run / "resource_summary.finished.json"
        write_new(tmp, resource)
        tmp.replace(run / "resource_summary.json")
    return resource


def bar_metrics(ax, vals, label, digits=3):
    bars = ax.bar(NAMES, vals, color=COLORS, width=0.57)
    ax.bar_label(bars, fmt=f"%.{digits}f", padding=3)
    ax.axhline(0, color="#606b74", lw=0.7)
    ax.margins(y=0.23)
    ax.set_ylabel(label)


def figures(run, test, levels, actual, importance, selected):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.labelcolor": "#253445", "axes.titleweight": "bold"})

    def save(fig, name):
        fig.savefig(run / name, dpi=230, bbox_inches="tight", pad_inches=.18)
        plt.close(fig)

    metrics = test["models"]
    fig, ax = plt.subplots(figsize=(7.3, 4.6), layout="constrained")
    bar_metrics(ax, [metrics[model]["wrmsse"] for model in MODELS], "WRMSSE")
    ax.set_title("WRMSSE Test  ·  " + WINDOW)
    save(fig, "test_wrmsse.png")

    fig, ax = plt.subplots(figsize=(9, 7.2), layout="constrained")
    y = np.arange(12)
    for model, label, color, offset in zip(MODELS, NAMES, COLORS, (-.18, .18)):
        vals = levels.loc[levels.model == model].sort_values("level").wrmsse_within_level
        ax.barh(y + offset, vals, .33, label=label, color=color)
    ax.set(yticks=y, yticklabels=[f"{n}. {label}" for n, label in enumerate(LEVEL_LABELS, 1)],
           xlabel="WRMSSE chuẩn hóa trong cấp", title="WRMSSE theo cấp  ·  " + WINDOW)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    save(fig, "test_wrmsse_by_level.png")

    daily = actual.groupby("date", sort=True)[["actual_sales", "random_forest_forecast", "seasonal_naive_forecast"]].sum()
    daily.index = pd.to_datetime(daily.index)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
    for series, label, color in zip(daily, ("Thực tế", *NAMES), ("#334155", *COLORS)):
        ax.plot(daily.index, daily[series], label=label, color=color, lw=2, marker="o", ms=3)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.set(title="Tổng lượng bán mỗi ngày  ·  " + WINDOW, xlabel="Ngày trong cửa sổ Test",
           ylabel="Số đơn vị bán")
    ax.legend(ncols=3)
    save(fig, "test_daily_sales.png")

    fig, ax = plt.subplots(figsize=(9.6, 5.5), layout="constrained")
    vals = [selected["wrmsse"], selected["seasonal_naive_wrmsse"]]
    for model, label, color, value, offset in zip(MODELS, NAMES, COLORS, vals, (-.18, .18)):
        bars = ax.bar(np.arange(2) + offset, [value, metrics[model]["wrmsse"]], .33, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(xticks=[0, 1], xticklabels=[
        "Validation 28/03–24/04/2016\nScale d_1–d_1885 · trọng số d_1858–d_1885",
        "Test " + WINDOW + "\nScale d_1–d_1913 · trọng số d_1886–d_1913",
    ], ylabel="WRMSSE", title="WRMSSE theo cửa sổ đánh giá và context", ylim=(0, 1.18 * max(vals)))
    ax.legend()
    save(fig, "validation_vs_test_wrmsse.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), layout="constrained")
    for ax, (key, title, ylabel) in zip(axes.flat, [
        ("mae", "MAE", "Số đơn vị bán"), ("rmse", "RMSE", "Số đơn vị bán"),
        ("wape_percent", "WAPE", "%"), ("forecast_bias_percent", "Forecast Bias", "%"),
    ]):
        bar_metrics(ax, [metrics[model][key] for model in MODELS], ylabel, digits=2)
        ax.set_title(title)
    fig.suptitle("Sai số và sai lệch trên Test  ·  " + WINDOW, fontweight="bold")
    save(fig, "test_business_metrics.png")

    fig, ax = plt.subplots(figsize=(9, 7.2), layout="constrained")
    imp = importance.sort_values()
    bars = ax.barh(imp.index, 100 * imp, color=["#17675e" if name == "snap_state" else "#65809a" for name in imp.index])
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.margins(x=.17)
    ax.set(title="Feature importance  ·  Random Forest cuối cùng", xlabel="Tỷ trọng giảm impurity (%)")
    fig.text(.99, .01, "Gộp cột one-hot theo đặc trưng logic  ·  Test " + WINDOW, ha="right")
    save(fig, "final_feature_importance.png")


def fmt(value, places=4):
    return f"{value:,.{places}f}".translate(str.maketrans(",.", ".,"))


def report(run, test, levels, config, importance, sampling, resources, selected):
    path = "../" + run.relative_to(ROOT).as_posix()
    metric_rows = []
    for key, label, places in (
        ("wrmsse", "WRMSSE", 10), ("mae", "MAE (đơn vị)", 4), ("rmse", "RMSE (đơn vị)", 4),
        ("wape_percent", "WAPE (%)", 4), ("forecast_bias_percent", "Forecast Bias (%)", 4),
        ("actual_sales", "Tổng lượng bán thực tế", 0), ("forecast_sales", "Tổng lượng bán dự báo", 2),
        ("sales_difference", "Dự báo trừ thực tế (đơn vị)", 2),
        ("overforecast_percent", "Tỷ lệ dự báo thừa (%)", 4),
        ("underforecast_percent", "Tỷ lệ dự báo thiếu (%)", 4),
    ):
        rf, sn = (test["models"][m][key] for m in MODELS)
        metric_rows.append(f"| {label} | {fmt(rf, places)} | {fmt(sn, places)} | {fmt(rf - sn, places)} |")
    scores = levels.pivot(index="level", columns="model", values="wrmsse_within_level")
    reduction_by_level = scores.seasonal_naive - scores.random_forest
    hierarchy_rows = []
    for level, (label, count) in enumerate(zip(LEVEL_LABELS, M5_COUNTS), 1):
        rf, sn = (scores.at[level, model] for model in MODELS)
        hierarchy_rows.append(f"| {level} | {label} | {fmt(count, 0)} | {fmt(rf, 6)} | {fmt(sn, 6)} | {fmt(rf - sn, 6)} |")
    importance_rows = [f"| {name} | {fmt(100 * weight, 4)} |" for name, weight in importance.sort_values(ascending=False).items()]
    params = "\n".join(f"| {key} | {config['params'][key]} |" for key in PARAMS)
    names = ", ".join(LOGICAL)
    releases = ", ".join(f"{name} {version}" for name, version in resources["versions"].items())
    chart_links = "\n".join(f"- [{label}]({path}/{name})" for name, label in CHARTS.items())
    rf_bias = test["models"]["random_forest"]["forecast_bias_percent"]
    sn_bias = test["models"]["seasonal_naive"]["forecast_bias_percent"]
    direction = lambda bias: "thiếu" if bias < 0 else "thừa" if bias > 0 else "đúng tổng"
    best = int(reduction_by_level.idxmax())
    worst = int(reduction_by_level.idxmin())
    win = int((reduction_by_level > 0).sum())
    loss = int((reduction_by_level < 0).sum())
    ntests = read(run / "preflight.json")["unit_tests_passed"]
    outcome = "giảm" if test["wrmsse_reduction"] >= 0 else "tăng"
    return f"""# Giai đoạn 4 · Huấn luyện cuối và đánh giá Test

RF-15 + FE-04 đạt **WRMSSE Test {fmt(test['models']['random_forest']['wrmsse'], 10)}**. Seasonal Naive đạt **{fmt(test['models']['seasonal_naive']['wrmsse'], 10)}**. Random Forest {outcome} WRMSSE **{fmt(abs(test['improvement_percent']), 2)}%** trên cửa sổ **{WINDOW}**.

## 1. Cấu hình cuối cùng

RF-15 + FE-04 được chọn theo WRMSSE Validation thấp nhất trong các cấu hình đã đánh giá. FE-04 đạt {fmt(selected['wrmsse'], 10)}, FE-09 đạt 0,6321382731 và FE-10 đạt 0,6360743504. Seasonal Naive Validation đạt {fmt(selected['seasonal_naive_wrmsse'], 10)}. Quá trình tối ưu kết thúc với 36 lượt gồm 32 fit và 4 reuse. [Báo cáo Giai đoạn 3](STAGE3_REPORT.md) · [Cấu hình selection v2]({path}/final_model_config.json).

| Siêu tham số | Giá trị |
|---|---:|
{params}

FE-04 gồm 15 đặc trưng logic theo thứ tự {names}. One-hot cho năm trường ID dùng vocabulary đã khóa, xác minh đầy đủ trên mẫu huấn luyện, cho 3.082 cột CSR float32.

## 2. Quy trình huấn luyện lại

Random Forest được khởi tạo mới và fit một lần trên **mẫu {fmt(sampling['rows'], 0)} quan sát lấy từ Train + Validation** trong d_1–d_1913. Giai đoạn này lấy mẫu từ lịch sử gồm hàng triệu quan sát. 600.000 khóa Train gốc được giữ đúng thứ tự và tái tạo chính xác. Mẫu thêm {fmt(sampling['validation_rows'], 0)} khóa từ d_1886–d_1913 theo mật độ 600.000 × 28 / 1.829 và làm tròn đến số nguyên gần nhất. Cả hai mẫu phân tầng theo cửa hàng × bộ phận × quý, lấy không hoàn lại với seed 42. Mỗi tầng nhận số lượng tối thiểu một dòng, phân bổ phần còn lại theo sức chứa và dùng phần dư lớn nhất để khớp tổng mẫu. Đã kiểm tra khóa trùng, phạm vi d và nhãn.

Nguồn doanh số Test feature tối đa là t−28, trong khoảng cutoff d_1886–d_1913. SNAP được dựng từ ngày trong tháng và bang với quy tắc kiểm chứng từ lịch đến d_1885. Lưới 30.490 chuỗi × 28 ngày và hai bộ dự báo được khóa SHA-256 trước khi mở actual Test. Seasonal Naive lặp d_1907–d_1913 cho cả cửa sổ.

## 3. Chi phí tính toán

| Chỉ tiêu | Giá trị |
|---|---:|
| Dòng huấn luyện | {fmt(sampling['rows'], 0)} |
| Đặc trưng logic / cột mã hóa | {config['logical_features']} / {fmt(config['encoded_features'], 0)} |
| Thời gian fit (giây) | {fmt(resources['fit_seconds'], 2)} |
| Thời gian predict Random Forest (giây) | {fmt(resources['predict_seconds'], 2)} |
| Thời gian predict Seasonal Naive (giây) | {fmt(resources['seasonal_naive_predict_seconds'], 4)} |
| RAM đỉnh worker (MiB) | {fmt(resources['peak_worker_rss_mib'], 2)} |
| RAM đỉnh cây tiến trình worker (MiB) | {fmt(resources['peak_tree_rss_mib'], 2)} |
| RAM khả dụng thấp nhất (MiB) | {fmt(resources['minimum_available_mib'], 2)} |
| CPU vật lý / logic / n_jobs | {resources['physical_cpu']} / {resources['logical_cpu']} / {resources['n_jobs']} |
| CPU đỉnh cây worker (% của một lõi) | {fmt(resources['peak_worker_tree_cpu_percent_one_core'], 1)} |

RSS được lấy mẫu mỗi giây. Phiên bản môi trường gồm {releases}.

## 4. Kết quả đánh giá Test

Test gồm 853.720 bản ghi item–store–day từ d_1914 đến d_1941. WRMSSE M5 là thước đo chính, với scale từ d_1–d_1913 và trọng số doanh thu lịch sử units × sell_price từ d_1886–d_1913. Cột chênh lệch tính Random Forest trừ Seasonal Naive.

| Chỉ số | Random Forest | Seasonal Naive | Chênh lệch |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

Chênh lệch WRMSSE tuyệt đối là {fmt(test['absolute_wrmsse_difference'], 10)}. Mức cải thiện tương đối tính bằng (WRMSSE Seasonal Naive − WRMSSE Random Forest) / WRMSSE Seasonal Naive × 100%, bằng {fmt(test['improvement_percent'], 4)}%. Bias âm biểu thị dự báo thiếu. Bias dương biểu thị dự báo thừa. Tỷ lệ thừa và thiếu lần lượt lấy phần dương và phần âm theo độ lớn của Bias. Khi tổng actual bằng 0, WAPE, Bias và tỷ lệ thừa hoặc thiếu được ghi null.

## 5. Đánh giá theo hệ thống phân cấp

| Cấp | Tên cấp | Số chuỗi | Random Forest | Seasonal Naive | RF trừ SN |
|---:|---|---:|---:|---:|---:|
{chr(10).join(hierarchy_rows)}

Random Forest có WRMSSE thấp hơn tại {win} cấp, cao hơn tại {loss} cấp và bằng nhau tại {12 - win - loss} cấp. Chênh lệch theo hướng có lợi cho Random Forest lớn nhất ở cấp {best} ({LEVEL_LABELS[best-1]}), với Seasonal Naive trừ Random Forest bằng {fmt(reduction_by_level.at[best], 6)}. Cấp {worst} ({LEVEL_LABELS[worst-1]}) có mức chênh lệch theo hướng đó thấp nhất, bằng {fmt(reduction_by_level.at[worst], 6)}. Mỗi cấp có trọng số 1/12. Tổng đóng góp của 42.840 chuỗi khớp WRMSSE toàn hệ thống ở cả hai mô hình.

## 6. Feature importance

| Đặc trưng logic | Tỷ trọng (%) |
|---|---:|
{chr(10).join(importance_rows)}

Importance trích từ mức giảm impurity trong model đã fit và cộng các cột one-hot theo đặc trưng logic gốc. Tổng sau gộp bằng 100% trước khi làm tròn. SNAP chiếm {fmt(100 * importance['snap_state'], 2)}%, các lag chiếm {fmt(100 * importance[importance.index.str.startswith('lag_')].sum(), 2)}% và hai rolling mean chiếm {fmt(100 * importance[importance.index.str.startswith('rolling_')].sum(), 2)}%. Các tỷ trọng mô tả cách rừng dùng đặc trưng để chia nút. Đặc trưng tương quan có thể chia sẻ importance, và các tỷ trọng này không xác lập quan hệ nhân quả.

## 7. Nhận xét kết quả

Trên tổng lượng bán Test, Random Forest dự báo {direction(rf_bias)} với Bias {fmt(rf_bias, 2)}%, còn Seasonal Naive dự báo {direction(sn_bias)} với Bias {fmt(sn_bias, 2)}%. Bảng 12 cấp cho thấy mức chênh lệch thay đổi theo độ chi tiết của chuỗi. Biểu đồ doanh số theo ngày giúp đối chiếu độ lệch tổng lượng với các ngày trong cả cửa sổ.

So sánh Validation và Test dùng hai context scale và trọng số khác nhau. Điểm Validation dùng mô hình chọn đã fit trên mẫu Train, còn điểm Test dùng mô hình cuối đã fit trên mẫu Train + Validation. MAE, RMSE, WAPE và Bias bổ sung góc nhìn sai số ở cấp item–store–day. Quyết định mô hình được khóa theo WRMSSE Validation.

## 8. Giới hạn nghiên cứu

- Mô hình dùng mẫu Train + Validation từ lịch sử, gồm {fmt(sampling['rows'], 0)} dòng.
- Tối ưu dựa trên một cửa sổ Validation 28 ngày.
- Test d_1914–d_1941 từng được dùng làm Validation trong nghiên cứu cũ. Actual Test chỉ được mở sau khi khóa mô hình và cả hai bộ dự báo trong pipeline WRMSSE mới. Cửa sổ này đã từng được quan sát trong lịch sử dự án.
- Kết quả phản ánh phạm vi dữ liệu, cách lấy mẫu và điều kiện thực nghiệm hiện tại.

## 9. Artifact

- [Mô hình]({path}/final_model.joblib) · [cấu hình]({path}/final_model_config.json) · [schema mã hóa]({path}/final_feature_metadata.json).
- [Mẫu huấn luyện]({path}/final_training_sample.parquet) · [tầng lấy mẫu]({path}/final_training_strata.parquet) · [manifest mẫu]({path}/final_training_sample_manifest.json).
- [Ma trận Test]({path}/test_feature_matrix.npz) · [khóa ma trận]({path}/test_feature_keys.parquet) · [Random Forest]({path}/test_predictions.parquet) · [Seasonal Naive]({path}/seasonal_naive_predictions.parquet).
- [Actual và hai dự báo cho Giai đoạn 5]({path}/test_actual_and_forecast.parquet) · [metrics JSON]({path}/test_metrics.json) · [so sánh CSV]({path}/test_model_comparison.csv).
- [WRMSSE theo cấp]({path}/test_wrmsse_by_level.csv) · [RMSSE, scale, trọng số và đóng góp từng chuỗi]({path}/test_wrmsse_by_series.parquet) · [importance CSV]({path}/feature_importance.csv).
- [Kiểm tra trước fit]({path}/preflight.json) · [giám sát tài nguyên]({path}/resource_summary.json) · [khóa dự báo]({path}/forecast_lock_manifest.json) · [biên bản mở actual]({path}/test_access.json).
- [Manifest hoàn thành]({path}/manifest_completed.json) · [hậu kiểm]({path}/postflight_verification.json).

{chart_links}

## 10. Xác nhận hoàn thành

Giai đoạn 4 hoàn thành với {ntests} kiểm thử đạt. Mô hình RF-15 + FE-04 và hai bộ dự báo được khóa trước khi đánh giá toàn bộ Test bằng WRMSSE. Training và Test đã đóng. Không có tuning hoặc refit sau Test.
"""


def main(run):
    test, levels, actual, config, importance, sampling = verify_outputs(run)
    resources = finalize_resources(run)
    selected = read(local_path(config["selected_model"]))
    figures(run, test, levels, actual, importance, selected)
    for name in CHARTS:
        with Image.open(run / name) as image:
            assert image.width > 1200 and image.height > 600
            image.verify()
    document = report(run, test, levels, config, importance, sampling, resources, selected)
    report_path = local_path("docs/STAGE4_REPORT.md")
    future = {run / name for name in ("manifest_completed.json", "postflight_verification.json")}
    for link in re.findall(r"\]\(([^)]+)\)", document):
        destination = (report_path.parent / link).resolve()
        assert destination.is_file() or destination in future, link
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(document)
    assert report_path.read_text(encoding="utf-8") == document
    write_new(run / "postflight_verification.json", {
        "status": "passed", "at": now(), "contract_id": CONTRACT_ID,
        "test_actual_opened_after_forecast_lock": True, "forecast_lock_verified": True,
        "sales_feature_history_end": 1913, "lag_and_rolling_match_origin_history_for_all_test_rows": True,
        "snap_rule_matches_all_test_rows": True,
        "training_sample_reproduced": True, "observations": ROWS, "hierarchical_series": 42840,
        "levels": 12, "weights_and_contributions_verified": True, "keys_and_predictions_verified": True,
        "business_metrics_reproduced_from_export": True, "feature_importance_verified": True,
        "plots_verified": list(CHARTS), "resource_summary_verified": True, "fit_count": 1,
        "metrics_match_csv_parquet_json_report_plots": True, "historical_outputs_and_sources_unchanged": True,
        "closed_gates": ["training", "tuning", "test_evaluation"], "post_test_tuning": False,
    })
    files = {path.relative_to(run).as_posix(): sha(path) for path in run.rglob("*") if path.is_file()}
    manifest = {
        "status": "completed", "stage": 4, "at": now(), "contract_id": CONTRACT_ID,
        "run": run.name, "candidate": "RF-15 + FE-04", "fit_count": 1, "files": files,
        "linked_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in
                          (report_path, Path(__file__).resolve())},
    }
    write_new(run / "manifest_completed.json", manifest)
    verify_files(run, manifest["files"])
    verify_files(ROOT, manifest["linked_sha256"])
    print(json.dumps({"run": run.name, "status": "completed", "models": test["models"],
                      "improvement_percent": test["improvement_percent"]}))


if __name__ == "__main__":
    main(local_path(sys.argv[1]))
