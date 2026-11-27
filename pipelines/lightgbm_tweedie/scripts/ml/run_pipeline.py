"""Script chạy toàn bộ sáu giai đoạn, từ dữ liệu M5 đến kết quả đánh giá."""

import argparse
import gc
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.ml.data_loader import (
    check_files,
    list_stores,
    load_calendar,
    load_prices,
    load_store,
)
from src.ml.feature_engineering import prepare_training_data
from src.ml.forecasting import SeasonalNaive, export_forecasts, recursive_forecast
from src.ml.preprocessing import history_matrix, prepare_history
from src.ml.reporting import evaluate_and_export
from src.ml.training import train_model

LOGGER = logging.getLogger("m5")


def load_config(path):
    """Đọc YAML và ngăn cấu hình làm sai thiết kế thí nghiệm."""
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (config["validation_origin"], config["test_origin"], config["horizon"]) != (
        1885,
        1913,
        28,
    ):
        raise ValueError("Thiết kế yêu cầu origin 1885/1913 và horizon 28.")
    model = config["model"]
    if model["objective"] != "tweedie" or not 1 < model["tweedie_variance_power"] < 2:
        raise ValueError("Cần objective=tweedie và 1 < variance_power < 2.")
    if model["device_type"] != "cpu" or model["n_jobs"] < 1:
        raise ValueError("Cần CPU và n_jobs là số luồng dương.")
    if config["price_policy"] != "last_known":
        raise ValueError("Phiên bản này chỉ hỗ trợ price_policy=last_known.")
    if config["chunk_size"] < 1 or config["smoke_items"] < 1:
        raise ValueError("chunk_size và smoke_items phải dương.")
    return config


def run_store(config, args, data_dir, output_root, store):
    """Chạy từ đọc dữ liệu đến xuất kết quả, xử lý riêng từng cửa hàng."""
    # Thiết lập phạm vi trước khi đọc dữ liệu, không chọn theo sai số holdout.
    split = "test" if args.mode == "test" else "validation"
    origin = config[f"{split}_origin"]
    scope = "smoke" if args.mode == "smoke" else args.scope
    sample = config["smoke_items"] if scope == "smoke" else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tag = f"{split}_{scope}_{store}_{run_id}"

    locations = {
        name: output_root / name / tag
        for name in [split, "models", "figures", "metrics"]
    }

    for directory in locations.values():
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Đọc dữ liệu
    # Chỉ lấy doanh số đến origin; chưa đọc nhãn của 28 ngày cần dự báo.
    LOGGER.info("1/6. Đọc dữ liệu | %s", store)
    calendar = load_calendar(data_dir)
    wide = load_store(
        data_dir,
        store,
        origin,
        chunk_size=config["chunk_size"],
        sample_items=sample,
        seed=config["seed"],
    )
    prices = load_prices(data_dir, store) if config["use_prices"] else None

    # 2. Chuẩn bị dữ liệu
    # Tạo bảng dài cho huấn luyện và ma trận lịch sử cho dự báo.
    LOGGER.info("2/6. Chuẩn bị dữ liệu")

    if scope == "full" and len(wide) != 3049:
        raise ValueError(
            f"Full M5 cần 3049 sản phẩm/cửa hàng; tìm thấy {len(wide)}. "
            "Không tự cắt mẫu."
        )

    states = wide.state_id.unique()

    if len(states) != 1 or states[0] not in ["CA", "TX", "WI"]:
        raise ValueError("Một cửa hàng phải thuộc đúng một bang CA/TX/WI.")

    state = states[0]
    items = wide.item_id.tolist()
    history = prepare_history(wide, calendar, origin, prices)
    matrix = history_matrix(wide, origin)
    last_prices = None
    last_price_changes = None

    if config["use_prices"]:
        # Giữ giá đã biết tại origin cố định cho toàn bộ 28 ngày dự báo.
        origin_snapshot = history.loc[history.day == origin].set_index("item_id")
        last_prices = origin_snapshot.loc[items, "sell_price"].to_numpy(dtype=np.float32)
        last_price_changes = origin_snapshot.loc[items, "price_change"].to_numpy(dtype=np.float32)

    del wide, prices
    gc.collect()

    # 3. Tạo đặc trưng
    # Lag và rolling chỉ dùng quá khứ; bỏ hàng chưa đủ 28 ngày lịch sử.
    LOGGER.info("3/6. Tạo đặc trưng")
    training_data = prepare_training_data(history)

    del history
    gc.collect()

    # 4. Huấn luyện
    # Fit một model chung cho các sản phẩm đã chọn của cửa hàng.
    LOGGER.info("4/6. Huấn luyện LightGBM–Tweedie")
    parameters = {**config["model"], "importance_type": "gain"}
    bundle, info = train_model(
        training_data,
        parameters,
        locations["models"] / "model.joblib",
    )

    del training_data
    gc.collect()

    # 5. Dự báo
    # Mỗi mô hình cập nhật lịch sử bằng dự báo của chính nó sau mỗi bước.
    LOGGER.info("5/6. Dự báo 28 ngày")
    common = {
        "history": matrix,
        "items": items,
        "store": store,
        "state": state,
        "calendar": calendar,
        "origin": origin,
        "features": bundle["features"],
        "categories": bundle["categories"],
        "split": split,
        "horizon": config["horizon"],
        "last_prices": last_prices,
        "last_price_changes": last_price_changes,
    }
    prediction = recursive_forecast(bundle["model"], **common)
    baseline = recursive_forecast(
        SeasonalNaive(),
        **common,
        model_name="seasonal_naive",
    )

    export_forecasts(prediction, locations[split] / "lightgbm_tweedie.csv")
    export_forecasts(baseline, locations[split] / "seasonal_naive.csv")

    # 6. Đánh giá và xuất kết quả
    # Chỉ đọc doanh số thật của holdout sau khi cả hai mô hình dự báo xong.
    LOGGER.info("6/6. Đánh giá và xuất kết quả")
    actual_wide = load_store(
        data_dir,
        store,
        origin + config["horizon"],
        chunk_size=config["chunk_size"],
        start_day=origin + 1,
        item_ids=items,
    )
    context = {
        "config": config,
        "store_id": store,
        "scope": scope,
        "split": split,
        "origin_day": origin,
        "run_id": run_id,
    }

    return evaluate_and_export(
        prediction,
        baseline,
        actual_wide,
        matrix,
        items,
        bundle,
        info,
        locations,
        context,
    )


def build_parser():
    """Định nghĩa các lệnh công khai để người mới có thể xem --help."""
    parser = argparse.ArgumentParser(description="LightGBM–Tweedie đệ quy M5, CPU")
    parser.add_argument(
        "--mode", choices=["smoke", "validation", "test"], default="smoke"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--store", default=None)
    group.add_argument("--all-stores", action="store_true")
    parser.add_argument("--scope", choices=["smoke", "full"], default="full")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser


def main(argv=None):
    """Kiểm tra ý định chạy Test và xử lý tuần tự để hạn chế RAM."""
    # Đọc lựa chọn chạy và kiểm tra điều kiện trước khi bắt đầu sáu giai đoạn.
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "test" and not args.confirm_final_test:
        parser.error("Test cuối cần --confirm-final-test sau khi đã chốt cấu hình.")
    if args.mode == "smoke" and args.all_stores:
        parser.error("Smoke chỉ chạy một cửa hàng; dùng --store CA_1.")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
    )
    try:
        config_path = Path(args.config)
        config = load_config(
            config_path if config_path.is_absolute() else ROOT / config_path
        )
        data_dir = ROOT / config["data_dir"]
        output_root = ROOT / config["results_dir"]
        check_files(data_dir, config["use_prices"])
        stores = list_stores(data_dir) if args.all_stores else [args.store or "CA_1"]
        if args.all_stores and len(stores) != 10:
            raise ValueError(f"M5 cần 10 cửa hàng, nhưng file hiện có {len(stores)}.")
        # run_store chạy đủ các giai đoạn 1–6 cho từng cửa hàng.
        # Chạy tuần tự để không giữ dữ liệu của 10 cửa hàng cùng lúc trong RAM.
        for store in stores:
            run_store(config, args, data_dir, output_root, store)
            gc.collect()
    except MemoryError:
        LOGGER.error(
            "Không đủ RAM. Đóng ứng dụng khác, giảm chunk_size hoặc n_jobs; "
            "thử scope smoke để kiểm tra. Full không tự cắt sản phẩm/lịch sử."
        )
        return 1
    except (FileNotFoundError, ValueError, KeyError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
