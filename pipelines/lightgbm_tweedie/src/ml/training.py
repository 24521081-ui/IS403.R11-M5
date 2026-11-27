"""Giai đoạn 4: huấn luyện một LightGBM chung cho sản phẩm của cửa hàng."""

import logging
from time import perf_counter

import joblib
from lightgbm import LGBMRegressor

LOGGER = logging.getLogger("m5")


def train_model(training_data, parameters, model_path):
    """4. Huấn luyện và lưu model từ bộ đặc trưng đã chuẩn bị."""
    X = training_data["X"]
    y = training_data["y"]
    features = training_data["features"]

    # 4.1. Khởi tạo và fit; chỉ sử dụng tập Train.
    # Tweedie phù hợp mục tiêu không âm có nhiều số 0; power=1.2 chỉ là khởi điểm.
    # learning_rate là bước học; n_estimators là số cây; num_leaves giới hạn độ phức tạp.
    # n_jobs giới hạn luồng CPU; random_state giúp tái lập thí nghiệm.
    model = LGBMRegressor(**parameters)
    LOGGER.info("Huấn luyện %d dòng, %d sản phẩm bằng CPU", len(X), X.item_id.nunique())
    start = perf_counter()
    model.fit(X, y, categorical_feature=["item_id"])
    elapsed = perf_counter() - start

    # 4.2. Lưu model cùng thứ tự đặc trưng và mã phân loại.
    bundle = {
        "model": model,
        "features": features,
        "categories": training_data["categories"],
    }
    joblib.dump(bundle, model_path)
    model.booster_.save_model(str(model_path.with_suffix(".txt")))

    info = {
        "training_time_seconds": elapsed,
        "n_train_rows": len(X),
        "n_items": int(X.item_id.nunique()),
        "features": features,
        "fit_start_day": 29,
        "fit_end_day": training_data["fit_end_day"],
        "dropped_warmup_rows": training_data["dropped_warmup_rows"],
        "effective_model_params": model.get_params(),
    }

    return bundle, info
