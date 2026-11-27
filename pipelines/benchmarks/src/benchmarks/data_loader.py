"""
Nạp và chuẩn bị dữ liệu lịch sử bán hàng M5 cho từng giai đoạn đánh giá.
"""

import time
import numpy as np
import pandas as pd

from src.benchmarks.config import (
    SALES_EVAL_FILE,
    TRAIN_END_VAL,
    VAL_START,
    VAL_END,
    ORIGIN_DAY_VAL,
    TRAIN_END_TEST,
    TEST_START,
    TEST_END,
    ORIGIN_DAY_TEST,
)


def load_sales_dataset(store_filter=None, nrows=None):
    """
    Đọc dữ liệu bán hàng từ file sales_train_evaluation.csv.
    Tối ưu bộ nhớ bằng cách chỉ chọn cột định danh và các cột ngày đến TEST_END.
    """
    print(f"Đang đọc dữ liệu từ {SALES_EVAL_FILE.name}...")
    t0 = time.time()

    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [f"d_{d}" for d in range(1, TEST_END + 1)]
    usecols = meta_cols + day_cols

    df = pd.read_csv(SALES_EVAL_FILE, usecols=usecols, nrows=nrows)

    if store_filter:
        if isinstance(store_filter, str):
            store_filter = [store_filter]
        df = df[df["store_id"].isin(store_filter)].reset_index(drop=True)
        print(f"-> Đã lọc theo cửa hàng: {store_filter}. Số chuỗi: {len(df):,}")
    else:
        print(f"-> Đã nạp toàn bộ 10 cửa hàng. Số chuỗi: {len(df):,}")

    print(f"-> Thời gian nạp dữ liệu: {time.time() - t0:.2f}s")
    return df


def prepare_split_matrices(df: pd.DataFrame, split: str):
    """
    Tách ma trận lịch sử huấn luyện (X_train) và ma trận thực tế đối chiếu (y_target)
    theo giai đoạn 'validation' hoặc 'test'.
    """
    assert split in ["validation", "test"], f"Split không hợp lệ: {split}"

    if split == "validation":
        origin_day = ORIGIN_DAY_VAL
        train_end = TRAIN_END_VAL
        target_start = VAL_START
        target_end = VAL_END
    else:
        origin_day = ORIGIN_DAY_TEST
        train_end = TRAIN_END_TEST
        target_start = TEST_START
        target_end = TEST_END

    train_cols = [f"d_{d}" for d in range(1, train_end + 1)]
    target_cols = [f"d_{d}" for d in range(target_start, target_end + 1)]

    X_train = df[train_cols].values.astype(np.float64)
    y_target = df[target_cols].values.astype(np.float64)

    return X_train, y_target, origin_day, target_cols

