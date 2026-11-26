"""Nhãn mức độ bán gián đoạn dùng chung cho dự báo và benchmark."""

import numpy as np


def compute_zero_ratio(history: np.ndarray, from_first_sale: bool = True) -> np.ndarray:
    """Tính tỷ lệ ngày bán bằng 0, mặc định từ ngày bán đầu tiên."""
    is_zero = history == 0
    if not from_first_sale:
        return is_zero.mean(axis=1)

    has_sold = np.maximum.accumulate(history > 0, axis=1)
    counts = has_sold.sum(axis=1)
    zeros = (is_zero & has_sold).sum(axis=1)
    return np.divide(zeros, counts, out=np.ones(len(history)), where=counts > 0)


def categorize_intermittency(zero_ratios: np.ndarray) -> np.ndarray:
    """Gán một trong bốn nhãn theo các ngưỡng 30%, 50% và 70%."""
    labels = np.array([
        "1. Thường xuyên (<30% 0)",
        "2. Trung bình (30-50% 0)",
        "3. Gián đoạn (50-70% 0)",
        "4. Rất gián đoạn (>=70% 0)",
    ], dtype=object)
    return labels[np.searchsorted([0.30, 0.50, 0.70], zero_ratios, side="right")]
