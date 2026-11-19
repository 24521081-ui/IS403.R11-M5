"""
Triển khai phương pháp Seasonal Naïve (chu kỳ 7 ngày).
Đóng vai trò mốc đối chứng cơ sở (benchmark) để đánh giá giá trị gia tăng của các phương pháp thống kê phức tạp hơn.
"""

import numpy as np
from numba import njit


@njit(fastmath=True)
def _seasonal_naive_jit(series_matrix: np.ndarray, horizon: int, season_len: int = 7) -> np.ndarray:
    """
    Dự báo bằng cách lặp lại chính xác các giá trị của chu kỳ tuần gần nhất.
    """
    n_series, n_timesteps = series_matrix.shape
    forecasts = np.zeros((n_series, horizon), dtype=np.float64)

    for i in range(n_series):
        if n_timesteps < season_len:
            last_val = series_matrix[i, -1] if n_timesteps > 0 else 0.0
            for h in range(horizon):
                forecasts[i, h] = max(0.0, last_val)
            continue
            
        for h in range(horizon):
            # Lấy giá trị tương ứng của chu kỳ tuần gần nhất
            offset = (h % season_len)
            src_idx = n_timesteps - season_len + offset
            forecasts[i, h] = max(0.0, float(series_matrix[i, src_idx]))

    return forecasts


def forecast_seasonal_naive(series_matrix: np.ndarray, horizon: int = 28, season_len: int = 7) -> np.ndarray:
    """
    Thực hiện dự báo Seasonal Naive chu kỳ 7 ngày.
    """
    matrix = np.ascontiguousarray(series_matrix, dtype=np.float64)
    return _seasonal_naive_jit(matrix, horizon, season_len)
