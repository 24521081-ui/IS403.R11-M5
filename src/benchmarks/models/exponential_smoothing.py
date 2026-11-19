"""
Triển khai phương pháp San bằng mũ (Exponential Smoothing) và cơ chế tổng hợp Bottom-Up (ES_bu).
Tối ưu hóa bằng Numba để xử lý nhanh hàng chục nghìn chuỗi thời gian phân cấp.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit


@njit(fastmath=True)
def _seasonal_es_jit(series_matrix: np.ndarray, horizon: int, alpha: float, gamma: float, season_len: int = 7) -> np.ndarray:
    """
    Cài đặt thuật toán San bằng mũ mùa vụ (Seasonal Exponential Smoothing - Additive)
    với chu kỳ tuần (season_len = 7).
    """
    n_series, n_timesteps = series_matrix.shape
    forecasts = np.zeros((n_series, horizon), dtype=np.float64)

    for i in range(n_series):
        # Xác định điểm bắt đầu có dữ liệu bán hàng
        first_idx = -1
        for t in range(n_timesteps):
            if series_matrix[i, t] > 0:
                first_idx = t
                break
                
        if first_idx == -1:
            continue

        valid_len = n_timesteps - first_idx
        
        # Nếu chuỗi quá ngắn (dưới 1 chu kỳ tuần), sử dụng trung bình đơn giản
        if valid_len < season_len:
            mean_val = 0.0
            for t in range(first_idx, n_timesteps):
                mean_val += series_matrix[i, t]
            mean_val /= max(1, valid_len)
            for h in range(horizon):
                forecasts[i, h] = mean_val
            continue

        # Khởi tạo mức cơ sở (Level) từ chu kỳ tuần đầu tiên
        initial_sum = 0.0
        for s in range(season_len):
            initial_sum += series_matrix[i, first_idx + s]
        level = initial_sum / season_len

        # Khởi tạo các hệ số mùa vụ ban đầu (Seasonal components)
        seasonal = np.zeros(season_len, dtype=np.float64)
        for s in range(season_len):
            seasonal[(first_idx + s) % season_len] = series_matrix[i, first_idx + s] - level

        # Cập nhật đệ quy qua từng bước thời gian
        for t in range(first_idx + season_len, n_timesteps):
            s_idx = t % season_len
            y_t = series_matrix[i, t]
            
            # Cập nhật Level và Mùa vụ
            prev_level = level
            level = alpha * (y_t - seasonal[s_idx]) + (1.0 - alpha) * prev_level
            seasonal[s_idx] = gamma * (y_t - level) + (1.0 - gamma) * seasonal[s_idx]

        # Dự báo ngoài mẫu cho horizon bước
        for h in range(horizon):
            future_t = n_timesteps + h
            s_idx = future_t % season_len
            pred = level + seasonal[s_idx]
            forecasts[i, h] = max(0.0, pred)

    return forecasts


def forecast_exponential_smoothing_bu(series_matrix: np.ndarray, horizon: int = 28, alpha: float = 0.2, gamma: float = 0.15) -> np.ndarray:
    """
    Dự báo bằng mô hình San bằng mũ có thành phần mùa vụ tuần cho từng chuỗi ở cấp thấp nhất (Level 12).
    """
    matrix = np.ascontiguousarray(series_matrix, dtype=np.float64)
    return _seasonal_es_jit(matrix, horizon, alpha, gamma, season_len=7)


def aggregate_bottom_up(forecast_df: pd.DataFrame, level_columns: Tuple[str, ...]) -> pd.DataFrame:
    """
    Tổng hợp dự báo từ cấp sản phẩm-cửa hàng (Level 12) lên các cấp phân cấp cao hơn (Bottom-Up aggregation).
    
    Tham số:
        forecast_df: DataFrame dự báo cấp chi tiết chứa các cột mã phân cấp và 'y_pred'.
        level_columns: Các cột định danh nhóm. Tuple rỗng biểu thị tổng toàn hệ thống.
        
    Trả về:
        DataFrame dự báo đã được cộng dồn theo thời gian.
    """
    group_cols = [*level_columns, "model", "split", "origin_day", "target_day", "horizon"]
    agg_df = forecast_df.groupby(group_cols, as_index=False)["y_pred"].sum()
    return agg_df
