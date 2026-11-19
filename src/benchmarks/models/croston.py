"""
Triển khai phương pháp Croston và Croston-SBA (Syntetos-Boylan Approximation).
Các thuật toán được tối ưu hóa bằng Numba để đạt tốc độ thực thi cao trên dữ liệu lớn.
"""

import numpy as np
from numba import njit


@njit(fastmath=True)
def _croston_jit(series_matrix: np.ndarray, horizon: int, alpha: float, is_sba: bool) -> np.ndarray:
    """
    Tính toán dự báo Croston hoặc Croston-SBA cho ma trận n chuỗi thời gian.
    
    Tham số:
        series_matrix: Ma trận numpy kích thước (n_series, n_timesteps) chứa lịch sử bán hàng.
        horizon: Số bước cần dự báo (mặc định 28 ngày).
        alpha: Hệ số làm mịn (thường là 0.1).
        is_sba: True nếu áp dụng hiệu chỉnh Syntetos-Boylan (SBA), False nếu dùng Croston cổ điển.
        
    Trả về:
        Ma trận dự báo kích thước (n_series, horizon).
    """
    n_series, n_timesteps = series_matrix.shape
    forecasts = np.zeros((n_series, horizon), dtype=np.float64)
    
    # Hệ số điều chỉnh SBA nhằm loại bỏ độ chệch dương (positive bias)
    sba_factor = (1.0 - 0.5 * alpha) if is_sba else 1.0

    for i in range(n_series):
        # Tìm thời điểm bán dương đầu tiên
        first_idx = -1
        for t in range(n_timesteps):
            if series_matrix[i, t] > 0:
                first_idx = t
                break
        
        # Nếu chuỗi không có bất kỳ ngày nào bán dương
        if first_idx == -1:
            continue
            
        z = float(series_matrix[i, first_idx])
        p = 1.0
        t_prev = first_idx
        has_multiple_sales = False
        
        for t in range(first_idx + 1, n_timesteps):
            val = series_matrix[i, t]
            if val > 0:
                has_multiple_sales = True
                interval = float(t - t_prev)
                z = z + alpha * (val - z)
                p = p + alpha * (interval - p)
                t_prev = t
                
        # Trường hợp chuỗi chỉ có đúng 1 ngày bán dương duy nhất
        if not has_multiple_sales:
            interval = float(n_timesteps - first_idx)
            p = max(1.0, interval)
            
        if p > 0:
            rate = sba_factor * (z / p)
        else:
            rate = 0.0
            
        rate = max(0.0, rate)
        for h in range(horizon):
            forecasts[i, h] = rate

    return forecasts


@njit(fastmath=True)
def _croston_sba_seasonal_jit(series_matrix: np.ndarray, horizon: int, alpha: float, season_len: int = 7) -> np.ndarray:
    """
    Croston-SBA kết hợp chỉ số mùa vụ tuần (7 ngày) phù hợp với hành vi bán lẻ.
    """
    n_series, n_timesteps = series_matrix.shape
    forecasts = np.zeros((n_series, horizon), dtype=np.float64)
    sba_factor = 1.0 - 0.5 * alpha

    for i in range(n_series):
        first_idx = -1
        for t in range(n_timesteps):
            if series_matrix[i, t] > 0:
                first_idx = t
                break
                
        if first_idx == -1:
            continue
            
        z = float(series_matrix[i, first_idx])
        p = 1.0
        t_prev = first_idx
        has_multiple = False
        
        # Tính toán phân bố theo thứ trong tuần (độ dài 7 ngày)
        day_sum = np.zeros(season_len, dtype=np.float64)
        day_cnt = np.zeros(season_len, dtype=np.float64)
        
        for t in range(first_idx, n_timesteps):
            dow = t % season_len
            day_sum[dow] += series_matrix[i, t]
            day_cnt[dow] += 1.0
            
            if t > first_idx and series_matrix[i, t] > 0:
                has_multiple = True
                interval = float(t - t_prev)
                z = z + alpha * (series_matrix[i, t] - z)
                p = p + alpha * (interval - p)
                t_prev = t
                
        if not has_multiple:
            interval = float(n_timesteps - first_idx)
            p = max(1.0, interval)
            
        base_rate = sba_factor * (z / p) if p > 0 else 0.0
        base_rate = max(0.0, base_rate)
        
        # Tính hệ số mùa vụ theo thứ trong tuần (seasonal indices)
        day_means = np.zeros(season_len, dtype=np.float64)
        total_mean = 0.0
        for s in range(season_len):
            if day_cnt[s] > 0:
                day_means[s] = day_sum[s] / day_cnt[s]
                total_mean += day_means[s]
        total_mean /= season_len
        
        seasonal_weights = np.ones(season_len, dtype=np.float64)
        if total_mean > 0:
            for s in range(season_len):
                seasonal_weights[s] = day_means[s] / total_mean

        for h in range(horizon):
            future_t = n_timesteps + h
            dow = future_t % season_len
            forecasts[i, h] = max(0.0, base_rate * seasonal_weights[dow])

    return forecasts


def forecast_croston(series_matrix: np.ndarray, horizon: int = 28, alpha: float = 0.1) -> np.ndarray:
    """
    Thực hiện dự báo Croston cổ điển (1972) cho tập các chuỗi thời gian.
    """
    matrix = np.ascontiguousarray(series_matrix, dtype=np.float64)
    return _croston_jit(matrix, horizon, alpha, is_sba=False)


def forecast_croston_sba(series_matrix: np.ndarray, horizon: int = 28, alpha: float = 0.1, with_seasonality: bool = False) -> np.ndarray:
    """
    Thực hiện dự báo Croston-SBA (Syntetos-Boylan Approximation, 2005).
    Có tùy chọn kết hợp chỉ số mùa vụ tuần để phản ánh hiệu ứng cuối tuần trong bán lẻ.
    """
    matrix = np.ascontiguousarray(series_matrix, dtype=np.float64)
    if with_seasonality:
        return _croston_sba_seasonal_jit(matrix, horizon, alpha, season_len=7)
    return _croston_jit(matrix, horizon, alpha, is_sba=True)
