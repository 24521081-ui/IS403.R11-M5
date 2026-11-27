"""
Các thước đo đánh giá sai số dự báo theo chuẩn bài toán M5 và phân tích kinh doanh.
Bao gồm:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- RMSSE (Root Mean Squared Scaled Error - chuẩn cuộc thi M5)
- Phân nhóm chuỗi theo mức độ gián đoạn (Intermittency Group)
"""

import numpy as np
import pandas as pd
from numba import njit


@njit(fastmath=True)
def _calc_m5_scale(history_matrix: np.ndarray) -> np.ndarray:
    """
    Tính thang đo chuẩn hóa mẫu số cho từng chuỗi theo định nghĩa M5:
    scale_i = mean((y_t - y_{t-1})^2) tính từ ngày bán dương đầu tiên.
    """
    n_series, n_timesteps = history_matrix.shape
    scales = np.zeros(n_series, dtype=np.float64)

    for i in range(n_series):
        first_idx = -1
        for t in range(n_timesteps):
            if history_matrix[i, t] > 0:
                first_idx = t
                break
                
        if first_idx == -1 or first_idx >= n_timesteps - 1:
            scales[i] = 1.0
            continue
            
        diff_sq_sum = 0.0
        count = 0
        for t in range(first_idx + 1, n_timesteps):
            diff = history_matrix[i, t] - history_matrix[i, t - 1]
            diff_sq_sum += diff * diff
            count += 1
            
        if count > 0 and diff_sq_sum > 0:
            scales[i] = diff_sq_sum / count
        else:
            scales[i] = 1.0

    return scales


@njit(fastmath=True)
def _calc_metrics_jit(history_matrix: np.ndarray, actual_matrix: np.ndarray, pred_matrix: np.ndarray):
    """
    Tính MAE, RMSE và RMSSE cho từng chuỗi.
    """
    n_series, horizon = actual_matrix.shape
    scales = _calc_m5_scale(history_matrix)
    
    mae_arr = np.zeros(n_series, dtype=np.float64)
    rmse_arr = np.zeros(n_series, dtype=np.float64)
    rmsse_arr = np.zeros(n_series, dtype=np.float64)
    
    for i in range(n_series):
        abs_err_sum = 0.0
        sq_err_sum = 0.0
        for h in range(horizon):
            diff = actual_matrix[i, h] - pred_matrix[i, h]
            abs_err_sum += abs(diff)
            sq_err_sum += diff * diff
            
        mse = sq_err_sum / horizon
        mae_arr[i] = abs_err_sum / horizon
        rmse_arr[i] = np.sqrt(mse)
        
        scale_val = scales[i]
        if scale_val > 1e-8:
            rmsse_arr[i] = np.sqrt(mse / scale_val)
        else:
            rmsse_arr[i] = rmse_arr[i]

    return mae_arr, rmse_arr, rmsse_arr


def compute_series_metrics(history: np.ndarray, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """
    Tính toán bảng chỉ số sai số MAE, RMSE, RMSSE cho từng chuỗi.
    """
    hist_mat = np.ascontiguousarray(history, dtype=np.float64)
    act_mat = np.ascontiguousarray(actual, dtype=np.float64)
    pred_mat = np.ascontiguousarray(pred, dtype=np.float64)
    
    mae, rmse, rmsse = _calc_metrics_jit(hist_mat, act_mat, pred_mat)
    
    return pd.DataFrame({
        "mae": mae,
        "rmse": rmse,
        "rmsse": rmsse
    })


def compute_zero_ratio(history: np.ndarray, from_first_sale: bool = True) -> np.ndarray:
    """
    Tính tỷ lệ ngày có lượng bán bằng 0 trên tập lịch sử.
    """
    n_series, n_timesteps = history.shape
    ratios = np.zeros(n_series, dtype=np.float64)
    
    for i in range(n_series):
        if from_first_sale:
            first_idx = -1
            for t in range(n_timesteps):
                if history[i, t] > 0:
                    first_idx = t
                    break
            if first_idx == -1:
                ratios[i] = 1.0
                continue
            sub_len = n_timesteps - first_idx
            zero_cnt = np.sum(history[i, first_idx:] == 0)
            ratios[i] = zero_cnt / max(1, sub_len)
        else:
            zero_cnt = np.sum(history[i, :] == 0)
            ratios[i] = zero_cnt / n_timesteps
            
    return ratios


def categorize_intermittency(zero_ratios: np.ndarray) -> np.ndarray:
    """
    Phân loại mức độ gián đoạn thành 4 nhóm theo tỷ lệ số ngày bằng 0:
    - Thường xuyên (Smooth/Regular): < 30%
    - Trung bình (Mild): 30% - 50%
    - Gián đoạn (Moderate): 50% - 70%
    - Rất gián đoạn (Severe/Lumpy): >= 70%
    """
    categories = np.empty(len(zero_ratios), dtype=object)
    for i, r in enumerate(zero_ratios):
        if r < 0.30:
            categories[i] = "1. Thường xuyên (<30% 0)"
        elif r < 0.50:
            categories[i] = "2. Trung bình (30-50% 0)"
        elif r < 0.70:
            categories[i] = "3. Gián đoạn (50-70% 0)"
        else:
            categories[i] = "4. Rất gián đoạn (>=70% 0)"
    return categories
