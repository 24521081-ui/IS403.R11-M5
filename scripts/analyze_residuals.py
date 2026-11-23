"""Script đọc toàn bộ các lần chạy validation trên CA_1, phân tích RMSSE và số dư (residuals)."""
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\Admin\Desktop\PTDLKD\m5-lightgbm-tweedie")
RESULTS_DIR = PROJECT_ROOT / "results"

def load_actuals():
    sales_file = PROJECT_ROOT / "data" / "raw" / "sales_train_evaluation.csv"
    # Origin 1885, horizon 28 -> d_1886 to d_1913
    cols = ["id", "item_id", "store_id"] + [f"d_{d}" for d in range(1886, 1914)]
    df = pd.read_csv(sales_file, usecols=cols)
    ca1 = df[df.store_id == "CA_1"].copy()
    
    melted = ca1.melt(
        id_vars=["item_id"],
        value_vars=[f"d_{d}" for d in range(1886, 1914)],
        var_name="target_day",
        value_name="actual"
    )
    return melted[["item_id", "target_day", "actual"]]

def analyze_all_runs():
    actuals = load_actuals()
    
    metric_folders = sorted(glob.glob(str(RESULTS_DIR / "metrics" / "validation_full_CA_1_*")))
    
    records = []
    
    for folder_str in metric_folders:
        folder = Path(folder_str)
        summary_file = folder / "summary.csv"
        if not summary_file.exists():
            continue
            
        summary_df = pd.read_csv(summary_file)
        lgb_row = summary_df[summary_df.model == "lightgbm_tweedie"]
        if lgb_row.empty:
            continue
            
        rmsse_col = "RMSSE_mean_unweighted" if "RMSSE_mean_unweighted" in lgb_row.columns else "RMSSE"
        rmsse = lgb_row[rmsse_col].values[0]
        mae = lgb_row["MAE"].values[0]
        rmse = lgb_row["RMSE"].values[0]
        t_time = lgb_row["training_time_seconds"].values[0]
        
        tag = folder.name
        val_forecast_file = RESULTS_DIR / "validation" / tag / "lightgbm_tweedie.csv"
        if not val_forecast_file.exists():
            continue
            
        forecast_df = pd.read_csv(val_forecast_file)
        
        merged = pd.merge(actuals, forecast_df[["item_id", "target_day", "y_pred"]], on=["item_id", "target_day"])
        merged["residual"] = merged["actual"] - merged["y_pred"]
        
        mean_res = merged["residual"].mean()
        median_res = merged["residual"].median()
        std_res = merged["residual"].std()
        iqr = merged["residual"].quantile(0.75) - merged["residual"].quantile(0.25)
        
        within_1 = (merged["residual"].abs() <= 1.0).mean() * 100
        
        zero_actuals = merged[merged["actual"] == 0]
        zero_pred_mean = zero_actuals["y_pred"].mean() if not zero_actuals.empty else np.nan

        model_meta = RESULTS_DIR / "models" / tag / "metadata.json"
        power = "N/A"
        leaves = "N/A"
        n_est = "N/A"
        use_prices = "N/A"
        if model_meta.exists():
            with open(model_meta, encoding="utf-8") as f:
                meta = json.load(f)
                cfg = meta.get("config", {})
                m_cfg = cfg.get("model", {})
                power = m_cfg.get("tweedie_variance_power", "N/A")
                leaves = m_cfg.get("num_leaves", "N/A")
                n_est = m_cfg.get("n_estimators", "N/A")
                use_prices = cfg.get("use_prices", "N/A")

        records.append({
            "tag": tag,
            "power": power,
            "leaves": leaves,
            "n_est": n_est,
            "use_prices": use_prices,
            "RMSSE": round(rmsse, 5),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Mean_Residual (Bias)": round(mean_res, 4),
            "Std_Residual": round(std_res, 4),
            "Median_Residual": round(median_res, 4),
            "IQR_Residual": round(iqr, 4),
            "Within_±1_pct": round(within_1, 2),
            "Zero_Demand_Avg_Pred": round(zero_pred_mean, 4),
            "Train_Time_s": round(t_time, 1)
        })
        
    res_df = pd.DataFrame(records)
    return res_df

if __name__ == "__main__":
    df = analyze_all_runs()
    df.to_csv(PROJECT_ROOT / "results" / "comparison_report.csv", index=False)
    print("\n--- BẢNG SO SÁNH KẾT QUẢ CÁC CẤU HÌNH ---")
    print(df.to_string())
