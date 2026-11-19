"""
Cấu hình tham số và đường dẫn cho Hợp phần Mô hình Thống kê Đối chứng (Statistical Benchmarks).
Thiết kế tự động nhận diện dữ liệu, độc lập, module hóa theo chuẩn chuyên nghiệp.
"""

from pathlib import Path

# Thư mục gốc dự án IS403.R11-M5
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Tìm kiếm thư mục chứa dữ liệu M5 theo thứ tự ưu tiên
CANDIDATE_DATA_DIRS = [
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "data",
    REPO_ROOT.parent / "data",
    REPO_ROOT.parent / "Đồ án" / "data",
    REPO_ROOT.parent / "m5-forecasting-accuracy",
]

DATA_DIR = None
for candidate in CANDIDATE_DATA_DIRS:
    if (candidate / "sales_train_evaluation.csv").exists():
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    DATA_DIR = REPO_ROOT / "data" / "raw"

SALES_EVAL_FILE = DATA_DIR / "sales_train_evaluation.csv"
SALES_VAL_FILE = DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE = DATA_DIR / "calendar.csv"
PRICES_FILE = DATA_DIR / "sell_prices.csv"

# Thư mục kết quả và tài liệu
RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
DOCS_DIR = REPO_ROOT / "docs"

# Đảm bảo các thư mục luôn tồn tại
for directory in [METRICS_DIR, PREDICTIONS_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Mốc phân chia tập dữ liệu theo quy chuẩn M5
TRAIN_END_VAL = 1885       # Giai đoạn train validation: d_1 đến d_1885
VAL_START = 1886           # Cửa sổ validation: d_1886 đến d_1913 (28 ngày)
VAL_END = 1913
ORIGIN_DAY_VAL = 1885

TRAIN_END_TEST = 1913      # Giai đoạn train test: d_1 đến d_1913
TEST_START = 1914          # Cửa sổ test cuối: d_1914 đến d_1941 (28 ngày)
TEST_END = 1941
ORIGIN_DAY_TEST = 1913

# Tham số bài toán dự báo
HORIZON = 28               # Độ dài cửa sổ dự báo (h = 28 ngày)
SEASONALITY = 7            # Chu kỳ tuần trong bán lẻ (m = 7 ngày)

# Tham số mô hình thống kê
ALPHA_CROSTON = 0.1        # Hệ số làm mịn cho Croston và Croston-SBA (theo chuẩn Syntetos & Boylan)
ALPHA_ES = 0.2             # Hệ số làm mịn mức (level) cho Exponential Smoothing
GAMMA_ES = 0.15            # Hệ số làm mịn thành phần mùa vụ (seasonal)

# Cấu hình danh mục cửa hàng và ngành hàng trong M5
STORES = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]
