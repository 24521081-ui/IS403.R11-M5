"""Build ten Vietnamese notebooks with genuine read-only execution output."""

import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat as nbf

from .wrmsse_stage5 import ROOT

BOOTSTRAP = """from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import Image, display

here = Path.cwd().resolve()
root = next(p for p in [here, *here.parents] if (p / "reports/stage5_inputs.json").is_file())
sys.path.insert(0, str(root))
from src.wrmsse_stage5 import STAGE2, STAGE3, STAGE4, SELECTED, TABLES, csv, read, check_gates
check_gates()

def table(name):
    return csv(TABLES / (name + ".csv"))

def figure(number):
    path, = (root / "reports/figures").glob(f"{number:02d}_*.png")
    display(Image(filename=str(path), width=950))
"""


def cells():
    return [
        ("00_project_overview", "Tổng quan đề tài", [
            ("md", "Mục tiêu là dự báo lượng bán M5 trong 28 ngày bằng Random Forest Regression và so sánh với Seasonal Naive chu kỳ 7. Thước đo chính là WRMSSE đầy đủ 12 cấp. Thứ tự nghiên cứu ở sơ đồ dưới đây."),
            ("code", 'display(table("split_timeline"))\ndisplay(table("experiment_workflow"))\nfigure(1)\nfigure(2)'),
            ("md", "Mỗi notebook mặc định đọc artifact của dự án. Chọn kernel Python của môi trường dự án và dùng Run All. Cấu hình, số liệu và cách xem kết quả có trong [README](../README.md)."),
            ("code", 'display(table("validation_test"))\ndisplay(read(STAGE4 / "manifest_completed.json")["status"])'),
        ]),
        ("01_data_preprocessing", "Chia dữ liệu và lấy mẫu", [
            ("md", "M5 lưu lượng bán hằng ngày theo sản phẩm và cửa hàng. Lịch hỗ trợ ánh xạ ngày và SNAP. Giá bán lịch sử dùng cho trọng số doanh thu. Train gốc lấy mẫu phân tầng cửa hàng × bộ phận × quý, không hoàn lại. Mỗi tầng nhận ít nhất một dòng rồi phân bổ phần còn lại theo sức chứa và số dư lớn nhất."),
            ("code", 'from src.wrmsse_stage2_data import sample_training\npre = read(STAGE4 / "preflight.json")\nprocessed = root / pre["processed"]\nmanifest = read(processed / "manifest.json")\ndisplay({k: manifest[k] for k in ("train", "validation", "bottom_series", "train_eligible_observations", "train_sample", "seed", "sampling")})\ndisplay(pd.read_parquet(processed / "strata.parquet").head(10))\nfigure(1)'),
            ("md", "Mẫu cuối giữ nguyên khóa Train. Phần bổ sung Validation lấy theo mật độ gốc, làm tròn đến số nguyên gần nhất. Hai tập lấy mẫu cố định theo seed. [Chuẩn bị dữ liệu](../docs/STAGE2_REPORT.md) · [Mã lấy mẫu](../src/wrmsse_stage2_data.py)."),
            ("code", 'sample = read(STAGE4 / "final_training_sample_manifest.json")\ndisplay(pd.DataFrame([sample]).drop(columns=["files"]))'),
        ]),
        ("02_wrmsse_evaluation_method", "WRMSSE M5 và hệ thống phân cấp", [
            ("md", """M5 tập hợp lưới sản phẩm × cửa hàng lên 12 cấp. Với mỗi chuỗi, scale là trung bình bình phương sai phân lịch sử sau lần bán khác 0 đầu tiên. RMSSE là căn bậc hai của MSE dự báo chia scale. Trọng số là tỷ trọng doanh thu lịch sử trong cấp chia 12. Contribution là trọng số nhân RMSSE và WRMSSE là tổng contribution.

Điểm trong cấp là 12 lần tổng contribution của cấp. Chuỗi trọng số 0 đóng góp 0. Evaluator dừng nếu scale không hữu hạn hoặc không dương ở chuỗi có trọng số dương. [Đặc tả và kiểm thử độc lập](../docs/WRMSSE_PREPARATION.md)."""),
            ("code", 'from src.wrmsse import LEVELS, M5_COUNTS, WRMSSEEvaluator\ndisplay(pd.DataFrame({"level": range(1, 13), "keys": LEVELS, "n_series": M5_COUNTS}))\naudit = read(root / "preparation/data_context_audit.json")\ndisplay(pd.DataFrame([{k: v for k, v in item.items() if k.endswith("_days")} | {"fold": name} for name, item in audit["contexts"].items()]))'),
            ("code", 'levels = table("wrmsse_12_levels")\ndisplay(levels)\nassert abs(levels.weight.sum() - 1) < 1e-12\nfigure(11)'),
        ]),
        ("03_feature_engineering", "Đặc trưng và giới hạn lịch sử", [
            ("md", "Lag 28/35/42/56 và trung bình trượt 7/28 ngày dùng doanh số tối đa đến t−28, đồng thời giới hạn ở forecast origin. SNAP từ bang và ngày trong tháng theo quy tắc khóa từ lịch đến d_1885. One-hot áp dụng cho năm định danh, giữ vocabulary và thứ tự cột. [Mã đặc trưng](../src/wrmsse_stage2_data.py) · [Hậu kiểm SNAP](../docs/STAGE3_SNAP_REVIEW.md)."),
            ("code", 'from src.wrmsse_stage2_data import CATS, BASE, numeric_block\nfeatures = read(STAGE4 / "final_feature_metadata.json")\nassert list(features["logical_features"]) == CATS + BASE + ["snap_state"]\ndisplay(pd.Series(features["logical_features"], name="thứ_tự_đặc_trưng"))\ndisplay({key: features[key] for key in ("logical_count", "encoded_count", "format", "history_cutoff_formula", "snap_rule")})'),
            ("md", "FE-09 hợp nhất RF-v4, giá quá khứ, SNAP và Behavior. FE-10 thêm Events. Hậu kiểm xác nhận Events trên cửa sổ Validation cho FE-05 và FE-10. FE-04 dùng RF-v1 + SNAP và có WRMSSE Validation thấp nhất trong các cấu hình đủ điều kiện."),
            ("code", 'experiments = table("experiment_results")\ndisplay(experiments[experiments.experiment.isin(["FE-04", "FE-09", "FE-10"])][["experiment", "feature_set", "logical", "encoded", "validation_wrmsse"]])\nfigure(7)'),
        ]),
        ("04_initial_hyperparameter_tuning", "Vòng siêu tham số ban đầu", [
            ("md", "RF-01–RF-05 dùng RF-v1 và cùng mẫu Train. Chọn WRMSSE Validation thấp nhất. Kết quả RF-04 cung cấp cấu hình cho ablation. [Mã điều phối và tạo các lượt](../scripts/run_stage3.py)."),
            ("code", 'initial = table("experiment_results").query("group == \'initial\'").sort_values("validation_wrmsse")\ndisplay(initial[["experiment", "validation_wrmsse", "fit_seconds", "n_jobs", "reuse"]])\nfigure(3)'),
        ]),
        ("05_feature_ablation", "Ablation RF-v1 đến RF-v4", [
            ("md", "Dùng cùng siêu tham số của RF-04 để kiểm tra các tầng đặc trưng mở rộng. RF-v2 bổ sung tỷ lệ không bán, RF-v3 bổ sung số ngày bán khác 0, RF-v4 thêm recency. RF-v1 cho WRMSSE thấp nhất và được giữ làm bộ cơ sở cho vòng nâng cao."),
            ("code", 'ablation = table("experiment_results").query("group == \'ablation\'").sort_values("validation_wrmsse")\ndisplay(ablation[["experiment", "logical", "encoded", "validation_wrmsse", "reuse"]])\nfigure(4)'),
        ]),
        ("06_advanced_hyperparameter_tuning", "Vòng siêu tham số nâng cao", [
            ("md", "RF-06–RF-22 dùng bộ cơ sở RF-v1. RF-15 có WRMSSE thấp nhất trong vòng nâng cao. So sánh thời gian dựa trên tài nguyên ghi nhận của lượt fit nguồn. Các lượt dùng 4 hoặc 8 luồng, và bảng ghi trường hợp reuse."),
            ("code", 'advanced = table("experiment_results").query("group == \'advanced\'").sort_values("validation_wrmsse")\ndisplay(advanced[["experiment", "validation_wrmsse", "fit_seconds", "n_jobs", "reuse"]])\nfigure(5)\nfigure(8)'),
        ]),
        ("07_feature_optimization", "Tối ưu đặc trưng và chốt selection v2", [
            ("md", "RF-15 cố định trong so sánh đặc trưng. Hậu kiểm lịch xác nhận SNAP cho Validation. FE-04 có WRMSSE thấp nhất và được chốt trước Test. Các lượt FE-09 và FE-10 kiểm tra bộ đặc trưng mở rộng trên cùng cửa sổ. [Báo cáo Giai đoạn 3](../docs/STAGE3_REPORT.md)."),
            ("code", 'fe = table("experiment_results").query("group == \'features\'")\ndisplay(fe[["experiment", "feature_set", "logical", "encoded", "validation_wrmsse", "selection_status"]])\nfigure(6)\nfigure(7)'),
            ("code", 'locked = read(SELECTED)\ndisplay({key: locked[key] for key in ("selection_version", "name", "hyperparameter_run", "params", "snap_rule")})'),
        ]),
        ("08_final_training_and_test", "Fit cuối và đánh giá Test", [
            ("md", "Giai đoạn 4 đã fit một rừng mới trên mẫu Train + Validation. Actual chỉ được mở qua gate sau khi khóa model, ma trận và hai dự báo. Seasonal Naive lặp bảy ngày cuối trước d_1914. [Pipeline fit và đánh giá](../src/wrmsse_stage4.py) · [Báo cáo Giai đoạn 4](../docs/STAGE4_REPORT.md)."),
            ("code", 'config = read(STAGE4 / "final_model_config.json")\nreceipt = read(STAGE4 / "resource_summary.json")\ndisplay(config)\ndisplay({key: receipt[key] for key in ("fit_seconds", "predict_seconds", "peak_worker_rss_mib", "n_jobs", "versions")})\ndisplay(table("model_comparison"))\nfigure(9)\nfigure(10)\nfigure(11)\nfigure(12)'),
            ("md", "Validation và Test dùng hai cửa sổ cùng các context scale/trọng số khác nhau. Test mới đã từng là Validation của nghiên cứu cũ. Notebook hiển thị kết quả đã khóa của lần chạy WRMSSE."),
        ]),
        ("09_business_insights_and_visualization", "Phân tích kinh doanh trên Test", [
            ("md", "Sai lệch dùng Forecast trừ Actual. WAPE và Bias tính lại từ tổng thành phần của từng nhóm. Tổng sai lệch trái dấu có thể bù trừ, nên cần xem cùng MAE/RMSE/WAPE. [Báo cáo kinh doanh](../docs/STAGE5_BUSINESS_REPORT.md) · [Danh mục biểu đồ](../reports/FIGURE_CATALOG.md)."),
            ("code", 'display(table("daily_sales")[["date", "actual_sales", "random_forest_forecast", "seasonal_naive_forecast", "rf_error"]])\nfigure(14)\nfigure(20)'),
            ("code", 'for name in ("state_analysis", "store_analysis", "category_analysis", "department_analysis"):\n    group = table(name)\n    display(group[[group.columns[0], "actual_sales", "rf_wape_pct", "sn_wape_pct", "rf_bias_pct"]])\nfigure(17)\nfigure(18)\nfigure(19)'),
            ("md", "Bốn chuỗi minh họa được chọn theo lượng bán, mức thiếu, mức thừa và contribution. Nếu trùng, lấy chuỗi tiếp theo trong tiêu chí. SNAP là so sánh mô tả ngày theo bang. Lượng bán quan sát không đủ để suy luận tồn kho hay quan hệ nhân quả."),
            ("code", 'display(table("representative_series"))\ndisplay(table("top_wrmsse_contribution")[["item_id", "store_id", "rf_contribution", "rf_rmsse"]].head(10))\nfigure(23)\nfigure(24)\nfigure(13)'),
        ]),
    ]


def presentation_source(name, title, sections):
    intro = (
        f"# {title}\n\n"
        "Mở từ thư mục dự án với kernel của môi trường dự án. Các ô đọc config, bảng và hình đã công bố."
    )
    make = {"md": nbf.v4.new_markdown_cell, "code": nbf.v4.new_code_cell}
    notebook = nbf.v4.new_notebook(
        cells=[make["md"](intro), make["code"](BOOTSTRAP), *[make[kind](source) for kind, source in sections]],
        metadata={"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "wrmsse_mode": "read_only_artifacts",
                  "wrmsse_notebook": name},
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {"fit", "partial_fit", "run", "Popen", "dump", "to_csv", "to_parquet"}
    nbf.validate(notebook)
    return notebook


def build_notebooks(execute=True):
    paths = []
    with TemporaryDirectory(prefix="wrmsse_stage5_kernel_") as directory:
        kernel_root = Path(directory)
        spec = kernel_root / "stage5-readonly"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}",
                     "--HistoryManager.enabled=False"],
            "display_name": "Stage 5 read-only", "language": "python",
            "env": {"IPYTHONDIR": str(kernel_root / "ipython")},
        }), encoding="utf-8")
        for name, title, sections in cells():
            notebook = presentation_source(name, title, sections)
            if execute:
                manager = KernelManager(kernel_name="stage5-readonly",
                                        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
                NotebookClient(notebook, km=manager, timeout=75, resources={"metadata": {"path": str(ROOT)}}).execute()
                notebook.metadata.kernelspec.name = "python3"
                notebook.metadata.kernelspec.display_name = "Python 3 (ipykernel)"
                nbf.validate(notebook)
            path = ROOT / "notebooks" / f"{name}.ipynb"
            with path.open("x", encoding="utf-8") as stream:
                nbf.write(notebook, stream)
            paths.append(path)
            print(f"Notebook verified: {path.name}", flush=True)
    return paths
