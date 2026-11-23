"""Chấm một mô hình trên Test M5 từ CSV đã lưu, không huấn luyện hay dự báo."""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from score_c7_wrmsse import score, META

ROOT = Path(__file__).resolve().parents[1]
VERSION = "WRMSSE_TEST_1913_V2"
KEYS = ["item_id", "store_id", "target_day"]
STORES = {"CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_forecasts(patterns):
    paths = []
    for pattern in patterns:
        candidate = Path(pattern)
        full = str(candidate if candidate.is_absolute() else ROOT / candidate)
        matches = sorted(Path(p).resolve() for p in glob.glob(full))
        require(bool(matches), f"Không tìm thấy dự báo: {pattern}")
        paths.extend(matches)
    require(len(paths) == len(set(paths)), "Một tệp dự báo được chọn nhiều lần")
    tables = []
    for path in paths:
        table = pd.read_csv(path, float_precision="round_trip")
        require(set(KEYS + ["y_pred"]).issubset(table.columns), f"Thiếu bốn cột chuẩn: {path}")
        for name, expected in [("split", "test"), ("origin_day", "d_1913")]:
            if name in table:
                require(table[name].eq(expected).all(), f"Sai {name}: {path}")
        if "horizon" in table:
            require(table.target_day.str.fullmatch(r"d_\d+").all(), f"Sai mã ngày: {path}")
            require(table.horizon.eq(table.target_day.str[2:].astype(int)-1913).all(), f"Sai horizon: {path}")
        tables.append(table[KEYS + ["y_pred"]])
    forecasts = pd.concat(tables, ignore_index=True)
    require(not forecasts[KEYS].isna().any().any(), "Khóa dự báo có giá trị trống")
    require(not forecasts.duplicated(KEYS).any(), "Trùng khóa item_id-store_id-target_day")
    forecasts["y_pred"] = pd.to_numeric(forecasts.y_pred, errors="raise")
    require(np.isfinite(forecasts.y_pred).all() and forecasts.y_pred.ge(0).all(), "Dự báo phải hữu hạn và không âm")
    require(set(forecasts.target_day) == {f"d_{d}" for d in range(1914,1942)}, "Kỳ Test phải là d_1914-d_1941")
    return forecasts, paths


def check_coverage(forecasts, wide):
    pairs = wide[["item_id", "store_id"]]
    require(len(pairs) == 30490 and not pairs.duplicated().any(), "M5 phải có 30.490 cặp duy nhất")
    require(set(wide.store_id) == STORES and wide.groupby("store_id").size().eq(3049).all(), "Sai tập cửa hàng M5")
    require(len(forecasts) == 853720, "Dự báo phải đủ 853.720 dòng")
    counts = forecasts.groupby(["item_id", "store_id"]).size().rename("n").reset_index()
    joined = pairs.merge(counts, on=["item_id", "store_id"], how="outer", validate="one_to_one", indicator=True)
    require(joined._merge.eq("both").all() and joined.n.eq(28).all(), "Dự báo phải khớp chính xác các cặp M5, đủ 28 ngày/cặp")


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True,help="Mã mô hình để đặt tên thư mục, ví dụ LightGBM_TwoPart_C7")
    parser.add_argument("--forecast",nargs="+",required=True,help="CSV hoặc mẫu glob được đặt trong dấu nháy")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/raw")
    parser.add_argument("--output-dir",type=Path,default=None)
    args=parser.parse_args(argv)
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",args.model)), "Mã mô hình chỉ gồm chữ ASCII, số, _ và -")
    data=args.data_dir if args.data_dir.is_absolute() else ROOT/args.data_dir
    output=args.output_dir or ROOT/".local_evidence/wrmsse_6_models"/args.model
    if not output.is_absolute():output=ROOT/output
    output=output.resolve()
    output_files=[output/n for n in ["diem_chung.csv","diem_theo_cap.csv","rmsse_theo_cap_san_pham.csv","forecast_test.csv","manifest.json"]]
    require(not any(p.exists() for p in output_files), "Thư mục đã có kết quả; chọn --output-dir mới để giữ nguyên kết quả cũ")
    forecasts,paths=load_forecasts(args.forecast)
    raw={n:data/n for n in ["sales_train_evaluation.csv","calendar.csv","sell_prices.csv"]}
    require(all(p.is_file() for p in raw.values()), "Thiếu tệp M5 trong --data-dir")
    require(not any(p==q.resolve() for p in output_files for q in [*paths,*raw.values()]), "Đầu ra trùng tệp đầu vào")
    wide=pd.read_csv(raw["sales_train_evaluation.csv"])
    require(set(META + [f"d_{d}" for d in range(1,1942)]).issubset(wide.columns), "Thiếu metadata hoặc ngày d_1-d_1941 trong M5")
    check_coverage(forecasts,wide)
    calendar=pd.read_csv(raw["calendar.csv"])
    require({"d","wm_yr_wk"}.issubset(calendar.columns) and calendar.d.is_unique, "Lịch phải có d duy nhất và wm_yr_wk")
    require(set(f"d_{d}" for d in range(1886,1914)).issubset(calendar.d), "Thiếu lịch cửa sổ doanh thu")
    weeks=set(calendar.loc[calendar.d.isin([f"d_{d}" for d in range(1886,1914)]),"wm_yr_wk"])
    prices=pd.concat([part.loc[part.wm_yr_wk.isin(weeks)] for part in pd.read_csv(raw["sell_prices.csv"],chunksize=250000)],ignore_index=True)
    require({"item_id","store_id","wm_yr_wk","sell_price"}.issubset(prices.columns), "Thiếu cột giá M5")
    require(not prices.duplicated(["item_id","store_id","wm_yr_wk"]).any(), "Trùng khóa bảng giá")
    require(np.isfinite(prices.sell_price).all() and prices.sell_price.ge(0).all(), "Giá phải hữu hạn và không âm")
    internal=forecasts.assign(origin_day="d_1913",split="test")
    levels,details=score(1913,wide,calendar,prices,internal,return_details=True)
    levels=pd.DataFrame(levels)
    levels.insert(0,"model",args.model)
    details.insert(0,"model",args.model)
    summary=pd.DataFrame([dict(model=args.model,split="test",origin=1913,start_day="d_1914",end_day="d_1941",n_series=30490,n_rows=len(forecasts),WRMSSE=float(levels.contribution_WRMSSE.sum()),scorer_version=VERSION)])
    output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output/"diem_chung.csv",index=False)
    levels.to_csv(output/"diem_theo_cap.csv",index=False)
    details.to_csv(output/"rmsse_theo_cap_san_pham.csv",index=False)
    forecasts.sort_values(KEYS).to_csv(output/"forecast_test.csv",index=False)
    manifest=dict(scorer_version=VERSION,model=args.model,revenue_days=[1886,1913],test_days=[1914,1941],
                  inputs_sha256={str(p.resolve()):sha256(p) for p in [*paths,*raw.values()]},
                  scorer_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name("score_c7_wrmsse.py")]},
                  pandas=pd.__version__,numpy=np.__version__,revenue_weights_sha256=hashlib.sha256(details[["item_id","store_id","revenue_train_28","revenue_weight"]].to_csv(index=False).encode("utf-8")).hexdigest())
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Gửi ba bảng tại: {output}")
    return output


if __name__=="__main__":
    main()
