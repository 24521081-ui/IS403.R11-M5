"""Chấm đồng nhất C0–C7 validation và C6/C7 Test bằng bộ chấm 12 cấp đã có."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from score_c7_wrmsse import score
from score_model_wrmsse import sha256

ROOT=Path(__file__).resolve().parents[1]
STORES=["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"]
KEY=["item_id","store_id","target_day"]


def write_json(obj,path):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def discover():
    cases=[]
    metrics=ROOT/"results/metrics"
    for i in range(7):
        manifest_name="validation_manifest.json" if i==0 else "manifest.json"
        matches=sorted(p for p in metrics.glob(f"validation_C{i}_*10_stores*") if (p/manifest_name).is_file())
        assert len(matches)==1,(i,matches)
        folder=matches[0]
        manifest=folder/("validation_manifest.json" if i==0 else "manifest.json")
        obj=json.loads(manifest.read_text(encoding="utf-8-sig"))
        assert obj["status"]=="completed" and obj.get("origin_day",obj.get("validation_origin"))==1885
        assert obj.get("target_days",obj.get("validation_days"))==[1886,1913]
        if i==6:
            paths=[ROOT/"results/validation"/f"{folder.name}_{store}"/"lightgbm_two_part.csv" for store in STORES]
        else:
            assert len(obj["sources"])==10 and {s["store_id"] for s in obj["sources"]}==set(STORES)
            paths=[ROOT/"results/validation"/s["run_tag"]/"lightgbm_two_part.csv" for s in sorted(obj["sources"],key=lambda s:s["store_id"])]
        cases.append(dict(model=f"C{i}",split="validation",origin=1885,manifest=manifest,files=paths))
    v=ROOT/"results/analysis/stage_C7_validation_d1885_lambda075"
    obj=json.loads((v/"manifest.json").read_text(encoding="utf-8-sig"))
    assert obj["status"]=="complete" and obj["origin"]==1885 and obj["lambda"]==.75 and obj["horizon"]==[1886,1913]
    cases.append(dict(model="C7",split="validation",origin=1885,manifest=v/"manifest.json",files=[v/"folds"/s/"forecast_C7.csv" for s in STORES]))
    t=ROOT/"results/analysis/final_test_C6_C7_frozen"
    for model,folder,filename in [("C6",t/"folds","forecast_C6.csv"),("C7",t/"C7/folds","forecast_C7.csv")]:
        manifest=t/("manifest_C6.json" if model=="C6" else "comparison_C6_C7/manifest_C6_C7.json")
        obj=json.loads(manifest.read_text(encoding="utf-8-sig"))
        assert obj["status"]=="complete" and obj["origin"]==1913 and obj["test_days"]==[1914,1941]
        if model=="C7":assert obj["lambda_C7"]==.75
        cases.append(dict(model=model,split="test",origin=1913,manifest=manifest,files=[folder/s/filename for s in STORES]))
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    output=(ROOT/args.output_dir).resolve()
    if output.exists():raise FileExistsError(f"Chọn thư mục mới: {output}")
    cases=discover()
    missing=[str(p) for c in cases for p in [c["manifest"],*c["files"]] if not p.is_file()]
    if missing:raise FileNotFoundError("Thiếu dự báo/manifest, chưa chấm: "+"; ".join(missing))
    data=ROOT.parents[1]/"data/raw"
    raw=[data/n for n in ["sales_train_evaluation.csv","calendar.csv","sell_prices.csv"]]
    dataset_hashes={p.name:sha256(p) for p in raw}
    wide=pd.read_csv(raw[0])
    assert len(wide)==30490 and not wide.duplicated(["item_id","store_id"]).any() and set(wide.store_id)==set(STORES)
    calendar=pd.read_csv(raw[1]);assert calendar.d.is_unique
    weeks=set(calendar.loc[calendar.d.isin([f"d_{d}" for d in range(1858,1914)]),"wm_yr_wk"])
    prices=pd.concat([p.loc[p.wm_yr_wk.isin(weeks)] for p in pd.read_csv(raw[2],chunksize=250000)],ignore_index=True)
    assert not prices.duplicated(["item_id","store_id","wm_yr_wk"]).any()
    output.mkdir(parents=True)
    inventory=[]
    for c in cases:
        inventory.append({**{k:v for k,v in c.items() if k not in ["manifest","files"]},"manifest":c["manifest"].relative_to(ROOT).as_posix(),"files":[p.relative_to(ROOT).as_posix() for p in c["files"]],"status":"saved_forecasts_available"})
    write_json(dict(status="running",cases=inventory,dataset_sha256=dataset_hashes,no_training_or_inference=True),output/"manifest.json")
    summary=[];common={}
    for c in cases:
        model,split,origin=c["model"],c["split"],c["origin"]
        folder=output/f"{split}_{model}";folder.mkdir()
        hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in [c["manifest"],*c["files"]]}
        tables=[]
        for store,path in zip(STORES,c["files"]):
            f=pd.read_csv(path,float_precision="round_trip")
            assert len(f)==85372 and f.store_id.eq(store).all()
            assert f.origin_day.eq(f"d_{origin}").all() and f.split.eq(split).all()
            assert f.model.eq("lightgbm_two_part").all()
            day=f.target_day.str[2:].astype(int)
            assert f.horizon.eq(day-origin).all() and set(day)==set(range(origin+1,origin+29))
            tables.append(f)
        forecast=pd.concat(tables,ignore_index=True)
        assert len(forecast)==853720 and not forecast.duplicated(KEY).any()
        assert np.isfinite(forecast.y_pred).all() and forecast.y_pred.ge(0).all()
        # Kiểm tra đủ đúng cặp và 28 ngày, không chỉ so số dòng.
        counts=forecast.groupby(["item_id","store_id"]).size().rename("n").reset_index()
        joined=wide[["item_id","store_id"]].merge(counts,on=["item_id","store_id"],how="outer",validate="one_to_one",indicator=True)
        assert joined._merge.eq("both").all() and joined.n.eq(28).all()
        levels,details=score(origin,wide,calendar,prices,forecast,return_details=True)
        levels=pd.DataFrame(levels).assign(model=model,split=split)
        assert len(levels)==12 and levels.n_series.sum()==42840
        np.testing.assert_allclose(levels.weight_sum,1,rtol=0,atol=1e-12)
        value=float(levels.weighted_RMSSE.mean())
        np.testing.assert_allclose(value,levels.contribution_WRMSSE.sum(),rtol=0,atol=1e-12)
        keys_hash=hashlib.sha256(forecast[KEY].sort_values(KEY).to_csv(index=False).encode()).hexdigest()
        scale_hash=hashlib.sha256(details[["item_id","store_id","scale"]].to_csv(index=False).encode()).hexdigest()
        weight_hash=hashlib.sha256(details[["item_id","store_id","revenue_train_28","revenue_weight"]].to_csv(index=False).encode()).hexdigest()
        checks=dict(keys_sha256=keys_hash,scale_sha256=scale_hash,weights_sha256=weight_hash)
        if split in common:assert checks==common[split],(model,checks,common[split])
        else:common[split]=checks
        details.assign(model=model,split=split).to_csv(folder/"rmsse_tung_chuoi.csv",index=False)
        levels.to_csv(folder/"diem_12_cap.csv",index=False)
        days=[f"d_{d}" for d in range(origin+1,origin+29)]
        daily=forecast.groupby("target_day").y_pred.sum().reindex(days).rename("predicted").to_frame()
        daily["actual"]=wide[days].sum().to_numpy()
        daily.to_csv(folder/"tong_luong_ngay.csv")
        actual=float(daily.actual.sum());predicted=float(daily.predicted.sum())
        # Đối chiếu tổng lượng với kết quả gốc của đúng kỳ, không lấy số trên ảnh.
        if split=="test":
            ref=pd.read_csv(ROOT/"results/analysis/final_test_C6_C7_frozen/comparison_C6_C7/01_by_origin.csv",float_precision="round_trip").iloc[0]
            expected_actual,expected_pred=ref.actual,ref[f"predicted_{model}"]
        elif model=="C0":
            ref=pd.read_csv(c["manifest"].parent/"Totals_all.csv",float_precision="round_trip").query("model == 'lightgbm_two_part'").iloc[0]
            expected_actual,expected_pred=ref.total_actual,ref.total_prediction
        elif model=="C7":
            ref=pd.read_csv(c["manifest"].parent/"01_by_origin.csv",float_precision="round_trip").iloc[0]
            expected_actual,expected_pred=ref.actual,ref.predicted_C7
        else:
            ref=pd.read_csv(c["manifest"].parent/"comparison_overall.csv",float_precision="round_trip").query("configuration == @model").iloc[0]
            expected_actual,expected_pred=ref.actual,ref.prediction
        np.testing.assert_allclose([actual,predicted],[expected_actual,expected_pred],rtol=0,atol=1e-7)
        for name,digest in hashes.items():assert sha256(ROOT/name)==digest,name
        row=dict(cau_hinh=model,ky=split,origin=origin,ngay_bat_dau=f"d_{origin+1}",ngay_ket_thuc=f"d_{origin+28}",so_cua_hang=10,so_chuoi=30490,so_dong=853720,so_cap=12,WRMSSE=value,
                 tong_thuc_te=actual,tong_du_bao=predicted,chenh_lech=predicted-actual,ty_le_du_thieu_pct=100*(predicted/actual-1),trang_thai="Đã chấm đủ",bang_12_cap=folder.relative_to(ROOT).as_posix()+"/diem_12_cap.csv")
        summary.append(row)
        write_json(dict(status="complete",**row,**checks,source_sha256=hashes,dataset_sha256=dataset_hashes,scorer_sha256=sha256(Path(__file__).with_name("score_c7_wrmsse.py")),revenue_days=[origin-27,origin],inference_executed=False),folder/"manifest.json")
        pd.DataFrame(summary).to_csv(output/"tong_hop.csv",index=False)
        print(f"COMPLETE {split} {model}: WRMSSE={value:.12f}; 12 cấp, 853720 dòng",flush=True)
    frame=pd.DataFrame(summary)
    validation=frame.query("ky == 'validation'")
    assert set(validation.cau_hinh)=={f"C{i}" for i in range(8)}
    winner=validation.sort_values("WRMSSE").iloc[0].cau_hinh
    for p in raw:assert sha256(p)==dataset_hashes[p.name]
    write_json(dict(status="complete",cases=inventory,validation_scored=8,test_scored=2,representative=winner,dataset_sha256=dataset_hashes,common_scope=common,
                    no_training_or_inference=True,missing_inputs=[],needs_replay=[],metric_added_after_prior_test_review=True),output/"manifest.json")
    print(f"Đại diện theo WRMSSE validation: {winner}",flush=True)


if __name__=="__main__":main()
