"""Score the saved Validation predictions, render the report, and retire transient matrices."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from PIL import Image

from run import HERE, LABELS, REF, REQUEST, WORK, Dataset, check_future_invariance, context, protected, read, score, sha, write

COLORS = ["#4e6783", "#007c74"]
MODELS = ["FE-04", "FE-11"]
GROUPS = ["frequent", "intermittent"]
MEASURES = {"mae":"MAE (đơn vị)", "rmse":"RMSE (đơn vị)", "wape_pct":"WAPE (%)", "bias_pct":"Bias (%)"}
LEVELS = ["Toàn hệ thống", "Bang", "Cửa hàng", "Ngành", "Bộ phận", "Bang × ngành", "Bang × bộ phận",
          "Cửa hàng × ngành", "Cửa hàng × bộ phận", "Sản phẩm", "Sản phẩm × bang", "Sản phẩm × cửa hàng"]


def metrics(actual, forecast):
    error = forecast-actual
    total = actual.sum(dtype=np.float64)
    return {"observations":actual.size, "actual_units":int(total), "forecast_units":float(forecast.sum()),
            "mae":np.abs(error).mean(), "rmse":np.sqrt(np.mean(error**2)),
            "wape_pct":100*np.abs(error).sum()/total if total else np.nan,
            "bias_pct":100*error.sum()/total if total else np.nan}


def compare(pred_path):
    ds = Dataset(REQUEST["dataset"])
    ev = context(ds)
    keys = ["item_id", "store_id"]
    labels = pd.read_csv(LABELS/"demand_groups_origin_1885.csv").set_index(keys).loc[ds.index,"demand_group"].to_numpy()
    actual = np.asarray(ds.actual,dtype=float)
    summary, grouped, levels = [], [], []
    for model, path, runtime in [(MODELS[0], REF/"validation_predictions.npy",read(REF/"result.json")),
                                 (MODELS[1], pred_path,read(HERE/"runtime.json"))]:
        pred = np.load(path,allow_pickle=False)
        assert pred.shape == actual.shape == (30490,28) and np.isfinite(pred).all() and (pred >= 0).all()
        scores = score(ds,ev,pred)
        level = scores.per_level.reset_index().assign(model=model)
        assert level.level.tolist() == list(range(1,13))
        np.testing.assert_allclose(level.contribution.sum(),scores.wrmsse,atol=1e-14)
        levels.append(level)
        row = {"model":model,"logical_features":15 if model=="FE-04" else 16,
               "encoded_features":3082 if model=="FE-04" else 3083,"wrmsse":scores.wrmsse,
               **metrics(actual,pred),**{k:runtime[k] for k in ("fit_seconds","predict_seconds","peak_rss_mib")}}
        summary.append(row)
        bottom = scores.per_series.query("level == 12").copy()
        bottom.index = pd.MultiIndex.from_tuples(list(map(tuple,bottom.key)),names=keys)
        bottom = bottom.loc[ds.index]
        for group in GROUPS:
            mask = labels==group
            part = bottom.loc[mask]
            grouped.append({"model":model,"demand_group":group,"series":int(mask.sum()),
                            **metrics(actual[mask],pred[mask]),"mean_rmsse":part.rmsse.mean(),
                            "weighted_rmsse_l12_within_group":part.contribution.sum()/part.weight.sum()})
        np.testing.assert_allclose(sum(g["forecast_units"] for g in grouped if g["model"]==model),row["forecast_units"])
    summary = pd.DataFrame(summary).set_index("model")
    baseline=summary.loc["FE-04","wrmsse"]
    assert abs(baseline-read(REF/"result.json")["wrmsse"])<1e-14
    summary["wrmsse_reduction"] = baseline-summary.wrmsse
    summary["wrmsse_improvement_pct"] = 100*summary.wrmsse_reduction/baseline
    return summary,pd.concat(levels,ignore_index=True),pd.DataFrame(grouped)


def save(fig,name):
    target=HERE/(name+".png")
    fig.savefig(target,dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    with Image.open(target) as im:
        im.verify()


def charts(s,levels,groups,importance):
    settings={"axes.spines.top":False,"axes.spines.right":False,"axes.axisbelow":True,"font.size":10}
    with plt.rc_context(settings):
        fig,ax=plt.subplots(figsize=(7.4,4.1),layout="constrained")
        bars=ax.bar(MODELS,s.wrmsse,color=COLORS,width=.52)
        ax.bar_label(bars,fmt="%.7f",padding=4)
        ax.set(title="WRMSSE Validation · 12 cấp M5",ylabel="WRMSSE (thấp hơn tốt hơn)",ylim=(0,s.wrmsse.max()*1.18))
        ax.grid(axis="y",alpha=.18)
        save(fig,"01_wrmsse")
        fig,(ax,delta)=plt.subplots(1,2,figsize=(12,6.8),layout="constrained",sharey=True,width_ratios=[2.2,1])
        p=levels.pivot(index="level",columns="model",values="wrmsse_within_level")
        y=np.arange(12)
        for i,m in enumerate(MODELS):
            ax.barh(y+(i-.5)*.35,p[m],height=.33,color=COLORS[i],label=m)
        ax.set(yticks=y,yticklabels=[f"{i+1}. {name}" for i,name in enumerate(LEVELS)],
               xlabel="RMSSE có trọng số trong cấp (12 × đóng góp)")
        ax.invert_yaxis()
        ax.legend(frameon=False)
        change=p["FE-04"]-p["FE-11"]
        delta.barh(y,change,color=np.where(change>=0,COLORS[1],"#ad573d"))
        delta.axvline(0,color="#384653",lw=1)
        delta.set_xlabel("FE-04 − FE-11\nDương = FE-11 tốt hơn")
        for a in (ax,delta):
            a.grid(axis="x",alpha=.18)
        fig.suptitle("Validation · so sánh từng cấp phân cấp")
        save(fig,"02_levels")
        fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout="constrained")
        for ax,(metric,label) in zip(axes.flat,MEASURES.items()):
            p=groups.pivot(index="demand_group",columns="model",values=metric).loc[GROUPS]
            for i,m in enumerate(MODELS):
                bars=ax.bar(np.arange(2)+(i-.5)*.38,p[m],width=.36,color=COLORS[i],label=m)
                ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=9)
            ax.set_xticks([0,1],["Frequent","Intermittent"])
            ax.set_title(label)
            ax.axhline(0,lw=.6,color="#384653")
            ax.margins(y=.22)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc="outside lower center",frameon=False,ncols=2)
        fig.suptitle("Sai số sản phẩm–cửa hàng–ngày · nhãn tại d_1885")
        save(fig,"03_demand_groups")
        fig,ax=plt.subplots(figsize=(8.6,6.3),layout="constrained")
        imp=importance.sort_values()
        bars=ax.barh(imp.index,100*imp,color=[COLORS[1] if k=="ADI" else COLORS[0] for k in imp.index])
        ax.bar_label(bars,fmt="%.2f%%",padding=4,fontsize=8)
        ax.margins(x=.15)
        ax.set(title="FE-11 · tầm quan trọng của 16 đặc trưng logic",xlabel="Tỷ trọng mức giảm impurity (%)")
        ax.grid(axis="x",alpha=.18)
        save(fig,"04_importance")


def vi(n,digits=3):
    return f"{n:,.{digits}f}".translate(str.maketrans(".,",",."))


def report(s,levels,groups,importance,pre,unchanged,ram):
    a,b=s.loc["FE-04"],s.loc["FE-11"]
    won=b.wrmsse<a.wrmsse
    fit_delta=(b.fit_seconds/a.fit_seconds-1)*100
    p=levels.pivot(index="level",columns="model",values="wrmsse_within_level")
    reduction=p["FE-04"]-p["FE-11"]
    rank=int(importance.rank(ascending=False,method="min")["ADI"])
    demand=groups.set_index(["model","demand_group"])
    decision=("FE-11 đạt WRMSSE Validation thấp hơn. Giữ kết quả ứng viên để người dùng quyết định có sử dụng FE-11."
              if won else "ADI chưa cải thiện WRMSSE Validation trong điều kiện thực nghiệm này. Đề xuất loại FE-11 và tiếp tục dùng RF-15 + FE-04.")
    parts=["# Thực nghiệm FE-11 · thêm ADI vào RF-15 + FE-04",decision,
           f"WRMSSE FE-04 là **{vi(a.wrmsse,10)}**, FE-11 là **{vi(b.wrmsse,10)}**. "
           f"Chênh theo chiều FE-04 trừ FE-11 là **{vi(b.wrmsse_reduction,10)}**, tương ứng **{vi(b.wrmsse_improvement_pct,4)}%** của điểm FE-04.",
           "## Thiết kế và kiểm tra lịch sử",
           "Dùng toàn bộ 600.000 khóa Train đã niêm phong và 853.720 dòng Validation d_1886–d_1913. "
           "Cùng RF-15, 150 cây, độ sâu 20, max_features và max_samples 0,5, min_samples_leaf 5, bootstrap bật, seed 42, 8 luồng. "
           "Đã tái tạo và khớp SHA-256 của hai ma trận RF-v1, hai cột SNAP và nhãn Train với bản gốc. "
           "FE-11 nối duy nhất cột số ADI sau 3.082 cột mã hóa. WRMSSE của dự báo FE-04 đã lưu khớp lại cả 12 cấp với cùng trọng số và scale Validation.",
           "Với ngày đích t, đặt c = t − 28. Nếu đã bán, ADI = (c − ngày bán dương đầu tiên + 1) / số ngày bán dương đến hết c. "
           "Quy ước này loại thời gian trước lần bán đầu khỏi tử số và khớp ADI trong hai tệp gốc. "
           "Nếu chưa bán thì tỷ số không xác định, mã hóa bằng 0 ở cả hai tập. Mọi ADI hữu hạn có bán đều ≥ 1, nên mã này phân biệt được bằng một cột.",
           f"Đã đối chiếu từng dòng với phép đếm độc lập bằng tìm kiếm nhị phân trên ngày bán dương. "
           f"Có {vi(pre['adi']['train']['no_positive_rows'],0)} dòng Train và {vi(pre['adi']['validation']['no_positive_rows'],0)} dòng Validation dùng mã 0. "
           "Cutoff của d_1000, d_1886 và d_1913 lần lượt là d_972, d_1858 và d_1885. "
           "Trong phép thử tại bốn ngày đích, thay toàn bộ lượng bán sau cutoff bằng 99 vẫn thu đúng ADI mong đợi, kể cả khi lần bán đầu rơi đúng cutoff. "
           "ADI của toàn bộ 30.490 chuỗi ở ngày đích d_1913 khớp nhãn số tại gốc d_1885 sau khi chuyển float32. "
           "Tệp gốc d_1913 chỉ dùng để xác nhận công thức. Hai nhóm dùng để phân tích sau dự báo được cố định tại d_1885. [Kết quả đối chiếu](preflight.json)",
           "## Kết quả Validation",
           "| Chỉ tiêu | FE-04 | FE-11 |\n|---|---:|---:|\n"+"\n".join(
               f"| {label} | {vi(a[k],d)} | {vi(b[k],d)} |" for k,label,d in [
               ("wrmsse","WRMSSE 12 cấp",10),("mae","MAE (đơn vị)",5),("rmse","RMSE (đơn vị)",5),
               ("wape_pct","WAPE (%)",4),("bias_pct","Bias (%)",4),("fit_seconds","Fit (giây)",2),
               ("predict_seconds","Dự báo (giây)",2),("peak_rss_mib","RAM đỉnh tiến trình RF (MiB)",2)]),
           "MAE và RMSE tính trên từng dòng chuỗi–ngày. WAPE = 100 × Σ|F−A| / ΣA, Bias = 100 × Σ(F−A) / ΣA. "
           "Chênh WRMSSE tương đối = 100 × (FE-04 − FE-11) / FE-04. [Bảng tổng hợp](comparison.csv)",
           "![WRMSSE hai cấu hình](01_wrmsse.png)",
           f"FE-11 có điểm trong cấp thấp hơn ở **{int((reduction>0).sum())}/12 cấp**. "
           f"Chênh FE-04 trừ FE-11 lớn nhất tại cấp {reduction.idxmax()} ({vi(reduction.max(),5)}), "
           f"nhỏ nhất tại cấp {reduction.idxmin()} ({vi(reduction.min(),5)}). Mỗi cấp chiếm 1/12 tổng trọng số. "
           "Bảng kèm cả điểm chuẩn hóa trong cấp và đóng góp vào WRMSSE chung. [Bảng 12 cấp](levels.csv)",
           "![Điểm theo cấp](02_levels.png)",
           "## Frequent và Intermittent",
           "| Nhóm | Cấu hình | Số chuỗi | MAE | RMSE | WAPE (%) | Bias (%) |\n|---|---|---:|---:|---:|---:|---:|\n"+"\n".join(
               f"| {r.demand_group.title()} | {r.model} | {vi(r.series,0)} | {vi(r.mae,4)} | {vi(r.rmse,4)} | {vi(r.wape_pct,3)} | {vi(r.bias_pct,3)} |"
               for r in groups.sort_values(["demand_group","model"]).itertuples())]
    for group in GROUPS:
        old,new=(demand.loc[m,group] for m in MODELS)
        parts.append(f"Ở {group.title()}, WAPE {'giảm' if new.wape_pct<old.wape_pct else 'tăng'} "
                     f"{vi(abs(new.wape_pct-old.wape_pct),3)} điểm phần trăm. Bias chuyển từ {vi(old.bias_pct,3)}% "
                     f"sang {vi(new.bias_pct,3)}%, tương ứng dự báo {'thiếu' if new.bias_pct<0 else 'thừa'} tổng lượng bán của nhóm.")
    parts+=["Tệp nhóm cũng ghi RMSSE trung bình cấp chuỗi và Weighted RMSSE cấp 12 với trọng số chuẩn hóa trong nhóm. [Bảng theo nhóm](groups.csv)",
            "![Sai số theo nhóm](03_demand_groups.png)","## Giá trị bổ sung của ADI và chi phí",
            f"ADI chiếm **{vi(importance['ADI']*100,3)}%** tổng mức giảm impurity, đứng thứ **{rank}/16**. "
            f"Các lag chiếm {vi(100*importance.filter(like='lag_').sum(),2)}% và hai rolling mean chiếm "
            f"{vi(100*importance.filter(like='rolling_mean_').sum(),2)}%. "
            "ADI mô tả tần suất bán trong toàn bộ lịch sử hoạt động, còn lag và rolling mean thể hiện lượng bán gần ngày dự báo. "
            "Importance mô tả cách cây dùng các đặc trưng có thể tương quan. Giá trị dự báo tăng thêm được đánh giá bằng chênh WRMSSE "
            "của hai cấu hình trên cùng cửa sổ. Một gốc Validation và một seed chỉ cho phép kết luận trong phạm vi lần so sánh này. [Bảng importance](importance.csv)",
            "![Tầm quan trọng đặc trưng](04_importance.png)",
            f"Thời gian fit {'tăng' if fit_delta>=0 else 'giảm'} {vi(abs(fit_delta),2)}% so với lượt FE-04 đã lưu. "
            f"Chi phí dự báo thay đổi {vi(b.predict_seconds-a.predict_seconds,2)} giây và RAM đỉnh thay đổi {vi(b.peak_rss_mib-a.peak_rss_mib,2)} MiB. "
            "Đây là hai lần chạy khác thời điểm. RAM là peak working set của tiến trình RF trên Windows, gồm cả dữ liệu và thư viện trong tiến trình. "
            "FE-11 tách chuẩn bị và chấm điểm sang tiến trình khác. Thời gian fit và dự báo chỉ đo hai thao tác tương ứng. "
            f"Đỉnh riêng của tiến trình chuẩn bị là {vi(ram['preparation_mib'],2)} MiB, chấm điểm và vẽ hình là "
            f"{vi(ram['evaluation_mib'],2)} MiB. Các đỉnh riêng có thể xuất hiện ở những thời điểm khác nhau.",
            "## Quyết định và tái lập",decision+" Mô hình chính thức vẫn là RF-15 + FE-04. Một lượt fit bổ sung, chỉ dự báo Validation.",
            f"Đã đối chiếu SHA-256 trước và sau trên {unchanged['files']} tệp dự án chính thức. "
            "Các gate, model, selected_model.json, dữ liệu, notebook, Excel, hình, báo cáo và artifact chính thức khớp bản trước thực nghiệm. "
            "Chỉ đọc byte của artifact Test để kiểm tra toàn vẹn. [Biên bản](verification.json)",
            "Để tái lập trong checkout tương đương, đặt mã và config vào thư mục thực nghiệm mới cùng cấp rồi chạy "
            "`.venv/Scripts/python.exe -B experiments/<thư_mục_mới>/run.py` từ gốc dự án. "
            "Dấu mốc fit_started.json ngăn huấn luyện lần hai trong cùng thư mục. [Cấu hình](config.json) · [Mã huấn luyện](run.py) · [Mã báo cáo](report.py)"]
    (HERE/"REPORT.md").write_text("\n\n".join(parts)+"\n",encoding="utf-8")


def main():
    check_future_invariance()
    pred=WORK/"prediction.npy"
    if not pred.is_file():
        pred=HERE/"validation_predictions.npy"
    pre=read(HERE/"preflight.json")
    s,l,g=compare(pred)
    imp=pd.read_csv(HERE/"importance.csv").set_index("feature").importance
    assert len(imp)==16 and set(imp.index)==set(pre["logical_features"])
    s.to_csv(HERE/"comparison.csv")
    l.to_csv(HERE/"levels.csv",index=False)
    g.to_csv(HERE/"groups.csv",index=False)
    charts(s,l,g,imp)
    after=protected()
    assert after==pre["protected_before"], "Official project fingerprint changed"
    won=s.at["FE-11","wrmsse"]<s.at["FE-04","wrmsse"]
    process=psutil.Process()
    # The launcher can add a small interpreter shim between report.py and run.py on Windows.
    prep_peak=pre.get("preparation_peak_mib",0)
    for parent in process.parents():
        try:
            if any(arg.endswith("run.py") for arg in parent.cmdline()):
                prep_peak=max(prep_peak,parent.memory_info().peak_wset/2**20)
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    ram={"evaluation_mib":process.memory_info().peak_wset/2**20,
         "rf_worker_mib":read(HERE/"runtime.json")["peak_rss_mib"],
         "preparation_mib":prep_peak}
    report(s,l,g,imp,pre,after,ram)
    pred_hash=sha(pred)
    if won and pred.parent==WORK:
        (WORK/"model.joblib").replace(HERE/"validation_model.joblib")
        pred.replace(HERE/"validation_predictions.npy")
    for name in ("train.npz","validation.npz","y.npy","prediction.npy","model.joblib"):
        (WORK/name).unlink(missing_ok=True)
    if WORK.exists():
        WORK.rmdir()
    write("verification.json",{"status":"passed","official_before":pre["protected_before"],"official_after":after,
          "fit_count":read(HERE/"runtime.json")["fit_count"],"validation_prediction_sha256":pred_hash,
          "validation_rows":853720,"future_perturbation_target_days":[57,1000,1886,1913],"groups_origin":1885,"official_model":"RF-15 + FE-04",
          "candidate_wrmsse_lower":bool(won),"peak_working_sets_mib":ram,"test_scored":False,"official_gates_modified":False})
    print(s.to_string(),flush=True)


if __name__=="__main__":
    main()
