"""Dựng báo cáo chính theo WRMSSE 12 cấp từ điểm đã chấm; không huấn luyện/dự báo."""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS={"C6":"#4477AA","C7":"#CC6677","actual":"#333333"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir",type=Path,default=Path("results/metrics/wrmsse_C0_C7_20261124"))
    args=parser.parse_args()
    source=(ROOT/args.scores_dir).resolve()
    manifest=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="complete" and manifest["validation_scored"]==8
    allscores=pd.read_csv(source/"tong_hop.csv",float_precision="round_trip")
    # Keep historical C0-C7 figures separate from the selected C7_EVENTS report.
    report=ROOT/"reports";figures=report/"archive"/source.name/"bieu_do";levelsdir=report/"wrmsse_12_cap"
    figures.mkdir(parents=True,exist_ok=True);levelsdir.mkdir(exist_ok=True)
    for split,model in zip(allscores.ky,allscores.cau_hinh):
        levels=pd.read_csv(source/f"{split}_{model}/diem_12_cap.csv",float_precision="round_trip")
        assert len(levels)==12
        value=allscores.loc[allscores.ky.eq(split)&allscores.cau_hinh.eq(model),"WRMSSE"].item()
        np.testing.assert_allclose(value,levels.weighted_RMSSE.mean(),rtol=0,atol=1e-12)
        shutil.copy2(source/f"{split}_{model}/diem_12_cap.csv",levelsdir/f"{split}_{model}.csv")
    allscores["bang_12_cap"]=[f"wrmsse_12_cap/{s}_{m}.csv" for s,m in zip(allscores.ky,allscores.cau_hinh)]
    validation=allscores.loc[allscores.ky.eq("validation")].sort_values("WRMSSE").copy()
    validation.insert(0,"hang",range(1,len(validation)+1))
    winner=validation.iloc[0].cau_hinh
    assert winner==manifest["representative"]
    validation["dai_dien"]=validation.cau_hinh.eq(winner)
    validation.to_csv(report/"xep_hang_wrmsse_validation.csv",index=False)
    allscores["dai_dien"]=allscores.cau_hinh.eq(winner)
    allscores.to_csv(report/"ket_qua_wrmsse_validation_test.csv",index=False)
    selected=allscores.loc[allscores.cau_hinh.eq(winner)].copy()
    selected["ten_bao_cao"]="Bản cuối"
    selected.to_csv(report/"ket_qua_cuoi.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.savefig(figures/name,dpi=220,bbox_inches="tight",facecolor="white");plt.close(fig)
    history=validation.sort_values("cau_hinh")
    labels=[f"{m} (bản cuối)" if m==winner else m for m in history.cau_hinh]
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(labels,history.WRMSSE,"o-",color="#4477aa")
    ax.scatter(labels,list(history.WRMSSE),c=["#cc6677" if m==winner else "#4477aa" for m in history.cau_hinh],s=65,zorder=3)
    for i,v in enumerate(history.WRMSSE):ax.annotate(f"{v:.6f}",(i,v),xytext=(0,9),textcoords="offset points",ha="center",fontsize=10)
    ax.set(title="Lịch sử WRMSSE 12 cấp M5 · thấp hơn là tốt hơn\nValidation d_1886–d_1913 · origin 1885 · cùng 30.490 chuỗi gốc",ylabel="WRMSSE chung (trung bình 12 điểm cấp)")
    ax.margins(y=.2);ax.grid(axis="y",alpha=.2);fig.tight_layout()
    save(fig,"lich_su_wrmsse_validation.png")
    fig,ax=plt.subplots(figsize=(12,5))
    ax.bar(labels,history.ty_le_du_thieu_pct,color=["#b85450" if v>0 else "#4477aa" for v in history.ty_le_du_thieu_pct])
    ax.axhline(0,color="black",lw=1)
    for i,v in enumerate(history.ty_le_du_thieu_pct):ax.annotate(f"{v:+.2f}%",(i,v),xytext=(0,6 if v>=0 else -16),textcoords="offset points",ha="center")
    ax.set(title="Sai lệch tổng lượng validation · chỉ tiêu phân tích bổ sung\nd_1886–d_1913 · 10 cửa hàng",ylabel="100 × (dự báo − thực tế) / thực tế (%)")
    ax.margins(y=.22);fig.tight_layout();save(fig,"lich_su_sai_lech_validation.png")
    comparisons=allscores.loc[allscores.cau_hinh.isin(["C6","C7"])]
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,split,title in zip(axes,["validation","test"],["Validation d_1886–d_1913","Test d_1914–d_1941"]):
        part=comparisons.loc[comparisons.ky.eq(split)].set_index("cau_hinh")
        for i,m in enumerate(["C6","C7"]):
            value=part.loc[m,"WRMSSE"]
            ax.scatter(i,value,color=COLORS[m],marker="o" if m=="C6" else "D",s=100)
            ax.annotate(f"{value:.6f}",(i,value),xytext=(0,10),textcoords="offset points",ha="center")
        delta=part.loc["C7","WRMSSE"]-part.loc["C6","WRMSSE"]
        ax.set(title=title+f"\nΔ (C7 − C6) = {delta:+.6f}",ylabel="WRMSSE chung · thấp hơn tốt hơn",xlim=(-.5,1.5))
        ax.set_xticks([0,1],["C6","C7"]);ax.margins(y=.35);ax.grid(axis="y",alpha=.2)
    fig.suptitle("C6 và C7 · cùng bộ chấm 12 cấp M5",fontsize=14);fig.tight_layout(rect=(0,0,1,.92));save(fig,"c6_c7_wrmsse_validation_test.png")
    test=allscores.loc[allscores.ky.eq("test")]
    if winner in set(test.cau_hinh):
        daily=pd.read_csv(source/f"test_{winner}/tong_luong_ngay.csv",float_precision="round_trip")
        x=daily.target_day.str[2:].astype(int)
        summary=test.loc[test.cau_hinh.eq(winner)].iloc[0]
        np.testing.assert_allclose([daily.actual.sum(),daily.predicted.sum()],[summary.tong_thuc_te,summary.tong_du_bao],rtol=0,atol=1e-8)
        fig,ax=plt.subplots(figsize=(12,5))
        ax.plot(x,daily.actual,label="Thực tế",color=COLORS["actual"],marker="o",ms=3,lw=2)
        ax.plot(x,daily.predicted,label=f"Bản cuối ({winner})",color=COLORS.get(winner,"#cc6677"),marker="D",ms=3,lw=2)
        ticks=x.iloc[::3];ax.set_xticks(ticks,[f"d_{d}" for d in ticks])
        ax.set(title=f"Bản cuối ({winner}): thực tế và dự báo 28 ngày Test\nd_1914–d_1941 · tổng lượng toàn hệ thống · 10 cửa hàng",xlabel="Ngày Test",ylabel="Lượng bán theo ngày (đơn vị)")
        ax.text(.99,.97,f"WRMSSE 12 cấp: {summary.WRMSSE:.6f}\nSai lệch tổng: {summary.ty_le_du_thieu_pct:+.2f}%",transform=ax.transAxes,ha="right",va="top",bbox=dict(facecolor="white",edgecolor="#ddd"))
        ax.legend(loc="upper left",frameon=False);ax.grid(alpha=.2);fig.tight_layout();save(fig,"ban_cuoi_test_28_ngay.png")
    fig,ax=plt.subplots(figsize=(11,7))
    names=["Toàn hệ thống","Bang","Cửa hàng","Ngành hàng","Nhóm hàng","Bang × ngành","Bang × nhóm","Cửa hàng × ngành","Cửa hàng × nhóm","Sản phẩm","Bang × sản phẩm","Cửa hàng × sản phẩm"]
    for model,offset in [("C6",-.13),("C7",.13)]:
        level=pd.read_csv(source/f"test_{model}/diem_12_cap.csv",float_precision="round_trip")
        ax.scatter(level.weighted_RMSSE,np.arange(12)+offset,label=model,color=COLORS[model],marker="o" if model=="C6" else "D",s=60)
    ax.set_yticks(np.arange(12),[f"{i+1}. {n}" for i,n in enumerate(names)]);ax.invert_yaxis()
    ax.set(title="Phân tích 12 cấp M5 · Test d_1914–d_1941",xlabel="Điểm cấp có trọng số (chưa chia 12) · thấp hơn tốt hơn")
    ax.legend(frameon=False);ax.grid(axis="x",alpha=.2);fig.tight_layout();save(fig,"c6_c7_diem_12_cap_test.png")
    # Bảng phụ tổng lượng theo cửa hàng, không gắn WRMSSE của toàn M5 vào từng cửa hàng.
    bystore=pd.read_csv(ROOT/"results/analysis/final_test_C6_C7_frozen/comparison_C6_C7/02_by_store.csv",float_precision="round_trip")
    bystore[["store_id","origin","actual","predicted_C6","predicted_C7","bias_C6_pct","bias_C7_pct"]].to_csv(report/"sai_lech_test_theo_cua_hang.csv",index=False)
    f=bystore.sort_values("bias_C7_pct");fig,ax=plt.subplots(figsize=(11,6));y=np.arange(10)
    for model in ["C6","C7"]:ax.scatter(f[f"bias_{model}_pct"],y,color=COLORS[model],marker="o" if model=="C6" else "D",label=model,s=65)
    ax.set_yticks(y,f.store_id);ax.invert_yaxis();ax.axvline(0,color="black",lw=1)
    ax.set(title="Sai lệch tổng lượng Test theo cửa hàng · phân tích bổ sung\nd_1914–d_1941 · mỗi cửa hàng 3.049 chuỗi",xlabel="Sai lệch tổng lượng (%) · thiếu (−), dư (+)")
    ax.legend(frameon=False);ax.grid(axis="x",alpha=.2);fig.tight_layout();save(fig,"c6_c7_sai_lech_test_theo_cua_hang.png")
    print(validation[["hang","cau_hinh","WRMSSE"]].to_string(index=False))
    print(f"Cấu hình đại diện: {winner}; không đưa RMSSE trung bình không trọng số vào bảng/hình chính.")


if __name__=="__main__":main()
