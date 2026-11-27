"""Export slide-ready SVG/PNG from the experiment's scored CSV tables only."""
from pathlib import Path
import argparse
import hashlib
import os
import json
import shutil
import textwrap
import sys
if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

OUT=ROOT/'results/analysis/c7_events_20261126'
FIG=OUT/'figures'
BLUE='#2864ad'
ORANGE='#e78336'
EXTENSIONS=('svg','png')
FINAL_NAMES={
    '01_so_do_C7':'so_do_two_part',
    '02_doi_chung_validation':'so_sanh_validation',
    '03_test_thuc_te_du_bao':'thuc_te_va_du_bao_test',
    '04_test_nhom_san_pham':'rmsse_theo_nganh_hang',
    '05_test_tan_suat_ban':'rmsse_theo_tan_suat_ban',
    '06_lich_su_cau_hinh':'lich_su_cau_hinh',
    '07_sai_lech_cua_hang':'sai_lech_test_theo_cua_hang',
}
FINAL_REPORT=False
PUBLISHED_REPORT=False

def export(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for ext in EXTENSIONS:
        target=FINAL_NAMES[name] if FINAL_REPORT else name
        fig.savefig(FIG/f'{target}.{ext}',dpi=250 if FINAL_REPORT else 200,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def diagram():
    cfg=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(16,10))
    ax.set(xlim=(0,16),ylim=(0,10))
    ax.axis('off')
    def box(x,y,w,h,text,color='#eaf1f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor=color,edgecolor=BLUE))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(a,b,label=None):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.7))
        if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.13,label,ha='center',fontsize=10)
    ax.text(0,9.6,'LightGBM Two-Part C7 | λ = 0,75',fontsize=23,weight='bold',color=BLUE)
    box(.1,6.6,3.7,1.8,'M5: lịch sử bán + giá quá khứ\ncalendar.csv: lịch / sự kiện / SNAP\nLag 1, 7, 14, 28; rolling 7, 28\nGiá tuần hoàn tất; cố định tại origin')
    box(4.6,7.5,2.9,1.0,'Classifier LightGBM\n300 cây • 48 lá • LR 0,05')
    box(4.6,5.7,2.9,1.0,'Regressor LightGBM\n200 cây • 31 lá • LR 0,05\nChỉ học các ngày sales > 0')
    arrow((3.9,7.9),(4.45,8))
    arrow((3.9,7),(4.45,6.2))
    box(8.3,7.5,2.3,1.0,'Platt: sigmoid\na × logit(p_raw) + b')
    arrow((7.65,8),(8.15,8),'p_raw')
    box(11.5,7.3,4.0,1.25,'p_blend = 0,25 × p_raw\n+ 0,75 × p_cal',color='#fff0e3')
    arrow((10.75,8),(11.35,8),'p_cal')
    ax.plot([7.8,7.8,13.5,13.5],[8.1,8.95,8.95,8.7],color=BLUE,lw=1.4)
    ax.text(10.6,9.02,'p_raw',fontsize=10,color=BLUE)
    box(11.5,5.3,4,1.1,'ŷ = p_blend × E[sales | sales > 0]\nCập nhật lag / rolling bằng ŷ',color='#e8f3eb')
    arrow((13.5,7.15),(13.5,6.55))
    arrow((7.65,6.2),(11.35,5.85),'Lượng bán kỳ vọng khi có bán')
    selected='Chưa chốt'
    if (OUT/'selection.json').exists():
        selected=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected']
    count=14 if selected=='C7' else 48 if selected=='C7_EVENTS' else None
    feature_note=f'{count} đặc trưng / nhánh' if count else 'Hai nhánh dùng cùng danh sách đặc trưng'
    ax.text(.1,4.8,f'Cấu hình đã chốt: {selected} | {feature_note}',fontsize=13,weight='bold')
    ax.text(.1,4.35,'C7 gốc: day_of_week, month, event_flag (có ít nhất một sự kiện), snap của bang; item_id; lag / rolling;\nsell_price, price_change_1w, price_ratio_4w.',fontsize=11,va='top')
    names=[v for k,v in cfg['event_features'].items() if k.startswith('event_name')]
    text='C7_EVENTS bổ sung: 30 chỉ báo tên sự kiện (gộp hai slot, one-hot): '+', '.join(names)+'.'
    ax.text(.1,3.55,textwrap.fill(text,165),fontsize=9.5,va='top',linespacing=1.5)
    ax.text(.1,2.55,'Thêm 4 chỉ báo loại: Cultural, National, Religious, Sporting. Từ điển cố định từ lịch đến d_1857.\nCác biến lịch tương lai lấy từ calendar.csv theo giả định lịch ngoại sinh của M5; không lấy doanh số / giá tương lai.',fontsize=10.5,va='top')
    box(.2,.7,7.2,1.1,'VALIDATION: mô hình phụ học ≤ d_1857\nPlatt: OOF d_1858–d_1885; mô hình chính học ≤ d_1885\nĐánh giá d_1886–d_1913 → chốt theo WRMSSE 12 cấp')
    box(8.1,.7,7.2,1.1,'TEST: mô hình phụ học ≤ d_1885\nPlatt: OOF d_1886–d_1913; mô hình chính học ≤ d_1913\nĐánh giá d_1914–d_1941 sau khi chốt cấu hình')
    export(fig,'01_so_do_C7')

def validation():
    table=pd.read_csv(OUT/'validation_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,column,title in zip(axes,['WRMSSE','weighted_RMSSE_level12'],['WRMSSE chung • 12 cấp M5','RMSSE trọng số doanh thu\nCấp 12: sản phẩm × cửa hàng']):
        ax.bar(table.configuration,table[column],color=[BLUE,ORANGE],width=.55)
        for i,value in enumerate(table[column]): ax.text(i,value+.008,f'{value:.6f}',ha='center',fontsize=12)
        delta=table[column].iloc[1]-table[column].iloc[0]
        ax.set(title=f'{title}\nΔ (thêm sự kiện − gốc) = {delta:+.6f}',ylabel='Điểm (thấp hơn tốt hơn)',ylim=(0,table[column].max()*1.22))
        ax.spines[['top','right']].set_visible(False)
    chosen=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected']
    fig.suptitle(f'Validation d_1886–d_1913 | λ = 0,75 | Chọn {chosen}',fontsize=15,weight='bold')
    fig.text(.5,.025,'Kỳ này không có ngày được gắn nhãn sự kiện; kết quả chưa đánh giá trực tiếp ngày lễ.',
             ha='center',fontsize=10,color='#555555')
    fig.tight_layout(rect=(0,.07,1,.95))
    export(fig,'02_doi_chung_validation')

def test():
    table=pd.read_csv(OUT/'test_daily.csv')
    meta=json.loads((OUT/'test_verified.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(28)
    ax.plot(x,table.actual,label='Tổng thực tế',color=BLUE,lw=2,marker='o',ms=4)
    ax.plot(x,table.predicted,label='Tổng dự báo',color=ORANGE,lw=2,marker='o',ms=4)
    ticks=np.arange(0,28,3)
    ax.set(xticks=ticks,xticklabels=table.target_day.iloc[ticks],ylabel='Tổng lượng bán / ngày',
        title=f"Test d_1914–d_1941 | {meta['selected']} | Tổng 30.490 chuỗi\nSai lệch tổng lượng = {meta['bias_pct']:+.2f}% (dương: dư; âm: thiếu)")
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    export(fig,'03_test_thuc_te_du_bao')

def groups():
    table=pd.read_csv(OUT/'test_group_scores.csv')
    for dimension,name in [('cat_id','04_test_nhom_san_pham'),('demand_group','05_test_tan_suat_ban')]:
        subset=table.loc[table.dimension.eq(dimension)]
        fig,ax=plt.subplots(figsize=(9,5))
        labels=subset.group.replace({'frequent':'Bán thường xuyên','intermittent':'Bán gián đoạn'})
        ax.bar(labels,subset.revenue_weighted_RMSSE,color=BLUE,width=.5)
        for i,row in enumerate(subset.itertuples()):
            ax.text(i,row.revenue_weighted_RMSSE+.015,f'{row.revenue_weighted_RMSSE:.6f}\nn = {row.n_scored:,}\nDư/thiếu {row.bias_pct:+.2f}%',ha='center',fontsize=11)
        ax.set(ylabel='RMSSE có trọng số doanh thu trong từng nhóm',ylim=(0,subset.revenue_weighted_RMSSE.max()*1.35),
            title=f'Phân tích {subset.configuration.iloc[0]} | Test d_1914–d_1941\nCấp 12: sản phẩm × cửa hàng\nNhãn cố định ≤ d_1885; trọng số doanh thu d_1886–d_1913')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        export(fig,name)

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def configuration_history():
    """Chronological configuration milestones, not a ranking of six methods."""
    source=ROOT/'results/metrics/wrmsse_C0_C7_20261124'
    if PUBLISHED_REPORT:
        data=pd.read_csv(ROOT/'reports/lich_su_cau_hinh_validation.csv',float_precision='round_trip')
        assert data.configuration.tolist()==[f'C{i}' for i in range(8)]+['C7_EVENTS']
        assert data.origin.eq(1885).all() and data.days.eq('d_1886-d_1913').all()
        return draw_history(data)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete' and manifest['validation_scored']==8
    table=pd.read_csv(source/'tong_hop.csv',float_precision='round_trip')
    history=table.loc[table.ky.eq('validation')].sort_values('cau_hinh').copy()
    assert history.cau_hinh.tolist()==[f'C{i}' for i in range(8)]
    assert history.origin.eq(1885).all() and history.ngay_bat_dau.eq('d_1886').all()
    assert history.ngay_ket_thuc.eq('d_1913').all() and history.so_chuoi.eq(30490).all()
    for row in history.itertuples():
        levels=pd.read_csv(source/f'validation_{row.cau_hinh}/diem_12_cap.csv',float_precision='round_trip')
        assert len(levels)==12
        assert np.isclose(levels.weighted_RMSSE.mean(),row.WRMSSE,rtol=0,atol=1e-12)
    events=pd.read_csv(OUT/'validation_comparison.csv',float_precision='round_trip').set_index('configuration')
    assert history.loc[history.cau_hinh.eq('C7'),'WRMSSE'].item()==events.loc['C7','WRMSSE']
    data=history[['cau_hinh','WRMSSE']].rename(columns={'cau_hinh':'configuration'})
    data=pd.concat([data,pd.DataFrame([{'configuration':'C7_EVENTS','WRMSSE':events.loc['C7_EVENTS','WRMSSE']}])],ignore_index=True)
    data['split']='validation'
    data['origin']=1885
    data['days']='d_1886-d_1913'
    data['level']='WRMSSE 12 cấp'
    data['selected']=data.configuration.eq('C7_EVENTS')
    data.to_csv(ROOT/'reports/lich_su_cau_hinh_validation.csv',index=False)
    return draw_history(data)


def draw_history(data):
    fig,ax=plt.subplots(figsize=(13,5.5))
    x=np.arange(len(data))
    ax.plot(x,data.WRMSSE,color=BLUE,lw=1.8,zorder=1)
    for i,row in enumerate(data.itertuples()):
        final=row.configuration=='C7_EVENTS'
        brief=row.configuration=='C6'
        color=ORANGE if final else '#888888' if brief else BLUE
        ax.scatter(i,row.WRMSSE,s=190 if final else 30 if brief else 55,
                   marker='*' if final else 'o',color=color,zorder=3)
        ax.annotate(f'{row.WRMSSE:.6f}',(i,row.WRMSSE),xytext=(0,-22 if final else 13),
                    textcoords='offset points',ha='center',fontsize=11 if final else 8 if brief else 10,
                    color=color,weight='bold' if final else 'normal')
    labels=data.configuration.tolist()
    labels[6]='C6\nPlatt'
    labels[-1]='C7_EVENTS\nBẢN CHỐT'
    ax.set_xticks(x,labels)
    ax.get_xticklabels()[-1].set_color(ORANGE)
    ax.get_xticklabels()[-1].set_weight('bold')
    ax.set(title='Lịch sử cấu hình LightGBM Two-Part: C0 → C7 → C7_EVENTS\nValidation d_1886–d_1913 · 30.490 chuỗi · chốt C7_EVENTS theo WRMSSE',
           ylabel='WRMSSE chung 12 cấp M5 (thấp hơn tốt hơn)',xlabel='Thứ tự cấu hình thực nghiệm')
    ax.set_ylim(.53,.83)
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    export(fig,'06_lich_su_cau_hinh')
    return data.to_dict('records')


def store_bias():
    """Aggregate saved C7_EVENTS Test predictions and saved actuals by store."""
    if PUBLISHED_REPORT:
        stores=pd.read_csv(ROOT/'reports/sai_lech_test_C7_EVENTS_theo_cua_hang.csv',float_precision='round_trip')
        assert len(stores)==10 and stores.n_pairs.eq(3049).all()
        assert stores.configuration.eq('C7_EVENTS').all() and stores.days.eq('d_1914-d_1941').all()
        np.testing.assert_allclose(stores.bias_pct,100*(stores.predicted/stores.actual-1),rtol=0,atol=1e-12)
        return draw_store_bias(stores)
    ids=['store_id','item_id']
    pred=pd.read_csv(OUT/'forecast_test_selected.csv',float_precision='round_trip')
    detail=pd.read_csv(OUT/'test_details.csv',float_precision='round_trip')
    assert not detail.duplicated(ids).any()
    stores=pred.groupby('store_id').agg(predicted=('y_pred','sum'),n_pairs=('item_id','nunique'))
    stores['actual']=detail.groupby('store_id').actual_units.sum()
    assert len(stores)==10 and stores.n_pairs.eq(3049).all() and stores.actual.gt(0).all()
    stores['bias_pct']=100*(stores.predicted/stores.actual-1)
    daily=pd.read_csv(OUT/'test_daily.csv',float_precision='round_trip')
    np.testing.assert_allclose([stores.predicted.sum(),stores.actual.sum()],
                               [daily.predicted.sum(),daily.actual.sum()],rtol=0,atol=1e-8)
    stores=stores.reset_index()
    stores['configuration']='C7_EVENTS'
    stores['split']='test'
    stores['origin']=1913
    stores['days']='d_1914-d_1941'
    stores.to_csv(ROOT/'reports/sai_lech_test_C7_EVENTS_theo_cua_hang.csv',index=False)
    return draw_store_bias(stores)


def draw_store_bias(stores):
    fig,ax=plt.subplots(figsize=(11,6))
    ax.barh(stores.store_id,stores.bias_pct,color=[BLUE if v<0 else ORANGE for v in stores.bias_pct])
    ax.invert_yaxis()
    ax.axvline(0,color='#333333',lw=1)
    for i,value in enumerate(stores.bias_pct):
        ax.text(value+(-.12 if value<0 else .12),i,f'{value:+.2f}%',
                va='center',ha='right' if value<0 else 'left',fontsize=10)
    ax.set(title='Phân tích C7_EVENTS: sai lệch tổng lượng theo 10 cửa hàng\nTest d_1914–d_1941 · 3.049 sản phẩm / cửa hàng',
           xlabel='100 × (tổng dự báo / tổng thực tế − 1) (%) · thiếu (−), dư (+)',ylabel='Cửa hàng')
    ax.margins(x=.25)
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    export(fig,'07_sai_lech_cua_hang')
    return stores.to_dict('records')


def verified_sources():
    """Check saved provenance and plotted numbers; no fitting or inference."""
    def read(name):
        return json.loads((OUT/name).read_text(encoding='utf-8'))
    selection=read('selection.json')
    verified=read('test_verified.json')
    cfg=read('config.json')
    assert selection['selected']==verified['selected']==cfg['name']=='C7_EVENTS'
    assert cfg['lambda_blend']==.75 and cfg['validation_days']==[1886,1913]
    assert cfg['test_days']==verified['days']==[1914,1941]
    assert len(cfg['original_features'])==14 and len(cfg['event_features'])==34
    assert sha256(OUT/'config.json')==verified['config_sha256']
    assert sha256(OUT/'forecast_test_selected.csv')==verified['sha256']
    for name,expected in verified['sources'].items():
        assert sha256(Path(name))==expected
    comparison=pd.read_csv(OUT/'validation_comparison.csv',float_precision='round_trip')
    assert comparison.configuration.tolist()==['C7','C7_EVENTS']
    for record in selection['validation']:
        row=comparison.set_index('configuration').loc[record['configuration']]
        for column in ['WRMSSE','weighted_RMSSE_level12']:
            assert row[column]==record[column]
    assert comparison.WRMSSE.iloc[1]<comparison.WRMSSE.iloc[0]
    assert np.isclose(comparison.delta_WRMSSE_vs_C7.iloc[1],comparison.WRMSSE.iloc[1]-comparison.WRMSSE.iloc[0],atol=1e-15,rtol=0)
    calendar=pd.read_csv(ROOT.parents[1]/'data/raw/calendar.csv')
    val_calendar=calendar.loc[calendar.d.isin([f'd_{d}' for d in range(1886,1914)])]
    assert len(val_calendar)==28 and val_calendar[['event_name_1','event_name_2']].isna().all().all()
    pred=pd.read_csv(OUT/'forecast_test_selected.csv',float_precision='round_trip')
    ids=['store_id','item_id']
    assert len(pred)==853720 and not pred.duplicated(ids+['target_day']).any()
    assert pred.store_id.nunique()==10 and pred.groupby(ids).size().eq(28).all()
    assert pred[ids].drop_duplicates().shape[0]==30490
    assert set(pred.target_day)=={f'd_{d}' for d in range(1914,1942)}
    assert np.isfinite(pred.y_pred).all() and pred.y_pred.ge(0).all()
    daily=pd.read_csv(OUT/'test_daily.csv',float_precision='round_trip')
    assert daily.target_day.tolist()==[f'd_{d}' for d in range(1914,1942)]
    totals=pred.groupby('target_day').y_pred.sum().reindex(daily.target_day)
    np.testing.assert_allclose(totals,daily.predicted,rtol=0,atol=1e-8)
    actual=pd.read_csv(ROOT.parents[1]/'data/raw/sales_train_evaluation.csv',usecols=daily.target_day.tolist()).sum()
    np.testing.assert_array_equal(actual.reindex(daily.target_day),daily.actual)
    bias=100*(totals.sum()/daily.actual.sum()-1)
    assert np.isclose(bias,verified['bias_pct'],rtol=0,atol=1e-10)
    labels=pd.read_csv(OUT/'labels/demand_groups_origin_1885.csv')
    details=pd.read_csv(OUT/'test_details.csv',float_precision='round_trip')
    assert len(labels)==len(details)==30490 and not labels.duplicated(ids).any()
    joined=details.merge(labels,on=ids,validate='one_to_one')
    assert len(joined)==30490 and joined.demand_group.notna().all()
    predicted=pred.groupby(ids).y_pred.sum().sort_index()
    np.testing.assert_allclose(predicted,details.set_index(ids).predicted_units.sort_index(),rtol=0,atol=1e-8)
    groups=pd.read_csv(OUT/'test_group_scores.csv',float_precision='round_trip')
    assert len(groups)==5 and groups.configuration.eq('C7_EVENTS').all()
    assert groups.split.eq('test').all() and groups.days.eq('d_1914-d_1941').all() and groups.level.eq(12).all()
    assert set(groups.loc[groups.dimension.eq('cat_id'),'group'])=={'FOODS','HOBBIES','HOUSEHOLD'}
    assert set(groups.loc[groups.dimension.eq('demand_group'),'group'])=={'frequent','intermittent'}
    for row in groups.itertuples():
        subset=joined.loc[joined[row.dimension].eq(row.group)]
        valid=subset.loc[np.isfinite(subset.RMSSE)&np.isfinite(subset.scale)&subset.scale.gt(0)]
        score=np.dot(valid.revenue_train_28/valid.revenue_train_28.sum(),valid.RMSSE)
        assert len(subset)==row.n_pairs and len(valid)==row.n_scored
        assert np.isclose(score,row.revenue_weighted_RMSSE,rtol=0,atol=1e-12)
    return {'selected':'C7_EVENTS','validation':comparison.to_dict('records'),
            'test_bias_pct':bias,'group_scores':groups.to_dict('records'),
            'checks':'PASS: hashes, periods, 28 days/pair, saved daily totals, actuals, historical labels and group scores'}


def final_report():
    """Collect seven report PNGs from saved results, preserving experiments."""
    global FIG,EXTENSIONS,FINAL_REPORT
    from PIL import Image
    protected={p:sha256(p) for p in OUT.rglob('*') if p.is_file()}
    FIG=ROOT/'reports/bieu_do'
    FIG.mkdir(parents=True,exist_ok=True)
    existing={p:sha256(p) for p in FIG.rglob('*') if p.is_file()
              and p.name not in {f'{name}.png' for name in FINAL_NAMES.values()}}
    evidence=verified_sources()
    EXTENSIONS=('png',)
    FINAL_REPORT=True
    for name in ['01_so_do_C7','03_test_thuc_te_du_bao']:
        source=OUT/'figures'/f'{name}.png'
        with Image.open(source) as im:
            im.verify()
        shutil.copy2(source,FIG/f'{FINAL_NAMES[name]}.png')
    validation()
    groups()
    evidence['configuration_history']=configuration_history()
    evidence['store_bias']=store_bias()
    evidence['figures']={}
    for original,name in FINAL_NAMES.items():
        path=FIG/f'{name}.png'
        with Image.open(path) as im:
            im.verify()
        evidence['figures'][path.relative_to(ROOT).as_posix()]={
            'sha256':sha256(path),'method':'reused' if original in ['01_so_do_C7','03_test_thuc_te_du_bao'] else 'redrawn_from_saved_tables'}
    assert all(sha256(p)==h for p,h in protected.items()),'Experiment artifacts changed'
    assert all(sha256(p)==h for p,h in existing.items()),'Other report figures changed'
    evidence['preserved_experiment_files']=len(protected)
    evidence['preserved_other_report_files']=len(existing)
    evidence['sources']={p.relative_to(ROOT).as_posix():h for p,h in protected.items()
                         if p.name in ['selection.json','config.json','validation_comparison.csv',
                                       'forecast_test_selected.csv','test_group_scores.csv','test_verified.json',
                                       'test_daily.csv','test_details.csv','demand_groups_origin_1885.csv']}
    history_source=ROOT/'results/metrics/wrmsse_C0_C7_20261124'
    history_files=[history_source/'manifest.json',history_source/'tong_hop.csv',
                   *[history_source/f'validation_C{i}/diem_12_cap.csv' for i in range(8)]]
    evidence['sources'].update({p.relative_to(ROOT).as_posix():sha256(p) for p in history_files})
    (ROOT/'reports/c7_events_bieu_do_nguon.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 7 final PNGs; experiment artifacts and other report figures unchanged.')
    for name in FINAL_NAMES.values():
        print(FIG/f'{name}.png')


def main():
    global OUT,FIG,EXTENSIONS,FINAL_REPORT,PUBLISHED_REPORT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-report',action='store_true',help='Collect seven report PNGs in reports/bieu_do without changing experiment outputs')
    parser.add_argument('--published-report',action='store_true',help='Rebuild seven PNGs using only small published tables; no raw data, models or forecasts required')
    args=parser.parse_args()
    if args.published_report:
        OUT=ROOT/'reports/c7_events'
        FIG=ROOT/'reports/bieu_do'
        EXTENSIONS=('png',)
        FINAL_REPORT=PUBLISHED_REPORT=True
        selected=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
        verified=json.loads((OUT/'test_verified.json').read_text(encoding='utf-8'))
        assert selected['selected']==verified['selected']=='C7_EVENTS' and verified['days']==[1914,1941]
        diagram()
        validation()
        test()
        groups()
        configuration_history()
        store_bias()
        print('PASS: seven PNGs rebuilt from published tables; no training or inference.')
        return
    if args.final_report:
        final_report()
        return
    diagram()
    if (OUT/'selection.json').exists(): validation()
    if (OUT/'test_verified.json').exists(): test()
    if (OUT/'test_group_scores.csv').exists(): groups()
    print('\n'.join(str(p) for p in sorted(FIG.iterdir())))


if __name__=='__main__':
    main()
