"""Isolated C7 event experiment. Explicit stages prevent premature test/group scoring."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "lightgbm_tweedie"))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('PYTHONUTF8', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
# Keep the sklearn/LightGBM import order stable on the tested Windows environment.
import sklearn
import joblib
import numpy as np
import pandas as pd
from src.data_loader import load_calendar, load_prices, load_store, list_stores
from src.feature_engineering import next_day_features
from src.preprocessing import history_matrix
from src.training import train_model, verify_saved_models
from run_c5_price_features import build_weekly_features, make_training_data, EXTRAS
from run_c6_platt import prepare_c5_features, apply_platt
from run_stage_b_robustness import historical_price_arrays, fit_platt
from score_c7_wrmsse import score
from score_model_wrmsse import load_forecasts, check_coverage, sha256

OUT = ROOT / 'results/analysis/c7_events_20261126'
BASE = ROOT / 'results/analysis/stage_C7_validation_d1885_lambda075/folds'
TEST = ROOT / 'results/analysis/final_test_C6_C7_frozen/C7/folds'
C5 = ROOT / 'results/metrics/validation_C5_full_10_stores_20261121T140034335643Z/manifest.json'

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def event_calendar():
    cal = load_calendar(ROOT.parents[1] / 'data/raw')
    past = cal.loc[cal.d.str[2:].astype(int).le(1857)]
    mapping = {}
    for kind in ['name', 'type']:
        cols = [f'event_{kind}_{i}' for i in [1, 2]]
        values = sorted(set(past[cols].stack().dropna()))
        for i, value in enumerate(values):
            feature = f'event_{kind}_{i:02d}'
            cal[feature] = cal[cols].eq(value).any(axis=1).astype('int8')
            mapping[feature] = value
    return cal, mapping

def configuration():
    cal, mapping = event_calendar()
    frozen = ROOT / 'configs/c7_events.json'
    if frozen.is_file():
        cfg = read(frozen)
        assert cfg['name'] == 'C7_EVENTS' and cfg['lambda_blend'] == .75
        assert cfg['event_features'] == mapping
        assert cfg['validation_days'] == [1886, 1913] and cfg['test_days'] == [1914, 1941]
        return cal, mapping, cfg
    original = read(C5)
    source = original['sources'][0]
    bundle = verify_saved_models(ROOT / 'results/models' / source['run_tag'] / 'model.joblib')
    for component in ['classifier', 'regressor']:
        assert getattr(bundle['model'], component).booster_.feature_name() == bundle['features']
    return cal, mapping, dict(name='C7_EVENTS', lambda_blend=.75,
        original_features=bundle['features'], event_features=mapping,
        parameters={n:{**original['config'][n], 'importance_type':'gain'} for n in ['classifier','regressor']},
        price_policy='previous_completed_week; frozen at forecast origin',
        vocabulary_end=1857, validation_aux_origin=1857, validation_platt_days=[1858,1885],
        validation_days=[1886,1913], test_aux_origin=1885, test_platt_days=[1886,1913], test_days=[1914,1941])

def forecast(bundle, matrix, items, store, state, cal, origin, prices, events, platt=None):
    history = matrix[:, -28:].copy()
    indexed = cal.set_index('d')
    rows, comps = [], []
    for h in range(1,29):
        day = f'd_{origin+h}'
        row = indexed.loc[day]
        x = next_day_features(history, items, row, state, bundle['categories'], last_prices=prices['sell_price'])
        for name in EXTRAS:
            x[name] = prices[name]
        for name in events:
            x[name] = np.int8(row[name])
        raw, mu = bundle['model'].predict_components(x[bundle['features']])
        calibrated = raw if platt is None else apply_platt(platt, raw)
        blended = raw if platt is None else .25*raw + .75*calibrated
        pred = blended*mu
        assert np.isfinite(pred).all() and (pred >= 0).all()
        rows.append(pd.DataFrame(dict(item_id=items,store_id=store,target_day=day,
            origin_day=f'd_{origin}',horizon=h,split='validation' if origin<1913 else 'test',y_pred=pred)))
        comps.append(pd.DataFrame(dict(item_id=items,store_id=store,target_day=day,
            probability_positive=raw,p_raw=raw,p_cal=calibrated,p_blend=blended,conditional_units=mu)))
        history = np.column_stack([history[:,1:],pred])
    return pd.concat(rows,ignore_index=True),pd.concat(comps,ignore_index=True)

def fit(store, origin, wide, weekly, cal, events, cfg):
    path = OUT / 'models' / store / f'd_{origin}/model.joblib'
    if path.exists():
        bundle = verify_saved_models(path)
        assert bundle['origin_day']==origin and bundle['features']==cfg['original_features']+list(events)
        assert read(path.parent/'training.json')['parameters']==cfg['parameters']
        return bundle
    logging.info('TRAIN %s origin=%s',store,origin)
    hist = prepare_c5_features(wide,cal,weekly,origin)
    train = make_training_data(hist)
    day_by_index = hist.loc[train['X'].index,'day']
    lookup = cal.assign(day=cal.d.str[2:].astype(int)).set_index('day')
    for name in events:
        train['X'][name] = day_by_index.map(lookup[name]).to_numpy(dtype='int8')
        train['features'].append(name)
    assert train['features']==cfg['original_features']+list(events)
    del hist
    gc.collect()
    bundle,info = train_model(train,cfg['parameters'],path)
    save(path.parent/'training.json',dict(**info,parameters=cfg['parameters']))
    del train
    gc.collect()
    return bundle

def train_predict(origin, stores=None):
    cal,events,cfg = configuration()
    config_path=OUT/'config.json'
    if config_path.exists():
        assert read(config_path)==cfg
    else:
        save(config_path,cfg)
    split='validation' if origin==1885 else 'test'
    if split=='test':
        assert read(OUT/'selection.json')['selected']=='C7_EVENTS'
    for store in stores or list_stores(ROOT.parents[1]/'data/raw'):
        folder=OUT/split/store
        done=folder/'completed.json'
        if done.exists():
            assert read(done)['forecast_sha256']==sha256(folder/'forecast.csv')
            logging.info('REUSE %s %s',split,store)
            continue
        wide=load_store(ROOT.parents[1]/'data/raw',store,origin)
        items=wide.item_id.tolist()
        assert len(items)==3049
        state=str(wide.state_id.iloc[0])
        weekly=build_weekly_features(load_prices(ROOT.parents[1]/'data/raw',store),cal)
        aux_origin=origin-28
        aux=fit(store,aux_origin,wide,weekly,cal,events,cfg)
        auxmatrix=history_matrix(wide,aux_origin)
        auxprices=historical_price_arrays(weekly,cal,items,aux_origin)
        oof,comps=forecast(aux,auxmatrix,items,store,state,cal,aux_origin,auxprices,events)
        folder.mkdir(parents=True,exist_ok=True)
        oof.to_csv(folder/'oof_forecast.csv',index=False)
        comps.to_csv(folder/'oof_components.csv',index=False)
        truth=wide[[f'd_{d}' for d in range(aux_origin+1,origin+1)]].to_numpy(dtype=float)
        platt,diag=fit_platt(comps,truth,aux_origin,origin,items)
        joblib.dump(platt,folder/'platt.joblib')
        save(folder/'calibration.json',diag)
        del aux,auxmatrix,oof,comps,truth
        gc.collect()
        main=fit(store,origin,wide,weekly,cal,events,cfg)
        matrix=history_matrix(wide,origin)
        prices=historical_price_arrays(weekly,cal,items,origin)
        pred,comp=forecast(main,matrix,items,store,state,cal,origin,prices,events,platt)
        pred.to_csv(folder/'forecast.csv',index=False)
        comp.to_csv(folder/'components.csv',index=False)
        save(done,dict(origin=origin,store=store,forecast_sha256=sha256(folder/'forecast.csv')))
        logging.info('COMPLETE %s %s',split,store)
        del wide,weekly,main,matrix,pred,comp
        gc.collect()

def inputs():
    # Sales are integer counts; int32 saves RAM while score() still converts to float64.
    wide=pd.read_csv(ROOT.parents[1]/'data/raw/sales_train_evaluation.csv',dtype={f'd_{d}':'int32' for d in range(1,1942)})
    cal=load_calendar(ROOT.parents[1]/'data/raw')
    weeks=set(cal.loc[cal.d.str[2:].astype(int).between(1858,1913),'wm_yr_wk'])
    prices=pd.concat([p.loc[p.wm_yr_wk.isin(weeks)] for p in pd.read_csv(ROOT.parents[1]/'data/raw/sell_prices.csv',chunksize=250000)],ignore_index=True)
    return wide,cal,prices

def combined(paths):
    paths=sorted(paths)
    assert len(paths)==10, f'Need ten forecasts, found {len(paths)}'
    return pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True)

def select():
    wide,cal,prices=inputs()
    summaries=[]
    for name,paths in [('C7',BASE.glob('*/forecast_C7.csv')),('C7_EVENTS',(OUT/'validation').glob('*/forecast.csv'))]:
        pred=combined(paths)
        check_coverage(pred,wide)
        levels,details=score(1885,wide,cal,prices,pred,return_details=True)
        levels=pd.DataFrame(levels)
        levels.to_csv(OUT/f'validation_{name}_levels.csv',index=False)
        details.to_csv(OUT/f'validation_{name}_details.csv',index=False)
        summaries.append(dict(configuration=name,WRMSSE=float(levels.contribution_WRMSSE.sum()),
            weighted_RMSSE_level12=float(levels.loc[levels.level.eq(12),'weighted_RMSSE'].iloc[0])))
    table=pd.DataFrame(summaries)
    table['delta_WRMSSE_vs_C7']=table.WRMSSE-table.WRMSSE.iloc[0]
    table['delta_level12_vs_C7']=table.weighted_RMSSE_level12-table.weighted_RMSSE_level12.iloc[0]
    table.to_csv(OUT/'validation_comparison.csv',index=False)
    selected='C7_EVENTS' if summaries[1]['WRMSSE']<summaries[0]['WRMSSE'] else 'C7'
    save(OUT/'selection.json',dict(selected=selected,rule='Strictly lower validation WRMSSE; ties keep C7',validation=summaries))
    print(table.to_string(index=False),flush=True)
    print('SELECTED',selected,flush=True)

def verify_test():
    selected=read(OUT/'selection.json')['selected']
    if selected=='C7':
        manifest=read(ROOT/'results/analysis/final_test_C6_C7_frozen/comparison_C6_C7/manifest_C6_C7.json')
        assert manifest['status']=='complete' and manifest['origin']==1913
        assert manifest['lambda_C7']==.75 and manifest['test_days']==[1914,1941]
        assert manifest['price_policy']=='previous_completed_week'
        cfg=read(OUT/'config.json')
        model_audit=[]
        for store in manifest['stores']:
            completed=read(TEST/store/'completed.json')
            assert completed['origin']==1913 and completed['lambda']==.75
            assert completed['store_id']==store
            model_path=ROOT/'results/analysis/final_test_C6_C7_frozen/models'/store/'model.joblib'
            bundle=verify_saved_models(model_path)
            assert bundle['origin_day']==1913 and bundle['features']==cfg['original_features']
            for component in ['classifier','regressor']:
                actual=getattr(bundle['model'],component).get_params()
                assert all(actual[key]==value for key,value in cfg['parameters'][component].items())
            calibration_folder=ROOT/'results/analysis/final_test_C6_C7_frozen/folds'/store
            calibration=read(calibration_folder/'completed.json')['calibration']
            assert calibration['aux_origin']==1885 and calibration['calibration_start']==1886 and calibration['calibration_end']==1913
            platt=joblib.load(calibration_folder/'platt.joblib')
            assert np.isclose(platt.coef_[0,0],calibration['platt_slope'])
            assert np.isclose(platt.intercept_[0],calibration['platt_intercept'])
            model_audit.append(dict(store=store,origin=1913,features=bundle['features'],model_sha256=sha256(model_path),
                calibration=calibration,platt_sha256=sha256(calibration_folder/'platt.joblib')))
            del bundle
        save(OUT/'audit/reused_test_models.json',model_audit)
        patterns=[str(TEST/'*/forecast_C7.csv')]
    else:
        cfg=read(OUT/'config.json')
        model_audit=[]
        for folder in sorted((OUT/'test').iterdir()):
            if not folder.is_dir(): continue
            completed=read(folder/'completed.json')
            assert completed['origin']==1913 and completed['store']==folder.name
            assert completed['forecast_sha256']==sha256(folder/'forecast.csv')
            model_path=OUT/'models'/folder.name/'d_1913/model.joblib'
            bundle=verify_saved_models(model_path)
            assert bundle['origin_day']==1913
            assert bundle['features']==cfg['original_features']+list(cfg['event_features'])
            for component in ['classifier','regressor']:
                actual=getattr(bundle['model'],component).get_params()
                assert all(actual[key]==value for key,value in cfg['parameters'][component].items())
            calibration=read(folder/'calibration.json')
            assert calibration['aux_origin']==1885 and calibration['calibration_start']==1886 and calibration['calibration_end']==1913
            platt=joblib.load(folder/'platt.joblib')
            assert np.isclose(platt.coef_[0,0],calibration['platt_slope'])
            assert np.isclose(platt.intercept_[0],calibration['platt_intercept'])
            model_audit.append(dict(store=folder.name,origin=1913,features=bundle['features'],model_sha256=sha256(model_path),
                calibration=calibration,platt_sha256=sha256(folder/'platt.joblib')))
            del bundle
        assert len(model_audit)==10
        save(OUT/'audit/selected_test_models.json',model_audit)
        patterns=[str(OUT/'test/*/forecast.csv')]
    pred,paths=load_forecasts(patterns)
    wide,cal,prices=inputs()
    check_coverage(pred,wide)
    levels,details=score(1913,wide,cal,prices,pred.assign(origin_day='d_1913',split='test'),return_details=True)
    pred.to_csv(OUT/'forecast_test_selected.csv',index=False)
    pd.DataFrame(levels).to_csv(OUT/'test_levels.csv',index=False)
    details.to_csv(OUT/'test_details.csv',index=False)
    daily=pred.groupby('target_day').y_pred.sum().reindex([f'd_{d}' for d in range(1914,1942)]).rename('predicted').reset_index()
    daily['actual']=[wide[f'd_{d}'].sum() for d in range(1914,1942)]
    daily.to_csv(OUT/'test_daily.csv',index=False)
    save(OUT/'test_verified.json',dict(selected=selected,forecast=str(OUT/'forecast_test_selected.csv'),
        rows=len(pred),pairs=30490,days=[1914,1941],unique_keys=True,finite_nonnegative=True,
        WRMSSE=sum(r['contribution_WRMSSE'] for r in levels),bias_pct=100*(daily.predicted.sum()/daily.actual.sum()-1),
        sources={str(p):sha256(p) for p in paths},sha256=sha256(OUT/'forecast_test_selected.csv'),config_sha256=sha256(OUT/'config.json')))

def groups():
    verified=read(OUT/'test_verified.json')
    assert sha256(OUT/'forecast_test_selected.csv')==verified['sha256']
    subprocess.run([sys.executable,str(ROOT/'scripts/build_demand_groups.py'),'--origins','1885',
        '--output-dir',str(OUT/'labels')],check=True)
    labels=pd.read_csv(OUT/'labels/demand_groups_origin_1885.csv')
    details=pd.read_csv(OUT/'test_details.csv')
    joined=details.merge(labels,on=['item_id','store_id'],how='left',validate='one_to_one')
    assert joined.demand_group.notna().all() and joined.cat_id.notna().all()
    rows=[]
    for column,values in [('cat_id',['FOODS','HOBBIES','HOUSEHOLD']),('demand_group',['frequent','intermittent'])]:
        for value in values:
            group=joined.loc[joined[column].eq(value)]
            valid=np.isfinite(group.RMSSE)&np.isfinite(group.scale)&group.scale.gt(0)
            sub=group.loc[valid]
            weights=sub.revenue_train_28/sub.revenue_train_28.sum()
            assert np.isclose(weights.sum(),1)
            rows.append(dict(configuration=verified['selected'],split='test',days='d_1914-d_1941',level=12,
                dimension=column,group=value,n_pairs=len(group),n_scored=len(sub),n_excluded=int((~valid).sum()),
                revenue_weighted_RMSSE=float(np.dot(weights,sub.RMSSE)),
                bias_pct=100*(sub.predicted_units.sum()/sub.actual_units.sum()-1)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'test_group_scores.csv',index=False)
    save(OUT/'labels_audit.json',dict(origin=1885,missing_labels=int(joined.demand_group.isna().sum()),
        counts=labels.demand_group.value_counts().to_dict(),
        never_sold=int(labels.ADI.isna().sum()),never_sold_policy='intermittent; exclude from RMSSE only when scale invalid'))
    print(result.to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['validation','select','test','verify-test','groups'])
    parser.add_argument('--stores',nargs='+')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if args.stage in ['validation','test']:
        train_predict(1885 if args.stage=='validation' else 1913,args.stores)
    elif args.stage=='select': select()
    elif args.stage=='verify-test': verify_test()
    elif args.stage=='groups': groups()

if __name__=='__main__':
    main()
