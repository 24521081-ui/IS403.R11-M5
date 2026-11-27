"""Ordered Validation-only workflows; batch tuning remains gated."""
import itertools,json,time,os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib,psutil
from .wrmsse_config import CONFIG,CONTRACT_ID,fold,require_stage_permission
from .wrmsse_stage2_data import Dataset,sha,write_new,load_matrix
from .wrmsse_data import load_weight_prices
from .wrmsse import WRMSSEEvaluator
from .project_paths import local_path

RF03=dict(n_estimators=100,max_depth=12,max_features=.3,max_samples=.5,min_samples_leaf=20,bootstrap=True,random_state=42,n_jobs=4)
RF06=RF03|{'n_estimators':150}
RF_FIRST=[{'name':f'RF-{i:02}','params':RF06|change} for i,change in enumerate([
 {'max_depth':16},{'max_depth':20},{'min_samples_leaf':10},{'min_samples_leaf':5},{'max_features':.1},{'max_features':.2},{'max_features':.5}],7)]
RF_LAST=[{'name':f'RF-{i:02}','params':RF06|{'n_jobs':8}|change} for i,change in enumerate([
 {'max_depth':16,'min_samples_leaf':5},{'max_depth':24,'min_samples_leaf':5},
 {'max_depth':20,'max_features':.5,'min_samples_leaf':5},{'max_depth':20,'min_samples_leaf':10}],19)]
FE_SINGLE={'FE-01':[],'FE-02':['recency'],'FE-03':['prices'],'FE-04':['snap'],'FE-05':['events'],'FE-06':['behavior']}
STAGES=['rf_v1_RF01_RF05','RF03_rf_v1_rf_v4','rf_v3_RF06','rf_v3_RF07_RF13','rf_v3_RF14_RF18_adaptive','rf_v3_RF19_RF22','FE01_FE06_new_selection','FE07_FE08_adaptive','select_validation_wrmsse']


def plan():
 return {'order':STAGES,'RF01_RF05':[{'name':f'RF-{i:02}','params':RF03|dict(n_estimators=n,max_depth=d,max_features=f,max_samples=s)} for i,(n,d,f,s) in enumerate([(30,10,.3,.5),(50,12,.3,.5),(100,12,.3,.5),(100,15,.5,.6),(150,15,.5,.6)],1)],'ablation':[{'version':v,'params':RF03} for v in ['rf_v1','rf_v2','rf_v3','rf_v4']],'RF06':RF06,'RF07_RF13':RF_FIRST,'RF19_RF22':RF_LAST,'FE01_FE06':FE_SINGLE,'selection':'Minimum new Validation WRMSSE; smoke and legacy records excluded','adaptive':'RF14_RF18 and FE07_FE08 require completed NEW single-factor results; never hardcode old winners'}


def check_record(row):
 if row.get('contract_id')!=CONTRACT_ID or row.get('split')!='validation' or row.get('metric')!='wrmsse' or row.get('purpose')!='tuning' or row.get('status')!='success':raise ValueError('New successful tuning record required')
 if not np.isfinite(row['wrmsse']) or row['wrmsse']<0:raise ValueError('Invalid WRMSSE')


def adaptive_rf(records,baseline):
 check_record(baseline);[check_record(r) for r in records]
 if {r['name'] for r in records}!={r['name'] for r in RF_FIRST}:raise ValueError('Complete RF07_RF13 before combinations')
 dims=['max_depth','min_samples_leaf','max_features'];options={d:[RF06[d]] for d in dims};gain={}
 for r in sorted(records,key=lambda x:x['wrmsse']):
  if r['wrmsse']<baseline['wrmsse']:
   for d in dims:
    v=r['params'][d]
    if v!=RF06[d]:
     if v not in options[d]:options[d].append(v)
     gain[d,v]=baseline['wrmsse']-r['wrmsse']
 candidates=[]
 for values in itertools.product(*(options[d] for d in dims)):
  changed=[(d,v) for d,v in zip(dims,values) if v!=RF06[d]]
  if len(changed)>=2:candidates.append((sum(gain[d,v] for d,v in changed),RF06|dict(zip(dims,values))))
 candidates.sort(key=lambda x:(-x[0],json.dumps(x[1],sort_keys=True)))
 return [{'name':f'RF-{i:02}','params':p} for i,(_,p) in enumerate(candidates[:5],14)]


def adaptive_fe(records,baseline):
 check_record(baseline);[check_record(r) for r in records]
 if {r['name'] for r in records}!=set(FE_SINGLE)-{'FE-01'}:raise ValueError('Complete FE02_FE06 first')
 candidates=[]
 for pair in itertools.combinations(sorted(records,key=lambda r:r['wrmsse']),2):
  gain=sum(baseline['wrmsse']-r['wrmsse'] for r in pair)
  groups=list(dict.fromkeys(g for r in pair for g in r['groups']))
  if gain>0 and any(r['wrmsse']<baseline['wrmsse'] for r in pair):candidates.append((gain,groups))
 candidates.sort(key=lambda x:(-x[0],x[1]))
 return [{'name':f'FE-{i:02}','groups':g} for i,(_,g) in enumerate(candidates[:2],7)]


def seasonal_naive(history,horizon=28,period=7):
 history=np.asarray(history)
 if history.ndim!=2 or history.shape[1]<period:raise ValueError('Insufficient history')
 return history[:,-period:][:,np.arange(horizon)%period]


def evaluator(dataset):
 m=dataset.series.set_index(['item_id','store_id'],drop=False)
 h=pd.DataFrame(np.asarray(dataset.history),index=dataset.index,columns=fold('validation').history_days)
 p=load_weight_prices(m,h,'validation')
 return WRMSSEEvaluator.fit(m,h,p,fold('validation'),strict_m5=True)


def score_saved(dataset,ev,prediction,output,name):
 assert prediction.shape==(30490,28)
 # Persist predictions before constructing the actual matrix used for scoring.
 with (output/'validation_predictions.npy').open('xb') as f:np.save(f,prediction,allow_pickle=False)
 prediction_sha=sha(output/'validation_predictions.npy')
 result=ev.score(pd.DataFrame(dataset.actual,index=dataset.index,columns=ev.fold.forecast_days),pd.DataFrame(prediction,index=dataset.index,columns=ev.fold.forecast_days))
 result.per_level.to_csv(output/'wrmsse_by_level.csv')
 result.per_series.to_parquet(output/'wrmsse_by_series.parquet',index=False)
 return result,prediction_sha


def run_rf(dataset,features,output,ev,params,version='rf_v1',groups=(),name='RF-03',purpose='smoke', *, selection_eligible=None):
 require_stage_permission('training')
 if purpose=='tuning':require_stage_permission('tuning')
 if purpose=='smoke' and (params!=RF03 or version!='rf_v1' or groups):raise ValueError('Smoke must use exact RF03 + RFv1')
 X=load_matrix(features,version,'train',groups);y=np.load(local_path(features)/'y_train.npy',allow_pickle=False)
 keys=dataset.train_keys
 if len(keys)!=600000 or X.shape[0]!=len(keys) or y.shape!=(600000,) or keys.duplicated().any() or not keys.day.between(57,1885).all():raise ValueError('Training grid invalid')
 if not np.array_equal(y,dataset.history[keys.series_id.to_numpy(),keys.day.to_numpy()-1]):raise ValueError('Labels misaligned')
 model=RandomForestRegressor(**params)
 print(f'phase: fit {name}',flush=True);start=time.perf_counter();model.fit(X,y);fit=time.perf_counter()-start
 columns=X.shape[1];del X,y
 X=load_matrix(features,version,'validation',groups)
 if X.shape!=(853720,columns):raise ValueError('Validation shape invalid')
 print('phase: predict Validation',flush=True);start=time.perf_counter()
 values=np.concatenate([model.predict(X[s:s+32768]) for s in range(0,len(dataset.validation_keys),32768)])
 predict=time.perf_counter()-start
 matrix=np.empty((30490,28));k=dataset.validation_keys
 if k.duplicated().any() or len(k)!=853720 or not k.day.between(1886,1913).all():raise ValueError('Validation keys invalid')
 matrix[k.series_id.to_numpy(),k.day.to_numpy()-1886]=values
 joblib.dump(model,output/'model.joblib')
 start=time.perf_counter();result,prediction_sha=score_saved(dataset,ev,matrix,output,name);evaluation=time.perf_counter()-start
 row={'name':name,'purpose':purpose,'status':'success','split':'validation','origin':1885,'metric':'wrmsse','wrmsse':result.wrmsse,'contract_id':CONTRACT_ID,'experiment_id':CONFIG['experiment_id'],'params':params,'version':version,'groups':list(groups),'train_rows':600000,'validation_rows':853720,'n_features':columns,'fit_seconds':fit,'predict_seconds':predict,'evaluation_seconds':evaluation,'prediction_sha256':prediction_sha,'peak_rss_mib':psutil.Process().memory_info().peak_wset/2**20 if os.name=='nt' else psutil.Process().memory_info().rss/2**20,'selection_eligible':(purpose=='tuning' if selection_eligible is None else selection_eligible)}
 write_new(output/'result.json',row);return row


def execute_stage(stage,completed,candidates,dataset,features,root,ev):
 require_stage_permission('tuning')
 index=STAGES.index(stage)
 if any(s not in completed for s in STAGES[:index]):raise ValueError('Historical workflow order must be respected')
 results=[]
 for candidate in candidates:
  name=candidate['name']
  if Path(name).name!=name or any(c in name for c in ('/','\\',':')):raise ValueError('Invalid run name')
  output=local_path(str(root)+'/'+stage+'/'+name);output.mkdir(parents=True,exist_ok=False)
  results.append(run_rf(dataset,features,output,ev,candidate['params'],candidate.get('version','rf_v1' if stage=='rf_v1_RF01_RF05' else 'rf_v3'),candidate.get('groups',()),name,'tuning'))
 return results


def select_new(records):
 for r in records:check_record(r)
 if not records:raise ValueError('No successful new tuning results')
 return min(records,key=lambda r:r['wrmsse'])


def run_ordered_tuning(dataset_path,features,output_root):
 """Future stage 3 entry point; cannot execute under stage 2 permissions."""
 require_stage_permission('tuning')
 ds=Dataset(dataset_path);ev=evaluator(ds);completed=[];all_rows=[];p=plan()
 root=local_path(output_root);root.mkdir(parents=True,exist_ok=False)
 def stage(name,candidates):
  write_new(root/(name+'_plan.json'),{'contract_id':CONTRACT_ID,'stage':name,'candidates':candidates})
  rows=execute_stage(name,completed,candidates,ds,features,output_root,ev)
  completed.append(name);all_rows.extend(rows)
  pd.DataFrame(rows).to_csv(root/(name+'_results.csv'),index=False)
  return rows
 first=stage(STAGES[0],p['RF01_RF05'])
 ablation=stage(STAGES[1],[{'name':'RF03_'+v,'version':v,'params':RF03} for v in ['rf_v1','rf_v2','rf_v3','rf_v4']])
 baseline=stage(STAGES[2],[{'name':'RF-06','params':RF06}])[0]
 single=stage(STAGES[3],RF_FIRST)
 combined=stage(STAGES[4],adaptive_rf(single,baseline))
 last=stage(STAGES[5],RF_LAST)
 selected=select_new([r for r in ablation if r['version']=='rf_v3']+[baseline]+single+combined+last)
 fe=stage(STAGES[6],[{'name':name,'params':selected['params'],'groups':groups} for name,groups in FE_SINGLE.items()])
 base=next(r for r in fe if r['name']=='FE-01')
 pairs=stage(STAGES[7],[x|{'params':selected['params']} for x in adaptive_fe([r for r in fe if r['name']!='FE-01'],base)])
 winner=select_new(fe+pairs)
 write_new(root/'selected_validation.json',winner)
 pd.DataFrame(all_rows).to_csv(root/'all_validation_results.csv',index=False)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(12,5));ax.bar(range(len(all_rows)),[row['wrmsse'] for row in all_rows]);ax.set_xticks(range(len(all_rows)),[row['name'] for row in all_rows],rotation=90);ax.set_ylabel('Validation WRMSSE');fig.tight_layout();fig.savefig(root/'validation_wrmsse.png',dpi=160);plt.close(fig)
 write_new(root/'manifest.json',{'contract_id':CONTRACT_ID,'completed':completed+[STAGES[8]],'dataset_manifest':sha(ds.root/'manifest.json'),'feature_manifest':sha(local_path(features)/'manifest.json'),'files':{q.relative_to(root).as_posix():sha(q) for q in root.rglob('*') if q.is_file()}})
 return winner
