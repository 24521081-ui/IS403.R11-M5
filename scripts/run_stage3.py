"""Stage 3 isolated workers, immutable checkpoints, fixed historical RF/FE plans."""
from pathlib import Path
import sys,os,json,time,gc,subprocess,traceback,hashlib,shutil
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import psutil,joblib
from src.project_paths import local_path
from src.wrmsse_config import CONTRACT_ID,stage3_tuning_scope
from src.wrmsse_stage2_data import Dataset,sha,write_new
from src.wrmsse_workflows import evaluator,run_rf


def effective_groups(version,groups):
 return [g for g in groups if not(g=='recency' and version=='rf_v4')]


def eligible(groups):return not set(groups)&{'snap','events'}


def best(rows):
 valid=[r for r in rows if r.get('status')=='success' and r.get('selection_eligible',False) and r.get('purpose')=='tuning' and r.get('contract_id')==CONTRACT_ID and np.isfinite(r['wrmsse'])]
 if not valid:raise RuntimeError('No eligible successful candidates')
 return min(valid,key=lambda r:r['wrmsse'])


def snapshot(folder):return {p.relative_to(folder).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in folder.rglob('*') if p.is_file()}


def verify_run(path):
 m=json.loads((path/'manifest.json').read_text())
 for rel,digest in m['files'].items():
  if sha(path/rel)!=digest:raise ValueError('Run artifact changed: '+str(path/rel))


def worker(rootrel,requestrel):
 root=local_path(rootrel);request=json.loads(local_path(requestrel).read_text());output=local_path(request['output'])
 # No Test CSV or legacy artifact can enter the worker.
 original=pd.read_csv
 def safe_csv(path,*a,**k):
  p=Path(path).resolve()
  if p not in {local_path('data/raw/'+f) for f in ['calendar.csv','sales_train_validation.csv','sell_prices.csv']}:raise AssertionError('Forbidden CSV read')
  return original(path,*a,**k)
 pd.read_csv=safe_csv
 ds=Dataset(request['dataset']);ev=joblib.load(root/'evaluator.joblib')
 with stage3_tuning_scope(request['evidence']):
  row=run_rf(ds,request['features'],output,ev,request['params'],request['version'],request['groups'],request['name'],'tuning')
 if len(pd.read_parquet(output/'wrmsse_by_series.parquet'))!=42840:raise AssertionError('Incomplete hierarchy')
 print(json.dumps({'name':row['name'],'wrmsse':row['wrmsse']}),flush=True)


def execute(root,request,previous):
 output=local_path(request['output']);record=output/'record.json'
 if record.exists():
  verify_run(output);return json.loads(record.read_text())
 output.mkdir(parents=True,exist_ok=True)
 key=hashlib.sha256(json.dumps({k:request[k] for k in ['params','version','groups','dataset_hash','feature_hash','contract_id']},sort_keys=True).encode()).hexdigest()
 for prior in previous:
  prior_params = {name: value for name, value in prior.get('params', {}).items() if name != 'n_jobs'}
  requested_params = {name: value for name, value in request['params'].items() if name != 'n_jobs'}
  equivalent = (
      prior_params == requested_params
      and all(prior.get(field) == request[field]
              for field in ('version', 'groups', 'dataset_hash', 'feature_hash', 'contract_id'))
  )
  if prior.get('status') == 'success' and prior.get('purpose') == 'tuning' and equivalent:
   source=local_path(prior['artifact_dir']);verify_run(source)
   reused=prior|{'name':request['name'],'requested_groups':request['requested_groups'],'group':request['group'],'artifact_dir':source.relative_to(ROOT).as_posix(),'reuse':True,'reuse_source':prior['name'],'requested_n_jobs':request['params']['n_jobs'],'reuse_reason':'Same model parameters, features, sample and seed; no refit solely for execution thread count','selection_eligible':eligible(request['requested_groups']),'reuse_key':key}
   write_new(record,reused);write_new(output/'manifest.json',{'files':{'record.json':sha(record)},'source_manifest_sha256':sha(source/'manifest.json')});return reused
 # Preserve any interrupted attempt, never overwrite its artifacts.
 attempts=sorted(output.glob('attempt_*'));attempt=output/f'attempt_{len(attempts)+1:03}';attempt.mkdir()
 request=request|{'output':attempt.relative_to(ROOT).as_posix()}
 write_new(attempt/'request.json',request)
 limits=json.loads((ROOT/'configs/stage3.json').read_text())['resource_limits']
 peak=0.;minimum=psutil.virtual_memory().available/2**20;stopped=False
 if minimum<limits['min_available_mib']+256:
  row=request|{'status':'resource_unavailable','error':'Insufficient starting reserve','selection_eligible':False,'purpose':'tuning'}
 else:
  env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['JOBLIB_TEMP_FOLDER']=str(attempt/'tmp');env['MPLCONFIGDIR']=str(root/'mplconfig')
  with (attempt/'worker.log').open('x',encoding='utf-8') as log,(attempt/'resources.jsonl').open('x',encoding='utf-8') as samples:
   proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',root.relative_to(ROOT).as_posix(),(attempt/'request.json').relative_to(ROOT).as_posix()],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    try:
     process=psutil.Process(proc.pid);rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True) if p.is_running())/2**20;available=psutil.virtual_memory().available/2**20
     peak=max(peak,rss);minimum=min(minimum,available)
     samples.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'rss_mib':rss,'available_mib':available})+'\n');samples.flush()
     if rss>limits['max_rss_mib'] or available<limits['min_available_mib']:
      stopped=True
      for child in process.children(recursive=True):child.terminate()
      proc.terminate();proc.wait(timeout=30);break
    except psutil.NoSuchProcess:pass
    time.sleep(2)
   proc.wait()
  if proc.returncode==0 and (attempt/'result.json').exists():row=json.loads((attempt/'result.json').read_text())
  else:row={'name':request['name'],'status':'resource_stopped' if stopped else 'failed','error':(attempt/'worker.log').read_text(encoding='utf-8')[-3000:],'selection_eligible':False,'purpose':'tuning'}
 row=row|{'contract_id':CONTRACT_ID,'group':request['group'],'version':request['version'],'params':request['params'],'groups':request['groups'],'requested_groups':request['requested_groups'],'selection_eligible':row.get('status')=='success' and eligible(request['requested_groups']),'reuse':False,'reuse_key':key,'artifact_dir':attempt.relative_to(ROOT).as_posix(),'monitor_peak_rss_mib':peak,'min_available_mib':minimum,'n_raw_features':request['logical'],'dataset_hash':request['dataset_hash'],'feature_hash':request['feature_hash']}
 write_new(record,row)
 write_new(attempt/'manifest.json',{'contract_id':CONTRACT_ID,'n_jobs':request['params']['n_jobs'],'files':{p.relative_to(attempt).as_posix():sha(p) for p in attempt.rglob('*') if p.is_file()}})
 write_new(output/'manifest.json',{'files':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}})
 return row


def report(root,rows,selected,summary2):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 table=pd.DataFrame(rows);table.to_csv(root/'all_results.csv',index=False);table.to_parquet(root/'all_results.parquet',index=False)
 for group in ['initial','ablation','advanced','features']:
  frame=table[table.group==group].copy()
  if group=='ablation':
   base=frame[frame.version=='rf_v1'].iloc[0].get('wrmsse',float('nan'));frame['delta_vs_rf_v1']=frame.get('wrmsse',float('nan'))-base
  frame.to_csv(root/(group+'_results.csv'),index=False)
  good=frame[frame.status=='success']
  fig,ax=plt.subplots(figsize=(11,4));ax.bar(good.name,good.wrmsse);ax.tick_params(axis='x',rotation=60);ax.set(title=group+' - Validation d_1886-d_1913',ylabel='WRMSSE (12 M5 levels)');fig.tight_layout();fig.savefig(root/(group+'_wrmsse.png'),dpi=150);plt.close(fig)
 good=table[table.status=='success']
 fig,ax=plt.subplots(figsize=(7,4));ax.scatter(good.fit_seconds,good.wrmsse);ax.set(xlabel='Fit seconds',ylabel='WRMSSE',title='Validation d_1886-d_1913');fig.tight_layout();fig.savefig(root/'wrmsse_vs_fit_time.png',dpi=150);plt.close(fig)
 # Controlled one-factor comparisons RF06..13, grouped by unchanged parameters.
 adv=[r for r in rows if r['group']=='advanced' and r['status']=='success']
 for parameter in ['n_estimators','max_depth','max_features','min_samples_leaf']:
  groups={}
  for row in adv:
   key=json.dumps({k:v for k,v in row['params'].items() if k not in [parameter,'n_jobs']},sort_keys=True)
   groups.setdefault(key,[]).append(row)
  fig,ax=plt.subplots(figsize=(8,4))
  for same in groups.values():
   if len(set(x['params'][parameter] for x in same))<2:continue
   same=sorted(same,key=lambda x:x['params'][parameter]);ax.plot([x['params'][parameter] for x in same],[x['wrmsse'] for x in same],marker='o',label='/'.join(x['name'] for x in same))
  ax.set(xlabel=parameter,ylabel='WRMSSE',title='Validation d_1886-d_1913; other model parameters fixed');ax.legend(fontsize=6);fig.tight_layout();fig.savefig(root/('wrmsse_vs_'+parameter+'.png'),dpi=150);plt.close(fig)
 source=local_path(selected['artifact_dir']);shutil.copyfile(source/'wrmsse_by_level.csv',root/'selected_wrmsse_by_level.csv')
 base=summary2['baseline']['wrmsse'];comparison=pd.DataFrame([{'model':'Seasonal Naive','wrmsse':base},{'model':selected['name'],'wrmsse':selected['wrmsse']}]);comparison.to_csv(root/'baseline_comparison.csv',index=False)
 fig,ax=plt.subplots(figsize=(7,4));ax.bar(comparison.model,comparison.wrmsse);ax.set(ylabel='WRMSSE',title='Validation d_1886-d_1913');fig.tight_layout();fig.savefig(root/'baseline_comparison.png',dpi=150);plt.close(fig)
 selected=selected|{'improvement_vs_naive_percent':100*(1-selected['wrmsse']/base),'smoke_excluded':True,'test_accessed':False,'final_retrain_performed':False,'selection_basis':'Minimum WRMSSE over all eligible successful NEW tuning candidates; SNAP/events exploratory excluded'}
 write_new(root/'selected_model.json',selected)
 return selected


def main():
 if len(sys.argv)>1 and sys.argv[1]=='--worker':
  try:worker(sys.argv[2],sys.argv[3])
  except Exception:traceback.print_exc();sys.exit(1)
  return
 policy_path=ROOT/'configs/stage3.json';policy=json.loads(policy_path.read_text())
 runs=sorted((ROOT/'results/wrmsse_v1').glob('stage3_*'))
 root=runs[-1] if '--resume' in sys.argv and runs else local_path('results/wrmsse_v1/stage3_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 if '--resume' in sys.argv and (root/'manifest_completed.json').exists():
  print('Stage 3 already completed: '+root.relative_to(ROOT).as_posix(),flush=True)
  return
 root.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(root/'mplconfig')
 stage2=ROOT/'results/wrmsse_v1/stage2_20261123T202104Z';summary2=json.loads((stage2/'summary.json').read_text())
 evidence=sorted((ROOT/'preparation').glob('unit_tests_summary_*.json'))[-1].relative_to(ROOT).as_posix()
 if not (root/'preflight.json').exists():
  before={'raw':snapshot(local_path('data/raw'))}
  write_new(root/'protected_before.json',before)
  ds=Dataset(summary2['processed']);fm=json.loads(local_path(summary2['features']+'/manifest.json').read_text())
  for rel,digest in fm['files'].items():assert sha(local_path(summary2['features']+'/'+rel))==digest,rel
  assert len(ds.train_keys)==600000 and len(ds.validation_keys)==853720
  ev=evaluator(ds);assert len(ev.weights)==42840;joblib.dump(ev,root/'evaluator.joblib');del ev,ds;gc.collect()
  write_new(root/'preflight.json',{'contract_id':CONTRACT_ID,'evaluator_sha256':sha(root/'evaluator.joblib'),'feature_manifest_sha256':sha(local_path(summary2['features']+'/manifest.json')),'dataset_manifest_sha256':sha(local_path(summary2['processed']+'/manifest.json')),'raw_sha256':{n:sha(local_path('data/raw/'+n)) for n in ['calendar.csv','sales_train_validation.csv','sell_prices.csv']},'evidence':evidence,'test_locked':True,'snap_events':'Exploratory; excluded from primary selection due to missing as-of publication metadata'})
  for folder in ['src','scripts','tests','configs']:
   for p in (ROOT/folder).glob('*'):
    if p.is_file():
     target=root/'code_snapshot'/folder/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 pre=json.loads((root/'preflight.json').read_text());assert sha(root/'evaluator.joblib')==pre['evaluator_sha256']
 policy['allow_tuning']=True;policy_path.write_text(json.dumps(policy,indent=2)+'\n',encoding='utf-8')
 rows=[]
 specs=json.loads(local_path(summary2['features']+'/manifest.json').read_text())['specs']
 def run(name,group,params,version,requested):
  if 'execution_n_jobs' in policy:
   params = params | {'n_jobs': policy['execution_n_jobs']}
  groups=effective_groups(version,requested);logical=specs[version]['n_raw_features']+sum(specs[g]['extra_logical'] for g in groups)
  request={'name':name,'group':group,'params':params,'version':version,'requested_groups':requested,'groups':groups,'logical':logical,'dataset':summary2['processed'],'features':summary2['features'],'dataset_hash':pre['dataset_manifest_sha256'],'feature_hash':pre['feature_manifest_sha256'],'contract_id':CONTRACT_ID,'evidence':evidence,'output':(root/group/name).relative_to(ROOT).as_posix()}
  row=execute(root,request,rows);rows.append(row)
  print(json.dumps({k:row.get(k) for k in ['name','version','groups','status','wrmsse','reuse','selection_eligible']}),flush=True)
  return row
 try:
  for c in policy['configs'][:5]:run(c['name'],'initial',c['params'],'rf_v1',[])
  initial=best(rows)
  for version in ['rf_v1','rf_v2','rf_v3','rf_v4']:run('ABL_'+version,'ablation',initial['params'],version,[])
  feature=best([r for r in rows if r['group']=='ablation'])['version']
  for c in policy['configs'][5:]:run(c['name'],'advanced',c['params'],feature,[])
  advanced=best([r for r in rows if r['group'] in ['advanced','ablation']])
  for name,groups in policy['FE'].items():run(name,'features',advanced['params'],feature,groups)
  selected=report(root,rows,best(rows),summary2)
  before=json.loads((root/'protected_before.json').read_text())
  assert snapshot(local_path('data/raw'))==before['raw']
  for n,digest in pre['raw_sha256'].items():assert sha(local_path('data/raw/'+n))==digest
  for row in rows:
   checkpoint=root/row['group']/row['name'];verify_run(checkpoint)
  write_new(root/'summary.json',{'status':'complete','selection':selected,'successful':sum(r['status']=='success' and not r['reuse'] for r in rows),'reused':sum(r['reuse'] for r in rows),'failed':sum(r['status']!='success' for r in rows),'raw_unchanged':True,'test_actual_accessed':False,'final_retrain':False,'ablation_hyperparameters':initial['params'],'base_feature':feature,'feature_tuning_hyperparameters':advanced['params']})
 finally:
  policy['allow_tuning']=False;policy_path.write_text(json.dumps(policy,indent=2)+'\n',encoding='utf-8')
 write_new(root/'manifest_completed.json',{'contract_id':CONTRACT_ID,'files':{p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ['supervisor.log']}})
 print('STAGE3_COMPLETE '+root.relative_to(ROOT).as_posix(),flush=True)

if __name__=='__main__':main()
