"""Stage 2: fresh data/features, Seasonal Naive, exactly one RF03 Validation fit."""
from pathlib import Path
import sys,os,json,time,subprocess,unittest,traceback
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import psutil
from src.project_paths import local_path
from src.wrmsse_config import CONTRACT_ID,CONFIG,stage2_training_scope
from src.wrmsse_stage2_data import build_processed,build_features,sha,write_new,sample_training
from src.wrmsse_workflows import plan,evaluator,seasonal_naive,score_saved,run_rf,RF03


def snapshot(root):
 return {p.relative_to(root).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in root.rglob('*') if p.is_file()}


def worker(run):
 output=local_path('results/wrmsse_v1/'+run)
 def phase(name):
  print('phase: '+name,flush=True)
  with (output/'phases.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'phase':name,'utc':datetime.now(timezone.utc).isoformat()})+'\n')
 # Enforce raw access: no Test sales file and no legacy artifact can be read.
 original_csv=pd.read_csv
 def safe_csv(path,*args,**kwargs):
  p=Path(path).resolve()
  allowed={local_path('data/raw/'+f) for f in ['calendar.csv','sales_train_validation.csv','sell_prices.csv']}
  if p not in allowed:raise AssertionError('Unauthorized CSV input: '+str(p))
  return original_csv(path,*args,**kwargs)
 pd.read_csv=safe_csv
 raw=local_path('data/raw');before_raw=snapshot(raw)
 write_new(output/'protected_metadata_before.json',{'raw':before_raw})
 phase('tests')
 evidence_rel='preparation/'+run+'_tests.json'
 suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_wrmsse*.py',top_level_dir=str(ROOT))
 with (output/'tests.log').open('x',encoding='utf-8') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 if not result.wasSuccessful():raise RuntimeError('Checks failed; training stays locked')
 code_hash={p.relative_to(ROOT).as_posix():sha(p) for folder in ['src','tests','scripts'] for p in (ROOT/folder).glob('*.py')}
 write_new(local_path(evidence_rel),{'status':'passed','contract_id':CONTRACT_ID,'tests':result.testsRun,'code_sha256':code_hash})
 phase('processed')
 processed='data/processed/wrmsse_v1/'+run;features='data/features/wrmsse_v1/'+run
 ds=build_processed(processed)
 # Full-size reproducibility and alignment checks before any training.
 repeated,_=sample_training(ds.series,ds.calendar)
 pd.testing.assert_frame_equal(repeated,ds.train_keys)
 assert ds.history.shape==(30490,1885) and ds.actual.shape==(30490,28)
 assert len(ds.validation_keys)==853720 and not ds.validation_keys.duplicated().any()
 assert not set(zip(ds.train_keys.series_id,ds.train_keys.day)) & set(zip(ds.validation_keys.series_id,ds.validation_keys.day))
 phase('features')
 specs=build_features(ds,features)
 write_new(output/'workflow_plan.json',plan())
 phase('evaluator')
 ev=evaluator(ds)
 assert ev.hierarchy.counts[-1]==30490 and len(ev.weights)==42840
 phase('Seasonal Naive')
 base=output/'seasonal_naive';base.mkdir();start=time.perf_counter();prediction=seasonal_naive(ds.history)
 pred_seconds=time.perf_counter()-start
 start=time.perf_counter();score,predsha=score_saved(ds,ev,prediction,base,'Seasonal Naive')
 baseline={'name':'Seasonal Naive','status':'success','purpose':'baseline','metric':'wrmsse','split':'validation','contract_id':CONTRACT_ID,'wrmsse':score.wrmsse,'predict_seconds':pred_seconds,'evaluation_seconds':time.perf_counter()-start,'prediction_sha256':predsha}
 write_new(base/'result.json',baseline)
 phase('enable scoped smoke training')
 policy_path=ROOT/'configs/stage2.json';policy=json.loads(policy_path.read_text(encoding='utf-8'))
 policy['training_enabled']=True
 policy_path.write_text(json.dumps(policy,indent=2)+'\n',encoding='utf-8')
 smoke=output/'RF03_smoke';smoke.mkdir()
 # No second fit even after a failed attempt without a new explicit instruction.
 prior=list((ROOT/'results/wrmsse_v1').glob('stage2_*/RF03_smoke/smoke_started.json'))
 if prior:raise RuntimeError('A smoke fit already started; automatic retry is disabled')
 write_new(smoke/'smoke_started.json',{'contract_id':CONTRACT_ID,'params':RF03,'purpose':'smoke','selection_eligible':False})
 try:
  with stage2_training_scope(evidence_rel):rf=run_rf(ds,features,smoke,ev,RF03)
 finally:
  policy['training_enabled']=False;policy_path.write_text(json.dumps(policy,indent=2)+'\n',encoding='utf-8')
 phase('integrity and reports')
 assert snapshot(raw)==before_raw,'Raw metadata changed'
 comparison=pd.DataFrame([baseline,rf]);comparison.to_csv(output/'comparison.csv',index=False)
 os.environ['MPLCONFIGDIR']=str(output/'mplconfig')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4));ax.bar(comparison.name,comparison.wrmsse,color=['#64748b','#0284c7']);ax.set_ylabel('Validation WRMSSE (12 M5 levels)');fig.tight_layout();fig.savefig(output/'validation_comparison.png',dpi=160);plt.close(fig)
 feature_rows=[]
 for name in ['rf_v1','rf_v2','rf_v3','rf_v4']:
  feature_rows.append({'set':name,'logical':specs[name]['n_raw_features'],'encoded':specs[name]['n_model_features']})
 for name,groups in plan()['FE01_FE06'].items():
  feature_rows.append({'set':name,'logical':17+sum(specs[g]['extra_logical'] for g in groups),'encoded':specs['rf_v3']['n_model_features']+sum(specs[g]['extra_encoded'] for g in groups)})
 pd.DataFrame(feature_rows).to_csv(output/'feature_counts.csv',index=False)
 summary={'status':'success','contract_id':CONTRACT_ID,'run':run,'processed':processed,'features':features,'tests_passed':result.testsRun,'train_rows':600000,'validation_rows':853720,'baseline':baseline,'rf03_smoke':rf,'feature_sets':feature_rows,'raw_unchanged':True,'test_actual_opened':False,'test_scored':False,'batch_tuning_executed':False,'final_model_trained':False,'smoke_selection_eligible':False}
 write_new(output/'summary.json',summary)
 write_new(output/'manifest.json',{'contract_id':CONTRACT_ID,'code_sha256':code_hash,'processed_manifest_sha256':sha(local_path(processed)/'manifest.json'),'features_manifest_sha256':sha(local_path(features)/'manifest.json'),'files':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and 'mplconfig' not in p.parts}})
 print(json.dumps(summary),flush=True)


def main():
 if len(sys.argv)>1 and sys.argv[1]=='--worker':
  try:worker(sys.argv[2])
  except Exception:
   output=local_path('results/wrmsse_v1/'+sys.argv[2]);write_new(output/'error.json',{'error':traceback.format_exc(),'status':'failed_safe'});raise
  return
 run='stage2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 output=local_path('results/wrmsse_v1/'+run);output.mkdir(parents=True,exist_ok=False)
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['JOBLIB_TEMP_FOLDER']=str(output/'tmp');env['MPLCONFIGDIR']=str(output/'mplconfig');env['OMP_NUM_THREADS']='4'
 policy=json.loads((ROOT/'configs/stage2.json').read_text())
 limits=policy['resource_limits'];peak=0;minimum=psutil.virtual_memory().available/2**20
 with (output/'worker.log').open('x',encoding='utf-8') as log,(output/'resources.jsonl').open('x',encoding='utf-8') as resource:
  proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',run],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   try:
    process=psutil.Process(proc.pid);rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True) if p.is_running())/2**20
    available=psutil.virtual_memory().available/2**20;peak=max(peak,rss);minimum=min(minimum,available)
    resource.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'rss_mib':rss,'available_mib':available})+'\n');resource.flush()
    if rss>limits['max_rss_mib'] or available<limits['min_available_mib']:
     for child in process.children(recursive=True):child.terminate()
     proc.terminate();proc.wait(timeout=30)
     write_new(output/'resource_stop.json',{'status':'stopped_safe','rss_mib':rss,'available_mib':available});break
   except psutil.NoSuchProcess:pass
   time.sleep(2)
 proc.wait()
 write_new(output/'resource_summary.json',{'peak_tree_rss_mib':peak,'min_available_mib':minimum,'returncode':proc.returncode})
 write_new(output/'manifest_completed.json',{'status':'completed' if proc.returncode==0 else 'failed_safe','contract_id':CONTRACT_ID,'files':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}})
 print(json.dumps({'run':run,'returncode':proc.returncode,'peak_tree_rss_mib':peak,'output':str(output)}))
 if proc.returncode:print((output/'worker.log').read_text(encoding='utf-8')[-8000:]);sys.exit(1)

if __name__=='__main__':main()
