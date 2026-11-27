"""Fresh raw-only processed dataset, exact stratified sample and batched RF features."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from .project_paths import local_path
from .wrmsse_config import CONFIG, CONTRACT_ID
from .wrmsse_data import load_history,load_calendar,load_actuals
from .wrmsse import Hierarchy
from .wrmsse_extensions import PriceHistory,impute_prices,known_calendar,event_features,snap_features

CATS=['item_id','dept_id','cat_id','store_id','state_id']
BASE=['wday','month','year','lag_28','lag_35','lag_42','lag_56','rolling_mean_7','rolling_mean_28']
EXTRA=['zero_ratio_28','zero_ratio_56','nonzero_count_28','days_since_last_sale','rolling_std_28','rolling_mean_56','sales_trend_nonoverlap_28']
PRICE=['historical_last_price','historical_price_change_ratio','historical_price_relative_mean','historical_price_age_days','historical_price_missing','historical_price_previous_missing','historical_price_week_missing']
GROUPS={'recency':['days_since_last_sale'],'prices':PRICE,'snap':['snap_state'],'behavior':EXTRA[4:]}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(2**20),b''):h.update(part)
 return h.hexdigest()


def write_new(path,value):
 path=Path(path); assert path.resolve().is_relative_to(local_path('data').parent)
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def sample_training(series,calendar,n=600000,seed=42,end=1885):
 cal=calendar[calendar.day.between(57,end)].copy()
 cal['quarter']=pd.to_datetime(cal.date).dt.to_period('Q').astype(str)
 quarters=[(str(q),g.day.to_numpy(dtype=np.int32)) for q,g in cal.groupby('quarter',sort=True)]
 records=[];members=[]
 for (store,dept),g in series.groupby(['store_id','dept_id'],observed=True,sort=True):
  for q,days in quarters:
   ids=np.sort(g.series_id.to_numpy(dtype=np.int32))
   records.append({'store_id':str(store),'dept_id':str(dept),'quarter':q,'available_rows':len(ids)*len(days)})
   members.append((ids,days))
 strata=pd.DataFrame(records);available=strata.available_rows.to_numpy(np.int64)
 if not len(strata)<=n<=available.sum():raise ValueError('Sample size cannot cover every stratum')
 ideal=(available-1)/(available-1).sum()*(n-len(strata))
 quota=np.floor(ideal).astype(np.int64)+1
 order=np.lexsort((np.random.default_rng(seed).random(len(strata)),-(ideal-np.floor(ideal))))
 quota[order[:int(n-quota.sum())]]+=1
 strata['sample_rows']=quota;blocks=[]
 for number,((ids,days),count) in enumerate(zip(members,quota)):
  rng=np.random.default_rng(np.random.SeedSequence([seed,number]))
  chosen=rng.choice(len(ids)*len(days),int(count),replace=False)
  blocks.append(pd.DataFrame({'series_id':ids[chosen%len(ids)],'day':days[chosen//len(ids)]}))
 keys=pd.concat(blocks,ignore_index=True).sort_values(['series_id','day']).reset_index(drop=True)
 if len(keys)!=n or keys.duplicated().any() or not keys.day.between(57,end).all():raise AssertionError('Invalid sample')
 return keys,strata


def numeric_block(values,rows,days):
 """Observed partial windows at early dates; no fictitious pre-d_1 zeros."""
 values=np.asarray(values,dtype=np.float64); rows=np.asarray(rows,dtype=int);days=np.asarray(days,dtype=int)
 right=days-28
 if (days<57).any() or (right>values.shape[1]).any():raise ValueError('Unsafe history cutoff')
 pref=np.pad(np.cumsum(values,axis=1),((0,0),(1,0)))
 zero=np.pad(np.cumsum(values==0,axis=1),((0,0),(1,0)))
 sq=np.pad(np.cumsum(values*values,axis=1),((0,0),(1,0)))
 last=np.maximum.accumulate(np.where(values>0,np.arange(1,values.shape[1]+1)[None,:],0),axis=1)
 left28=np.maximum(0,right-28);left56=np.maximum(0,right-56)
 mean28=(pref[rows,right]-pref[rows,left28])/(right-left28)
 previous=(pref[rows,left28]-pref[rows,left56])/(left28-left56)
 z28=(zero[rows,right]-zero[rows,left28])/(right-left28)
 z56=(zero[rows,right]-zero[rows,left56])/(right-left56)
 since=right-last[rows,right-1]
 std=np.sqrt(np.maximum(0,(sq[rows,right]-sq[rows,left28])/(right-left28)-mean28**2))
 data={f'lag_{k}':values[rows,days-k-1] for k in (28,35,42,56)}
 data.update(rolling_mean_7=(pref[rows,right]-pref[rows,right-7])/7,rolling_mean_28=mean28,
  zero_ratio_28=z28,zero_ratio_56=z56,nonzero_count_28=(right-left28)*(1-z28),days_since_last_sale=since,
  rolling_std_28=std,rolling_mean_56=(pref[rows,right]-pref[rows,left56])/(right-left56),sales_trend_nonoverlap_28=mean28-previous)
 return pd.DataFrame(data,dtype=np.float32)


class Dataset:
 def __init__(self,directory):
  self.root=local_path(str(directory))
  manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
  if manifest['contract_id']!=CONTRACT_ID:raise ValueError('Stale dataset contract')
  for name,digest in manifest['files'].items():
   if sha(self.root/name)!=digest:raise ValueError('Processed file changed: '+name)
  self.series=pd.read_parquet(self.root/'series.parquet')
  self.calendar=pd.read_parquet(self.root/'calendar.parquet')
  self.history=np.load(self.root/'train_sales.npy',mmap_mode='r')
  self.actual=np.load(self.root/'validation_sales.npy',mmap_mode='r')
  self.train_keys=pd.read_parquet(self.root/'train_keys.parquet')
  self.validation_keys=pd.read_parquet(self.root/'validation_keys.parquet')
  self.index=pd.MultiIndex.from_frame(self.series[['item_id','store_id']])


def build_processed(relative):
 directory=local_path(relative);directory.mkdir(parents=True,exist_ok=False)
 metadata,history=load_history('validation');Hierarchy(metadata,strict_m5=True)
 values=history.to_numpy(dtype=float)
 if not (values==np.floor(values)).all() or values.max()>65535:raise ValueError('Invalid count data')
 actual=load_actuals('validation',metadata.index)
 if actual.shape!=(30490,28):raise ValueError('Incomplete Validation')
 series=metadata.reset_index(drop=True);series.insert(0,'series_id',np.arange(len(series),dtype=np.int32))
 calendar=load_calendar();calendar['day']=calendar.d.str[2:].astype(int)
 keys,strata=sample_training(series,calendar)
 val=pd.DataFrame({'series_id':np.repeat(np.arange(len(series),dtype=np.int32),28),'day':np.tile(np.arange(1886,1914,dtype=np.int32),len(series))})
 for name,array in [('train_sales',values.astype(np.uint16)),('validation_sales',actual.to_numpy(dtype=np.float32))]:
  with (directory/(name+'.npy')).open('xb') as f:np.save(f,array,allow_pickle=False)
 for name,table in [('series',series),('calendar',calendar),('train_keys',keys),('validation_keys',val),('strata',strata)]:table.to_parquet(directory/(name+'.parquet'),index=False)
 inputs={p.name:sha(p) for p in [local_path(CONFIG['sales_source']),local_path('data/raw/calendar.csv')]}
 write_new(directory/'manifest.json',{'status':'complete','contract_id':CONTRACT_ID,'inputs':inputs,'train':[1,1885],'validation':[1886,1913],'test':{'days':[1914,1941],'status':'locked_unmaterialized'},'bottom_series':30490,'train_total_observations':30490*1885,'train_eligible_observations':30490*(1885-57+1),'train_sample':600000,'validation_observations':853720,'seed':42,'sampling':'store x dept x calendar quarter; one per stratum then proportional remaining capacity; largest remainder; SeedSequence per stratum','files':{p.name:sha(p) for p in directory.iterdir()}})
 return Dataset(relative)


class FeatureBuilder:
 def __init__(self,dataset):
  self.ds=dataset;self.series=dataset.series;self.calendar=dataset.calendar
  self.encoder=OneHotEncoder(categories=[sorted(self.series[c].unique()) for c in CATS],handle_unknown='error',sparse_output=True,dtype=np.float32).fit(self.series[CATS])
  self.cat_names=self.encoder.get_feature_names_out(CATS).tolist()
  cal=known_calendar(self.calendar)
  train=cal[cal.day<=1885]
  self.vocabulary={kind:sorted(set().union(*(set(train[f'event_{kind}_{i}'].fillna('NoEvent').astype(str)) for i in (1,2)))-{'NoEvent'}) for kind in ('name','type')}
  self.events_calendar=cal
  self.event_names=[f'event_{kind}__{category}' for kind,categories in self.vocabulary.items() for category in categories+['unknown']]
  self.groups=GROUPS|{'events':self.event_names}

 def sales_features(self,keys):
  if keys.duplicated(['series_id','day']).any() or not keys.day.between(57,1913).all() or not keys.series_id.isin(self.series.series_id).all():raise ValueError('Invalid feature keys')
  result=np.empty((len(keys),len(BASE)+len(EXTRA)),np.float32)
  cal=self.calendar.set_index('day')
  for start in range(0,len(self.series),256):
   selected=np.flatnonzero(keys.series_id.between(start,start+255).to_numpy())
   if not len(selected):continue
   block=keys.iloc[selected];rows=block.series_id.to_numpy()-start;days=block.day.to_numpy()
   num=numeric_block(self.ds.history[start:start+256],rows,days)
   for c in ('wday','month','year'):num[c]=cal.loc[days,c].to_numpy()
   result[selected]=num[BASE+EXTRA].to_numpy()
  if not np.isfinite(result).all():raise ValueError('Invalid numeric features')
  return pd.DataFrame(result,columns=BASE+EXTRA)

 def base(self,keys,numeric,version):
  extra={'rf_v1':[],'rf_v2':EXTRA[:2],'rf_v3':EXTRA[:3],'rf_v4':EXTRA[:4]}[version]
  ids=self.series.iloc[keys.series_id.to_numpy()][CATS]
  names=self.cat_names+BASE+extra
  X=sparse.hstack([self.encoder.transform(ids),sparse.csr_matrix(numeric[BASE+extra].to_numpy())],format='csr',dtype=np.float32)
  return X,{'raw_features':CATS+BASE+extra,'feature_names':names,'n_raw_features':len(CATS+BASE+extra),'n_model_features':len(names)}

 def price_history(self):
  cal=self.calendar;weeks=cal.groupby('wm_yr_wk').day.max();allowed=set(weeks[weeks<=1885].index)
  index=pd.MultiIndex.from_frame(self.series[['store_id','item_id']])
  def batches():
   with pd.read_csv(local_path('data/raw/sell_prices.csv'),chunksize=100000) as reader:
    for batch in reader:
     batch=batch[batch.wm_yr_wk.isin(allowed)].copy()
     positions=index.get_indexer(pd.MultiIndex.from_frame(batch[['store_id','item_id']]))
     if (positions<0).any():raise ValueError('Unknown price keys')
     batch['series_id']=positions;yield batch
  return PriceHistory(cal,self.series,batches(),history_end=1885)

 def extensions(self,train_numeric,val_numeric):
  trk=self.ds.train_keys;vak=self.ds.validation_keys
  ph=self.price_history();trprice,trprov=ph.features(trk);vaprice,vaprov=ph.features(vak,origin=1885)
  trprice,vaprice,medians=impute_prices(trprice,vaprice)
  outputs=[]
  for keys,numeric,price in [(trk,train_numeric,trprice),(vak,val_numeric,vaprice)]:
   values=numeric[EXTRA[3:]].copy()
   for i,name in enumerate(PRICE):values[name]=price[:,i]
   values['snap_state']=snap_features(self.calendar,self.series,keys).ravel()
   events,names=event_features(self.events_calendar.fillna('NoEvent'),keys,self.vocabulary)
   for i,name in enumerate(names):values[name]=events[:,i]
   outputs.append(values)
  return outputs,{'price_train_medians':medians,'event_vocabulary':self.vocabulary,'events_policy':'Verified civil-calendar holidays only; no sporting or unverified lunar schedules','snap_assumption':'Target-day SNAP schedule assumed announced before origin','max_price_source_day_train':int(trprov.max()),'max_price_source_day_validation':int(vaprov.max())}


def build_features(dataset,relative):
 directory=local_path(relative);directory.mkdir(parents=True,exist_ok=False)
 builder=FeatureBuilder(dataset);tr=builder.sales_features(dataset.train_keys);va=builder.sales_features(dataset.validation_keys)
 specs={}
 for version in ('rf_v1','rf_v2','rf_v3','rf_v4'):
  dest=directory/version;dest.mkdir()
  for split,keys,nums in [('train',dataset.train_keys,tr),('validation',dataset.validation_keys,va)]:
   X,spec=builder.base(keys,nums,version);sparse.save_npz(dest/('X_'+split+'.npz'),X)
  specs[version]=spec
  write_new(dest/'feature_spec.json',spec)
 extras,details=builder.extensions(tr,va)
 for group,columns in builder.groups.items():
  dest=directory/group;dest.mkdir()
  for split,frame in zip(('train','validation'),extras):
   sparse.save_npz(dest/('extra_'+split+'.npz'),sparse.csr_matrix(frame[columns].to_numpy(dtype=np.float32)))
  specs[group]={'extra_names':columns,'extra_encoded':len(columns),'extra_logical':4 if group=='events' else len(columns)}
 labels=dataset.history[dataset.train_keys.series_id.to_numpy(),dataset.train_keys.day.to_numpy()-1].astype(np.float32)
 with (directory/'y_train.npy').open('xb') as f:np.save(f,labels,allow_pickle=False)
 write_new(directory/'manifest.json',{'contract_id':CONTRACT_ID,'dataset':dataset.root.relative_to(local_path('data').parent).as_posix(),'dataset_manifest_sha256':sha(dataset.root/'manifest.json'),'specs':specs,'details':details,'files':{p.relative_to(directory).as_posix():sha(p) for p in directory.rglob('*') if p.is_file()}})
 return specs


def load_matrix(directory,version,split,groups=()):
 directory=local_path(directory)
 manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
 if manifest['contract_id']!=CONTRACT_ID:raise ValueError('Wrong feature contract')
 paths=[directory/version/('X_'+split+'.npz')]+[directory/g/('extra_'+split+'.npz') for g in groups]
 for p in paths:
  if sha(p)!=manifest['files'][p.relative_to(directory).as_posix()]:raise ValueError('Feature file modified')
 X=sparse.hstack([sparse.load_npz(p) for p in paths],format='csr',dtype=np.float32)
 if not np.isfinite(X.data).all():raise ValueError('Nonfinite feature matrix')
 return X
