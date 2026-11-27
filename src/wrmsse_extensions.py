"""Ported causal price and civil-calendar definitions; original modules never imported."""

import numpy as np

import pandas as pd

def causal_cutoff(keys,origin=None):
    days=keys.day.to_numpy(dtype=np.int64)
    cutoff=days-28
    if origin is not None:
        if ((days<=origin)|(days>origin+28)).any():raise ValueError('Keys outside origin horizon')
        cutoff=np.minimum(cutoff,origin)
    if (cutoff<1).any():raise ValueError('No historical days')
    return cutoff

def snap_features(calendar,series,keys):
    states=series.set_index('series_id').loc[keys.series_id,'state_id'].astype(str).to_numpy()
    if not set(states).issubset({'CA','TX','WI'}):raise ValueError('Unknown state_id for SNAP join')
    lookup=calendar.set_index('day')
    result=np.empty(len(keys),np.float32)
    for state in ['CA','TX','WI']:
        mask=states==state
        result[mask]=lookup.loc[keys.day.to_numpy()[mask],'snap_'+state].to_numpy()
    if not np.isin(result,[0,1]).all():raise ValueError('Invalid SNAP flags')
    return result[:,None]

def event_features(calendar,keys,vocabulary):
    selected=calendar.set_index('day').loc[keys.day]
    columns=[];names=[]
    for kind,categories in vocabulary.items():
        left=selected[f'event_{kind}_1'].astype(str).to_numpy()
        right=selected[f'event_{kind}_2'].astype(str).to_numpy()
        for category in categories:
            columns.append((left==category)|(right==category));names.append(f'event_{kind}__{category}')
        known=set(categories)|{'NoEvent'}
        columns.append(~np.isin(left,list(known))|~np.isin(right,list(known)))
        names.append(f'event_{kind}__unknown')
    return np.column_stack(columns).astype(np.float32),names

class PriceHistory:
    """As-of join in actual week order, not arithmetic on wm_yr_wk codes."""
    def __init__(self,calendar,series,price_batches,history_end=1885):
        if series.series_id.duplicated().any() or series.duplicated(['item_id','store_id']).any():
            raise ValueError('series_id must map one-to-one to (item_id, store_id)')
        self.ids=pd.Index(series.series_id)
        weeks=calendar.groupby('wm_yr_wk',observed=True).agg(start=('day','min'),end=('day','max')).sort_values('start')
        weeks=weeks[(weeks.end<=history_end)&((weeks.end-weeks.start+1)==7)]
        self.weeks=pd.Index(weeks.index)
        self.ends=weeks.end.to_numpy(dtype=np.int64)
        self.values=np.full((len(series),len(weeks)),np.nan,np.float64)
        seen=np.zeros(self.values.shape,bool)
        self.rows_read=0;self.rows_used=0
        for batch in price_batches:
            if batch.duplicated(['series_id','wm_yr_wk']).any():raise ValueError('Duplicate price week keys')
            positions=self.ids.get_indexer(batch.series_id)
            if (positions<0).any():raise ValueError('Unknown series price join key')
            wi=self.weeks.get_indexer(batch.wm_yr_wk)
            self.rows_read+=len(batch)
            keep=wi>=0
            ri=positions[keep];ci=wi[keep]
            prices=batch.sell_price.to_numpy()[keep]
            if not np.isfinite(prices).all() or (prices<=0).any():raise ValueError('Observed prices must be positive finite values')
            if seen[ri,ci].any():raise ValueError('Duplicate price across batches')
            self.values[ri,ci]=prices;seen[ri,ci]=True;self.rows_used+=int(keep.sum())
        valid=np.isfinite(self.values)
        axis=np.arange(len(weeks),dtype=np.int32)
        self.last=np.maximum.accumulate(np.where(valid,axis[None,:],-1),axis=1)
        self.sums=np.cumsum(np.nan_to_num(self.values,nan=0.),axis=1)  # Internal masked sums, never raw-price imputation.
        self.counts=np.cumsum(valid,axis=1,dtype=np.int32)

    def features(self,keys,origin=None):
        cutoff=causal_cutoff(keys,origin)
        rows=self.ids.get_indexer(keys.series_id)
        if (rows<0).any():raise ValueError('Unknown series key')
        wi=np.searchsorted(self.ends,cutoff,side='right')-1
        safe_wi=np.maximum(wi,0)
        li=np.where(wi>=0,self.last[rows,safe_wi],-1)
        present=li>=0; safe_li=np.maximum(li,0)
        last=np.where(present,self.values[rows,safe_li],np.nan)
        pi=np.where(li>0,self.last[rows,np.maximum(li-1,0)],-1)
        previous=np.where(pi>=0,self.values[rows,np.maximum(pi,0)],np.nan)
        counts=np.where(wi>=0,self.counts[rows,safe_wi],0)
        means=np.divide(self.sums[rows,safe_wi],counts,out=np.full(len(keys),np.nan),where=counts>0)
        ratio=np.divide(last,previous,out=np.full(len(keys),np.nan),where=np.isfinite(previous))
        relative=np.divide(last,means,out=np.full(len(keys),np.nan),where=np.isfinite(means))
        age=np.where(present,cutoff-self.ends[safe_li],np.nan)
        current_missing=(wi<0)|~np.isfinite(self.values[rows,safe_wi])
        result=np.column_stack([last,ratio,relative,age,~present,pi<0,current_missing]).astype(np.float32)
        provenance=np.where(present,self.ends[safe_li],0)
        if (provenance>cutoff).any():raise ValueError('Future price accessed')
        return result,provenance

def impute_prices(train,validation):
    train=train.copy();validation=validation.copy();medians=[]
    for column in range(4):
        finite=np.isfinite(train[:,column])
        if not finite.any():raise ValueError('Train-only price median cannot be estimated')
        median=float(np.median(train[finite,column]));medians.append(median)
        if column<3 and median<=0:raise ValueError('Raw price/ratio imputation must be positive')
        for matrix in [train,validation]:matrix[~np.isfinite(matrix[:,column]),column]=median
    if not np.isfinite(train).all() or not np.isfinite(validation).all():raise ValueError('Invalid imputed price features')
    return train,validation,medians

FIXED={'NewYear':(1,1),'OrthodoxChristmas':(1,7),'ValentinesDay':(2,14),
       'StPatricksDay':(3,17),'Cinco De Mayo':(5,5),'IndependenceDay':(7,4),
       'Halloween':(10,31),'VeteransDay':(11,11),'Christmas':(12,25)}

NTH={'MartinLutherKingDay':(1,0,3),'PresidentsDay':(2,0,3),
     "Mother's day":(5,6,2),"Father's day":(6,6,3),
     'LaborDay':(9,0,1),'ColumbusDay':(10,0,2),'Thanksgiving':(11,3,4)}

def rule_matches(name,date):
    date=pd.Timestamp(date)
    if name in FIXED:return (date.month,date.day)==FIXED[name]
    if name in NTH:
        month,weekday,nth=NTH[name]
        return date.month==month and date.weekday()==weekday and (date.day-1)//7+1==nth
    if name=='MemorialDay':return date.month==5 and date.weekday()==0 and date.day>=25
    return False

def known_calendar(calendar):
    result=calendar.copy()
    for slot in [1,2]:
        name=f'event_name_{slot}';kind=f'event_type_{slot}'
        result[name]=result[name].astype(str);result[kind]=result[kind].astype(str)
        keep=np.array([rule_matches(n,d) for n,d in zip(result[name],result.date)],bool)
        result.loc[~keep,[name,kind]]='NoEvent'
    return result
