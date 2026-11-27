"""Audit real histories, hierarchy, revenue and scales; never score any forecasts.

Test actuals d_1914..d_1941 are neither loaded nor evaluated. The two historical
contexts are validated using only columns <= their forecast origins.
"""
from pathlib import Path
import csv
import gc
import json
import math
import sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.wrmsse_config import fold, CONFIG, CONTRACT_ID
from src.wrmsse_data import load_history, load_weight_prices
from src.wrmsse import WRMSSEEvaluator, M5_COUNTS
from src.project_paths import local_path, write_json_new


def manual_scale(values):
    first=next((i for i,x in enumerate(values) if x != 0),None)
    if first is None or len(values)-first<2: return float('nan')
    active=values[first:]
    return math.fsum((float(active[i])-float(active[i-1]))**2 for i in range(1,len(active)))/(len(active)-1)


def main():
    report={'contract_id':CONTRACT_ID,'status':'passed','kind':'history_only_preparation_audit',
            'training_executed':False,'tuning_executed':False,'test_actuals_loaded':False,
            'forecast_scoring_executed':False,'contexts':{}}
    for split in ['validation','test']:
        metadata,history=load_history(split)
        prices=load_weight_prices(metadata,history,split)
        spec=fold(split)
        evaluator=WRMSSEEvaluator.fit(metadata,history,prices,spec,strict_m5=True)
        nodes=evaluator.hierarchy.nodes
        # Separate pandas grouping revenue oracle (no sparse roll-up).
        money=(history[spec.weight_columns]*prices).sum(axis=1).to_numpy()
        revenue_table=metadata.reset_index(drop=True).assign(dollars=money)
        revenue_columns=[[],['state_id'],['store_id'],['cat_id'],['dept_id'],
                         ['state_id','cat_id'],['state_id','dept_id'],
                         ['store_id','cat_id'],['store_id','dept_id'],
                         ['item_id'],['item_id','state_id'],['item_id','store_id']]
        references=[]
        for level,columns in enumerate(revenue_columns,1):
            if not columns: mapping={('Total',):float(money.sum())}
            else:
                grouped=revenue_table.groupby(columns,observed=True,sort=True).dollars.sum()
                mapping={((str(k),) if not isinstance(k,tuple) else tuple(map(str,k))):float(v) for k,v in grouped.items()}
            references.extend(mapping[key]/money.sum()/12 for key in nodes.loc[nodes.level==level,'key'])
        max_weight_error=float(np.max(np.abs(evaluator.weights-np.asarray(references))))
        assert max_weight_error < 1e-12
        # Independent scalar scale checks: total + first/middle/last bottom series.
        sample_positions=[0,len(history)//2,len(history)-1]
        bottom_rows=nodes.index[nodes.level==12]
        key_to_row={nodes.loc[i,'key']:i for i in bottom_rows}
        scale_checks=[]
        total_scale=manual_scale(history.to_numpy().sum(axis=0).tolist())
        scale_checks.append({'node':'Total','absolute_error':abs(float(evaluator.scales[0])-total_scale)})
        for position in sample_positions:
            row=metadata.iloc[position]
            key=(str(row.item_id),str(row.store_id))
            expected=manual_scale(history.iloc[position].tolist())
            measured=float(evaluator.scales[key_to_row[key]])
            assert np.isclose(measured,expected,rtol=1e-12,atol=1e-12)
            scale_checks.append({'node':list(key),'absolute_error':abs(measured-expected)})
        assert max(c['absolute_error'] for c in scale_checks)<1e-9
        weights_by_level=pd.Series(evaluator.weights).groupby(nodes.level).sum()
        np.testing.assert_allclose(weights_by_level,1/12,rtol=1e-12)
        context={'history_days':[1,spec.origin], 'weight_days':[spec.origin-27,spec.origin],
            'forecast_days':[spec.origin+1,spec.origin+28], 'bottom_series':len(metadata),
            'hierarchy_series':len(nodes),'counts_by_level':list(evaluator.hierarchy.counts),
            'weight_sum':float(evaluator.weights.sum()),'weights_by_level':weights_by_level.to_dict(),
            'dollar_sales':float(money.sum()),'zero_weight_nodes':int((evaluator.weights==0).sum()),
            'scale_status_counts':pd.Series(evaluator.statuses).value_counts().to_dict(),
            'independent_weight_max_absolute_error':max_weight_error,'independent_scale_checks':scale_checks}
        if split=='test':
            # This official M5 file belongs to origin d_1913. It is a reference fixture,
            # never an input to fitting the evaluator; its weights sum to one per level.
            official=pd.read_csv(local_path('data/raw/weights_validation.csv'))
            lookup={(int(r.Level_id.replace('Level','')),str(r.Agg_Level_1),str(r.Agg_Level_2)):(float(r.Dollar_Sales),float(r.weight)) for r in official.itertuples(index=False)}
            dollars_error=[];weight_error=[]
            for i,node in nodes.iterrows():
                key=tuple(map(str,node['key']))
                agg1=key[0];agg2=key[1] if len(key)>1 else 'X'
                # Level1's literal label differs in the supplied official CSV.
                if node.level==1:
                    candidates=[k for k in lookup if k[0]==1]
                    fixture_key=candidates[0]
                elif node.level==11: fixture_key=(11,agg2,agg1)  # Official CSV stores state before item.
                else: fixture_key=(int(node.level),agg1,agg2)
                if fixture_key not in lookup:
                    raise ValueError(f'Unknown official reference key {fixture_key}; verify key convention')
                dollars,weight=lookup[fixture_key]
                recreated=evaluator.weights[i]*12*money.sum()
                dollars_error.append(abs(recreated-dollars))
                weight_error.append(abs(evaluator.weights[i]*12-weight))
            assert max(dollars_error)<.011
            assert max(weight_error)<1e-7
            context['official_reference']={'file':'data/raw/weights_validation.csv','fixture_origin':1913,
                'compared_nodes':len(nodes),'max_dollar_absolute_error':max(dollars_error),
                'max_within_level_weight_absolute_error':max(weight_error),
                'note':'Rounding tolerance for supplied CSV; weights recreated from raw unit sales and weekly prices.'}
        report['contexts'][split]=context
        print(json.dumps({'split':split,'nodes':len(nodes),'scale_status':context['scale_status_counts'],'dollar_sales':context['dollar_sales']}),flush=True)
        del metadata,history,prices,evaluator,revenue_table
        gc.collect()
    write_json_new('preparation/data_context_audit.json',report)
    print('Historical context audit passed; no model predictions or Test labels used.')

if __name__=='__main__': main()
