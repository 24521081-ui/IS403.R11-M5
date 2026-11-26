"""Check event plumbing, independent recursion and probability blend against frozen C7."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_c7_events import forecast
from experiment_common import predict_blended

class Echo:
    def predict_components(self,x):
        return np.full(len(x),.5), x.lag_1.to_numpy()

class Platt:
    def predict_proba(self,x):
        return np.tile([.75,.25],(len(x),1))

def test_event_runner_matches_original_when_new_variables_are_unused():
    bundle=dict(model=Echo(),features=['lag_1'],categories=['B','A'])
    matrix=np.array([np.arange(60),np.arange(60)+10],dtype='float32')
    calendar=pd.DataFrame(dict(d=[f'd_{d}' for d in range(61,89)],day_of_week=0,month=1,event_flag=0,snap_CA=0))
    prices=dict(sell_price=np.ones(2),price_change_1w=np.zeros(2),price_ratio_4w=np.ones(2))
    original,_=predict_blended(bundle,Platt(),matrix,['B','A'],'CA_1','CA',calendar,60,prices,.75)
    got,components=forecast(bundle,matrix,['B','A'],'CA_1','CA',calendar,60,prices,[],Platt())
    np.testing.assert_array_equal(got.y_pred,original.y_pred)
    np.testing.assert_array_equal(components.p_blend,.25*components.p_raw+.75*components.p_cal)
    np.testing.assert_array_equal(matrix[0],np.arange(60))

def test_named_event_reaches_both_components_on_target_day():
    class EventModel:
        def predict_components(self,x):
            return x.event_name_00.to_numpy()*.5,np.full(len(x),4.)
    bundle=dict(model=EventModel(),features=['event_name_00'],categories=['A'])
    calendar=pd.DataFrame(dict(d=[f'd_{d}' for d in range(61,89)],day_of_week=0,month=1,event_flag=0,snap_CA=0,
        event_name_00=[1]+[0]*27))
    prices=dict(sell_price=np.ones(1),price_change_1w=np.zeros(1),price_ratio_4w=np.ones(1))
    pred,_=forecast(bundle,np.ones((1,60)),['A'],'CA_1','CA',calendar,60,prices,['event_name_00'])
    np.testing.assert_array_equal(pred.y_pred,[2.]+[0.]*27)
