import unittest,json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.wrmsse_stage2_data import sample_training,numeric_block,FeatureBuilder
from src.wrmsse_workflows import seasonal_naive,plan,adaptive_fe,select_new,execute_stage
from src.wrmsse_extensions import PriceHistory,known_calendar
from src.wrmsse_config import require_stage_permission,stage2_training_scope


def PathRoot():
 from src.wrmsse_config import ROOT
 return str(ROOT/'preparation')


def fixture():
 n=4
 series=pd.DataFrame({'series_id':range(n),'item_id':['a','b','a','b'],'dept_id':['d']*n,'cat_id':['c']*n,'store_id':['s1','s1','s2','s2'],'state_id':['CA','CA','TX','TX']})
 dates=pd.date_range('2011-01-29',periods=1913)
 calendar=pd.DataFrame({'day':np.arange(1,1914),'d':[f'd_{i}' for i in range(1,1914)],'date':dates,'wm_yr_wk':np.arange(1913)//7,'wday':(dates.dayofweek+2)%7+1,'month':dates.month,'year':dates.year})
 for c in ['snap_CA','snap_TX','snap_WI']:calendar[c]=0
 for kind in ['name','type']:
  for i in [1,2]:calendar[f'event_{kind}_{i}']='NoEvent'
 ds=SimpleNamespace(series=series,calendar=calendar,history=np.tile(np.arange(1,1886),(n,1)))
 return ds


class Sampling(unittest.TestCase):
 def test_seed_repeatable_strata_and_unique_new_train(self):
  ds=fixture();a,strata=sample_training(ds.series,ds.calendar,300,42,end=150)
  b,_=sample_training(ds.series,ds.calendar,300,42,end=150)
  c,_=sample_training(ds.series,ds.calendar,300,43,end=150)
  pd.testing.assert_frame_equal(a,b);self.assertFalse(a.equals(c))
  self.assertEqual(len(a),300);self.assertFalse(a.duplicated().any());self.assertTrue(a.day.between(57,150).all())
  self.assertTrue((strata.sample_rows>=1).all());self.assertEqual(strata.sample_rows.sum(),300)
 def test_sample_cannot_exceed_population(self):
  ds=fixture()
  with self.assertRaises(ValueError):sample_training(ds.series,ds.calendar,1000000,end=60)


class CausalFeatures(unittest.TestCase):
 def test_exact_early_partial_windows_and_no_sale_recency(self):
  h=np.zeros((1,1885));h[0,0]=2
  f=numeric_block(h,[0],[57])
  self.assertAlmostEqual(f.zero_ratio_56.iloc[0],28/29,places=6)
  self.assertEqual(f.days_since_last_sale.iloc[0],28)
  self.assertEqual(f.rolling_mean_56.iloc[0],np.float32(2/29))
  self.assertEqual(numeric_block(np.zeros_like(h),[0],[57]).days_since_last_sale.iloc[0],29)
 def test_partial_behavior_population_std_and_trend(self):
  h=np.arange(1,1886)[None,:]
  f=numeric_block(h,[0],[57])
  self.assertAlmostEqual(f.rolling_std_28.iloc[0],np.std(np.arange(2,30)),places=5)
  self.assertEqual(f.sales_trend_nonoverlap_28.iloc[0],14.5)
 def test_mutating_after_cutoff_does_not_affect_validation_day(self):
  h=np.arange(1,1886)[None,:];a=numeric_block(h,[0],[1886]);h[:,1858:]=999
  pd.testing.assert_frame_equal(a,numeric_block(h,[0],[1886]))
 def test_history_cannot_cover_test_feature_keys(self):
  with self.assertRaises(ValueError):numeric_block(np.ones((1,1885)),[0],[1914])
 def test_all_rf_schemas_and_identical_encoding(self):
  ds=fixture();builder=FeatureBuilder(ds)
  keys=pd.DataFrame({'series_id':[0,1,2,3],'day':[57,100,1886,1913]})
  nums=builder.sales_features(keys)
  for v,count in [('rf_v1',14),('rf_v2',16),('rf_v3',17),('rf_v4',18)]:
   matrix,spec=builder.base(keys,nums,v)
   self.assertEqual(spec['n_raw_features'],count)
   self.assertEqual(matrix.shape[1],len(builder.cat_names)+count-5)
   self.assertTrue(np.isfinite(matrix.data).all())
  reordered=keys.iloc[::-1].reset_index(drop=True)
  matrix2,_=builder.base(reordered,builder.sales_features(reordered),'rf_v1')
  matrix1,_=builder.base(keys,nums,'rf_v1')
  np.testing.assert_allclose(matrix1.toarray()[::-1],matrix2.toarray())
 def test_price_only_complete_weeks_no_future_and_missing(self):
  ds=fixture();cal=ds.calendar[ds.calendar.day<=28]
  batch=pd.DataFrame({'series_id':[0,0,0],'wm_yr_wk':[0,1,3],'sell_price':[2.,4.,999.]})
  ph=PriceHistory(cal,ds.series,[batch],history_end=21)
  keys=pd.DataFrame({'series_id':[0,1],'day':[49,49]})
  features,source=ph.features(keys)
  self.assertEqual(features[0,0],4)
  self.assertEqual(source[0],14)
  self.assertTrue(np.isnan(features[1,0]));self.assertEqual(features[1,4],1)
 def test_events_unknown_calendar_policy(self):
  ds=fixture();cal=ds.calendar.copy();cal.loc[0,'event_name_1']='SuperBowl'
  self.assertEqual(known_calendar(cal).loc[0,'event_name_1'],'NoEvent')
 def test_unknown_series_key_rejected(self):
  builder=FeatureBuilder(fixture())
  with self.assertRaises(ValueError):builder.sales_features(pd.DataFrame({'series_id':[99],'day':[1886]}))
 def test_features_reject_validation_actual_input(self):
  ds=fixture();builder=FeatureBuilder(ds)
  with self.assertRaises(ValueError):builder.sales_features(pd.DataFrame({'series_id':[0],'day':[1914]}))


class BaselineAndWorkflow(unittest.TestCase):
 def test_seasonal_repeats_last_seven_for_all_28_days(self):
  h=np.arange(1,1886)[None,:]
  expected=np.tile(h[:,-7:],(1,4));np.testing.assert_array_equal(seasonal_naive(h),expected)
 def test_all_selection_entrypoints_reject_smoke(self):
  from src.wrmsse_pipeline import select_best_validation
  from src.wrmsse_config import CONTRACT_ID,CONFIG
  row={'name':'RF-03','purpose':'smoke','contract_id':CONTRACT_ID,'experiment_id':CONFIG['experiment_id'],'metric':'wrmsse','split':'validation','origin':1885,'status':'success','wrmsse':.5}
  with self.assertRaises(ValueError):select_best_validation([row])
 def test_future_tuning_scope_is_disabled(self):
  from src.wrmsse_config import stage3_tuning_scope
  with self.assertRaises(RuntimeError):
   with stage3_tuning_scope('preparation/not_used.json'):pass
 def test_ordered_batch_entrypoint_stays_locked(self):
  from src.wrmsse_workflows import run_ordered_tuning
  with self.assertRaises(RuntimeError):run_ordered_tuning(None,None,None)
 def test_first_stage_uses_rf_v1(self):
  from src.wrmsse_workflows import execute_stage
  from tempfile import TemporaryDirectory
  from unittest.mock import patch
  with patch('src.wrmsse_workflows.require_stage_permission'),patch('src.wrmsse_workflows.run_rf',return_value={}) as fit:
   with TemporaryDirectory(dir=PathRoot()) as folder:
    from src.wrmsse_config import ROOT
    relative=__import__('pathlib').Path(folder).relative_to(ROOT).as_posix()
    execute_stage('rf_v1_RF01_RF05',[],[{'name':'RF-01','params':{}}],None,'unused',relative,None)
    self.assertEqual(fit.call_args.args[5],'rf_v1')
 def test_smoke_not_eligible_for_selection(self):
  with self.assertRaises(ValueError):select_new([{'purpose':'smoke'}])
 def test_workflow_order_and_expected_parameters(self):
  p=plan();self.assertEqual(p['RF01_RF05'][2]['params']['n_estimators'],100)
  self.assertEqual(len(p['RF07_RF13']),7);self.assertEqual(len(p['RF19_RF22']),4)
  self.assertEqual(p['order'][0],'rf_v1_RF01_RF05')
  with self.assertRaises(RuntimeError):execute_stage('rf_v3_RF06',[],[],None,None,None,None)
 def test_training_outside_scope_and_test_stay_locked(self):
  for action in ['training','test_evaluation','tuning']:
   with self.assertRaises(RuntimeError):require_stage_permission(action)
 def test_all_real_raw_access_blocks_test_and_legacy(self):
  from src.wrmsse_data import load_actuals
  from src.project_paths import local_path
  with patch('pandas.read_csv',side_effect=AssertionError('Actual Test opened')):
   with self.assertRaises(RuntimeError):load_actuals('test',None)
  with self.assertRaises(ValueError):local_path('legacy/data/features')

if __name__=='__main__':unittest.main()
