import unittest
from unittest.mock import patch
from uuid import uuid4
import numpy as np
import pandas as pd
from src.wrmsse_config import CONFIG, CONTRACT_ID, fold, Fold, require_stage_permission
from src.wrmsse_data import weight_prices_from_tables, load_actuals, normalize_calendar
from src.wrmsse_pipeline import select_best_validation, fit_random_forest, run_validation_tuning, evaluate_forecast, assemble_forecast, validate_training_rows
from src.project_paths import ROOT, local_path, write_json_new
from src.wrmsse_features import feature_rows
from tests.test_wrmsse import fixture


class FoldAndStageSafety(unittest.TestCase):
    def test_exact_split_boundaries(self):
        self.assertEqual(CONFIG['train'],[1,1885])
        self.assertEqual(CONFIG['validation'],[1886,1913])
        self.assertEqual(CONFIG['test'],[1914,1941])
        self.assertEqual(fold('validation').history_days[-1],'d_1885')
        self.assertEqual(fold('test').history_days[-1],'d_1913')
        self.assertEqual(fold('validation').weight_columns,[f'd_{i}' for i in range(1858,1886)])
        self.assertEqual(fold('test').weight_columns,[f'd_{i}' for i in range(1886,1914)])
        self.assertEqual(len(fold('validation').forecast_days),28)
        self.assertEqual(len(fold('test').forecast_days),28)

    def test_training_and_tuning_blocked_before_data_or_model_access(self):
        for action in ['training','tuning','test_evaluation']:
            with self.assertRaises(RuntimeError): require_stage_permission(action)
        with self.assertRaises(RuntimeError): fit_random_forest(None,None,None)
        with self.assertRaises(RuntimeError): run_validation_tuning(None,None,None,None)

    def test_test_labels_and_scoring_blocked_before_any_file_read(self):
        with patch('src.wrmsse_data.pd.read_csv',side_effect=AssertionError('Test accessed')):
            with self.assertRaises(RuntimeError): load_actuals('test',None)
            with self.assertRaises(RuntimeError): evaluate_forecast('test',None)

    def test_selection_rejects_legacy_scores_and_wrong_folds(self):
        valid={'contract_id':CONTRACT_ID,'experiment_id':CONFIG['experiment_id'],'metric':'wrmsse',
            'split':'validation','origin':1885,'status':'success','wrmsse':1.0,'name':'new'}
        self.assertEqual(select_best_validation([valid,valid|{'wrmsse':.5,'name':'best'}])['name'],'best')
        for change in [{'metric':'mean_rmsse'},{'origin':1913},{'split':'test'},
                       {'contract_id':'legacy'},{'status':'failed'},{'wrmsse':float('nan')},
                       {'wrmsse':-1},{'experiment_id':'old'}]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError): select_best_validation([valid|change])
        with self.assertRaises(ValueError): select_best_validation([])


class RevenueNoLeakage(unittest.TestCase):
    def tables(self):
        m,h,p,s,a=fixture([[1,2,3,4]],origin=4)
        calendar=pd.DataFrame({'d':[f'd_{i}' for i in range(1,7)],'wm_yr_wk':[1,1,1,2,3,3]})
        prices=pd.DataFrame({'store_id':['s']*3,'item_id':['i0']*3,'wm_yr_wk':[1,2,3],'sell_price':[2.,3.,9999.]})
        return m,h,calendar,prices,s

    def test_raw_calendar_without_d_maps_from_date_even_when_shuffled(self):
        calendar=pd.DataFrame({'date':['2011-01-31','2011-01-29','2011-01-30'],'wm_yr_wk':[1,1,1]})
        normalized=normalize_calendar(calendar)
        self.assertEqual(normalized.d.tolist(),['d_3','d_1','d_2'])

    def test_calendar_date_and_day_disagreement_rejected(self):
        calendar=pd.DataFrame({'date':['2011-01-29'],'d':['d_2']})
        with self.assertRaises(ValueError): normalize_calendar(calendar)

    def test_price_join_uses_correct_week_and_only_past_window(self):
        m,h,c,p,s=self.tables()
        result=weight_prices_from_tables(m,h,c,p,s)
        np.testing.assert_allclose(result.to_numpy(),[[2,3]])
        self.assertEqual(float((h[s.weight_columns]*result).sum(axis=1).iloc[0]),18)

    def test_future_price_changes_do_not_affect_weights(self):
        m,h,c,p,s=self.tables()
        first=weight_prices_from_tables(m,h,c,p,s)
        p.loc[p.wm_yr_wk==3,'sell_price']=-999999
        second=weight_prices_from_tables(m,h,c,p,s)
        pd.testing.assert_frame_equal(first,second)

    def test_missing_price_for_positive_sales_rejected(self):
        m,h,c,p,s=self.tables();p=p[p.wm_yr_wk!=2]
        with self.assertRaisesRegex(ValueError,'Missing price'): weight_prices_from_tables(m,h,c,p,s)

    def test_missing_price_for_zero_units_allowed(self):
        m,h,c,p,s=self.tables();p=p[p.wm_yr_wk!=2];h['d_4']=0
        result=weight_prices_from_tables(m,h,c,p,s)
        self.assertEqual(result.iloc[0,1],0)

    def test_duplicate_prices_and_calendar_rejected(self):
        m,h,c,p,s=self.tables()
        with self.assertRaises(ValueError): weight_prices_from_tables(m,h,c,pd.concat([p,p.iloc[:1]]),s)
        with self.assertRaises(ValueError): weight_prices_from_tables(m,h,pd.concat([c,c.iloc[2:3]]),p,s)

    def test_invalid_in_window_price_rejected(self):
        m,h,c,p,s=self.tables();p.loc[p.wm_yr_wk==2,'sell_price']=-1
        with self.assertRaises(ValueError): weight_prices_from_tables(m,h,c,p,s)


class FeatureLeakage(unittest.TestCase):
    def tables(self):
        m,h,p,s,a=fixture([list(range(1,81))],origin=80,horizon=28)
        calendar=pd.DataFrame({'d':[f'd_{i}' for i in range(1,109)],'wday':[1]*108,'month':[1]*108,'year':[2011]*108})
        return m,h,calendar

    def test_direct_horizon_first_last_cutoffs(self):
        m,h,c=self.tables();keys,features=feature_rows(m,h,c,[81,108],origin=80)
        np.testing.assert_allclose(features['lag_28'],[53,80])
        np.testing.assert_allclose(features['rolling_mean_7'],[50,77])
        self.assertEqual(keys.day.tolist(),[81,108])

    def test_sales_after_first_day_cutoff_cannot_change_features(self):
        m,h,c=self.tables();_,first=feature_rows(m,h,c,[81],origin=80)
        h.loc[:,[f'd_{i}' for i in range(54,81)]]=99999
        _,second=feature_rows(m,h,c,[81],origin=80)
        pd.testing.assert_frame_equal(first,second)

    def test_future_history_and_invalid_targets_rejected(self):
        m,h,c=self.tables();h['d_81']=1
        with self.assertRaises(ValueError): feature_rows(m,h,c,[81],origin=80)
        m,h,c=self.tables()
        for days in [[109],[81,81],[1]]:
            with self.assertRaises(ValueError): feature_rows(m,h,c,days,origin=80)


class ForecastAndTrainingKeys(unittest.TestCase):
    def test_shuffled_predictions_follow_keys_instead_of_row_order(self):
        spec=Fold('unit',4,2,2)
        index=pd.MultiIndex.from_tuples([('i0','s'),('i1','s')])
        keys=pd.DataFrame({'bottom_key':[('i1','s'),('i0','s'),('i0','s'),('i1','s')],'day':[6,5,6,5]})
        result=assemble_forecast(keys,[26,15,16,25],index,spec)
        np.testing.assert_allclose(result.to_numpy(),[[15,16],[25,26]])

    def test_duplicate_forecast_grid_rejected(self):
        spec=Fold('unit',4,2,2)
        keys=pd.DataFrame({'bottom_key':['b0','b0'],'day':[5,5]})
        with self.assertRaises(ValueError): assemble_forecast(keys,[1,2],pd.Index(['b0']),spec)

    def test_training_keys_reject_old_validation_rows(self):
        keys=pd.DataFrame({'bottom_key':['b0'],'day':[1886]})
        with self.assertRaises(ValueError): validate_training_rows(keys,pd.DataFrame({'x':[1]}),[1])
        keys.day=1885
        validate_training_rows(keys,pd.DataFrame({'x':[1]}),[1])


class WriteIsolation(unittest.TestCase):
    def test_escaping_absolute_and_legacy_paths_rejected(self):
        for path in ['../escape.json','legacy/results/new.json',str(ROOT.parent/'other.json')]:
            with self.assertRaises(ValueError): local_path(path)

    def test_exclusive_creation_does_not_overwrite(self):
        relative='preparation/unit_'+uuid4().hex+'.json'
        path=write_json_new(relative,{'one':1})
        try:
            with self.assertRaises(FileExistsError): write_json_new(relative,{'two':2})
            self.assertIn('"one"',path.read_text())
        finally:
            self.assertTrue(path.resolve().is_relative_to(ROOT/'preparation'))
            path.unlink()

if __name__=='__main__': unittest.main()
