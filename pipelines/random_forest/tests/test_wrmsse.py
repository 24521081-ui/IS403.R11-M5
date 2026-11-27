import math
import unittest
import numpy as np
import pandas as pd
from src.wrmsse import Hierarchy, WRMSSEEvaluator, UndefinedScaleError
from src.wrmsse_config import Fold
from tests.reference_wrmsse import reference_wrmsse


def fixture(values=None, *, origin=4, horizon=2, weight_days=2):
    values = [[1,2,3,4]] if values is None else values
    n = len(values)
    metadata = pd.DataFrame({"item_id":[f"i{i}" for i in range(n)], "dept_id":["d"]*n,
        "cat_id":["c"]*n,"store_id":["s"]*n,"state_id":["state"]*n},index=[f"b{i}" for i in range(n)])
    spec = Fold("unit", origin, horizon, weight_days)
    history = pd.DataFrame(values, index=metadata.index, columns=spec.history_days,dtype=float)
    prices = pd.DataFrame(1.0,index=metadata.index,columns=spec.weight_columns)
    actual = pd.DataFrame(2.0,index=metadata.index,columns=spec.forecast_days)
    return metadata,history,prices,spec,actual


def fitted(parts):
    m,h,p,s,a=parts
    return WRMSSEEvaluator.fit(m,h,p,s)


class HandCalculations(unittest.TestCase):
    def test_perfect_forecast(self):
        parts=fixture()
        self.assertEqual(fitted(parts).score(parts[-1],parts[-1]).wrmsse,0.0)

    def test_unit_error(self):
        parts=fixture()
        result=fitted(parts).score(parts[-1],parts[-1]+1)
        self.assertAlmostEqual(result.wrmsse,1.0,places=14)
        np.testing.assert_allclose(result.per_level.weight,1/12)
        np.testing.assert_allclose(result.per_level.wrmsse_within_level,1)

    def test_leading_zeros_and_internal_zeros(self):
        parts=fixture([[0,0,2,0,4]],origin=5)
        evaluator=fitted(parts)
        np.testing.assert_allclose(evaluator.scales,10)
        actual=parts[-1]
        predicted=actual+np.array([[1,-1]])
        self.assertAlmostEqual(evaluator.score(actual,predicted).wrmsse,1/math.sqrt(10),places=14)

    def test_revenue_weight_hand_result_point_three(self):
        m,h,p,s,a=fixture([[1,2,3,4],[2,4,6,8]])
        p.loc['b1']=2
        evaluator=WRMSSEEvaluator.fit(m,h,p,s)
        predicted=a.copy();predicted.loc['b0']+=1
        # Nine upper levels: 9/12 * 1/3. Three lower levels: 3/12 * .2.
        self.assertAlmostEqual(evaluator.score(a,predicted).wrmsse,.3,places=14)
        np.testing.assert_allclose(evaluator.weights[-2:],[.2/12,.8/12])

    def test_opposite_errors_cancel_only_after_aggregation(self):
        parts=fixture([[1,2,3,4],[2,3,4,5]])
        a=parts[-1];p=a.copy();p.loc['b0']+=1;p.loc['b1']-=1
        result=fitted(parts).score(a,p)
        self.assertAlmostEqual(result.wrmsse,.25,places=14)
        np.testing.assert_allclose(result.per_level.loc[1:9,'wrmsse_within_level'],0)

    def test_scale_invariance_when_units_and_forecasts_scaled(self):
        m,h,p,s,a=fixture([[1,2,3,4],[2,4,6,8]])
        first=WRMSSEEvaluator.fit(m,h,p,s).score(a,a+1).wrmsse
        second=WRMSSEEvaluator.fit(m,h*7,p,s).score(a*7,(a+1)*7).wrmsse
        self.assertAlmostEqual(first,second,places=14)

    def test_signed_forecasts_no_implicit_clipping(self):
        parts=fixture()
        self.assertAlmostEqual(fitted(parts).score(parts[-1],parts[-1]*0-1).wrmsse,3)


class SpecialSeries(unittest.TestCase):
    def test_zero_sales_series_zero_weight_not_renormalized(self):
        parts=fixture([[1,2,3,4],[0,0,0,0]])
        evaluator=fitted(parts)
        self.assertAlmostEqual(evaluator.weights.sum(),1)
        result=evaluator.score(parts[-1],parts[-1]+1)
        self.assertTrue(np.isfinite(result.wrmsse))
        zero=result.per_series.query('weight == 0')
        self.assertEqual(len(zero),3)
        self.assertTrue((zero.contribution==0).all())
        self.assertTrue((zero.scale_status=='all_zero_history').all())

    def test_constant_positive_weight_raises(self):
        with self.assertRaises(UndefinedScaleError): fitted(fixture([[2,2,2,2]]))

    def test_single_active_observation_raises(self):
        with self.assertRaises(UndefinedScaleError): fitted(fixture([[0,0,0,2]]))

    def test_constant_zero_revenue_has_zero_contribution(self):
        m,h,p,s,a=fixture([[1,2,3,4],[2,2,2,2]])
        p.loc['b1']=0
        result=WRMSSEEvaluator.fit(m,h,p,s).score(a,a)
        self.assertEqual(result.wrmsse,0)
        self.assertEqual((result.per_series.scale_status=='zero_scale').sum(),3)

    def test_total_zero_revenue_raises(self):
        with self.assertRaisesRegex(ValueError,'dollar sales'): fitted(fixture([[0,0,0,0]]))


class InputValidation(unittest.TestCase):
    def test_row_permutation_is_key_aligned(self):
        parts=fixture([[1,2,3,4],[2,4,6,8]])
        evaluator=fitted(parts);a=parts[-1];p=a.copy();p.loc['b0']+=1
        self.assertAlmostEqual(evaluator.score(a,p.iloc[::-1]).wrmsse,evaluator.score(a,p).wrmsse)

    def test_duplicate_bottom_rows_rejected(self):
        m,h,p,s,a=fixture([[1,2,3,4],[2,4,6,8]])
        m.loc['b1','item_id']=m.loc['b0','item_id']
        with self.assertRaises(ValueError): WRMSSEEvaluator.fit(m,h,p,s)

    def test_inconsistent_hierarchy_rejected(self):
        m,h,p,s,a=fixture([[1,2,3,4],[2,4,6,8]])
        m.loc['b1','cat_id']='other'
        with self.assertRaises(ValueError): Hierarchy(m)

    def test_multiindex_bottom_keys_match_real_data_schema(self):
        m,h,p,s,a=fixture([[1,2,3,4],[2,4,6,8]])
        index=pd.MultiIndex.from_frame(m[['item_id','store_id']])
        for frame in [m,h,p,a]: frame.index=index
        result=WRMSSEEvaluator.fit(m,h,p,s).score(a,a+1)
        self.assertTrue(np.isfinite(result.wrmsse))

    def test_partial_m5_dataset_rejected(self):
        m,h,p,s,a=fixture()
        with self.assertRaisesRegex(ValueError,'Incomplete M5'): WRMSSEEvaluator.fit(m,h,p,s,strict_m5=True)

    def test_missing_or_duplicate_forecast_grid_rejected(self):
        parts=fixture([[1,2,3,4],[2,4,6,8]])
        evaluator=fitted(parts);a=parts[-1]
        for bad in [a.iloc[:1],a.iloc[:,:1],a.iloc[:,::-1],pd.concat([a,a.iloc[:1]])]:
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError): evaluator.score(a,bad)

    def test_invalid_history_and_prices_rejected(self):
        for value in [float('nan'),float('inf'),-1]:
            m,h,p,s,a=fixture();h.iloc[0,0]=value
            with self.subTest(value=value):
                with self.assertRaises(ValueError): WRMSSEEvaluator.fit(m,h,p,s)
            m,h,p,s,a=fixture();p.iloc[0,0]=value
            with self.assertRaises(ValueError): WRMSSEEvaluator.fit(m,h,p,s)

    def test_invalid_forecast_rejected(self):
        parts=fixture()
        for value in [float('nan'),float('inf')]:
            p=parts[-1].copy();p.iloc[0,0]=value
            with self.assertRaises(ValueError): fitted(parts).score(parts[-1],p)

    def test_future_history_columns_rejected(self):
        m,h,p,s,a=fixture();h['d_5']=99999
        with self.assertRaises(ValueError): WRMSSEEvaluator.fit(m,h,p,s)

    def test_invalid_fold_rejected(self):
        for args in [('unit',0,2,1),('unit',2,2,3),('unit',2,0,1)]:
            with self.assertRaises(ValueError): Fold(*args)


class IndependentReference(unittest.TestCase):
    def test_randomized_multistore_full_hierarchy_against_scalar_oracle(self):
        for seed in range(10):
            rng=np.random.default_rng(seed)
            rows=[]
            for state in range(2):
                for store in range(2):
                    for item in range(4):
                        rows.append({'state_id':f'st{state}','store_id':f's{state}{store}',
                                     'item_id':f'i{item}','dept_id':f'd{item//2}','cat_id':f'c{item//2}'})
            m=pd.DataFrame(rows,index=[f'b{i}' for i in range(16)])
            spec=Fold('random',40,28,28)
            h=pd.DataFrame(rng.integers(1,10,(16,40)),index=m.index,columns=spec.history_days,dtype=float)
            h.iloc[0,:5]=0
            prices=pd.DataFrame(rng.uniform(.1,10,(16,28)),index=m.index,columns=spec.weight_columns)
            a=pd.DataFrame(rng.integers(0,10,(16,28)),index=m.index,columns=spec.forecast_days)
            predicted=pd.DataFrame(rng.uniform(-1,12,(16,28)),index=m.index,columns=spec.forecast_days)
            result=WRMSSEEvaluator.fit(m,h,prices,spec,batch_size=7).score(a,predicted)
            expected,detail=reference_wrmsse(m,h,prices,a,predicted,28)
            with self.subTest(seed=seed):
                self.assertAlmostEqual(result.wrmsse,expected,places=12)
                np.testing.assert_allclose(result.per_series.weight,[r[2] for r in detail],rtol=1e-12)
                np.testing.assert_allclose(result.per_series.scale,[r[3] for r in detail],rtol=1e-12)
                np.testing.assert_allclose(result.per_series.contribution,[r[4] for r in detail],rtol=1e-12,atol=1e-14)

if __name__=='__main__': unittest.main()
