import unittest,json
from scripts.run_stage3 import effective_groups,eligible,best
from src.wrmsse_config import CONTRACT_ID,ROOT

class Stage3Policy(unittest.TestCase):
 def test_fixed_fe07_fe08(self):
  p=json.loads((ROOT/'configs/stage3.json').read_text())
  self.assertEqual(p['FE']['FE-07'],['behavior','snap'])
  self.assertEqual(p['FE']['FE-08'],['behavior','prices'])
 def test_all_historical_configs_present(self):
  p=json.loads((ROOT/'configs/stage3.json').read_text())
  self.assertEqual([x['name'] for x in p['configs']],[f'RF-{i:02}' for i in range(1,23)])
  self.assertEqual(p['configs'][13]['params']['max_features'],.5)
  self.assertEqual(p['configs'][13]['params']['min_samples_leaf'],5)
 def test_recency_not_duplicated_with_rf_v4(self):
  self.assertEqual(effective_groups('rf_v4',['recency','prices']),['prices'])
  self.assertEqual(effective_groups('rf_v1',['recency']),['recency'])
 def test_calendar_assumptions_excluded_from_primary(self):
  self.assertFalse(eligible(['snap']));self.assertFalse(eligible(['events']))
  self.assertFalse(eligible(['behavior','snap']));self.assertTrue(eligible(['behavior','prices']))
 def test_lowest_eligible_new_wrmsse_only(self):
  row={'status':'success','purpose':'tuning','contract_id':CONTRACT_ID,'selection_eligible':True,'wrmsse':.8}
  winner=best([row,row|{'wrmsse':.7},row|{'wrmsse':.1,'purpose':'smoke'},row|{'wrmsse':.2,'selection_eligible':False}])
  self.assertEqual(winner['wrmsse'],.7)
 def test_no_valid_success_fails_closed(self):
  with self.assertRaises(RuntimeError):best([])

if __name__=='__main__':unittest.main()
