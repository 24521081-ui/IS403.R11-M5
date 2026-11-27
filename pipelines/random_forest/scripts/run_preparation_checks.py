"""Run only synthetic unit tests. Does not discover legacy tests or fit models."""
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone
import json
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project_paths import local_path


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_wrmsse*.py',top_level_dir=str(ROOT))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = local_path(f'preparation/unit_tests_{stamp}.log')
    with output.open('x',encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    summary={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'log':output.relative_to(ROOT).as_posix()}
    from src.wrmsse_config import CONTRACT_ID
    import hashlib
    summary['contract_id']=CONTRACT_ID
    summary['code_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts','tests'] for p in (ROOT/folder).glob('*.py')}
    from src.project_paths import write_json_new
    write_json_new(f'preparation/unit_tests_summary_{stamp}.json',summary)
    print(output.read_text(encoding='utf-8'))
    print(json.dumps(summary))
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__=='__main__': main()
