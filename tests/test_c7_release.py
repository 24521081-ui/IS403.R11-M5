"""Published C7_EVENTS evidence and clean-clone configuration checks; no fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/c7_events'


def read(name):
    return json.loads((REPORT / name).read_text(encoding='utf-8'))


def test_published_selection_period_and_scores():
    selection = read('selection.json')
    verified = read('test_verified.json')
    assert selection['selected'] == verified['selected'] == 'C7_EVENTS'
    assert verified['days'] == [1914, 1941]
    assert verified['rows'] == 853720 and verified['pairs'] == 30490
    table = pd.read_csv(REPORT / 'validation_comparison.csv', float_precision='round_trip')
    assert table.configuration.tolist() == ['C7', 'C7_EVENTS']
    assert table.WRMSSE.iloc[1] < table.WRMSSE.iloc[0]
    for record in selection['validation']:
        assert table.set_index('configuration').loc[record['configuration'], 'WRMSSE'] == record['WRMSSE']
    levels = pd.read_csv(REPORT / 'test_levels.csv', float_precision='round_trip')
    assert len(levels) == 12
    np.testing.assert_allclose(levels.contribution_WRMSSE.sum(), verified['WRMSSE'], atol=1e-14, rtol=0)


def test_published_test_totals_store_bias_and_groups():
    daily = pd.read_csv(REPORT / 'test_daily.csv', float_precision='round_trip')
    assert daily.target_day.tolist() == [f'd_{d}' for d in range(1914, 1942)]
    stores = pd.read_csv(ROOT / 'reports/sai_lech_test_C7_EVENTS_theo_cua_hang.csv', float_precision='round_trip')
    assert len(stores) == stores.store_id.nunique() == 10
    assert stores.n_pairs.eq(3049).all() and stores.configuration.eq('C7_EVENTS').all()
    np.testing.assert_allclose([stores.predicted.sum(), stores.actual.sum()],
                               [daily.predicted.sum(), daily.actual.sum()], rtol=0, atol=1e-8)
    np.testing.assert_allclose(stores.bias_pct, 100*(stores.predicted/stores.actual-1), atol=1e-12, rtol=0)
    groups = pd.read_csv(REPORT / 'test_group_scores.csv')
    assert len(groups) == 5 and groups.level.eq(12).all()
    assert groups.configuration.eq('C7_EVENTS').all() and groups.days.eq('d_1914-d_1941').all()
    assert groups.groupby('dimension').n_pairs.sum().eq(30490).all()


def test_release_files_match_verified_hashes():
    manifest = read('published_manifest.json')
    for name, digest in manifest['published_files'].items():
        data = (ROOT / name).read_bytes()
        if Path(name).suffix in {'.csv', '.json'}:
            data = data.replace(b'\r\n', b'\n')
        assert hashlib.sha256(data).hexdigest() == digest, name
    assert len(list((ROOT / 'reports/bieu_do').glob('*.png'))) == 7
    assert not list((ROOT / 'reports/bieu_do').glob('*.svg'))


def test_configuration_does_not_require_legacy_c5_models(monkeypatch):
    import run_c7_events as events
    frozen = json.loads((ROOT / 'configs/c7_events.json').read_text(encoding='utf-8'))
    sentinel = object()
    monkeypatch.setattr(events, 'event_calendar', lambda: (sentinel, frozen['event_features']))
    def forbid_legacy_model(*args, **kwargs):
        raise AssertionError('Clean-clone configuration must not depend on a C5 artifact')
    monkeypatch.setattr(events, 'verify_saved_models', forbid_legacy_model)
    calendar, mapping, actual = events.configuration()
    assert calendar is sentinel and mapping == frozen['event_features'] and actual == frozen
    assert len(actual['original_features']) + len(mapping) == 48
