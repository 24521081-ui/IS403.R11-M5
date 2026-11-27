"""Kiểm tra quy tắc scale M5, đặc biệt leading zeros và cấp tổng hợp."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_c7_wrmsse import LEVELS, rmsse_scale
from score_model_wrmsse import load_forecasts


def test_scale_excludes_jump_from_leading_zeros_and_keeps_later_zeros():
    scale, valid = rmsse_scale(np.array([[0., 0., 2., 4., 0.], [2., 4., 0., 0., 0.]]))
    np.testing.assert_allclose(scale, [10., 5.])
    assert valid.all()


def test_scale_rejects_empty_constant_and_single_observation_series():
    _, valid = rmsse_scale(np.array([[0., 0., 0.], [2., 2., 2.], [0., 0., 2.]]))
    assert not valid.any()


def test_hierarchy_contains_twelve_unique_levels_and_bottom_level():
    assert len(LEVELS) == len({tuple(cols) for cols in LEVELS}) == 12
    assert LEVELS[0] == [] and set(LEVELS[-1]) == {"store_id", "item_id"}


def forecast_frame():
    return pd.DataFrame({"item_id": ["FOODS_1_001"]*28, "store_id": ["CA_1"]*28,
                         "target_day": [f"d_{d}" for d in range(1914,1942)], "y_pred": np.ones(28)})


def test_shared_reader_combines_store_files_and_preserves_predictions(tmp_path):
    first=forecast_frame()
    second=first.assign(store_id="CA_2",y_pred=2.5)
    first.to_csv(tmp_path/"CA_1.csv",index=False)
    second.to_csv(tmp_path/"CA_2.csv",index=False)
    loaded,paths=load_forecasts([str(tmp_path/"CA_*.csv")])
    assert len(paths)==2 and len(loaded)==56
    assert loaded.loc[loaded.store_id.eq("CA_2"),"y_pred"].eq(2.5).all()


@pytest.mark.parametrize("problem",["duplicate","negative","infinite","wrong_period"])
def test_shared_reader_rejects_invalid_submissions(tmp_path,problem):
    frame=forecast_frame()
    if problem=="duplicate":frame=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
    if problem=="negative":frame.loc[0,"y_pred"]=-1
    if problem=="infinite":frame.loc[0,"y_pred"]=np.inf
    if problem=="wrong_period":frame.loc[0,"target_day"]="d_1886"
    path=tmp_path/"forecast.csv"
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError):load_forecasts([str(path)])


def test_shared_reader_rejects_same_file_selected_twice(tmp_path):
    path=tmp_path/"forecast.csv"
    forecast_frame().to_csv(path,index=False)
    with pytest.raises(ValueError,match="nhiều lần"):
        load_forecasts([str(path),str(path)])


def test_export_split_by_model_separates_benchmark_models(tmp_path, monkeypatch):
    bench_root = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(bench_root / "scripts" / "benchmarks"))
    monkeypatch.syspath_prepend(str(bench_root))
    monkeypatch.delitem(sys.modules, "src", raising=False)
    from score_benchmarks_wrmsse import export_split_by_model

    combined = pd.concat(
        [
            forecast_frame().assign(model="ES_bu", y_pred=1.5),
            forecast_frame().assign(model="Croston", y_pred=0.5),
        ],
        ignore_index=True,
    )
    src = tmp_path / "forecasts.csv"
    combined.to_csv(src, index=False)

    out_dir = tmp_path / "by_model"
    exported = export_split_by_model(src, out_dir, models=["ES_bu", "Croston"])
    assert {p.name for p in exported} == {"ES_bu.csv", "Croston.csv"}

    es_loaded, _ = load_forecasts([str(out_dir / "ES_bu.csv")])
    assert len(es_loaded) == 28 and es_loaded["y_pred"].eq(1.5).all()
