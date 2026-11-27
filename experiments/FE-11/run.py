"""One Validation-only FE-11 fit. All generated files stay beside this script."""
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.wrmsse import Hierarchy, WRMSSEEvaluator
from src.wrmsse_config import fold
from src.wrmsse_extensions import snap_features
from src.wrmsse_stage2_data import BASE, CATS, Dataset, FeatureBuilder, sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


CFG = read(HERE / "config.json")
REF = ROOT / CFG["baseline"]
REQUEST = read(REF / "request.json")
WORK = HERE / "work"
LABELS = ROOT / "reports/demand_groups"


def protected():
    # Hash opaque bytes for integrity, including Test artifacts. Only Validation enters scoring.
    paths = [p for d in ("configs", "data", "docs", "notebooks", "preparation", "reports", "results", "scripts", "src", "tests")
             for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths.extend(p for p in ROOT.iterdir() if p.is_file())
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update((path.relative_to(ROOT).as_posix() + "\0" + sha(path) + "\n").encode())
    return {"files": len(paths), "sha256": h.hexdigest()}


def context(ds):
    official = pd.read_parquet(REF / "wrmsse_by_series.parquet")
    h = Hierarchy(ds.series.set_index(["item_id", "store_id"], drop=False), strict_m5=True)
    assert h.nodes.level.equals(official.level) and h.nodes.key.tolist() == list(map(tuple, official.key))
    return WRMSSEEvaluator(h, fold("validation"), official.scale.to_numpy(), official.scale_status.to_numpy(),
                           official.weight.to_numpy())


def score(ds, ev, prediction):
    args = {"index": ds.index, "columns": ev.fold.forecast_days}
    return ev.score(pd.DataFrame(ds.actual, **args), pd.DataFrame(prediction, **args))


def adi_feature(history, keys):
    """Inclusive cutoff t-28. Zero is outside the domain [1, +inf) of observed ADI."""
    ids, days = keys.series_id.to_numpy(), keys.day.to_numpy()
    cutoff = days - CFG["adi"]["lag"]
    assert cutoff.min() >= 1 and cutoff.max() <= history.shape[1] and np.all(ids[:-1] <= ids[1:])
    out = np.empty(len(keys), np.float32)
    for start in range(0, len(history), 256):
        left, right = ids.searchsorted([start, start + 256])
        if left == right:
            continue
        c = cutoff[left:right]
        active = history[start:start+256, :c.max()] > 0
        row = ids[left:right] - start
        count = np.cumsum(active, axis=1, dtype=np.int32)[row, c-1]
        first_zero_based = active.argmax(axis=1)[row]
        out[left:right] = np.divide(c - first_zero_based, count, out=np.zeros(len(c)), where=count > 0)
    # Independent oracle: binary-search the observed positive sale dates for each row.
    last_cutoff = int(cutoff.max())
    for series, (left, right) in enumerate(zip(ids.searchsorted(np.arange(len(history))),
                                               ids.searchsorted(np.arange(len(history))+1))):
        positive = np.flatnonzero(history[series, :last_cutoff] > 0) + 1
        c = cutoff[left:right]
        count = positive.searchsorted(c, side="right")
        first = positive[0] if len(positive) else 1
        expected = np.divide(c-first+1, count, out=np.zeros(len(c)), where=count > 0)
        np.testing.assert_array_equal(out[left:right], expected.astype(np.float32))
    assert np.isfinite(out).all() and ((out == 0) | (out >= 1)).all()
    return out


def origin_check(ds, origin):
    keys = ["item_id", "store_id"]
    ref = pd.read_csv(LABELS / f"demand_groups_origin_{origin}.csv", float_precision="round_trip")
    ref = ds.series[keys].merge(ref, on=keys, validate="one_to_one")
    assert len(ref) == len(ds.series)
    counts, first = [], []
    for start in range(0, len(ds.series), 256):
        history = ds.history[start:start+256]
        if origin == 1913:  # Method audit only, never used to build input features.
            history = np.column_stack([history, ds.actual[start:start+256]])
        active = history > 0
        counts.extend(active.sum(axis=1))
        first.extend(active.argmax(axis=1))
    counts = np.array(counts)
    reference = np.divide(origin-np.array(first), counts, out=np.full(len(ref), np.inf), where=counts > 0)
    np.testing.assert_allclose(reference, ref.ADI, rtol=1e-13)
    np.testing.assert_array_equal(np.where(reference < 1.32, "frequent", "intermittent"), ref.demand_group)
    summary = pd.read_csv(LABELS / "demand_groups_summary.csv").query("origin == @origin").set_index("demand_group")
    stats = ref.groupby("demand_group").ADI.agg(["size", "median"])
    np.testing.assert_array_equal(stats["size"], summary.loc[stats.index, "n_series"])
    np.testing.assert_allclose(stats["median"], summary.loc[stats.index, "median_ADI"])
    return ref


def check_future_invariance():
    for day in (57,1000,1886,1913):
        cutoff = day-28
        history = np.zeros((3,1913),np.uint16)
        history[0,[cutoff-4,cutoff-1]] = 2
        history[1,cutoff-1] = 1
        keys = pd.DataFrame({"series_id":[0,1,2],"day":[day]*3})
        expected = [2,1,0]
        np.testing.assert_array_equal(adi_feature(history,keys),expected)
        history[:,cutoff:] = 99
        np.testing.assert_array_equal(adi_feature(history,keys),expected)


def prepare():
    if (HERE / "fit_started.json").exists():
        raise RuntimeError("FE-11 already started in this directory")
    WORK.mkdir(exist_ok=True)
    before = protected()
    check_future_invariance()
    ds = Dataset(REQUEST["dataset"])
    assert REQUEST["params"] == CFG["params"] and REQUEST["groups"] == ["snap"] and REQUEST["logical"] == 15
    assert sha(ds.root/"manifest.json") == REQUEST["dataset_hash"]
    manifest = read(REF/"manifest.json")["files"]
    for name in ("request.json", "result.json", "validation_predictions.npy", "wrmsse_by_series.parquet", "wrmsse_by_level.csv"):
        assert sha(REF/name) == manifest[name], name
    golden = pd.read_csv(ROOT/"docs/PROJECT_CLEANUP_INVENTORY.csv").set_index("path").sha256
    hashes = {}
    def verify_bytes(key, array):
        buf = io.BytesIO()
        if sparse.issparse(array):
            sparse.save_npz(buf, array)
        else:
            np.save(buf, array, allow_pickle=False)
        hashes[key] = hashlib.sha256(buf.getbuffer()).hexdigest()
        assert hashes[key] == golden.loc[REQUEST["features"] + "/" + key], key
    builder = FeatureBuilder(ds)
    audit = {}
    ref = origin_check(ds, 1885)
    origin_check(ds, 1913)
    np.testing.assert_array_equal(ds.validation_keys.series_id, np.repeat(ds.series.series_id,28))
    np.testing.assert_array_equal(ds.validation_keys.day, np.tile(np.arange(1886,1914),len(ds.series)))
    for split, keys, size in (("train",ds.train_keys,600000), ("validation",ds.validation_keys,853720)):
        assert len(keys) == size and not keys.duplicated().any()
        base,spec = builder.base(keys, builder.sales_features(keys), "rf_v1")
        snap = sparse.csr_matrix(snap_features(ds.calendar,ds.series,keys))
        verify_bytes(f"rf_v1/X_{split}.npz",base)
        verify_bytes(f"snap/extra_{split}.npz",snap)
        adi = adi_feature(ds.history,keys)
        if split == "validation":
            np.testing.assert_array_equal(adi[keys.day.eq(1913)],ref.ADI.to_numpy(np.float32))
        X = sparse.hstack([base,snap,sparse.csr_matrix(adi[:,None])],format="csr",dtype=np.float32)
        assert X.shape == (size,3083) and np.isfinite(X.data).all()
        sparse.save_npz(WORK / f"{split}.npz",X)
        audit[split] = {"rows":size,"min_cutoff":int(keys.day.min()-28),"max_cutoff":int(keys.day.max()-28),
                        "no_positive_rows":int((adi==0).sum()), "max_ADI":float(adi.max()), "oracle_mismatches":0}
        del X,base,snap,adi
    y = ds.history[ds.train_keys.series_id, ds.train_keys.day-1].astype(np.float32)
    verify_bytes("y_train.npy", y)
    np.save(WORK / "y.npy", y)
    ev = context(ds)
    baseline = score(ds,ev,np.load(REF/"validation_predictions.npy"))
    np.testing.assert_allclose(baseline.wrmsse,read(REF/"result.json")["wrmsse"],rtol=0,atol=1e-14)
    np.testing.assert_allclose(baseline.per_level,pd.read_csv(REF/"wrmsse_by_level.csv",index_col="level"),atol=1e-13)
    logical_names = [name for name, cats in zip(CATS,builder.encoder.categories_) for _ in cats] + BASE + ["snap_state","ADI"]
    assert len(logical_names) == 3083 and len(set(logical_names)) == 16
    write("preflight.json",{"baseline_wrmsse_reproduced":baseline.wrmsse,"original_matrix_hashes":hashes,
          "dataset_manifest_sha256":REQUEST["dataset_hash"],"origin_ADI_verified":[1885,1913],"adi":audit,
          "logical_features":list(dict.fromkeys(logical_names)),"protected_before":before,
          "scikit_learn":sklearn.__version__,"preparation_peak_mib":psutil.Process().memory_info().peak_wset/2**20,
          "columns_per_feature":{name:logical_names.count(name) for name in dict.fromkeys(logical_names)}})


def train():
    pre = read(HERE/"preflight.json")
    X,y = sparse.load_npz(WORK/"train.npz"),np.load(WORK/"y.npy")
    with (HERE/"fit_started.json").open("x",encoding="utf-8") as marker:
        json.dump({"configuration":"FE-11","params":CFG["params"]},marker)
    model = RandomForestRegressor(**CFG["params"])
    print("FE-11: one fit started",flush=True)
    start = time.perf_counter()
    model.fit(X,y)
    fit_time = time.perf_counter()-start
    del X,y
    gc.collect()
    X = sparse.load_npz(WORK/"validation.npz")
    start = time.perf_counter()
    pred = np.concatenate([model.predict(X[i:i+32768]) for i in range(0,X.shape[0],32768)]).reshape(30490,28)
    predict_time = time.perf_counter()-start
    np.save(WORK/"prediction.npy",pred)
    names = np.repeat(list(pre["columns_per_feature"]),list(pre["columns_per_feature"].values()))
    importance = pd.Series(model.feature_importances_,index=names).groupby(level=0).sum().sort_values(ascending=False)
    np.testing.assert_allclose(importance.sum(),1)
    importance.rename_axis("feature").rename("importance").to_csv(HERE/"importance.csv")
    joblib.dump(model, WORK/"model.joblib")
    write("runtime.json",{"fit_count":1,"fit_seconds":fit_time,"predict_seconds":predict_time,
                          "peak_rss_mib":psutil.Process().memory_info().peak_wset/2**20})
    print("FE-11: fit and Validation forecast saved",flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        train()
    elif not sys.argv[1:]:
        prepare()
        print("Preflight passed", flush=True)
        subprocess.run([sys.executable,"-B",str(Path(__file__)),"--worker"],check=True)
        subprocess.run([sys.executable,"-B",str(HERE/"report.py")],check=True)
    else:
        raise SystemExit("Usage: python -B run.py")
