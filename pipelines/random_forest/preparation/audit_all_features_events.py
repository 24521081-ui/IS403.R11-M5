"""Verify Events from pre-origin calendar rules without using future event labels."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wrmsse_extensions import FIXED, NTH, known_calendar, rule_matches
from src.wrmsse_stage2_data import sha, write_new

DEST = ROOT / "results/wrmsse_v1/stage3_all_features"


def main():
    from verify_stage3 import read
    stage2 = read(DEST.parent / "stage2_20261123T202104Z/summary.json")
    processed, features = (ROOT / stage2[key] for key in ("processed", "features"))
    manifest = read(features / "manifest.json")
    origin = 1885
    # The only event names and categories read are at or before the forecast origin.
    train_calendar = pd.read_parquet(processed / "calendar.parquet", filters=[("day", "<=", origin)])
    train = known_calendar(train_calendar)
    types_by_event = {}
    for slot in (1, 2):
        for name, category in train[[f"event_name_{slot}", f"event_type_{slot}"]].itertuples(index=False, name=None):
            if name != "NoEvent":
                types_by_event.setdefault(name, set()).add(category)
    assert set(types_by_event) == set(FIXED) | set(NTH) | {"MemorialDay"}
    assert all(len(categories) == 1 for categories in types_by_event.values())
    categories = {name: next(iter(kinds)) for name, kinds in types_by_event.items()}
    vocabulary = {"name": sorted(categories), "type": sorted(set(categories.values()))}
    assert vocabulary == manifest["details"]["event_vocabulary"]
    names = [
        f"event_{kind}__{value}" for kind, values in vocabulary.items()
        for value in [*values, "unknown"]
    ]
    assert names == manifest["specs"]["events"]["extra_names"]
    calendar = pd.read_parquet(
        processed / "calendar.parquet", columns=["day", "date"], filters=[("day", "<=", 1913)]
    )
    np.testing.assert_array_equal(calendar.day, np.arange(1, 1914))
    expected = np.zeros((len(calendar), len(names)), dtype=np.float32)
    for position, date in enumerate(calendar.date):
        names_on_date = {name for name in categories if rule_matches(name, date)}
        types_on_date = {categories[name] for name in names_on_date}
        expected[position] = [
            value in activated
            for kind, activated in (("name", names_on_date), ("type", types_on_date))
            for value in [*vocabulary[kind], "unknown"]
        ]
    split_results = {}
    for split in ("train", "validation"):
        keys = pd.read_parquet(processed / f"{split}_keys.parquet", columns=["day"])
        path = features / "events" / f"extra_{split}.npz"
        assert sha(path) == manifest["files"][path.relative_to(features).as_posix()]
        stored = sparse.load_npz(path)
        assert stored.shape == (len(keys), len(names))
        mismatches = 0
        for start in range(0, len(keys), 32768):
            end = min(start + 32768, len(keys))
            days = keys.day.iloc[start:end].to_numpy()
            values = stored[start:end].toarray()
            mismatches += int(np.count_nonzero(expected[days - 1] != values))
        split_results[split] = {"rows": len(keys), "encoded_columns": len(names),
                                "mismatched_values": mismatches, "stored_nonzero": int(stored.count_nonzero())}
    eligibility = all(check["mismatched_values"] == 0 for check in split_results.values())
    outcome = {
        "status": "verified_for_this_window" if eligibility else "exploratory",
        "calendar_history_end": origin, "validation_days": [1886, 1913],
        "events_on_days_in_validation": int(np.count_nonzero(expected[1885:].any(axis=1))),
        "rule": "Fixed and nth-weekday civil holiday rules and Memorial Day; categories learned before origin",
        "event_types": categories, "vocabulary": vocabulary, "checks": split_results,
        "eligible_by_origin_availability": eligibility, "test_accessed": False,
        "feature_manifest_sha256": sha(features / "manifest.json"),
    }
    write_new(DEST / "events_origin_audit.json", outcome)
    print(outcome)


if __name__ == "__main__":
    main()

