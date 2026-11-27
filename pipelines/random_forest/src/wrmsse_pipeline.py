"""WRMSSE preparation and future RF entry points; execution gates stay closed.

This module creates no results on import. All experiments must be rerun from raw
inputs and must carry this experiment's contract ID. Legacy artifacts are rejected.
"""
import math
from pathlib import Path
from .wrmsse_config import CONFIG, CONTRACT_ID, fold, require_stage_permission
from .wrmsse_data import load_history, load_weight_prices, load_actuals
from .wrmsse import WRMSSEEvaluator
from .project_paths import write_json_new


def prepare_evaluator(split):
    metadata, history = load_history(split)
    prices = load_weight_prices(metadata, history, split)
    evaluator = WRMSSEEvaluator.fit(metadata, history, prices, fold(split), strict_m5=True)
    return evaluator, history


def evaluate_forecast(split, forecast):
    if split == "test":
        require_stage_permission("test_evaluation")
    evaluator, _ = prepare_evaluator(split)
    actual = load_actuals(split, evaluator.hierarchy.bottom_index)
    return evaluator.score(actual, forecast)


def select_best_validation(records):
    records = list(records)
    if not records:
        raise ValueError("No new Validation records")
    for row in records:
        if (row.get("purpose") == "smoke" or row.get("contract_id") != CONTRACT_ID or row.get("metric") != "wrmsse" or
                row.get("split") != "validation" or row.get("origin") != fold("validation").origin or
                row.get("status") != "success" or row.get("experiment_id") != CONFIG["experiment_id"] or
                not isinstance(row.get("wrmsse"), (int, float)) or not math.isfinite(row["wrmsse"]) or row["wrmsse"] < 0):
            raise ValueError("Record is not a new-split Validation WRMSSE result")
    return min(records, key=lambda row: row["wrmsse"])


def fit_random_forest(features, labels, params):
    require_stage_permission("training")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
    from .wrmsse_features import CATEGORICAL, NUMERIC
    encoder = ColumnTransformer([
        ("ids", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
        ("numeric", "passthrough", NUMERIC),
    ], sparse_threshold=1.0)
    model = Pipeline([("features", encoder), ("rf", RandomForestRegressor(**params))])
    return model.fit(features[CATEGORICAL + NUMERIC], labels)


def assemble_forecast(keys, values, expected_index, spec):
    import numpy as np
    import pandas as pd
    if not isinstance(keys, pd.DataFrame) or keys.columns.tolist() != ["bottom_key", "day"]:
        raise ValueError("Explicit bottom/day keys required")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(keys) or not np.isfinite(values).all():
        raise ValueError("Prediction length or values invalid")
    if keys.duplicated(["bottom_key", "day"]).any() or len(keys) != len(expected_index) * spec.horizon:
        raise ValueError("Incomplete or duplicate prediction grid")
    positions = {key: i for i, key in enumerate(expected_index)}
    rows = keys["bottom_key"].map(positions)
    if rows.isna().any() or not keys["day"].isin(range(spec.origin + 1, spec.origin + spec.horizon + 1)).all():
        raise ValueError("Prediction keys outside fold")
    matrix = np.full((len(expected_index), spec.horizon), np.nan)
    matrix[rows.to_numpy(dtype=int), keys["day"].to_numpy(dtype=int) - spec.origin - 1] = values
    if not np.isfinite(matrix).all():
        raise ValueError("Incomplete forecast grid")
    return pd.DataFrame(matrix, index=expected_index, columns=spec.forecast_days)


def validate_training_rows(keys, features, labels):
    import numpy as np
    import pandas as pd
    if not isinstance(keys, pd.DataFrame) or keys.columns.tolist() != ["bottom_key", "day"]:
        raise ValueError("Explicit training bottom/day keys required")
    labels = np.asarray(labels, dtype=float)
    if (len(keys) == 0 or len(keys) != len(features) or labels.shape != (len(keys),) or
            keys.duplicated(["bottom_key", "day"]).any() or
            not keys["day"].isin(range(57, CONFIG["train"][1] + 1)).all() or
            not np.isfinite(labels).all() or (labels < 0).any()):
        raise ValueError("Training rows must belong to the new Train range")


def run_validation_tuning(candidates, train_features, train_labels, validation_features, *, train_keys=None, validation_keys=None):
    require_stage_permission("tuning")
    require_stage_permission("training")
    from .wrmsse_features import CATEGORICAL, NUMERIC
    import pandas as pd
    evaluator, _ = prepare_evaluator("validation")
    actual = load_actuals("validation", evaluator.hierarchy.bottom_index)
    import numpy as np
    validate_training_rows(train_keys, train_features, train_labels)
    assemble_forecast(validation_keys, np.zeros(len(validation_features)), actual.index, fold("validation"))
    records = []
    for candidate in candidates:
        safe_name = candidate["name"]
        if not safe_name or Path(safe_name).name != safe_name or any(x in safe_name for x in ("/", "\\", ":")):
            raise ValueError("Invalid candidate name")
        model = fit_random_forest(train_features, train_labels, candidate["params"])
        values = model.predict(validation_features[CATEGORICAL + NUMERIC])
        forecast = assemble_forecast(validation_keys, values, actual.index, fold("validation"))
        result = evaluator.score(actual, forecast)
        row = {"name": candidate["name"], "params": candidate["params"], "wrmsse": result.wrmsse,
               "contract_id": CONTRACT_ID, "experiment_id": CONFIG["experiment_id"], "metric": "wrmsse",
               "split": "validation", "origin": fold("validation").origin, "status": "success"}
        # Exclusive creation prevents accidentally replacing any historical/new run.
        write_json_new(f"{CONFIG['outputs']}/{safe_name}.json", row)
        records.append(row)
    return select_best_validation(records)


def main():
    print("Preparation only. No tuning, training, forecasts or Test evaluation executed.")
    print({"train": CONFIG["train"], "validation": CONFIG["validation"], "test": CONFIG["test"], "metric": "wrmsse", "contract_id": CONTRACT_ID})

if __name__ == "__main__":
    main()
