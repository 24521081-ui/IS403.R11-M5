"""Direct 28-day RF features with history cutoff min(t - 28, origin).

No labels inside the forecast window enter features, rolling windows, or encoding.
Metadata/calendar covariates are known in advance. No demand/price imputation.
"""
import numpy as np
import pandas as pd
from .wrmsse_config import CONFIG

LAGS = (28, 35, 42, 56)
WINDOWS = (7, 28)
CATEGORICAL = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
NUMERIC = ["wday", "month", "year"] + [f"lag_{lag}" for lag in LAGS] + [f"rolling_mean_{window}" for window in WINDOWS]


def feature_rows(metadata, history, calendar, target_days, *, origin):
    if (history.index.has_duplicates or not history.index.equals(metadata.index) or
            history.columns.tolist() != [f"d_{d}" for d in range(1, origin + 1)]):
        raise ValueError("Features require keyed chronological history ending at origin")
    if not np.isfinite(history.to_numpy(dtype=float)).all() or (history.to_numpy(dtype=float) < 0).any():
        raise ValueError("Invalid feature history")
    if calendar["d"].duplicated().any():
        raise ValueError("Duplicate calendar day")
    if not target_days or len(set(target_days)) != len(target_days):
        raise ValueError("Unique nonempty target days required")
    dates = calendar.set_index("d")
    values = history.to_numpy(dtype=float)
    blocks, keys = [], []
    for day in target_days:
        if type(day) is not int or day < 57 or day > origin + CONFIG["horizon"]:
            raise ValueError("Target day outside supported origin window")
        cutoff = min(day - 28, origin)
        block = metadata[CATEGORICAL].copy()
        for col in ("wday", "month", "year"):
            block[col] = dates.loc[f"d_{day}", col]
        for lag in LAGS:
            block[f"lag_{lag}"] = values[:, day - lag - 1]
        for window in WINDOWS:
            block[f"rolling_mean_{window}"] = values[:, cutoff - window:cutoff].mean(axis=1)
        blocks.append(block.reset_index(drop=True))
        keys.extend((key, day) for key in metadata.index)
    return pd.DataFrame(keys, columns=["bottom_key", "day"]), pd.concat(blocks, ignore_index=True)
