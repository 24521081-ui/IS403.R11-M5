"""Raw-only loaders. No old parquet, weight, model or forecast is an active input."""
import numpy as np
import pandas as pd
from .project_paths import local_path
from .wrmsse_config import CONFIG, fold, require_stage_permission
from .wrmsse import META_COLUMNS, _matrix


def normalize_calendar(calendar):
    result = calendar.copy()
    if "date" in result:
        dates = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="raise")
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError("Missing or duplicate calendar dates")
        day = (dates - pd.Timestamp("2011-01-29")).dt.days + 1
        if (day < 1).any():
            raise ValueError("Calendar precedes M5 day one")
        labels = "d_" + day.astype(str)
        if "d" in result and not result["d"].equals(labels):
            raise ValueError("Calendar d labels disagree with actual dates")
        result["d"] = labels
    elif "d" not in result:
        raise ValueError("Calendar requires dates or explicit day labels")
    if result["d"].duplicated().any():
        raise ValueError("Duplicate calendar days")
    return result


def load_calendar():
    return normalize_calendar(pd.read_csv(local_path("data/raw/calendar.csv")))


def load_history(split):
    spec = fold(split)
    columns = list(META_COLUMNS) + spec.history_days
    frame = pd.read_csv(local_path(CONFIG["sales_source"]), usecols=columns,
                        dtype={c: "float64" for c in spec.history_days})
    metadata = frame[list(META_COLUMNS)].copy()
    metadata.index = pd.MultiIndex.from_frame(metadata[["item_id", "store_id"]])
    history = frame[spec.history_days].copy()
    history.index = metadata.index
    _matrix(history, metadata.index, spec.history_days, "history")
    return metadata, history


def load_actuals(split, expected_index):
    if split == "test":
        require_stage_permission("test_evaluation")
    spec = fold(split)
    source = CONFIG["test_actual_source"] if split == "test" else CONFIG["sales_source"]
    frame = pd.read_csv(local_path(source), usecols=["item_id", "store_id"] + spec.forecast_days)
    frame.index = pd.MultiIndex.from_frame(frame[["item_id", "store_id"]])
    actual = frame[spec.forecast_days]
    _matrix(actual, expected_index, spec.forecast_days, "actual")
    return actual.reindex(expected_index)


def weight_prices_from_tables(metadata, history, calendar, prices, spec):
    calendar = normalize_calendar(calendar)
    units = _matrix(history, metadata.index, spec.history_days, "history")[:, -spec.weight_days:]
    days = spec.weight_columns
    selected = calendar.loc[calendar["d"].isin(days), ["d", "wm_yr_wk"]]
    if selected["d"].duplicated().any() or len(selected) != len(days) or selected.isna().any().any():
        raise ValueError("Weight-window calendar must contain each day exactly once")
    weeks = selected.set_index("d").loc[days, "wm_yr_wk"]
    current = prices.loc[prices["wm_yr_wk"].isin(weeks.unique())].copy()
    price_key = ["store_id", "item_id", "wm_yr_wk"]
    if current.duplicated(price_key).any():
        raise ValueError("Duplicate weekly price keys")
    table = current.set_index(price_key)["sell_price"]
    result = np.empty_like(units, dtype=float)
    for column, week in enumerate(weeks):
        keys = pd.MultiIndex.from_arrays([metadata["store_id"], metadata["item_id"], np.repeat(week, len(metadata))], names=price_key)
        result[:, column] = table.reindex(keys).to_numpy(dtype=float)
    if np.isinf(result).any() or (result < 0).any():
        raise ValueError("Invalid weight-window prices")
    missing = np.isnan(result)
    if (missing & (units > 0)).any():
        raise ValueError("Missing price for positive unit sales; revenue cannot be inferred")
    result[missing] = 0
    return pd.DataFrame(result, index=metadata.index, columns=days)


def load_weight_prices(metadata, history, split):
    spec = fold(split)
    calendar = load_calendar()
    weeks = calendar.loc[calendar["d"].isin(spec.weight_columns), "wm_yr_wk"].unique()
    parts = []
    with pd.read_csv(local_path("data/raw/sell_prices.csv"), chunksize=100_000) as reader:
        for part in reader:
            parts.append(part.loc[part["wm_yr_wk"].isin(weeks)])
    prices = pd.concat(parts, ignore_index=True)
    return weight_prices_from_tables(metadata, history, calendar, prices, spec)
