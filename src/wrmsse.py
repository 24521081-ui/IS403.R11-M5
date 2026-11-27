"""Full M5 hierarchy WRMSSE. No model fitting, forecast generation, or file writes.

Each level receives weight 1/12. Dollar sales use only the final 28 history
observations; scale uses differences after the first nonzero historical sale.
Undefined positive-weight scales fail closed rather than changing the metric.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from .wrmsse_config import Fold

LEVELS = (
    (), ("state_id",), ("store_id",), ("cat_id",), ("dept_id",),
    ("state_id", "cat_id"), ("state_id", "dept_id"),
    ("store_id", "cat_id"), ("store_id", "dept_id"),
    ("item_id",), ("item_id", "state_id"), ("item_id", "store_id"),
)
M5_COUNTS = (1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490)
META_COLUMNS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")


def _matrix(frame, index, columns, label, nonnegative=True):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label}: labeled DataFrame required")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{label}: duplicate labels")
    if (len(frame.index) != len(index) or not frame.index.isin(index).all() or
            frame.columns.tolist() != list(columns)):
        raise ValueError(f"{label}: requires complete bottom keys and exact chronological days")
    values = frame.reindex(index).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (nonnegative and (values < 0).any()):
        raise ValueError(f"{label}: invalid values")
    return values


class UndefinedScaleError(ValueError):
    """Official ratio is undefined for a positive-weight aggregate."""


class Hierarchy:
    def __init__(self, metadata, *, strict_m5=False):
        if not isinstance(metadata, pd.DataFrame) or metadata.empty or metadata.index.has_duplicates:
            raise ValueError("Unique nonempty bottom-series metadata required")
        if metadata.index.to_frame(index=False).isna().any().any():
            raise ValueError("Missing bottom keys")
        if any(c not in metadata for c in META_COLUMNS) or metadata[list(META_COLUMNS)].isna().any().any():
            raise ValueError("Missing hierarchy metadata")
        if metadata.duplicated(["item_id", "store_id"]).any():
            raise ValueError("Duplicate item-store series")
        for child, parent in (("item_id", "dept_id"), ("dept_id", "cat_id"), ("store_id", "state_id")):
            if (metadata.reset_index(drop=True).groupby(child, observed=True)[parent].nunique() != 1).any():
                raise ValueError(f"Inconsistent hierarchy mapping: {child} -> {parent}")
        self.metadata = metadata.copy()
        self.bottom_index = metadata.index.copy()
        blocks, rows, counts = [], [], []
        for level, keys in enumerate(LEVELS, 1):
            labels = [tuple(row) for row in metadata[list(keys)].astype(str).itertuples(index=False, name=None)] if keys else [("Total",)] * len(metadata)
            unique = sorted(set(labels))
            mapping = {key: i for i, key in enumerate(unique)}
            codes = np.fromiter((mapping[key] for key in labels), dtype=np.int64)
            blocks.append(sparse.csr_matrix((np.ones(len(metadata)), (codes, np.arange(len(metadata)))), shape=(len(unique), len(metadata))))
            counts.append(len(unique))
            rows.extend({"level": level, "key": key} for key in unique)
        self.rollup = sparse.vstack(blocks, format="csr")
        self.nodes = pd.DataFrame(rows)
        self.counts = tuple(counts)
        if strict_m5 and self.counts != M5_COUNTS:
            raise ValueError(f"Incomplete M5 hierarchy: {self.counts}")

    def aggregate(self, frame, days, *, label="values", nonnegative=True):
        values = _matrix(frame, self.bottom_index, days, label, nonnegative)
        return self.rollup @ values


def _scales(values):
    first = (values > 0).argmax(axis=1)
    has_sales = (values > 0).any(axis=1)
    active = values.shape[1] - first
    count = np.where(has_sales, active - 1, 0)
    differences = np.diff(values, axis=1)
    included = np.arange(1, values.shape[1])[None, :] > first[:, None]
    total = np.where(included, differences ** 2, 0).sum(axis=1)
    scale = np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)
    status = np.full(len(values), "ok", dtype=object)
    status[scale == 0] = "zero_scale"
    status[count == 0] = "insufficient_active_history"
    status[~has_sales] = "all_zero_history"
    return scale, status


@dataclass
class WRMSSEResult:
    wrmsse: float
    per_level: pd.DataFrame
    per_series: pd.DataFrame


class WRMSSEEvaluator:
    def __init__(self, hierarchy, fold, scales, statuses, weights):
        self.hierarchy = hierarchy
        self.fold = fold
        self.scales = scales.copy()
        self.statuses = statuses.copy()
        self.weights = weights.copy()

    @classmethod
    def fit(cls, metadata, history, weight_prices, fold: Fold, *, strict_m5=False, batch_size=256):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        hierarchy = Hierarchy(metadata, strict_m5=strict_m5)
        values = _matrix(history, hierarchy.bottom_index, fold.history_days, "history")
        prices = _matrix(weight_prices, hierarchy.bottom_index, fold.weight_columns, "weight prices", nonnegative=False)
        units = values[:, -fold.weight_days:]
        # Missing prices are allowed only when no unit was sold; validated by price loader.
        if (prices < 0).any():
            raise ValueError("Negative prices")
        revenue = (units * prices).sum(axis=1)
        total = revenue.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Weight-window dollar sales must be positive and finite")
        weights = np.asarray(hierarchy.rollup @ revenue).ravel() / total / 12
        n = hierarchy.rollup.shape[0]
        scales, statuses = np.empty(n), np.empty(n, dtype=object)
        for start in range(0, n, batch_size):
            part = hierarchy.rollup[start:start + batch_size] @ values
            scale, status = _scales(part)
            scales[start:start + len(part)] = scale
            statuses[start:start + len(part)] = status
        bad = (weights > 0) & ((statuses != "ok") | ~np.isfinite(scales) | (scales <= 0))
        if bad.any():
            details = hierarchy.nodes.loc[bad].assign(status=statuses[bad]).head(10).to_dict("records")
            raise UndefinedScaleError(f"Positive-weight aggregate has undefined M5 scale: {details}")
        if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
            raise ValueError("Hierarchy weights do not sum to one")
        return cls(hierarchy, fold, scales, statuses, weights)

    def score(self, actual, predicted):
        days = self.fold.forecast_days
        truth = self.hierarchy.aggregate(actual, days, label="actual")
        forecast = self.hierarchy.aggregate(predicted, days, label="forecast", nonnegative=False)
        mse = np.mean((truth - forecast) ** 2, axis=1)
        positive = self.weights > 0
        rmsse = np.full(len(mse), np.nan)
        valid_scale = np.isfinite(self.scales) & (self.scales > 0)
        rmsse[valid_scale] = np.sqrt(mse[valid_scale] / self.scales[valid_scale])
        if not np.isfinite(rmsse[positive]).all():
            raise ValueError("Nonfinite positive-weight RMSSE")
        contribution = np.zeros(len(mse))
        contribution[positive] = self.weights[positive] * rmsse[positive]
        detail = self.hierarchy.nodes.copy()
        detail["weight"] = self.weights
        detail["scale"] = self.scales
        detail["scale_status"] = self.statuses
        detail["mse"] = mse
        detail["rmsse"] = rmsse
        detail["contribution"] = contribution
        levels = detail.groupby("level", sort=True).agg(n_series=("level", "size"), weight=("weight", "sum"), contribution=("contribution", "sum"))
        levels["wrmsse_within_level"] = levels["contribution"] * 12
        return WRMSSEResult(float(contribution.sum()), levels, detail)
