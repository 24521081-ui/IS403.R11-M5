"""Independent scalar oracle: pure Python grouping, math.fsum, no production imports.

Fresh test-only implementation of the M5 equations, structurally independent
from the sparse roll-up/vectorized implementation. Does not use shipped weights.
"""
import math


def reference_wrmsse(metadata, history, prices, actual, predicted, weight_days):
    levels = [[], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"],
              ["state_id", "cat_id"], ["state_id", "dept_id"],
              ["store_id", "cat_id"], ["store_id", "dept_id"],
              ["item_id"], ["item_id", "state_id"], ["item_id", "store_id"]]
    keys = metadata.index.tolist()
    h = history.loc[keys].to_numpy().tolist()
    p = prices.loc[keys].to_numpy().tolist()
    a = actual.loc[keys].to_numpy().tolist()
    f = predicted.loc[keys].to_numpy().tolist()
    money = [math.fsum(x*y for x,y in zip(row[-weight_days:], price)) for row,price in zip(h,p)]
    total = math.fsum(money)
    if total <= 0: raise ValueError("No revenue")
    score = 0.0
    details = []
    for level, columns in enumerate(levels, 1):
        groups = {}
        for position, key in enumerate(keys):
            group = tuple(str(metadata.loc[key, c]) for c in columns) if columns else ("Total",)
            groups.setdefault(group, []).append(position)
        for group in sorted(groups):
            positions = groups[group]
            weight = math.fsum(money[i] for i in positions)/total/12
            series = [math.fsum(h[i][d] for i in positions) for d in range(len(h[0]))]
            first = next((i for i,x in enumerate(series) if x != 0), None)
            active = [] if first is None else series[first:]
            scale = math.fsum((active[t]-active[t-1])**2 for t in range(1,len(active)))/(len(active)-1) if len(active)>1 else float("nan")
            if weight == 0: contribution = 0.0
            elif not math.isfinite(scale) or scale <= 0: raise ValueError("Undefined positive-weight scale")
            else:
                error = [math.fsum(a[i][d] for i in positions)-math.fsum(f[i][d] for i in positions) for d in range(len(a[0]))]
                contribution = weight*math.sqrt(math.fsum(x*x for x in error)/len(error)/scale)
            score += contribution
            details.append((level,group,weight,scale,contribution))
    return score, details
