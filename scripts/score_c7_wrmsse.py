"""WRMSSE bß╗ò sung: 12 cß║Ñp M5, doanh thu 28 ng├áy tr╞░ß╗¢c origin, kh├┤ng ─æß╗òi RMSSE gß╗æc.

Nguß╗ôn: https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [[], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"],
          ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"],
          ["store_id", "dept_id"], ["item_id"], ["state_id", "item_id"], ["store_id", "item_id"]]
META = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def rmsse_scale(history):
    first = np.argmax(history > 0, axis=1)
    count = history.shape[1] - first - 1
    diff = np.diff(history, axis=1)
    diff[np.arange(history.shape[1]-1)[None, :] < first[:, None]] = 0
    scale = np.square(diff).sum(axis=1) / np.maximum(count, 1)
    valid = (count > 0) & (scale > 0) & np.isfinite(scale)
    return scale, valid


def score(origin, wide, calendar, prices, forecasts, *, return_details=False):
    assert len(wide) == 30490 and not wide.duplicated(["item_id", "store_id"]).any()
    days = [f"d_{d}" for d in range(origin+1, origin+29)]
    assert len(forecasts) == 30490 * 28
    assert forecasts.origin_day.eq(f"d_{origin}").all()
    assert set(forecasts.target_day) == set(days)
    assert forecasts.split.eq("validation" if origin == 1885 else "test").all()
    assert not forecasts.duplicated(["item_id", "store_id", "target_day"]).any()
    matrix = forecasts.pivot(index=["item_id", "store_id"], columns="target_day", values="y_pred")
    index = pd.MultiIndex.from_frame(wide[["item_id", "store_id"]])
    pred = matrix.reindex(index=index, columns=days).to_numpy(dtype=float)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    history = wide[[f"d_{d}" for d in range(1, origin+1)]].to_numpy(dtype=float)
    actual = wide[days].to_numpy(dtype=float)
    revenue = np.zeros(len(wide))
    weekmap = calendar.set_index("d").wm_yr_wk
    for day in range(origin-27, origin+1):
        week = weekmap[f"d_{day}"]
        price = wide[["store_id", "item_id"]].merge(
            prices.loc[prices.wm_yr_wk.eq(week), ["store_id", "item_id", "sell_price"]],
            on=["store_id", "item_id"], how="left", validate="one_to_one").sell_price.to_numpy()
        units = wide[f"d_{day}"].to_numpy(dtype=float)
        assert np.isfinite(price[units > 0]).all(), f"Thiß║┐u gi├í c├│ doanh thu d_{day}"
        revenue += units * np.nan_to_num(price)
    assert revenue.sum() > 0
    rows = []
    details = None
    for level, cols in enumerate(LEVELS, 1):
        codes = pd.factorize(pd.MultiIndex.from_frame(wide[cols]), sort=True)[0] if cols else np.zeros(len(wide), dtype=int)
        n = int(codes.max()) + 1
        h, a, p, dollars = np.zeros((n, origin)), np.zeros((n, 28)), np.zeros((n, 28)), np.zeros(n)
        np.add.at(h, codes, history)
        np.add.at(a, codes, actual)
        np.add.at(p, codes, pred)
        np.add.at(dollars, codes, revenue)
        # Bß╗Å leading zeros tr╞░ß╗¢c lß║ºn b├ín ─æß║ºu ti├¬n ß╗ƒ tß╗½ng chuß╗ùi tß╗òng hß╗úp.
        scale, valid = rmsse_scale(h)
        # Kh├┤ng tß╗▒ ├╜ ─æß╗òi mß║½u sß╗æ hoß║╖c ph├ón phß╗æi lß║íi trß╗ìng sß╗æ khi chuß╗ùi kh├┤ng hß╗úp lß╗ç.
        assert valid.all(), f"Cß║Ñp {level}: {int((~valid).sum())} chuß╗ùi kh├┤ng c├│ scale hß╗úp lß╗ç"
        rmsse = np.sqrt(np.square(a-p).mean(axis=1) / scale)
        weight = dollars / dollars.sum()
        if return_details and level == 12:
            details = wide[["item_id", "store_id"]].assign(group_code=codes).drop_duplicates().sort_values("group_code").drop(columns="group_code").reset_index(drop=True)
            details["RMSSE"] = rmsse
            details["scale"] = scale
            details["actual_units"] = a.sum(axis=1)
            details["predicted_units"] = p.sum(axis=1)
            details["revenue_train_28"] = dollars
            details["total_revenue_train_28"] = dollars.sum()
            details["revenue_weight"] = weight
            details["contribution_level_12"] = weight * rmsse
            details["contribution_WRMSSE"] = weight * rmsse / 12
        rows.append({"origin": origin, "level": level, "group_by": "+".join(cols) or "Total",
                     "n_series": n, "weight_sum": float(weight.sum()),
                     "weighted_RMSSE": float(np.dot(weight, rmsse)),
                     "contribution_WRMSSE": float(np.dot(weight, rmsse)/12)})
        del h, a, p
    assert sum(row["n_series"] for row in rows) == 42840
    return (rows, details) if return_details else rows


def main():
    data = ROOT / "data/raw"
    wide = pd.read_csv(data / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data / "calendar.csv")
    weeks = set(calendar.loc[calendar.d.isin([f"d_{d}" for d in range(1858, 1914)]), "wm_yr_wk"])
    prices = pd.concat([part.loc[part.wm_yr_wk.isin(weeks)] for part in pd.read_csv(data / "sell_prices.csv", chunksize=250000)], ignore_index=True)
    results = ROOT / "results/analysis"
    rows, summaries = [], []
    for split, origin, folder in [("validation", 1885, results / "stage_C7_validation_d1885_lambda075/folds"),
                                   ("test", 1913, results / "final_test_C6_C7_frozen/C7/folds")]:
        files = sorted(folder.glob("*/forecast_C7.csv"))
        assert len(files) == 10
        forecasts = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in files], ignore_index=True)
        scores = score(origin, wide, calendar, prices, forecasts)
        rows.extend(dict(ky=split, **row) for row in scores)
        summaries.append({"ky": split, "origin": origin, "WRMSSE": sum(row["contribution_WRMSSE"] for row in scores)})
    output = ROOT / ".local_evidence/wrmsse"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "c7_wrmsse_theo_cap.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "c7_wrmsse.csv", index=False)
    (output / "nguon_wrmsse.json").write_text(json.dumps({
        "method": "M5: 12 cß║Ñp, tß╗òng 42.840 chuß╗ùi; trß╗ìng sß╗æ chuß║⌐n h├│a mß╗ùi cß║Ñp rß╗ôi chia 12; scale sau leading zeros.",
        "reference": "https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf",
        "inputs": ["data/raw/sales_train_evaluation.csv", "data/raw/calendar.csv", "data/raw/sell_prices.csv"],
        "forecasts": ["results/analysis/stage_C7_validation_d1885_lambda075/folds/*/forecast_C7.csv", "results/analysis/final_test_C6_C7_frozen/C7/folds/*/forecast_C7.csv"],
        "revenue_days": {"validation": [1858, 1885], "test": [1886, 1913]},
        "scope": "T├¡nh tß║íi m├íy theo ph╞░╞íng ph├íp M5; ch╞░a ─æß╗æi chiß║┐u vß╗¢i evaluator b├¬n ngo├ái; kh├┤ng phß║úi ─æiß╗âm leaderboard."}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    main()
