"""Origin-specific, post-forecast demand analysis of the sealed M5 forecasts."""

import numpy as np
import pandas as pd

from .wrmsse_business_v2 import MAE_TOL
from .wrmsse_stage4 import read, verify_files
from .wrmsse_stage5 import (
    DETAIL, FORECAST, REPORTS, ROOT, STAGE2, STAGE3, STAGE4, TABLES,
    WEEKDAYS, components, csv, ratio, summarize,
)

KEY = ["item_id", "store_id"]
META = KEY + ["state_id", "cat_id", "dept_id"]
GROUP = "demand_group"
INPUT = REPORTS / "demand_groups"
PROCESSED = ROOT / read(STAGE4 / "preflight.json")["processed"]
VALID_RF = STAGE3 / "features/FE-04/attempt_001"
VALID_SN = STAGE2 / "seasonal_naive"
ENRICHED = TABLES / "demand_group_test.parquet"
BINS = ["[1, 1.32)", "[1.32, 2)", "[2, 4)", "[4, 8)", "[8, +∞)", "+∞", "NA"]


def join_labels(rows, labels):
    enriched = rows.merge(labels[KEY + ["ADI", GROUP]], on=KEY, how="left", sort=False,
                          validate="many_to_one", indicator=True)
    if not enriched.pop("_merge").eq("both").all():
        raise ValueError("Missing demand labels")
    pd.testing.assert_frame_equal(rows.reset_index(drop=True), enriched[rows.columns], check_exact=True)
    return enriched


def historical_revenue(members, train, validation):
    calendar = pd.read_parquet(PROCESSED / "calendar.parquet").set_index("day")
    weeks = calendar.loc[1858:1913, "wm_yr_wk"]
    price_key = [*KEY, "wm_yr_wk"]
    parts = []
    with pd.read_csv(ROOT / "data/raw/sell_prices.csv", chunksize=200_000) as reader:
        for chunk in reader:
            parts.append(chunk.loc[chunk.wm_yr_wk.isin(weeks.unique())])
    prices = pd.concat(parts).set_index(price_key, verify_integrity=True).sell_price
    revenue = {}
    for origin, units in [(1885, train[:, -28:]), (1913, validation)]:
        selected = weeks.loc[origin - 27:origin].to_numpy()
        total = np.zeros(len(members))
        for week in np.unique(selected):
            qty = units[:, selected == week].sum(axis=1)
            keys = pd.MultiIndex.from_frame(members[KEY].assign(wm_yr_wk=week))
            price = prices.reindex(keys)
            if (price.isna().to_numpy() & (qty > 0)).any():
                raise ValueError("Missing historical price on positive sales")
            total += price.fillna(0).to_numpy() * qty
        revenue[origin] = total
    return revenue


def inspect_inputs():
    verify_files(PROCESSED, read(PROCESSED / "manifest.json")["files"])
    members = pd.read_parquet(PROCESSED / "series.parquet").sort_values("series_id").reset_index(drop=True)
    np.testing.assert_array_equal(members.series_id, np.arange(30490))
    train = np.load(PROCESSED / "train_sales.npy", mmap_mode="r")
    actual = np.load(PROCESSED / "validation_sales.npy", mmap_mode="r")
    revenue = historical_revenue(members, train, actual)
    summary = csv(INPUT / "demand_groups_summary.csv")
    if summary.duplicated(["origin", GROUP]).any() or len(summary) != 4 or summary.isna().any().any():
        raise ValueError("Invalid origin summary")
    labels, audit = {}, {}
    for origin in (1885, 1913):
        supplied = csv(INPUT / f"demand_groups_origin_{origin}.csv")
        if supplied.duplicated(KEY).any() or supplied[META + [GROUP]].isna().any().any():
            raise ValueError(f"Invalid series keys or group at {origin}")
        ordered = members.merge(supplied, on=META, how="outer", validate="one_to_one", indicator=True)
        if len(ordered) != len(members) or not ordered.pop("_merge").eq("both").all():
            raise ValueError("Series membership differs from the official grid")
        ordered = ordered.sort_values("series_id").reset_index(drop=True)
        positive = (train if origin == 1885 else np.column_stack([train, actual])) > 0
        count = positive.sum(axis=1)
        adi = np.divide(origin - positive.argmax(axis=1), count, out=np.full(len(count), np.inf), where=count > 0)
        np.testing.assert_allclose(ordered.ADI, adi, rtol=1e-13, atol=0)
        np.testing.assert_array_equal(ordered[GROUP], np.where(adi < 1.32, "frequent", "intermittent"))
        stats = ordered.groupby(GROUP).ADI.agg(["size", "median"])
        given = summary.loc[summary.origin.eq(origin)].set_index(GROUP).loc[stats.index]
        np.testing.assert_array_equal(stats["size"], given.n_series)
        np.testing.assert_allclose(stats["size"] * 100 / len(members), given.series_share_pct)
        np.testing.assert_allclose(stats["median"], given.median_ADI)
        dollar_sales = pd.Series(revenue[origin]).groupby(ordered[GROUP]).sum()
        np.testing.assert_allclose(dollar_sales, given.revenue_28d, rtol=0, atol=1e-7)
        labels[origin] = ordered
        audit[str(origin)] = {"rows": len(supplied), "dtypes": supplied.dtypes.astype(str).to_dict(),
                              "nulls": supplied.isna().sum().to_dict(), "duplicate_keys": 0,
                              "infinite_ADI": int(np.isinf(adi).sum()), "at_threshold": int((adi == 1.32).sum()),
                              "max_ADI_reconstruction_error": float(np.max(np.abs(adi - ordered.ADI)))}
    transitions = pd.crosstab(labels[1885][GROUP], labels[1913][GROUP])
    audit.update(transitions=transitions.to_dict(orient="index"),
                 summary={"rows": len(summary), "dtypes": summary.dtypes.astype(str).to_dict(),
                          "nulls": summary.isna().sum().to_dict()})
    return labels, summary, audit


def official_scores(origin):
    if origin == 1913:
        official = pd.read_parquet(STAGE4 / "test_wrmsse_by_series.parquet")
    else:
        official = pd.concat([pd.read_parquet(folder / "wrmsse_by_series.parquet").assign(model=model)
                              for model, folder in (("random_forest", VALID_RF), ("seasonal_naive", VALID_SN))])
    scores = official[official.level.eq(12)].reset_index(drop=True).copy()
    scores[KEY] = pd.DataFrame(scores.key.tolist(), columns=KEY)
    np.testing.assert_allclose(scores.contribution, scores.weight * scores.rmsse, atol=1e-15)
    return scores


def validation_rows(labels):
    native = pd.read_parquet(VALID_SN / "validation_predictions.parquet")
    rows = native.sort_values(["series_id", "day"]).reset_index(drop=True)
    np.testing.assert_array_equal(rows.series_id, np.repeat(labels.series_id, 28))
    np.testing.assert_array_equal(rows.day, np.tile(np.arange(1886, 1914), len(labels)))
    rows = rows.rename(columns={"prediction": FORECAST["sn"]})
    for name, path in [(FORECAST["rf"], VALID_RF / "validation_predictions.npy"),
                       ("actual_sales", PROCESSED / "validation_sales.npy")]:
        values = np.load(path, mmap_mode="r")
        assert values.shape == (len(labels), 28)
        rows[name] = values.ravel()
    return rows.merge(labels[["series_id", *META]], on="series_id", how="left", validate="many_to_one")


def group_metrics(rows, labels, origin, given):
    series = summarize(rows, [*KEY, GROUP, "series_id"])
    agg = summarize(rows, [GROUP])
    scores = official_scores(origin)
    stats = labels.groupby(GROUP).ADI.agg(n_series="size", mean_ADI="mean", median_ADI="median")
    stats = stats.join(agg.set_index(GROUP)[["observations", "actual_sales", "actual_share_pct"]])
    stats["series_share_pct"] = 100 * stats.n_series / len(series)
    zeros = rows.actual_sales.eq(0).groupby(rows[GROUP]).sum()
    stats["zero_day_pct"] = ratio(zeros, stats.observations)
    stats["units_per_series_28d"] = stats.actual_sales / stats.n_series
    positive_days = stats.observations - zeros
    stats["units_per_selling_day"] = stats.actual_sales / positive_days.where(positive_days > 0)
    stats = stats.join(given.set_index(GROUP)[["revenue_28d", "revenue_share_pct"]])
    np.testing.assert_allclose(stats.revenue_28d * 100 / stats.revenue_28d.sum(), stats.revenue_share_pct)
    diff = series.rf_mae - series.sn_mae
    for col, flag in (("rf_win_pct", diff < -MAE_TOL), ("tie_pct", diff.abs() <= MAE_TOL),
                      ("rf_loss_pct", diff > MAE_TOL)):
        stats[col] = flag.groupby(series[GROUP]).mean() * 100
    np.testing.assert_allclose(stats[["rf_win_pct", "tie_pct", "rf_loss_pct"]].sum(axis=1), 100)
    metrics = agg.set_index(GROUP)
    metrics["absolute_bias_improvement_pct"] = ratio(metrics.sn_bias_pct.abs() - metrics.rf_bias_pct.abs(),
                                                      metrics.sn_bias_pct.abs())
    for key, model in (("rf", "random_forest"), ("sn", "seasonal_naive")):
        by_key = scores.loc[scores.model.eq(model), KEY + ["weight", "rmsse", "contribution"]]
        tagged = labels.merge(by_key, on=KEY, how="left", validate="one_to_one")
        assert tagged.weight.notna().all() and len(tagged) == len(by_key)
        sums = tagged.groupby(GROUP)[["weight", "contribution"]].sum()
        np.testing.assert_allclose(sums.weight.sum(), 1 / 12)
        np.testing.assert_allclose(1200 * sums.weight, stats.revenue_share_pct, atol=1e-8)
        metrics[key + "_L12_contribution"] = sums.contribution
        metrics[key + "_mean_rmsse"] = tagged.groupby(GROUP).rmsse.mean()
        metrics[key + "_valid_rmsse_series"] = tagged.groupby(GROUP).rmsse.count()
        metrics[key + "_conditional_weighted_rmsse"] = sums.contribution / sums.weight.where(sums.weight > 0)
        metrics["L12_weight"] = sums.weight
        levels_file = (STAGE4 / "test_wrmsse_by_level.csv" if origin == 1913 else
                       (VALID_RF if key == "rf" else VALID_SN) / "wrmsse_by_level.csv")
        levels = csv(levels_file)
        if origin == 1913:
            levels = levels[levels.model.eq(model)]
        np.testing.assert_allclose(sums.contribution.sum(), levels.loc[levels.level.eq(12), "contribution"].item())
        target = csv(TABLES / "validation_test.csv").set_index("window").at[
            "Test" if origin == 1913 else "Validation", model]
        np.testing.assert_allclose(levels.contribution.sum(), target, atol=1e-12, rtol=0)
    return stats.reset_index(), metrics.reset_index(), series


def adi_bin(adi):
    cat = pd.cut(adi, bins=[1, 1.32, 2, 4, 8, np.inf], labels=BINS[:5], right=False).astype("string")
    return cat.mask(np.isposinf(adi), "+∞").fillna("NA")


def time_tables(rows):
    """Pool item-store-day errors before division, including zero-sale observations."""
    calendar = rows[["day", "date"]].drop_duplicates().sort_values("day").copy()
    dates = pd.to_datetime(calendar.date)
    np.testing.assert_array_equal(calendar.day, np.arange(1914, 1942))
    np.testing.assert_array_equal(dates, pd.date_range("2016-04-25", periods=28))
    calendar["week"] = (calendar.day - 1914) // 7 + 1
    calendar["weekday_no"] = dates.dt.day_of_week + 1
    calendar["weekday"] = calendar.weekday_no.map(dict(enumerate(WEEKDAYS, 1)))
    rows = rows.merge(calendar, on=["day", "date"], validate="many_to_one")
    values = ["actual_sales", FORECAST["rf"], "rf_error", "rf_absolute_error", "rf_squared_error",
              "observations", "rf_wape_pct", "rf_mae", "rf_rmse", "rf_bias_pct"]
    result = {}
    for name, keys in (("daily", ["day", "date"]), ("weekly", ["week"]), ("weekday", ["weekday_no", "weekday"])):
        keys = [GROUP, *keys]
        result[name] = summarize(rows, keys)[keys + values]
    daily = result["daily"]
    for col in ["actual_sales", FORECAST["rf"], "rf_error"]:
        daily[col + "_cumulative"] = daily.groupby(GROUP)[col].cumsum()
    weeks = calendar.groupby("week").date.agg(week_start="min", week_end="max")
    result["weekly"] = result["weekly"].join(weeks, on="week")
    system = csv(TABLES / "daily_sales.csv").set_index("day")
    totals = daily.groupby("day")[values[:6]].sum()
    np.testing.assert_allclose(totals, system.loc[totals.index, totals.columns], atol=1e-8, rtol=1e-13)
    return result


def representatives(rows, series):
    eligible = series[series.actual_sales.gt(0)]
    chosen = []
    for _, group in eligible.groupby(GROUP):
        for quantile in (.1, .9):
            target = group.rf_wape_pct.quantile(quantile)
            ranked = group.assign(distance=(group.rf_wape_pct - target).abs()).sort_values(["distance", *KEY])
            selected = ranked.iloc[0]
            days = rows[rows.series_id.eq(selected.series_id)].sort_values("day")
            peak = days.loc[days.actual_sales.idxmax()]
            chosen.append({**selected[[*KEY, GROUP, "series_id", "rf_wape_pct", "rf_bias_pct",
                                       "sn_wape_pct", "sn_bias_pct"]].to_dict(),
                           "quantile": quantile, "target_wape_pct": target, "eligible": len(group),
                           "ADI": peak.ADI, "zero_days": int(days.actual_sales.eq(0).sum()),
                           "peak_date": peak.date, "peak_units": peak.actual_sales,
                           "peak_rf": peak[FORECAST["rf"]], "peak_sn": peak[FORECAST["sn"]],
                           "rf_units_on_zero_days": days.loc[days.actual_sales.eq(0), FORECAST["rf"]].sum()})
    return pd.DataFrame(chosen)


def build():
    for run in (VALID_RF, VALID_SN):
        rf = run == VALID_RF
        names = ["wrmsse_by_level.csv", "wrmsse_by_series.parquet",
                 "validation_predictions.npy" if rf else "validation_predictions.parquet"]
        files = read(VALID_RF / "manifest.json" if rf else STAGE2 / "manifest_completed.json")["files"]
        verify_files(run, {name: files[name if rf else "seasonal_naive/" + name] for name in names})
    verify_files(STAGE4, read(STAGE4 / "manifest_completed.json")["files"])
    labels, supplied, audit = inspect_inputs()
    output = {}
    summaries, metrics = [], []
    for origin, split in ((1885, "Validation"), (1913, "Test")):
        original = validation_rows(labels[origin]) if origin == 1885 else pd.read_parquet(DETAIL)
        detail = join_labels(original, labels[origin])
        assert len(detail) == 853720 and not detail.duplicated(KEY + ["day"]).any()
        assert detail.day.between(origin + 1, origin + 28).all()
        rows = components(detail)
        summary, perf, series = group_metrics(rows, labels[origin], origin, supplied[supplied.origin.eq(origin)])
        for frame in (summary, perf):
            frame.insert(0, "split", split)
            frame.insert(1, "origin", origin)
        assert summary.n_series.sum() == 30490 and summary.observations.sum() == len(rows)
        summaries.append(summary)
        metrics.append(perf)
    assert summary.actual_sales.sum() == 1231764
    system = csv(TABLES / "system.csv").iloc[0]
    for col in ["actual_sales", *FORECAST.values(), "rf_absolute_error", "sn_absolute_error", "rf_error", "sn_error"]:
        np.testing.assert_allclose(perf[col].sum(), system[col], rtol=1e-13)
    output["summary"] = pd.concat(summaries, ignore_index=True)
    output["metrics"] = pd.concat(metrics, ignore_index=True)
    for name, key in (("store", "store_id"), ("category", "cat_id")):
        segment = summarize(rows, [key, GROUP])
        segment["n_series"] = segment.observations // 28
        output[name] = segment
    bins = summarize(rows.assign(ADI_bin=adi_bin(rows.ADI)), ["ADI_bin"]).set_index("ADI_bin").reindex(BINS)
    bins["n_series"] = bins.observations.fillna(0).astype(int) // 28
    bins["observations"] = bins.observations.fillna(0).astype(int)
    output["adi"] = bins.reset_index()
    output["representatives"] = representatives(rows, series)
    output.update(time_tables(rows))
    return detail, output, audit
