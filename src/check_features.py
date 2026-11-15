"""Self-check of the feature pipeline: no NaNs, no leakage path, unseen lanes handled, December path works,
and a split-half test that the lane effect is a real, stable signal rather than noise.

Run:  python src/check_features.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeaturePipeline, LaneEncoder, clean,
                      daily_market_index, load_december_inputs, load_train, load_validation)
from validation import fold_indices

tr_raw, va_raw, dec_raw = load_train(), load_validation(), load_december_inputs()
tr, va = clean(tr_raw), clean(va_raw)
y = tr["posted_rate"]

print("1. cleaning")
print("   negative weights after clean:", (tr["abs_weight"] < 0).sum(), "| market_index NaN after fill: train", tr["market_index"].isna().sum(), "val", va["market_index"].isna().sum())
print("   abs_weight NaN (to be imputed in pipeline): train", tr["abs_weight"].isna().sum(), "val", va["abs_weight"].isna().sum())

print("2. fold-wise pipeline: fit on train window only, transform both")
for name, itr, iva in fold_indices(tr["date"]):
    pipe = FeaturePipeline().fit(tr.iloc[itr], y.iloc[itr])
    Xtr, Xva = pipe.transform(tr.iloc[itr]), pipe.transform(tr.iloc[iva])
    assert list(Xtr.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES, Xtr.columns
    nan_num = Xva[NUMERIC_FEATURES].isna().sum().sum()
    unseen_lane = (Xva["lane_count"] == 0).mean()
    print(f"   {name}: train {len(itr):,} valid {len(iva):,} | numeric NaNs in valid: {nan_num} | valid rows on lane unseen in window: {unseen_lane:.3f} | max train date {tr['date'].iloc[itr].max().date()} < min valid date {tr['date'].iloc[iva].min().date()}: {tr['date'].iloc[itr].max() < tr['date'].iloc[iva].min()}")

print("3. full fit -> validation.csv and December chart rows")
pipe = FeaturePipeline().fit(tr, y)
Xva = pipe.transform(va)
print("   validation numeric NaNs:", Xva[NUMERIC_FEATURES].isna().sum().sum(), "| unseen pickup (NaN category):", Xva["pickup"].isna().sum(), "| unseen delivery:", Xva["delivery"].isna().sum(), "| rows on unseen lane:", int((Xva["lane_count"] == 0).sum()))
dec = clean(dec_raw.drop(columns=["predicted_rate"]), daily_mi=daily_market_index(va))
# December rows need coordinates: take them from the training city table
coords = pd.concat([tr[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
                    tr[["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})]).drop_duplicates().set_index("city")
for side in ["pickup", "delivery"]:
    dec[f"{side}_lat"] = dec[side].map(coords["lat"]); dec[f"{side}_lon"] = dec[side].map(coords["lon"])
Xdec = pipe.transform(dec)
print("   December rows:", len(Xdec), "| numeric NaNs:", Xdec[NUMERIC_FEATURES].isna().sum().sum(), "| market_index range:", round(Xdec["market_index"].min(), 3), round(Xdec["market_index"].max(), 3), "| lane_count:", Xdec["lane_count"].iloc[0], "| holidays flagged:", int(Xdec["is_holiday"].sum()))

print("4. lane effect: is it a real signal?  split-half reliability on training data (clean band)")
rng = np.random.default_rng(42)
half = rng.integers(0, 2, len(tr))
a = LaneEncoder(smoothing=0).fit(tr[half == 0], y[half == 0]).lane_stats_
b = LaneEncoder(smoothing=0).fit(tr[half == 1], y[half == 1]).lane_stats_
j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
for mincount in [1, 5, 8]:
    jj = j[(j["count_a"] >= mincount) & (j["count_b"] >= mincount)]
    print(f"   lanes with >= {mincount} loads in both halves: n={len(jj):4d}  corr(effect_a, effect_b) = {jj['effect_a'].corr(jj['effect_b']):.3f}")
print("   sd of raw lane effects (half a, >=5 loads):", round(j[j["count_a"] >= 5]["effect_a"].std(), 4), "(log scale; 0.05 = 5% price premium spread)")
# city-level: origin and destination effects separately (more loads per city -> more stable)
def city_eff(d, yy, col):
    enc = LaneEncoder(smoothing=0)
    rpm = yy / d["distance"]; ok = (rpm > 1.2) & (rpm < 5)
    dd = pd.DataFrame({"c": d[col], "eq": d["equipment"], "band": enc._band(d), "lr": np.log(rpm)})[ok]
    base = dd.groupby(["eq", "band"])["lr"].median()
    dd["res"] = dd["lr"] - dd.set_index(["eq", "band"]).index.map(base)
    return dd.groupby("c")["res"].mean()
for col in ["pickup", "delivery"]:
    ca, cb = city_eff(tr[half == 0], y[half == 0], col), city_eff(tr[half == 1], y[half == 1], col)
    print(f"   {col} city effect split-half corr: {ca.corr(cb):.3f} | sd: {ca.std():.4f}")
print("5. does the lane effect add beyond pickup + delivery city effects? (variance of lane effect after removing city effects)")
enc = LaneEncoder(smoothing=0).fit(tr, y)
ls = enc.lane_stats_.copy(); ls["p"] = [k.split("->")[0] for k in ls.index]; ls["d"] = [k.split("->")[1] for k in ls.index]
pe, de = city_eff(tr, y, "pickup"), city_eff(tr, y, "delivery")
ls["resid"] = ls["effect"] - ls["p"].map(pe) - ls["d"].map(de)
big = ls[ls["count"] >= 15]
print(f"   lanes >=15 loads: sd(lane effect)={big['effect'].std():.4f}  sd(after city effects)={big['resid'].std():.4f}  expected sampling sd~{(0.09 / np.sqrt(big['count'])).mean():.4f}")
print("OK")
