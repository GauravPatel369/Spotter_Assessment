"""Why do all models under-predict the last backtest fold (Sep-Oct) by 3-4%?

Isolates the daily 'market + time' component of log($/mile) net of load mix, then tests which signal
explains it best: same-day market_index, trailing means of market_index, a linear trend, or combinations.
Also replicates the backtest situation (fit on Jan-Aug, extrapolate Sep-Oct) for each candidate.

Run:  python src/temporal_analysis.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FeaturePipeline, clean, load_train, load_validation, daily_market_index

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
pd.set_option("display.width", 200)

# NOTE: no lane_effect here. An in-sample lane encoding leaks each day's own level into the mix model
# (its daily mean correlates 0.44 with market_index) and halves the estimated market coefficient.
MIX_FEATURES = ["log_distance", "abs_weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "haversine", "circuity",
                "bearing_sin", "bearing_cos", "dow", "is_holiday", "equipment", "pickup", "delivery"]


def main():
    tr = clean(load_train()); va = clean(load_validation())
    y = tr["posted_rate"]; rpm = y / tr["distance"]
    ok = ((rpm > 1.2) & (rpm < 5)).values
    tr, y, rpm = tr[ok].reset_index(drop=True), y[ok].reset_index(drop=True), rpm[ok].reset_index(drop=True)

    # 1. mix-adjusted daily level: fit a model WITHOUT any time/market feature, take mean residual per day
    pipe = FeaturePipeline().fit(tr, y)
    X = pipe.transform(tr)[MIX_FEATURES]
    z = np.log(rpm)
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=400, categorical_features="from_dtype", random_state=0).fit(X, z)
    tr["res"] = z - m.predict(X)
    d = tr.groupby("date").agg(level=("res", "mean"), n=("res", "size"), mi=("market_index", "mean"))
    d["t"] = np.arange(len(d))
    full_mi = pd.concat([daily_market_index(tr), daily_market_index(va)]).sort_index()
    for w in [7, 14, 30, 45, 60, 90]:
        d[f"mi_ma{w}"] = full_mi.rolling(w, min_periods=1).mean().reindex(d.index).values
    for lag in [7, 14, 30]:
        d[f"mi_lag{lag}"] = full_mi.shift(lag).bfill().reindex(d.index).values
    print("=" * 100 + "\n1. Which signal explains the mix-adjusted daily level of log($/mile)?  (in-sample R2 over all 304 days)\n" + "=" * 100)

    def r2(cols):
        Xd = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        b = np.linalg.lstsq(Xd, d["level"].values, rcond=None)[0]
        res = d["level"].values - Xd @ b
        return 1 - res.var() / d["level"].var(), b

    rows = []
    for cols in [["mi"], ["t"], ["mi", "t"], ["mi_ma7"], ["mi_ma14"], ["mi_ma30"], ["mi_ma45"], ["mi_ma60"], ["mi_ma90"], ["mi_lag7"], ["mi_lag14"], ["mi_lag30"],
                 ["mi", "mi_ma30"], ["mi", "mi_ma60"], ["mi_ma30", "t"], ["mi_ma60", "t"], ["mi", "mi_ma30", "t"], ["mi", "mi_ma60", "t"]]:
        rr, b = r2(cols)
        rows.append({"signal": " + ".join(cols), "R2": round(rr, 3), "coefs": np.round(b[1:], 4).tolist()})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nmonthly mean of mix-adjusted level (log scale; 0.01 = 1%) and market_index:")
    mm = d.groupby(d.index.to_period("M")).agg(level=("level", "mean"), mi=("mi", "mean"), mi_ma60=("mi_ma60", "mean")).round(4)
    print(mm.to_string())

    print("\n" + "=" * 100 + "\n2. Backtest replication: fit the daily-level model on Jan-Aug, extrapolate Sep-Oct. Mean error in % (negative = under-prediction)\n" + "=" * 100)
    fit = d.index <= "2025-08-31"; test = ~fit
    rows = []
    for cols in [["mi"], ["mi", "t"], ["mi_ma30"], ["mi_ma60"], ["mi_ma30", "t"], ["mi_ma60", "t"], ["mi", "mi_ma60"], ["mi", "mi_ma60", "t"]]:
        Xd = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        b = np.linalg.lstsq(Xd[fit], d["level"].values[fit], rcond=None)[0]
        pred = Xd @ b
        e_test = (pred - d["level"].values)[test]
        e_fit = (pred - d["level"].values)[fit]
        # also what a 'saturating' tree does: carry the last-month level forward
        rows.append({"signal": " + ".join(cols), "fit_rmse%": round(100 * np.sqrt(np.mean(e_fit ** 2)), 2), "SepOct_bias%": round(100 * e_test.mean(), 2), "SepOct_rmse%": round(100 * np.sqrt(np.mean(e_test ** 2)), 2)})
    aug = d.loc["2025-08-01":"2025-08-31", "level"].mean(); last = d.loc[test, "level"]
    rows.append({"signal": "carry August level forward (tree-like)", "fit_rmse%": np.nan, "SepOct_bias%": round(100 * (aug - last).mean(), 2), "SepOct_rmse%": round(100 * np.sqrt(np.mean((aug - last) ** 2)), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    # same exercise for earlier folds to check the pattern is stable, not a one-off
    print("\nsame test for every backtest fold (bias % on the 2-month horizon):")
    rows = []
    for end, hor in [("2025-04-30", ("2025-05-01", "2025-06-30")), ("2025-06-30", ("2025-07-01", "2025-08-31")), ("2025-07-31", ("2025-08-01", "2025-09-30")), ("2025-08-31", ("2025-09-01", "2025-10-31"))]:
        fit = d.index <= end; test = (d.index >= hor[0]) & (d.index <= hor[1])
        row = {"train_end": end}
        for cols in [["mi"], ["mi", "t"], ["mi_ma60"], ["mi_ma60", "t"], ["mi", "mi_ma60", "t"]]:
            Xd = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
            b = np.linalg.lstsq(Xd[fit], d["level"].values[fit], rcond=None)[0]
            row[" + ".join(cols)] = round(100 * ((Xd @ b) - d["level"].values)[test].mean(), 2)
        lastm = d.loc[fit, "level"].iloc[-30:].mean()
        row["carry last 30d level"] = round(100 * (lastm - d.loc[test, "level"]).mean(), 2)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 100 + "\n3. What do the candidate signals imply for Nov-Dec?  (validation.csv features only)\n" + "=" * 100)
    vd = pd.DataFrame(index=sorted(va["date"].unique()))
    vd["mi"] = full_mi.reindex(vd.index).values
    vd["mi_ma60"] = full_mi.rolling(60, min_periods=1).mean().reindex(vd.index).values
    vd["mi_ma30"] = full_mi.rolling(30, min_periods=1).mean().reindex(vd.index).values
    vd["t"] = np.arange(len(d), len(d) + len(vd))
    print("validation monthly means:", vd.groupby(vd.index.to_period("M")).mean().round(4).to_string())
    print("train Sep-Oct means:     ", d.loc["2025-09-01":, ["mi", "mi_ma60", "mi_ma30"]].mean().round(4).to_dict())

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(d.index, 100 * d["level"], color="#0B6E7A", lw=1, label="mix-adjusted daily level (%)"); ax[0].plot(d.index, 100 * d["level"].rolling(7, center=True).mean(), color="#D9822B", lw=2, label="7-day mean"); ax[0].legend(); ax[0].set_ylabel("% vs model w/o time")
    ax[1].plot(full_mi.index, full_mi.values, color="#0B6E7A", lw=1, label="market_index (daily)"); ax[1].plot(full_mi.index, full_mi.rolling(60, min_periods=1).mean(), color="#D9822B", lw=2, label="60-day trailing mean"); ax[1].axvline(pd.Timestamp("2025-11-01"), color="grey", ls=":"); ax[1].legend(); ax[1].set_ylabel("market_index")
    ax[0].set_title("Market/time component of rates vs market_index", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "14_temporal_component.png"); plt.close(fig)
    d.to_csv(ROOT / "outputs" / "daily_level.csv")


if __name__ == "__main__":
    main()
