"""Phase 9 - Error analysis of backtest predictions saved by experiments.py (last fold: train Jan-Aug -> predict Sep-Oct).

Usage:  python src/evaluate.py outputs/oof/<file>.csv [--label name]
Writes figures to reports/figures/err_<label>_*.png and prints segment tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import clean, load_train
from validation import metrics

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
pd.set_option("display.width", 200)


def seg_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "MAE": g.apply(lambda d: np.mean(np.abs(d.pred - d.y))),
        "MAPE%": g.apply(lambda d: np.mean(np.abs(d.pred - d.y) / d.y) * 100),
        "bias": g.apply(lambda d: np.mean(d.pred - d.y)),
        "bias%": g.apply(lambda d: np.median((d.pred - d.y) / d.y) * 100),
        "MAE_clean": g.apply(lambda d: np.mean(np.abs(d.pred - d.y)[d.clean])),
    })
    return out.round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    label = args.label or Path(args.oof).stem
    oof = pd.read_csv(args.oof, parse_dates=["date"])
    tr = clean(load_train())
    df = oof.merge(tr.drop(columns=["date"]), on="load_id", how="left")
    df["rpm"] = df.y / df.distance
    df["pred_rpm"] = df.pred / df.distance
    df["clean"] = (df.rpm > 1.2) & (df.rpm < 5)
    df["resid"] = df.pred - df.y
    df["resid_pct"] = 100 * df.resid / df.y
    df["dist_band"] = pd.cut(df.distance, [0, 250, 500, 750, 1000, 1500, 2000, 2500, 4000])
    df["wt_band"] = pd.cut(df.abs_weight, [0, 15000, 25000, 35000, 45000, 50000])
    df["week"] = df.date.dt.to_period("W").astype(str)
    df["month"] = df.date.dt.to_period("M").astype(str)
    df["dow"] = df.date.dt.dayofweek
    df["lane"] = df.pickup + "->" + df.delivery
    lane_cnt = tr[tr.date <= "2025-08-31"].assign(lane=lambda d: d.pickup + "->" + d.delivery).lane.value_counts()
    df["lane_hist"] = pd.cut(df.lane.map(lane_cnt).fillna(0), [-1, 0, 4, 9, 19, 100], labels=["0", "1-4", "5-9", "10-19", "20+"])
    df["y_band"] = pd.cut(df.y, [0, 500, 1000, 2000, 3000, 4500, 6000, 30000])

    print("=" * 100 + f"\nERROR ANALYSIS: {label}  (n={len(df):,})\n" + "=" * 100)
    m_all, m_clean = metrics(df.y, df.pred), metrics(df.y[df.clean], df.pred[df.clean])
    print("all rows:  ", {k: round(v, 3) for k, v in m_all.items()})
    print("clean rows:", {k: round(v, 3) for k, v in m_clean.items()})
    print(f"share over-predicted: {(df.resid > 0).mean():.3f} | median signed error %: {df.resid_pct.median():+.2f}% | clean median signed error %: {df.resid_pct[df.clean].median():+.2f}%")
    print("\nresidual % quantiles (clean rows):", df.resid_pct[df.clean].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2).to_dict())
    for by in ["month", "equipment", "dist_band", "wt_band", "lane_hist", "y_band", "dow"]:
        print(f"\n-- by {by} --")
        print(seg_table(df, by).to_string())
    print("\n-- weekly bias % (clean rows), to see drift within the horizon --")
    print(df[df.clean].groupby("week").resid_pct.median().round(2).to_string())
    print("\n-- worst 10 clean-row errors --")
    print(df[df.clean].reindex(df[df.clean].resid.abs().sort_values(ascending=False).index).head(10)[["load_id", "lane", "equipment", "distance", "abs_weight", "date", "market_index", "y", "pred", "resid_pct"]].to_string(index=False))
    print("\n-- unseen-lane rows --")
    print(seg_table(df, df.lane_hist.eq("0").map({True: "unseen lane", False: "seen lane"})).to_string())

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    c1, c2 = "#0B6E7A", "#D9822B"
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    s = df.sample(min(6000, len(df)), random_state=0)
    ax[0, 0].scatter(s.y, s.pred, s=4, alpha=0.3, color=c1); lim = [0, s.y.quantile(0.995)]
    ax[0, 0].plot(lim, lim, "--", color=c2); ax[0, 0].set_xlim(lim); ax[0, 0].set_ylim(lim); ax[0, 0].set_xlabel("actual posted_rate"); ax[0, 0].set_ylabel("predicted"); ax[0, 0].set_title("predicted vs actual (axes cut at 99.5th pct)")
    ax[0, 1].hist(df.resid_pct[df.clean].clip(-30, 30), bins=100, color=c1); ax[0, 1].set_title("residual % (clean rows, clipped ±30%)"); ax[0, 1].axvline(0, color=c2)
    ax[1, 0].scatter(s.pred, s.resid_pct.clip(-60, 60), s=4, alpha=0.3, color=c1); ax[1, 0].axhline(0, color=c2); ax[1, 0].set_xlabel("predicted"); ax[1, 0].set_ylabel("residual %"); ax[1, 0].set_title("residual % vs predicted")
    wk = df[df.clean].groupby("date").resid_pct.median()
    ax[1, 1].plot(wk.index, wk.values, color=c1); ax[1, 1].axhline(0, color=c2); ax[1, 1].set_title("daily median signed error % (clean rows)"); ax[1, 1].tick_params(axis="x", rotation=30)
    fig.suptitle(f"Error analysis: {label}", x=0.01, ha="left", fontweight="bold"); fig.tight_layout()
    fig.savefig(FIG / f"err_{label}_residuals.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, by in zip(ax, ["dist_band", "equipment", "lane_hist"]):
        t = seg_table(df, by)
        a.bar(range(len(t)), t["MAPE%"], color=c1); a.set_xticks(range(len(t))); a.set_xticklabels([str(i) for i in t.index], rotation=35, ha="right"); a.set_title(f"MAPE % by {by}")
    fig.tight_layout(); fig.savefig(FIG / f"err_{label}_segments.png"); plt.close(fig)
    print(f"\nfigures: err_{label}_residuals.png, err_{label}_segments.png")


if __name__ == "__main__":
    main()
