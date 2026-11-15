"""Phase 4 - Development (train_test.csv) vs final validation (validation.csv) distribution comparison.

Uses only validation FEATURES (no labels exist). Compares validation against (a) the whole training set
and (b) the last two training months (Sep-Oct), which is the natural reference for a chronological model.

Run:  python src/drift_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
pd.set_option("display.width", 200)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the expected (reference) distribution."""
    e, a = expected.dropna(), actual.dropna()
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_cat(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected) | set(actual))
    pe = expected.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-6)
    pa = actual.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-6)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def main():
    tr = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    va = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    recent = tr[tr.date >= "2025-09-01"]
    for d in (tr, va, recent):
        d["abs_weight"] = d.weight.abs()
        d["lane"] = d.pickup + " -> " + d.delivery

    section("1. Temporal coverage")
    print("train:      ", tr.date.min().date(), "->", tr.date.max().date(), f"({tr.date.nunique()} days, {len(tr) / tr.date.nunique():.0f} loads/day)")
    print("validation: ", va.date.min().date(), "->", va.date.max().date(), f"({va.date.nunique()} days, {len(va) / va.date.nunique():.0f} loads/day)")
    print("gap between last train day and first validation day:", (va.date.min() - tr.date.max()).days, "day(s) -> validation is strictly the two months after training; no overlap")
    print("validation loads per day quantiles:", va.groupby("date").size().quantile([0, 0.25, 0.5, 0.75, 1]).to_dict())
    print("validation dow share:", va.date.dt.dayofweek.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("train dow share:     ", tr.date.dt.dayofweek.value_counts(normalize=True).sort_index().round(3).to_dict())

    section("2. Numerical features: PSI and KS (validation vs all-train / vs Sep-Oct)")
    rows = []
    for c in ["distance", "abs_weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        rows.append({
            "feature": c,
            "train_mean": tr[c].mean(), "val_mean": va[c].mean(),
            "train_p50": tr[c].median(), "val_p50": va[c].median(),
            "PSI_vs_all": psi(tr[c], va[c]), "PSI_vs_SepOct": psi(recent[c], va[c]),
            "KS_vs_all": stats.ks_2samp(tr[c].dropna(), va[c].dropna()).statistic,
            "KS_vs_SepOct": stats.ks_2samp(recent[c].dropna(), va[c].dropna()).statistic,
        })
    out = pd.DataFrame(rows).set_index("feature")
    print(out.round(4).to_string())
    print("\nPSI rule of thumb: <0.10 stable, 0.10-0.25 moderate shift, >0.25 major shift")
    # reference: natural month-to-month PSI inside training, so we know what 'normal' looks like
    months = sorted(tr.date.dt.to_period("M").unique())
    ref = []
    for c in ["distance", "abs_weight", "market_index"]:
        vals = [psi(tr[tr.date.dt.to_period("M") == months[i]][c], tr[tr.date.dt.to_period("M") == months[i + 1]][c]) for i in range(len(months) - 1)]
        ref.append((c, round(np.median(vals), 4), round(max(vals), 4)))
    print("reference month-to-month PSI inside training (median, max):", ref)

    print("\nranges:")
    for c in ["distance", "weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon"]:
        print(f"  {c:13s} train [{tr[c].min():.3f}, {tr[c].max():.3f}]  validation [{va[c].min():.3f}, {va[c].max():.3f}]  "
              f"val rows outside train range: {((va[c] < tr[c].min()) | (va[c] > tr[c].max())).sum()}")

    section("3. Missing values and anomalies")
    for c in ["weight", "market_index"]:
        print(f"  {c:13s} missing: train {100 * tr[c].isna().mean():.2f}%  validation {100 * va[c].isna().mean():.2f}%")
    print(f"  negative weight: train {100 * (tr.weight < 0).mean():.2f}%  validation {100 * (va.weight < 0).mean():.2f}%")
    print(f"  weight at +/-47500: train {100 * (tr.weight.abs() == 47500).mean():.2f}%  validation {100 * (va.weight.abs() == 47500).mean():.2f}%")
    print(f"  weight at exactly 47500 share of max: train max {tr.weight.max()}, validation max {va.weight.max()}")

    section("4. Categorical features")
    print("equipment share train:     ", tr.equipment.value_counts(normalize=True).round(4).to_dict())
    print("equipment share validation:", va.equipment.value_counts(normalize=True).round(4).to_dict(), f"| PSI={psi_cat(tr.equipment, va.equipment):.4f}")
    new_cities = sorted((set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery)))
    print("\ncities in validation not in train:", new_cities)
    inv = va[va.pickup.isin(new_cities) | va.delivery.isin(new_cities)]
    print(f"validation rows touching an unseen city: {len(inv):,} ({100 * len(inv) / len(va):.2f}%) | both ends unseen: {(va.pickup.isin(new_cities) & va.delivery.isin(new_cities)).sum()}")
    nc = pd.concat([va[va.pickup.isin(new_cities)][["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
                    va[va.delivery.isin(new_cities)][["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})]).drop_duplicates()
    print("unseen city coordinates and load counts:")
    cnt = pd.concat([va.pickup, va.delivery]).value_counts()
    nc["loads_as_pickup_or_delivery"] = nc.city.map(cnt)
    print(nc.sort_values("city").to_string(index=False))
    print("\nunseen city timing (first/last date, loads per month):")
    print(inv.groupby(inv.date.dt.to_period("M")).size().to_dict(), "| first:", inv.date.min().date())
    print("\nvalidation lanes:", va.lane.nunique(), "| share of validation rows whose lane exists in train:", round(va.lane.isin(set(tr.lane)).mean(), 4))
    print("share of validation rows whose lane has >=10 train loads:", round(va.lane.map(tr.lane.value_counts()).fillna(0).ge(10).mean(), 4))
    print("pickup city PSI (train vs val, shared cities only):", round(psi_cat(tr.pickup, va[~va.pickup.isin(new_cities)].pickup), 4),
          "| delivery:", round(psi_cat(tr.delivery, va[~va.delivery.isin(new_cities)].delivery), 4))
    top_shift = (va.pickup.value_counts(normalize=True) - tr.pickup.value_counts(normalize=True)).dropna().sort_values()
    print("largest pickup share changes (val - train):", top_shift.head(3).round(4).to_dict(), top_shift.tail(3).round(4).to_dict())

    section("5. market_index regime")
    tm = tr.groupby(tr.date.dt.to_period("M")).market_index.mean().round(3)
    vm = va.groupby(va.date.dt.to_period("M")).market_index.mean().round(3)
    print("monthly mean market_index train:", tm.to_dict())
    print("monthly mean market_index validation:", vm.to_dict())
    print("validation market_index lies within the training range:", va.market_index.min() >= tr.market_index.min() and va.market_index.max() <= tr.market_index.max())
    print("training months with similar market level (|mean - 0.93| < 0.07):", [str(k) for k, v in tm.items() if abs(v - 0.93) < 0.07])

    section("6. Feature combinations: equipment x distance band shares")
    for d, n in [(tr, "train"), (va, "validation")]:
        d["dist_band"] = pd.cut(d.distance, [0, 250, 500, 1000, 1500, 2500, 4000])
    ct_tr = pd.crosstab(tr.equipment, tr.dist_band, normalize=True).round(3)
    ct_va = pd.crosstab(va.equipment, va.dist_band, normalize=True).round(3)
    print("train:\n", ct_tr.to_string()); print("validation:\n", ct_va.to_string())
    print("max abs cell difference:", float((ct_tr - ct_va).abs().max().max()))

    section("7. Figures")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for a, c in zip(ax.ravel(), ["distance", "abs_weight", "market_index", "quote_signal"]):
        a.hist(tr[c].dropna(), bins=60, density=True, alpha=0.55, color="#0B6E7A", label="train (Jan-Oct)")
        a.hist(va[c].dropna(), bins=60, density=True, alpha=0.55, color="#D9822B", label="validation (Nov-Dec)")
        a.set_title(c); a.legend(fontsize=8)
    fig.suptitle("Train vs validation feature distributions", x=0.01, ha="left", fontweight="bold"); fig.tight_layout()
    fig.savefig(FIG / "12_drift_numeric.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    allc = pd.concat([tr[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"})]).drop_duplicates()
    ax.scatter(allc.lon, allc.lat, s=16, color="#0B6E7A", label="train cities")
    ax.scatter(nc.lon, nc.lat, s=40, color="#D9822B", marker="^", label="validation-only cities")
    for _, r in nc.iterrows():
        ax.annotate(r.city, (r.lon, r.lat), fontsize=7, color="#D9822B")
    ax.legend(); ax.set_title("Unseen validation cities relative to training cities", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "13_unseen_cities.png"); plt.close(fig)
    print("written 12_drift_numeric.png, 13_unseen_cities.png")


if __name__ == "__main__":
    main()
