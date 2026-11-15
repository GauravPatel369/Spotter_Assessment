"""Phase 2 - Exploratory Data Analysis for the Spotter freight-rate assessment.

Reads data/train_test.csv (and validation.csv for schema comparison only),
prints a structured text summary and writes figures to reports/figures/.

Run:  python src/eda.py
"""
from __future__ import annotations

import sys
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
FIG.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

NUM_COLS = ["distance", "weight", "market_index", "quote_signal", "posted_rate"]
CAT_COLS = ["pickup", "delivery", "equipment"]


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def haversine(lat1, lon1, lat2, lon2):
    r = 3958.8  # miles
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main() -> None:
    tr = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    va = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    tr["rpm"] = tr["posted_rate"] / tr["distance"]
    tr["lane"] = tr["pickup"] + " -> " + tr["delivery"]
    tr["month"] = tr["date"].dt.to_period("M")
    tr["dow"] = tr["date"].dt.dayofweek

    # ------------------------------------------------------------------ A. overview
    section("A. DATASET OVERVIEW (train_test.csv)")
    print(f"shape: {tr.shape[0]:,} rows x {tr.shape[1] - 4} original columns")
    print(f"memory: {tr.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("\ndtypes / nunique / nulls / null%:")
    ov = pd.DataFrame(
        {
            "dtype": tr.dtypes.astype(str),
            "nunique": tr.nunique(),
            "nulls": tr.isna().sum(),
            "null_pct": (100 * tr.isna().mean()).round(2),
        }
    )
    print(ov.to_string())
    print("\nconstant / near-constant columns:",
          [c for c in tr.columns if tr[c].nunique() <= 1] or "none")

    # ------------------------------------------------------------------ B. data quality
    section("B. DATA QUALITY")
    print("duplicate load_id:", tr["load_id"].duplicated().sum())
    print("exact duplicate rows (excluding load_id):", tr.drop(columns=["load_id"]).duplicated().sum())
    feat_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
    print("duplicate feature rows (pickup,delivery,distance,equipment,weight,date):",
          tr.duplicated(subset=feat_cols).sum())
    print("load_id sequential & monotone with date:",
          (tr["load_id"].str[3:].astype(int).diff().dropna() == 1).all(), "/",
          tr["date"].is_monotonic_increasing)
    print("same pickup and delivery city:", (tr["pickup"] == tr["delivery"]).sum())
    print("distance <= 0:", (tr["distance"] <= 0).sum(), "| posted_rate <= 0:", (tr["posted_rate"] <= 0).sum())
    print("quote_signal <= 0:", (tr["quote_signal"] <= 0).sum(), "| market_index <= 0:", (tr["market_index"] <= 0).sum())

    print("\n-- weight anomalies --")
    w = tr["weight"]
    print("negative:", (w < 0).sum(), "| zero:", (w == 0).sum(), "| missing:", w.isna().sum())
    print("abs(weight)==47500:", (w.abs() == 47500).sum(), f"({100 * (w.abs() == 47500).mean():.2f}%)",
          "| weight==+47500:", (w == 47500).sum(), "| weight==-47500:", (w == -47500).sum())
    print("abs(weight) < 5000:", (w.abs() < 5000).sum(), "| min positive:", w[w > 0].min(), "| max:", w.max())
    neg = tr[w < 0]
    pos = tr[w > 0]
    print("abs(negative weight) quantiles:", neg["weight"].abs().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).tolist())
    print("positive weight quantiles:       ", pos["weight"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).tolist())
    ks = stats.ks_2samp(neg["weight"].abs(), pos["weight"])
    print(f"KS test abs(neg) vs pos weight: stat={ks.statistic:.4f} p={ks.pvalue:.4f}  (high p => sign flip of same distribution)")
    print("median rpm | weight negative / positive / missing:",
          round(neg["rpm"].median(), 3), round(pos["rpm"].median(), 3), round(tr.loc[w.isna(), "rpm"].median(), 3))
    print("negative weight share by equipment:", (neg["equipment"].value_counts(normalize=True).round(3)).to_dict())
    print("overall equipment share:            ", tr["equipment"].value_counts(normalize=True).round(3).to_dict())
    print("negative weight by month:", neg.groupby("month").size().to_dict())

    print("\n-- market_index missingness --")
    mi_na = tr["market_index"].isna()
    print("missing by month:", tr[mi_na].groupby("month").size().to_dict())
    print("days with ALL market_index missing:", (tr.groupby("date")["market_index"].count() == 0).sum(),
          "of", tr["date"].nunique())
    print("median rpm | market_index missing / present:",
          round(tr.loc[mi_na, "rpm"].median(), 3), round(tr.loc[~mi_na, "rpm"].median(), 3))
    daily = tr.groupby("date")["market_index"].agg(["mean", "std", "count"])
    print(f"market_index within-day std: mean={daily['std'].mean():.4f}, max={daily['std'].max():.4f}; "
          f"across-day std of daily mean={daily['mean'].std():.4f}")
    print("=> market_index is effectively a daily market-level series; missing values can be filled from the daily mean.")
    print("day-to-day abs change in daily mean market_index: median",
          round(daily["mean"].diff().abs().median(), 4), "max", round(daily["mean"].diff().abs().max(), 4))

    print("\n-- coordinates vs distance --")
    hv = haversine(tr["pickup_lat"], tr["pickup_lon"], tr["delivery_lat"], tr["delivery_lon"])
    ratio = tr["distance"] / hv
    print("distance / haversine ratio quantiles:", ratio.quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(3).to_dict())
    print("corr(distance, haversine):", round(np.corrcoef(tr["distance"], hv)[0, 1], 4))
    city_coords = pd.concat([
        tr[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
        tr[["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}),
    ]).drop_duplicates()
    print("city -> coordinate mapping is one-to-one:", city_coords["city"].is_unique, f"({len(city_coords)} cities)")
    dup_lon = city_coords[city_coords.duplicated("lon", keep=False)].sort_values("lon")
    print("cities sharing a longitude:\n", dup_lon.to_string(index=False) if len(dup_lon) else "none")
    print("sample city coordinates (sanity check vs real geography):")
    print(city_coords.set_index("city").loc[["Philadelphia", "Chicago" if "Chicago" in set(city_coords.city) else "Richmond", "Los Angeles", "Atlanta", "Lexington", "Fort Wayne"]].to_string())
    lane_dist = tr.groupby("lane")["distance"].agg(["min", "median", "max", "count"])
    lane_dist["spread_pct"] = 100 * (lane_dist["max"] - lane_dist["min"]) / lane_dist["median"]
    print("per-lane distance spread % (max-min)/median quantiles:",
          lane_dist["spread_pct"].quantile([0.5, 0.9, 0.99, 1.0]).round(1).to_dict())

    # ------------------------------------------------------------------ C. statistics
    section("C. NUMERICAL STATISTICS")
    desc = tr[NUM_COLS + ["rpm"]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["skew"] = tr[NUM_COLS + ["rpm"]].skew()
    desc["kurtosis"] = tr[NUM_COLS + ["rpm"]].kurtosis()
    print(desc.round(3).to_string())

    section("C2. CATEGORICAL STATISTICS")
    for c in CAT_COLS:
        vc = tr[c].value_counts()
        print(f"\n{c}: {vc.size} levels | top5 {vc.head(5).to_dict()} | bottom5 {vc.tail(5).to_dict()}")
    lanes = tr["lane"].value_counts()
    print(f"\nlanes: {lanes.size} distinct of {64 * 63} possible directed pairs; "
          f"median loads/lane={lanes.median():.0f}, min={lanes.min()}, max={lanes.max()}; lanes with <5 loads: {(lanes < 5).sum()}")
    print("share of loads in lanes with <10 training loads:", round((tr['lane'].map(lanes) < 10).mean(), 4))
    lane_eq = tr.groupby(["lane", "equipment"]).size()
    print("lane x equipment combos:", lane_eq.size, "| combos with <5 loads:", (lane_eq < 5).sum())

    # ------------------------------------------------------------------ D. target
    section("D. TARGET ANALYSIS: posted_rate")
    y = tr["posted_rate"]
    print(f"mean={y.mean():.1f} median={y.median():.1f} std={y.std():.1f} skew={y.skew():.3f} log-skew={np.log(y).skew():.3f}")
    print("top 15 posted_rate rows:")
    print(tr.nlargest(15, "posted_rate")[["load_id", "lane", "equipment", "distance", "weight", "date", "market_index", "quote_signal", "posted_rate", "rpm"]].to_string(index=False))
    print("\nrpm extremes: rpm>4:", (tr["rpm"] > 4).sum(), "| rpm>6:", (tr["rpm"] > 6).sum(), "| rpm<1:", (tr["rpm"] < 1).sum(), "| rpm<1.2:", (tr["rpm"] < 1.2).sum())
    print("bottom 10 rpm rows:")
    print(tr.nsmallest(10, "rpm")[["load_id", "lane", "equipment", "distance", "weight", "date", "quote_signal", "posted_rate", "rpm"]].to_string(index=False))
    # how do the extreme rows relate to quote_signal?
    tr["qs_ratio"] = tr["rpm"] / tr["quote_signal"]
    print("\nrpm / quote_signal ratio quantiles:", tr["qs_ratio"].quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]).round(3).to_dict())
    hi = tr[tr["rpm"] > 4]
    print("among rpm>4 rows, rpm/quote_signal ratio median:", round(hi["qs_ratio"].median(), 3),
          "| quote_signal median:", round(hi["quote_signal"].median(), 3), "(overall", round(tr["quote_signal"].median(), 3), ")")
    print("=> if ratio is ~2x while quote_signal is normal, extreme rates are NOT reflected in the quote and look like label noise / special loads.")
    print("\nmedian rpm by equipment:", tr.groupby("equipment")["rpm"].median().round(3).to_dict())
    print("median rpm by distance band:")
    tr["dist_band"] = pd.cut(tr["distance"], [0, 250, 500, 750, 1000, 1500, 2000, 2500, 4000])
    print(tr.groupby("dist_band", observed=True)["rpm"].agg(["median", "mean", "count"]).round(3).to_string())
    tr["wt_band"] = pd.cut(tr["weight"].abs(), [0, 15000, 25000, 35000, 45000, 50000])
    print("median rpm by |weight| band:")
    print(tr.groupby("wt_band", observed=True)["rpm"].agg(["median", "count"]).round(3).to_string())

    # ------------------------------------------------------------------ E. relationships
    section("E. FEATURE RELATIONSHIPS")
    num = tr[["distance", "weight", "market_index", "quote_signal", "posted_rate", "rpm"]].copy()
    num["weight"] = num["weight"].abs()
    print("Pearson:\n", num.corr().round(3).to_string())
    print("\nSpearman:\n", num.corr(method="spearman").round(3).to_string())
    trimmed = tr[(tr["rpm"] > 1.2) & (tr["rpm"] < 4)]
    print(f"\nPearson rpm~quote_signal on trimmed 1.2<rpm<4 ({len(trimmed):,} rows): {trimmed['rpm'].corr(trimmed['quote_signal']):.4f}")
    print(f"Pearson rpm~market_index on trimmed: {trimmed['rpm'].corr(trimmed['market_index']):.4f}")
    # variance explained by naive formula distance*quote_signal
    naive = tr["distance"] * tr["quote_signal"]
    resid = tr["posted_rate"] - naive
    print(f"\nnaive prediction distance*quote_signal: MAE={resid.abs().mean():.1f}, median AE={resid.abs().median():.1f}, "
          f"R2={1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum():.4f}, MAPE={(resid.abs() / y).mean() * 100:.2f}%")
    print("naive residual quantiles:", resid.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(1).to_dict())
    # does the residual still depend on anything?
    tr["naive_resid_pct"] = resid / naive
    print("residual% by equipment:", tr.groupby("equipment")["naive_resid_pct"].median().round(4).to_dict())
    print("residual% by month:", tr.groupby("month")["naive_resid_pct"].median().round(4).to_dict())
    print("residual% by dow:", tr.groupby("dow")["naive_resid_pct"].median().round(4).to_dict())
    print("corr(residual%, market_index):", round(tr["naive_resid_pct"].corr(tr["market_index"]), 4))
    print("corr(residual%, |weight|):", round(tr["naive_resid_pct"].corr(tr["weight"].abs()), 4))
    print("corr(residual%, distance):", round(tr["naive_resid_pct"].corr(tr["distance"]), 4))
    lane_res = tr.groupby("lane")["naive_resid_pct"].agg(["median", "count"])
    print("lane-level residual% median spread (std across lanes with >=20 loads):",
          round(lane_res[lane_res["count"] >= 20]["median"].std(), 4))

    # ------------------------------------------------------------------ F. temporal
    section("F. TEMPORAL PATTERNS")
    monthly = tr.groupby("month").agg(loads=("load_id", "size"), rpm_med=("rpm", "median"), rate_med=("posted_rate", "median"),
                                      mi=("market_index", "mean"), qs=("quote_signal", "mean"), dist_med=("distance", "median"))
    print(monthly.round(3).to_string())
    print("\nday-of-week (0=Mon): loads, median rpm")
    print(tr.groupby("dow").agg(loads=("load_id", "size"), rpm_med=("rpm", "median")).round(3).T.to_string())
    print("\nloads per day: mean", round(tr.groupby("date").size().mean(), 1), "min", tr.groupby("date").size().min(), "max", tr.groupby("date").size().max())
    # US holidays within range
    hol = {"2025-01-01": "New Year", "2025-01-20": "MLK", "2025-02-17": "Presidents", "2025-05-26": "Memorial",
           "2025-07-04": "July4", "2025-09-01": "Labor", "2025-10-13": "Columbus"}
    print("\nholiday days: loads / median rpm / daily-mean rpm vs +-7day window:")
    for d, n in hol.items():
        dd = pd.Timestamp(d)
        day = tr[tr["date"] == dd]
        win = tr[(tr["date"] >= dd - pd.Timedelta(days=7)) & (tr["date"] <= dd + pd.Timedelta(days=7)) & (tr["date"] != dd)]
        print(f"  {n:10s} {d}: loads={len(day):3d} rpm_med={day['rpm'].median():.3f} window_rpm_med={win['rpm'].median():.3f}")
    # rpm vs market_index at daily level
    d_agg = tr.groupby("date").agg(rpm=("rpm", "median"), mi=("market_index", "mean"), qs=("quote_signal", "mean"), n=("load_id", "size"))
    print("\ndaily-level corr: rpm~market_index", round(d_agg["rpm"].corr(d_agg["mi"]), 3),
          "| rpm~quote_signal", round(d_agg["rpm"].corr(d_agg["qs"]), 3), "| rpm~day index", round(d_agg["rpm"].corr(pd.Series(np.arange(len(d_agg)), index=d_agg.index)), 3))
    # linear trend fit on daily median rpm with market index
    X = np.column_stack([np.ones(len(d_agg)), np.arange(len(d_agg)), d_agg["mi"].values])
    beta, *_ = np.linalg.lstsq(X, d_agg["rpm"].values, rcond=None)
    print(f"daily rpm ~ a + b*day + c*market_index: b={beta[1] * 30:.4f} $/mile per 30 days, c={beta[2]:.4f} $/mile per unit market_index")

    # ------------------------------------------------------------------ G. figures
    section("G. FIGURES -> reports/figures/")
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    c1, c2, c3 = "#0B6E7A", "#D9822B", "#6B7280"

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(y, bins=100, color=c1); ax[0].set_title("posted_rate ($)"); ax[0].set_xlabel("$")
    ax[1].hist(np.log(y), bins=100, color=c1); ax[1].set_title("log(posted_rate)")
    ax[2].hist(tr["rpm"].clip(upper=5), bins=100, color=c1); ax[2].set_title("rate per mile (clipped at 5)"); ax[2].set_xlabel("$/mile")
    fig.suptitle("Target distribution", x=0.01, ha="left", fontweight="bold"); fig.tight_layout()
    fig.savefig(FIG / "01_target_distribution.png"); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].hist(tr["distance"], bins=80, color=c1); ax[0, 0].set_title("distance (miles)")
    ax[0, 1].hist(tr["weight"].dropna(), bins=80, color=c1); ax[0, 1].set_title("weight (lb) - note negatives and spike at ±47,500")
    ax[1, 0].hist(tr["market_index"].dropna(), bins=80, color=c1); ax[1, 0].set_title("market_index")
    ax[1, 1].hist(tr["quote_signal"], bins=80, color=c1); ax[1, 1].set_title("quote_signal")
    fig.suptitle("Numerical feature distributions", x=0.01, ha="left", fontweight="bold"); fig.tight_layout()
    fig.savefig(FIG / "02_numeric_distributions.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    s = tr.sample(8000, random_state=0)
    ax[0].scatter(s["distance"], s["posted_rate"], s=4, alpha=0.3, color=c1); ax[0].set_xlabel("distance"); ax[0].set_ylabel("posted_rate"); ax[0].set_title("posted_rate vs distance")
    ax[1].scatter(s["quote_signal"], s["rpm"].clip(upper=5), s=4, alpha=0.3, color=c1); ax[1].set_xlabel("quote_signal"); ax[1].set_ylabel("rate per mile (clipped)"); ax[1].set_title("rate per mile vs quote_signal")
    ax[1].plot([1, 3.5], [1, 3.5], color=c2, lw=1, ls="--", label="y = x"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "03_target_vs_distance_quote.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = num.corr(method="spearman")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right"); ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Spearman correlation", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "04_correlation_heatmap.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    eq_order = ["Dry Van", "Reefer", "Flatbed"]
    ax[0].boxplot([tr.loc[tr["equipment"] == e, "rpm"].clip(upper=4) for e in eq_order], tick_labels=eq_order, showfliers=False)
    ax[0].set_title("rate per mile by equipment"); ax[0].set_ylabel("$/mile")
    bands = tr.groupby("dist_band", observed=True)["rpm"]
    ax[1].boxplot([g.clip(upper=4) for _, g in bands], tick_labels=[str(b) for b in bands.groups], showfliers=False)
    ax[1].set_title("rate per mile by distance band"); ax[1].tick_params(axis="x", rotation=45)
    ax[2].boxplot([y, np.log(y)], tick_labels=["posted_rate", "log"], showfliers=True)
    ax[2].set_yscale("log"); ax[2].set_title("posted_rate outliers (log axis)")
    fig.tight_layout(); fig.savefig(FIG / "05_boxplots.png"); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(d_agg.index, d_agg["rpm"], color=c1, lw=1); ax[0].plot(d_agg.index, d_agg["rpm"].rolling(7, center=True).mean(), color=c2, lw=2, label="7-day mean"); ax[0].set_ylabel("daily median $/mile"); ax[0].legend()
    ax[1].plot(d_agg.index, d_agg["mi"], color=c1, lw=1.5); ax[1].set_ylabel("daily mean market_index")
    ax[2].plot(d_agg.index, d_agg["qs"], color=c1, lw=1); ax[2].set_ylabel("daily mean quote_signal")
    for d in hol:
        for a in ax:
            a.axvline(pd.Timestamp(d), color=c3, lw=0.8, ls=":")
    ax[0].set_title("Temporal patterns (dotted = US holidays)", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "06_temporal.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(range(7), tr.groupby("dow")["rpm"].median(), color=c1); ax[0].set_xticks(range(7)); ax[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); ax[0].set_ylim(2.0, 2.25); ax[0].set_title("median $/mile by day of week")
    top = tr["pickup"].value_counts().head(20)
    ax[1].barh(top.index[::-1], top.values[::-1], color=c1); ax[1].set_title("top 20 pickup cities (loads)")
    fig.tight_layout(); fig.savefig(FIG / "07_dow_and_cities.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(city_coords["lon"], city_coords["lat"], s=18, color=c1)
    for _, r in city_coords.iterrows():
        ax.annotate(r["city"], (r["lon"], r["lat"]), fontsize=6, alpha=0.8)
    ax.set_title("City coordinates as given (check against real geography)", loc="left", fontweight="bold"); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout(); fig.savefig(FIG / "08_city_map.png"); plt.close(fig)

    print("written:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
