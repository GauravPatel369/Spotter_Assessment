"""Phase 3 - Leakage audit, with a deep dive on quote_signal and on the extreme-rate rows.

Run:  python src/leakage_audit.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
pd.set_option("display.width", 200)


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def main():
    tr = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    va = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    tr["rpm"] = tr.posted_rate / tr.distance
    tr["lane"] = tr.pickup + " -> " + tr.delivery
    tr["month"] = tr.date.dt.to_period("M")
    tr["ratio"] = tr.rpm / tr.quote_signal
    tr["dev"] = (tr.ratio - 1).abs()

    section("1. quote_signal: how often does it equal the realised rate per mile?")
    for thr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        print(f"  |rpm/quote - 1| < {thr:<5}: {100 * (tr.dev < thr).mean():5.1f}% of rows")
    tr["match"] = tr.dev < 0.01
    print("\nmatch rate by month:")
    print(tr.groupby("month")["match"].mean().round(3).to_string())
    print("\nmatch rate by equipment:", tr.groupby("equipment")["match"].mean().round(3).to_dict())
    print("match rate by day of week:", tr.groupby(tr.date.dt.dayofweek)["match"].mean().round(3).to_dict())
    tr["dist_band"] = pd.cut(tr.distance, [0, 250, 500, 1000, 1500, 2500, 4000])
    print("match rate by distance band:", tr.groupby("dist_band", observed=True)["match"].mean().round(3).to_dict())
    print("match rate when weight negative / missing / market_index missing:",
          round(tr.loc[tr.weight < 0, "match"].mean(), 3), round(tr.loc[tr.weight.isna(), "match"].mean(), 3),
          round(tr.loc[tr.market_index.isna(), "match"].mean(), 3))

    # daily match fraction: is the regime a calendar block or random per load?
    daily = tr.groupby("date").agg(match=("match", "mean"), n=("match", "size"), qs=("quote_signal", "mean"), rpm=("rpm", "median"), mi=("market_index", "mean"))
    print("\ndaily match fraction quantiles:", daily["match"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).round(3).to_dict())
    print("days with match fraction > 0.8:", (daily["match"] > 0.8).sum(), "| < 0.2:", (daily["match"] < 0.2).sum(), "| in between:", ((daily["match"] >= 0.2) & (daily["match"] <= 0.8)).sum())
    # contiguous regimes
    hi = (daily["match"] > 0.5).astype(int)
    runs = (hi != hi.shift()).cumsum()
    seg = daily.groupby(runs).agg(start=("match", lambda s: s.index.min().date()), end=("match", lambda s: s.index.max().date()), days=("match", "size"), match=("match", "mean"))
    print("\ncontiguous regimes (match>0.5 vs not):")
    print(seg.round(3).to_string(index=False))

    section("2. In the NON-matching regime, what is quote_signal?")
    nm = tr[~tr.match].copy()
    m = tr[tr.match]
    print(f"non-match rows: {len(nm):,} | match rows: {len(m):,}")
    print("non-match ratio rpm/quote quantiles:", nm.ratio.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3).to_dict())
    print("non-match: Pearson rpm~quote", round(nm.rpm.corr(nm.quote_signal), 4), "| Spearman", round(nm.rpm.corr(nm.quote_signal, method="spearman"), 4))
    print("match:     Pearson rpm~quote", round(m.rpm.corr(m.quote_signal), 4))
    # candidate explanations for non-match quote: lane median rpm? previous-day rate? scaled rpm?
    lane_med = tr.groupby("lane")["rpm"].median()
    nm["lane_med"] = nm.lane.map(lane_med)
    print("non-match: corr(quote, lane median rpm):", round(nm.quote_signal.corr(nm.lane_med), 4))
    print("non-match: corr(quote, distance):", round(nm.quote_signal.corr(nm.distance), 4), "| corr(rpm, distance):", round(nm.rpm.corr(nm.distance), 4))
    print("non-match: corr(quote, market_index):", round(nm.quote_signal.corr(nm.market_index), 4))
    print("non-match: quote_signal quantiles:", nm.quote_signal.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(3).to_dict())
    print("match:     quote_signal quantiles:", m.quote_signal.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(3).to_dict())
    print("non-match: rpm quantiles:", nm.rpm.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(3).to_dict())
    print("match:     rpm quantiles:", m.rpm.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(3).to_dict())
    # does quote in non-match periods equal rpm of the SAME lane at an earlier time (stale quote)?
    for lag in [0, 7, 14, 30, 60, 90]:
        shifted = tr[["date", "lane", "equipment", "rpm"]].copy()
        shifted["date"] = shifted["date"] + pd.Timedelta(days=lag)
        agg = shifted.groupby(["date", "lane"])["rpm"].median().rename(f"lag{lag}")
        j = nm.merge(agg, left_on=["date", "lane"], right_index=True, how="left")
        ok = j[f"lag{lag}"].notna()
        if ok.sum() > 50:
            print(f"  stale-quote test lag={lag:3d}d: rows={ok.sum():5d} corr(quote, same-lane rpm {lag}d earlier)={j.loc[ok, 'quote_signal'].corr(j.loc[ok, f'lag{lag}']):.4f}")
    # is non-match quote = rpm / (1 + something) with something tied to market_index?
    print("non-match: corr(ratio, market_index):", round(nm.ratio.corr(nm.market_index), 4), "| corr(ratio, distance):", round(nm.ratio.corr(nm.distance), 4))
    # quote vs rpm scatter in each regime
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, d, t in [(ax[0], m.sample(min(5000, len(m)), random_state=0), "matching regime"), (ax[1], nm.sample(min(5000, len(nm)), random_state=0), "non-matching regime")]:
        a.scatter(d.quote_signal, d.rpm.clip(upper=4), s=4, alpha=0.3, color="#0B6E7A"); a.plot([1, 3.5], [1, 3.5], "--", color="#D9822B"); a.set_xlabel("quote_signal"); a.set_ylabel("rate per mile (clipped)"); a.set_title(t)
    fig.tight_layout(); fig.savefig(FIG / "09_quote_regimes.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(daily.index, daily["match"], color="#0B6E7A"); ax.set_ylabel("share of loads where quote = realised $/mile"); ax.set_title("Daily share of loads with quote_signal within 1% of realised rate per mile", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "10_quote_match_daily.png"); plt.close(fig)

    section("3. Does validation (Nov-Dec) look like the matching or non-matching regime?")
    vd = va.groupby("date").agg(qs=("quote_signal", "mean"), qs_sd=("quote_signal", "std"), mi=("market_index", "mean"), n=("load_id", "size"))
    td = tr.groupby("date").agg(qs=("quote_signal", "mean"), qs_sd=("quote_signal", "std"), match=("match", "mean"))
    print("train daily quote_signal mean/sd in matching days (match>0.5):", td[td.match > 0.5][["qs", "qs_sd"]].mean().round(3).to_dict())
    print("train daily quote_signal mean/sd in non-matching days:        ", td[td.match <= 0.5][["qs", "qs_sd"]].mean().round(3).to_dict())
    print("validation daily quote_signal mean/sd (Nov):", vd[vd.index.month == 11][["qs", "qs_sd"]].mean().round(3).to_dict())
    print("validation daily quote_signal mean/sd (Dec):", vd[vd.index.month == 12][["qs", "qs_sd"]].mean().round(3).to_dict())
    # within-regime signature: in matching regime quote correlates with distance (because rpm does); in non-matching it may not
    print("corr(quote, distance) | train match regime:", round(m.quote_signal.corr(m.distance), 4), "| train non-match:", round(nm.quote_signal.corr(nm.distance), 4), "| validation:", round(va.quote_signal.corr(va.distance), 4))
    print("corr(quote, equipment is Reefer) | train match:", round(m.quote_signal.corr((m.equipment == "Reefer").astype(float)), 4), "| non-match:", round(nm.quote_signal.corr((nm.equipment == "Reefer").astype(float)), 4), "| validation:", round(va.quote_signal.corr((va.equipment == "Reefer").astype(float)), 4))
    # per-month in validation
    for mo in [11, 12]:
        v = va[va.date.dt.month == mo]
        print(f"  validation month {mo}: corr(quote, distance)={v.quote_signal.corr(v.distance):.4f}, corr(quote, Reefer)={v.quote_signal.corr((v.equipment == 'Reefer').astype(float)):.4f}, corr(quote, |weight|)={v.quote_signal.corr(v.weight.abs()):.4f}")
    print("  train match regime corr(quote, |weight|):", round(m.quote_signal.corr(m.weight.abs()), 4), "| non-match:", round(nm.quote_signal.corr(nm.weight.abs()), 4))
    # weekly view for validation
    vw = va.groupby(va.date.dt.to_period("W")).apply(lambda g: pd.Series({"corr_dist": g.quote_signal.corr(g.distance), "n": len(g)}))
    print("validation weekly corr(quote, distance):")
    print(vw.round(3).T.to_string())
    tw = tr.groupby(tr.date.dt.to_period("W")).apply(lambda g: pd.Series({"corr_dist": g.quote_signal.corr(g.distance), "match": g.match.mean()}))
    print("train weekly corr(quote, distance) in weeks with match>0.8:", tw[tw.match > 0.8].corr_dist.describe()[["mean", "min", "max"]].round(3).to_dict())
    print("train weekly corr(quote, distance) in weeks with match<0.2:", tw[tw.match < 0.2].corr_dist.describe()[["mean", "min", "max"]].round(3).to_dict())

    section("4. Extreme-rate rows: are they predictable from any feature?")
    tr["extreme_hi"] = tr.rpm > 5
    tr["extreme_lo"] = tr.rpm < 1.2
    ext = tr[tr.extreme_hi | tr.extreme_lo]
    print(f"extreme rows: {len(ext):,} ({100 * len(ext) / len(tr):.2f}%) | hi={tr.extreme_hi.sum()} lo={tr.extreme_lo.sum()}")
    print("hi ratio rpm/quote quantiles:", tr.loc[tr.extreme_hi, "ratio"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).to_dict())
    print("lo ratio rpm/quote quantiles:", tr.loc[tr.extreme_lo, "ratio"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_dict())
    print("hi: implied multiplier vs lane-median rpm:", (tr.loc[tr.extreme_hi, "rpm"] / tr.loc[tr.extreme_hi, "lane"].map(lane_med)).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).to_dict())
    print("lo: implied multiplier vs lane-median rpm:", (tr.loc[tr.extreme_lo, "rpm"] / tr.loc[tr.extreme_lo, "lane"].map(lane_med)).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_dict())
    for col in ["equipment", "month"]:
        print(f"extreme share by {col}:", (ext.groupby(col).size() / tr.groupby(col).size()).round(4).to_dict())
    print("extreme share by distance band:", (ext.groupby("dist_band", observed=True).size() / tr.groupby("dist_band", observed=True).size()).round(4).to_dict())
    print("extreme share by match regime:", (ext.groupby("match").size() / tr.groupby("match").size()).round(4).to_dict())
    print("extreme share when weight negative:", round(tr.loc[tr.weight < 0, ["extreme_hi", "extreme_lo"]].any(axis=1).mean(), 4),
          "| weight missing:", round(tr.loc[tr.weight.isna(), ["extreme_hi", "extreme_lo"]].any(axis=1).mean(), 4),
          "| market_index missing:", round(tr.loc[tr.market_index.isna(), ["extreme_hi", "extreme_lo"]].any(axis=1).mean(), 4))
    # the middle band between 1.2 and 5: is it a clean distribution? gap analysis
    print("rpm counts in bands:", pd.cut(tr.rpm, [0, 0.5, 0.8, 1.0, 1.2, 1.5, 3.5, 4, 5, 6, 8, 20]).value_counts().sort_index().to_dict())
    # how extreme rows affect the mean: share of total variance
    y = tr.posted_rate
    clean = tr[~(tr.extreme_hi | tr.extreme_lo)].posted_rate
    print(f"posted_rate mean all={y.mean():.1f} clean={clean.mean():.1f} | std all={y.std():.1f} clean={clean.std():.1f}")

    section("5. Identifier, entity and temporal leakage checks")
    print("load_id encodes time order (corr of numeric id with date ordinal):", round(pd.Series(tr.load_id.str[3:].astype(int)).corr(pd.Series(tr.date.map(pd.Timestamp.toordinal))), 5), "-> must be excluded as a feature")
    print("validation load_ids also sequential by date:", (va.load_id.str[3:].astype(int).diff().dropna() == 1).all(), va.date.is_monotonic_increasing)
    print("no customer / carrier / shipper identifiers present -> no entity leakage beyond lanes")
    print("lanes in validation that also appear in train:", round(va.assign(lane=va.pickup + ' -> ' + va.delivery).lane.isin(set(tr.lane)).mean(), 4),
          "| validation lanes total:", va.assign(lane=va.pickup + ' -> ' + va.delivery).lane.nunique())
    print("distance within a lane is near-constant -> distance/lane carry no label information")
    print("market_index: feature available at prediction time for validation rows (present in validation.csv); NOT present for December chart rows")
    print("quote_signal: present in validation.csv; NOT present for December chart rows")
    # could aggregates over the full train leak into a chronological validation? note for pipeline
    print("-> all target-derived encodings (lane/city means) must be fitted inside each backtest fold on its training window only.")

    section("6. Validation quote_signal: any structure at all?")
    vq = va.groupby("date")["quote_signal"].agg(["mean", "std", "size"])
    expected_sd = (va.quote_signal.std() / np.sqrt(vq["size"])).mean()
    print(f"validation daily-mean quote sd across days = {vq['mean'].std():.4f} vs expected sd if iid = {expected_sd:.4f}")
    print(f"validation lag-1 autocorrelation of daily-mean quote = {vq['mean'].autocorr(1):.4f}")
    tq = tr.groupby("date")["quote_signal"].mean()
    print(f"train    daily-mean quote sd across days = {tq.std():.4f}, lag-1 autocorr = {tq.autocorr(1):.4f}")
    print("validation corr(quote, market_index):", round(va.quote_signal.corr(va.market_index), 4),
          "| corr(quote, pickup_lat):", round(va.quote_signal.corr(va.pickup_lat), 4), "| corr(quote, delivery_lon):", round(va.quote_signal.corr(va.delivery_lon), 4))
    vl = va.assign(lane=va.pickup + " -> " + va.delivery)
    lane_q = vl.groupby("lane")["quote_signal"].agg(["mean", "size"])
    lane_q = lane_q[lane_q["size"] >= 4]
    # split-half reliability of lane means: if quote carried lane information, halves would agree
    rng = np.random.default_rng(0)
    vl["half"] = rng.integers(0, 2, len(vl))
    h = vl.pivot_table(index="lane", columns="half", values="quote_signal", aggfunc="mean").dropna()
    print(f"validation split-half correlation of lane-mean quote (lanes with both halves, n={len(h)}): {h[0].corr(h[1]):.4f}")
    tl = tr[tr.match].assign(half=rng.integers(0, 2, tr.match.sum()))
    th = tl.pivot_table(index="lane", columns="half", values="quote_signal", aggfunc="mean").dropna()
    print(f"train matching-regime split-half correlation of lane-mean quote: {th[0].corr(th[1]):.4f}  (reference: a real price signal)")
    # non-matching regime hypothesis: quote ~ C - rpm ?
    nmc = nm[(nm.rpm > 1.2) & (nm.rpm < 5)]
    b = np.polyfit(nmc.rpm, nmc.quote_signal, 1)
    print(f"non-match regime (clean rows): quote ~ {b[1]:.3f} + {b[0]:.3f} * rpm ; corr = {nmc.rpm.corr(nmc.quote_signal):.4f}; sd(quote + rpm) = {(nmc.quote_signal + nmc.rpm).std():.4f} vs sd(quote) = {nmc.quote_signal.std():.4f}")

    section("7. market_index: is validation market_index the same kind of series as in training?")
    for name, d in [("train", tr), ("validation", va)]:
        g = d.groupby("date")["market_index"].agg(["mean", "std", "count"])
        print(f"{name:10s}: days={len(g)} within-day sd mean={g['std'].mean():.4f} max={g['std'].max():.4f} | daily-mean sd={g['mean'].std():.4f} "
              f"| day-to-day |diff| median={g['mean'].diff().abs().median():.4f} max={g['mean'].diff().abs().max():.4f} | lag-1 autocorr={g['mean'].autocorr(1):.4f} | missing rate={d.market_index.isna().mean():.4f}")
        print(f"{'':10s}  corr(market_index, distance)={d.market_index.corr(d.distance):.4f} corr(market_index, |weight|)={d.market_index.corr(d.weight.abs()):.4f} corr(market_index, Reefer)={d.market_index.corr((d.equipment == 'Reefer').astype(float)):.4f}")
    # lead/lag relation between daily market_index and daily realised rate in train
    dd = tr.groupby("date").agg(rpm=("rpm", "median"), mi=("market_index", "mean"))
    print("train cross-correlation corr(mi(t), rpm(t+k)) for k = -14..14:")
    print({k: round(dd["mi"].corr(dd["rpm"].shift(-k)), 3) for k in [-14, -7, -3, -1, 0, 1, 3, 7, 14]})
    # does the daily mean of market_index relate to the daily mean of the *clean* rpm more than to the noisy one?
    clean = tr[(tr.rpm > 1.2) & (tr.rpm < 5)].groupby("date")["rpm"].mean()
    print(f"train corr(daily mean market_index, daily MEAN clean rpm) = {dd['mi'].corr(clean):.4f}; with daily MEDIAN rpm = {dd['mi'].corr(dd['rpm']):.4f}")
    # is market_index missingness informative?
    from scipy import stats as st
    a, b_ = tr.loc[tr.market_index.isna(), "rpm"], tr.loc[tr.market_index.notna(), "rpm"]
    print(f"market_index missing vs present: median rpm {a.median():.3f} vs {b_.median():.3f}; Mann-Whitney p={st.mannwhitneyu(a, b_).pvalue:.3f}")
    a, b_ = tr.loc[tr.weight.isna(), "rpm"], tr.loc[tr.weight.notna(), "rpm"]
    print(f"weight missing vs present:       median rpm {a.median():.3f} vs {b_.median():.3f}; Mann-Whitney p={st.mannwhitneyu(a, b_).pvalue:.3f}")
    a, b_ = tr.loc[tr.weight < 0, "rpm"], tr.loc[tr.weight > 0, "rpm"]
    print(f"weight negative vs positive:     median rpm {a.median():.3f} vs {b_.median():.3f}; Mann-Whitney p={st.mannwhitneyu(a, b_).pvalue:.3f}")
    fig, ax = plt.subplots(figsize=(13, 3.8))
    g1 = tr.groupby("date")["market_index"].mean(); g2 = va.groupby("date")["market_index"].mean()
    ax.plot(g1.index, g1.values, color="#0B6E7A", label="train"); ax.plot(g2.index, g2.values, color="#D9822B", label="validation (features only)")
    ax.set_ylabel("daily mean market_index"); ax.legend(); ax.set_title("market_index: training period vs validation period", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "11_market_index_train_vs_val.png"); plt.close(fig)


if __name__ == "__main__":
    main()
