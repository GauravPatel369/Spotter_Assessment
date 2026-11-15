"""Data loading, cleaning and feature engineering for the Spotter freight-rate model.

Design rules
------------
* Nothing in here looks at the label except `LaneEncoder`, which is fitted explicitly on a
  training window and applied to anything else, so it can never leak across a backtest fold.
* `quote_signal` is deliberately excluded (see reports: it is the label in half the training
  months, inverted in the other half, and i.i.d. noise in validation.csv).
* `load_id` is excluded (it is a time index).
* market_index is a daily market-level series; missing values are filled with the daily mean of the
  same file.  For the December chart rows (which carry no market_index) the caller supplies the
  daily means taken from validation.csv.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EQUIPMENT_LEVELS = ["Dry Van", "Reefer", "Flatbed"]
US_HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25",
])
NUMERIC_FEATURES = [
    "distance", "log_distance", "abs_weight", "weight_missing", "weight_capped",
    "market_index", "mi_missing", "dow", "dom", "is_weekend", "is_holiday", "day_index",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "haversine", "circuity", "bearing_sin", "bearing_cos",
    "lane_effect", "lane_count",
]
CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery"]
DAY0 = pd.Timestamp("2025-01-01")


# ----------------------------------------------------------------------------- loading / cleaning
def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(DATA / "validation.csv", parse_dates=["date"])


def load_december_inputs() -> pd.DataFrame:
    return pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])


def daily_market_index(df: pd.DataFrame) -> pd.Series:
    """Daily mean market_index of a frame (index = date). Used to fill gaps and to feed the chart rows."""
    return df.groupby("date")["market_index"].mean()


def clean(df: pd.DataFrame, daily_mi: pd.Series | None = None) -> pd.DataFrame:
    """Deterministic, label-free cleaning. Returns a copy with helper columns added.

    daily_mi: optional external daily market_index series (needed when the frame has no market_index
    of its own, e.g. the December chart rows). If None, the frame's own daily means are used.
    """
    out = df.copy()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["abs_weight"] = out["weight"].abs()  # negatives are sign flips (KS p=0.70 vs positives)
    out["weight_capped"] = (out["abs_weight"] == 47_500).astype(int)
    if "market_index" not in out:
        out["market_index"] = np.nan
    out["mi_missing"] = out["market_index"].isna().astype(int)
    dm = daily_market_index(out) if daily_mi is None else daily_mi
    fill = out["date"].map(dm)
    out["market_index"] = out["market_index"].fillna(fill)
    if out["market_index"].isna().any():  # a day with no observation at all: carry nearest day
        dm_full = dm.reindex(pd.date_range(dm.index.min(), dm.index.max())).interpolate().ffill().bfill()
        out["market_index"] = out["market_index"].fillna(out["date"].map(dm_full))
    return out


# ----------------------------------------------------------------------------- geo helpers
def haversine(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.arctan2(x, y)  # radians, 0 = north


# ----------------------------------------------------------------------------- lane encoder (label-aware)
@dataclass
class LaneEncoder:
    """Smoothed lane effect = shrunken mean of log(rpm) residual after an equipment baseline.

    Fitted on the training window ONLY.  Unseen lanes get 0 (the prior) and count 0.
    The residual is taken against equipment x distance-band medians so the lane effect captures
    origin/destination premium rather than restating distance or equipment.
    """
    smoothing: float = 20.0
    clip_rpm: tuple[float, float] = (1.2, 5.0)  # exclude the injected extreme labels from the encoding
    lane_stats_: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_: pd.Series = field(default_factory=pd.Series)

    @staticmethod
    def _key(df: pd.DataFrame) -> pd.Series:
        return df["pickup"] + "->" + df["delivery"]

    @staticmethod
    def _band(df: pd.DataFrame) -> pd.Series:
        return pd.cut(df["distance"], [0, 250, 500, 750, 1000, 1500, 2000, 2500, 5000], labels=False)

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "LaneEncoder":
        rpm = y / df["distance"]
        ok = (rpm > self.clip_rpm[0]) & (rpm < self.clip_rpm[1])
        d = pd.DataFrame({"lane": self._key(df), "eq": df["equipment"].astype(str), "band": self._band(df), "lr": np.log(rpm)})[ok]
        self.baseline_ = d.groupby(["eq", "band"])["lr"].median()
        d["res"] = d["lr"] - d.set_index(["eq", "band"]).index.map(self.baseline_)
        g = d.groupby("lane")["res"].agg(["sum", "count"])
        g["effect"] = g["sum"] / (g["count"] + self.smoothing)  # shrink toward 0
        self.lane_stats_ = g[["effect", "count"]]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        key = self._key(df)
        eff = key.map(self.lane_stats_["effect"]).fillna(0.0)
        cnt = key.map(self.lane_stats_["count"]).fillna(0).astype(float)
        return pd.DataFrame({"lane_effect": eff.values, "lane_count": cnt.values}, index=df.index)


# ----------------------------------------------------------------------------- feature matrix
def build_features(df: pd.DataFrame, lane_encoder: LaneEncoder | None = None, city_levels: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Label-free feature construction from a cleaned frame. Lane features come from a fitted encoder."""
    f = pd.DataFrame(index=df.index)
    f["distance"] = df["distance"]
    f["log_distance"] = np.log(df["distance"])
    f["abs_weight"] = df["abs_weight"].fillna(df["abs_weight"].median() if df["abs_weight"].notna().any() else 31_000.0)
    f["weight_missing"] = df["weight_missing"]
    f["weight_capped"] = df["weight_capped"]
    f["market_index"] = df["market_index"]
    f["mi_missing"] = df["mi_missing"]
    f["dow"] = df["date"].dt.dayofweek
    f["dom"] = df["date"].dt.day
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["is_holiday"] = df["date"].isin(US_HOLIDAYS_2025).astype(int)
    f["day_index"] = (df["date"] - DAY0).dt.days
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        f[c] = df[c]
    hv = haversine(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    f["haversine"] = hv
    f["circuity"] = df["distance"] / hv
    br = bearing(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    f["bearing_sin"], f["bearing_cos"] = np.sin(br), np.cos(br)
    if lane_encoder is not None:
        f[["lane_effect", "lane_count"]] = lane_encoder.transform(df)
    else:
        f["lane_effect"], f["lane_count"] = 0.0, 0.0
    levels = city_levels or {}
    f["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT_LEVELS)
    f["pickup"] = pd.Categorical(df["pickup"], categories=levels.get("pickup"))
    f["delivery"] = pd.Categorical(df["delivery"], categories=levels.get("delivery"))
    return f


def city_levels_from(df: pd.DataFrame) -> dict[str, list[str]]:
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    return {"pickup": cities, "delivery": cities}


@dataclass
class LevelModel:
    """Explicit model of the daily market/time level of log($/mile), fitted on the training window only.

    Step 1: mix-adjusted daily level  L_t = mean over the day's clean loads of
            log(rpm) - median_log_rpm(equipment, distance band) - lane_effect
    Step 2: L_t = a * market_index_t + month_effect(m(t)) + noise   (OLS)
            'a' is identified from WITHIN-month variation, so it is not confounded with the slow drift.
    Forecast for dates after the window: offset = a * market_index_t + mean(month_effect of last k months).
    In-sample offset (used to de-level the training target): a * market_index_t + month_effect(m(t)).

    Rationale (see reports/figures/14_temporal_component.png and outputs/temporal_analysis.txt): month
    effects step up in Q1 2025 and are flat afterwards; a linear trend over-shoots badly when extrapolated
    (+6% on one fold), while 'last-k-month level + market' has the lowest bias across all backtest folds.
    """
    k_months: int = 3
    drift_damping: float = 0.0   # 0 = pure local level; 0.5 = add half of the in-window monthly drift per month ahead
    clip_rpm: tuple[float, float] = (1.2, 5.0)
    a_: float = 0.0
    drift_: float = 0.0
    last_month_: pd.Period | None = None
    month_effects_: pd.Series = field(default_factory=pd.Series)
    future_level_: float = 0.0
    baseline_: pd.Series = field(default_factory=pd.Series)
    lane_encoder_: LaneEncoder | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "LevelModel":
        rpm = y / df["distance"]
        ok = ((rpm > self.clip_rpm[0]) & (rpm < self.clip_rpm[1])).values
        d, yy = df[ok], y[ok]
        # mix adjustment = equipment x distance-band medians only. Lane mix is stationary over time (PSI ~ 0),
        # and an in-sample lane encoding would leak each day's own level into the adjustment.
        lr = np.log(yy / d["distance"])
        eq, band = d["equipment"].astype(str), LaneEncoder._band(d)
        self.baseline_ = pd.DataFrame({"eq": eq, "band": band, "lr": lr}).groupby(["eq", "band"])["lr"].median()
        base = pd.MultiIndex.from_arrays([eq, band]).map(self.baseline_)
        res = lr.values - np.asarray(base, float)
        daily = pd.DataFrame({"date": d["date"].values, "res": res, "mi": d["market_index"].values}).groupby("date").agg(level=("res", "mean"), mi=("mi", "mean"))
        daily["m"] = daily.index.to_period("M").astype(str)
        dummies = pd.get_dummies(daily["m"]).astype(float)
        X = np.column_stack([daily["mi"].values, dummies.values])
        b = np.linalg.lstsq(X, daily["level"].values, rcond=None)[0]
        self.a_ = float(b[0])
        self.month_effects_ = pd.Series(b[1:], index=dummies.columns)
        self.future_level_ = float(self.month_effects_.iloc[-self.k_months:].mean())
        # in-window drift of the month effects (OLS slope per month); applied damped, see offset()
        me = self.month_effects_.values
        self.drift_ = float(np.polyfit(np.arange(len(me)), me, 1)[0]) if len(me) >= 3 else 0.0
        self.last_month_ = pd.Period(self.month_effects_.index[-1], "M")
        return self

    def offset(self, df: pd.DataFrame) -> np.ndarray:
        """Additive offset in log($/mile) units. Months inside the fitted window use their own effect; later months use the forecast level."""
        per = df["date"].dt.to_period("M")
        m = per.astype(str)
        eff = m.map(self.month_effects_)
        if self.drift_damping > 0:
            # months from the centre of the last-k window to the target month, e.g. k=3: Nov=+2, Dec=+3
            ahead = (per - self.last_month_).map(lambda x: x.n).astype(float) + (self.k_months - 1) / 2
            future = self.future_level_ + self.drift_damping * self.drift_ * ahead
        else:
            future = pd.Series(self.future_level_, index=df.index)
        eff = eff.fillna(future).values
        return self.a_ * df["market_index"].values + eff

    def describe(self) -> dict:
        return {"a_market": round(self.a_, 4), "month_effects_pct": {k: round(100 * v, 2) for k, v in self.month_effects_.items()}, "future_level_pct": round(100 * self.future_level_, 2),
                "k_months": self.k_months, "drift_pct_per_month": round(100 * self.drift_, 3), "drift_damping": self.drift_damping}


@dataclass
class FeaturePipeline:
    """Fit on a training window, transform anything. Holds everything that is learned from data."""
    smoothing: float = 20.0
    lane_encoder: LaneEncoder | None = None
    city_levels: dict[str, list[str]] | None = None
    weight_median_: float = 31_000.0

    def fit(self, df_clean: pd.DataFrame, y: pd.Series) -> "FeaturePipeline":
        self.lane_encoder = LaneEncoder(smoothing=self.smoothing).fit(df_clean, y)
        self.city_levels = city_levels_from(df_clean)
        self.weight_median_ = float(df_clean["abs_weight"].median())
        return self

    def transform(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        d = df_clean.copy()
        d["abs_weight"] = d["abs_weight"].fillna(self.weight_median_)
        return build_features(d, self.lane_encoder, self.city_levels)

    def fit_transform(self, df_clean: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
        """Fit on the window and return TRAINING features with an out-of-fold lane encoding.

        In-sample target encoding leaks each row's own label into its lane_effect (the daily mean of an
        in-sample lane_effect correlates 0.44 with market_index). Out-of-fold encoding gives training rows
        the same quality of lane information that unseen rows will have.
        """
        self.fit(df_clean, y)
        X = self.transform(df_clean)
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, n_splits, len(df_clean))
        eff = np.zeros(len(df_clean)); cnt = np.zeros(len(df_clean))
        for k in range(n_splits):
            enc = LaneEncoder(smoothing=self.smoothing).fit(df_clean[fold != k], y[fold != k])
            part = enc.transform(df_clean[fold == k])
            eff[fold == k], cnt[fold == k] = part["lane_effect"].values, part["lane_count"].values
        X["lane_effect"], X["lane_count"] = eff, cnt * n_splits / (n_splits - 1)  # rescale count to full-window scale
        return X
