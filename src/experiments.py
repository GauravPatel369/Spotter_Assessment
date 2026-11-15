"""Backtest harness + experiment log.

Usage examples
  python src/experiments.py --models dummy_median dummy_mean linear ridge --targets raw log
  python src/experiments.py --models hgb_l2 hgb_l1 --targets raw log rpm log_rpm --train-filter all clean
  python src/experiments.py --models lgbm_l1 xgb_gpu cat_gpu --targets log_rpm

Every (model, target, train_filter, feature_set) combination is evaluated on the rolling-origin folds in
validation.py; one row per fold plus an aggregate row is appended to outputs/experiment_log.csv.
Predictions on the last fold (train Jan-Aug -> Sep-Oct) are saved to outputs/oof/<exp>.csv for error analysis.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeaturePipeline, LevelModel, clean, load_train
from validation import fold_indices, metrics, metrics_clean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OOF = OUT / "oof"
OOF.mkdir(parents=True, exist_ok=True)
import os
LOG = OUT / os.environ.get("EXP_LOG", "experiment_log.csv")
ES_MODE = os.environ.get("ES_MODE", "random")

FEATURE_SETS = {
    "full": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "no_lane": [c for c in NUMERIC_FEATURES if c not in ("lane_effect", "lane_count")] + CATEGORICAL_FEATURES,
    "no_time": [c for c in NUMERIC_FEATURES if c not in ("day_index",)] + CATEGORICAL_FEATURES,
    "no_time_no_dom": [c for c in NUMERIC_FEATURES if c not in ("day_index", "dom")] + CATEGORICAL_FEATURES,
    "no_cities": NUMERIC_FEATURES + ["equipment"],
    "core": ["distance", "log_distance", "abs_weight", "market_index", "dow", "is_holiday", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "haversine", "circuity", "bearing_sin", "bearing_cos", "lane_effect", "lane_count", "equipment", "pickup", "delivery"],
    "minimal": ["distance", "log_distance", "abs_weight", "market_index", "dow", "equipment"],
    # for use with an explicit LevelModel: market level and calendar drift are handled by the offset, not the tree
    "level": [c for c in NUMERIC_FEATURES if c not in ("market_index", "day_index", "dom")] + CATEGORICAL_FEATURES,
    "level_no_lane": [c for c in NUMERIC_FEATURES if c not in ("market_index", "day_index", "dom", "lane_effect", "lane_count")] + CATEGORICAL_FEATURES,
    "level_no_cities": [c for c in NUMERIC_FEATURES if c not in ("market_index", "day_index", "dom")] + ["equipment"],
    # lean: drops features with ~zero permutation importance out of time (bearing, circuity, calendar flags, missing flags)
    "lean": ["distance", "log_distance", "haversine", "abs_weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "lane_effect", "lane_count", "dow", "equipment", "pickup", "delivery"],
}


# ----------------------------------------------------------------------------- target transforms
def parse_level(level: str) -> LevelModel:
    """'fe3' -> last-3-month level; 'fe3d5' -> same plus 0.5 x in-window drift per month ahead."""
    k, damp = level[2:], 0.0
    if "d" in k:
        k, d = k.split("d"); damp = int(d) / 10
    return LevelModel(k_months=int(k), drift_damping=damp)


class Target:
    """Maps posted_rate <-> model space. 'rpm' variants divide by distance so the model learns $/mile."""

    def __init__(self, kind: str):
        self.kind = kind

    def forward(self, y: pd.Series, X: pd.DataFrame) -> np.ndarray:
        y = np.asarray(y, float)
        d = np.asarray(X["distance"], float)
        if self.kind == "raw":
            return y
        if self.kind == "log":
            return np.log(y)
        if self.kind == "rpm":
            return y / d
        return np.log(y / d)

    def inverse(self, z: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        d = np.asarray(X["distance"], float)
        if self.kind == "raw":
            return z
        if self.kind == "log":
            return np.exp(z)
        if self.kind == "rpm":
            return z * d
        return np.exp(z) * d


# ----------------------------------------------------------------------------- models
def _linear_preprocessor(features):
    num = [c for c in features if c in NUMERIC_FEATURES]
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def make_model(name: str, features: list[str], params: dict | None = None):
    p = params or {}
    if name == "dummy_median":
        return DummyRegressor(strategy="median"), {}
    if name == "dummy_mean":
        return DummyRegressor(strategy="mean"), {}
    if name == "linear":
        return Pipeline([("prep", _linear_preprocessor(features)), ("m", LinearRegression())]), {}
    if name == "ridge":
        return Pipeline([("prep", _linear_preprocessor(features)), ("m", Ridge(alpha=p.get("alpha", 1.0)))]), {}
    if name.startswith("hgb"):
        loss = {"hgb_l2": "squared_error", "hgb_l1": "absolute_error", "hgb_huber": "squared_error"}[name]
        kw = dict(loss=loss, learning_rate=p.get("learning_rate", 0.05), max_iter=p.get("max_iter", 600), max_leaf_nodes=p.get("max_leaf_nodes", 31),
                  min_samples_leaf=p.get("min_samples_leaf", 40), l2_regularization=p.get("l2", 0.0), categorical_features="from_dtype",
                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=p.get("seed", 0))
        return HistGradientBoostingRegressor(**kw), {}
    if name.startswith("lgbm"):
        import lightgbm as lgb
        obj = {"lgbm_l2": "regression", "lgbm_l1": "regression_l1", "lgbm_huber": "huber"}[name]
        kw = dict(objective=obj, learning_rate=p.get("learning_rate", 0.03), n_estimators=p.get("n_estimators", 3000), num_leaves=p.get("num_leaves", 31),
                  min_child_samples=p.get("min_child_samples", 40), subsample=p.get("subsample", 0.8), subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.8),
                  reg_lambda=p.get("reg_lambda", 1.0), random_state=p.get("seed", 0), verbose=-1, n_jobs=8)
        return lgb.LGBMRegressor(**kw), {"early_stopping": "lgbm"}
    if name.startswith("xgb"):
        import xgboost as xgb
        obj = {"xgb_gpu": "reg:squarederror", "xgb_gpu_l1": "reg:absoluteerror", "xgb_gpu_huber": "reg:pseudohubererror"}[name]
        kw = dict(objective=obj, learning_rate=p.get("learning_rate", 0.03), n_estimators=p.get("n_estimators", 3000), max_depth=p.get("max_depth", 6),
                  min_child_weight=p.get("min_child_weight", 20), subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.8),
                  reg_lambda=p.get("reg_lambda", 1.0), tree_method="hist", device="cuda", enable_categorical=True, max_cat_to_onehot=1,
                  early_stopping_rounds=100, random_state=p.get("seed", 0))
        return xgb.XGBRegressor(**kw), {"early_stopping": "xgb"}
    if name.startswith("cat"):
        from catboost import CatBoostRegressor
        loss = {"cat_gpu": "RMSE", "cat_gpu_mae": "MAE", "cat_gpu_huber": "Huber:delta=0.1"}[name]
        kw = dict(loss_function=loss, learning_rate=p.get("learning_rate", 0.05), iterations=p.get("iterations", 4000), depth=p.get("depth", 6),
                  l2_leaf_reg=p.get("l2_leaf_reg", 3.0), task_type="GPU", devices="0", random_seed=p.get("seed", 0), verbose=0,
                  od_type="Iter", od_wait=150, cat_features=[c for c in features if c in CATEGORICAL_FEATURES])
        return CatBoostRegressor(**kw), {"early_stopping": "cat"}
    raise ValueError(name)


def fit_predict(name, model, info, Xtr, ztr, Xva, features):
    """Fit with early stopping on a chronological tail of the training window (never on the validation fold)."""
    Xtr_, Xva_ = Xtr[features], Xva[features]
    es = info.get("early_stopping")
    if es is None:
        model.fit(Xtr_, ztr)
        return model.predict(Xva_), getattr(model, "n_iter_", None)
    # early-stopping set drawn from INSIDE the training window (never the validation fold).
    # ES_MODE=random (default): random 10% holdout -> measures fit quality on the training distribution.
    # ES_MODE=chrono: last 10% chronologically -> unstable, because the tail sits at a different price level.
    n = len(Xtr_)
    if ES_MODE == "chrono":
        cut = int(n * 0.9)
        ia, ib = np.arange(cut), np.arange(cut, n)
    else:
        rng = np.random.default_rng(12345)
        perm = rng.permutation(n)
        ib, ia = np.sort(perm[: n // 10]), np.sort(perm[n // 10:])
    Xa, za, Xb, zb = Xtr_.iloc[ia], ztr[ia], Xtr_.iloc[ib], ztr[ib]
    if es == "lgbm":
        import lightgbm as lgb
        model.fit(Xa, za, eval_set=[(Xb, zb)], callbacks=[lgb.early_stopping(100, verbose=False)])
        best = model.best_iteration_
    elif es == "xgb":
        model.fit(Xa, za, eval_set=[(Xb, zb)], verbose=False)
        best = model.best_iteration
    elif es == "cat":
        Xa, Xb, Xva_ = (_cat_frame(d) for d in (Xa, Xb, Xva_))
        model.fit(Xa, za, eval_set=(Xb, zb), use_best_model=True)
        best = model.get_best_iteration()
    return model.predict(Xva_), best


def _cat_frame(X):
    X = X.copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype(str).where(X[c].notna(), "UNKNOWN")
    return X


# ----------------------------------------------------------------------------- runner
def run_experiment(tr, y, model_name, target_kind, train_filter, feature_set, params=None, tag="", save_oof=True, level="none"):
    features = FEATURE_SETS[feature_set]
    target = Target(target_kind)
    exp_id = f"{model_name}|{target_kind}|{train_filter}|{feature_set}" + (f"|lvl{level}" if level != "none" else "") + (f"|{tag}" if tag else "")
    if level != "none":
        assert target_kind in ("log", "log_rpm"), "level offsets are additive in log space"
    rows, last_pred = [], None
    for fold_name, itr, iva in fold_indices(tr["date"]):
        t0 = time.time()
        dtr, dva = tr.iloc[itr], tr.iloc[iva]
        ytr, yva = y.iloc[itr], y.iloc[iva]
        if train_filter == "clean":
            rpm = ytr / dtr["distance"]
            keep = ((rpm > 1.2) & (rpm < 5)).values
            dtr, ytr = dtr[keep], ytr[keep]
        pipe = FeaturePipeline()
        Xtr, Xva = pipe.fit_transform(dtr, ytr), pipe.transform(dva)
        ztr = target.forward(ytr, Xtr)
        off_tr = off_va = 0.0
        if level != "none":
            lm = parse_level(level).fit(dtr, ytr)
            off_tr, off_va = lm.offset(dtr), lm.offset(dva)
            ztr = ztr - off_tr
        # a model name like "xgb_gpu+lgbm_l2+ridge" fits each member and averages their predictions in model space
        zhats, iters = [], []
        for member in model_name.split("+"):
            model, info = make_model(member, features, params)
            zh, bi = fit_predict(member, model, info, Xtr, ztr, Xva, features)
            zhats.append(np.asarray(zh, float)); iters.append(bi)
        zhat = np.mean(zhats, axis=0)
        best_iter = iters[0] if len(iters) == 1 else None
        pred = target.inverse(zhat + off_va, Xva)
        pred = np.clip(pred, 1.0, None)  # the scorer rejects non-positive predictions; never triggers in practice
        m = metrics(yva, pred)
        m.update(metrics_clean(yva, pred, dva["distance"]))
        m.update(exp=exp_id, model=model_name, target=target_kind, train_filter=train_filter, feature_set=feature_set, fold=fold_name,
                 n_train=len(dtr), n_valid=len(dva), best_iter=best_iter, fit_seconds=round(time.time() - t0, 1), params=json.dumps(params or {}))
        rows.append(m)
        last_pred = pd.DataFrame({"load_id": dva["load_id"].values, "date": dva["date"].values, "y": yva.values, "pred": pred})
    df = pd.DataFrame(rows)
    agg = df[["MAE", "RMSE", "R2", "MAPE", "MedAE", "bias", "MAE_clean", "RMSE_clean", "MAPE_clean", "bias_clean"]].mean().to_dict()
    agg.update(exp=exp_id, model=model_name, target=target_kind, train_filter=train_filter, feature_set=feature_set, fold="MEAN", n_train=0, n_valid=0,
               best_iter=df["best_iter"].mean() if df["best_iter"].notna().any() else None, fit_seconds=df["fit_seconds"].sum(), params=json.dumps(params or {}))
    df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    df["logged_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    if save_oof and last_pred is not None:
        last_pred.to_csv(OOF / (exp_id.replace("|", "__").replace(" ", "") + ".csv"), index=False)
    last = df[df.fold != "MEAN"].iloc[-1]
    print(f"{exp_id:55s} MEAN  MAE={agg['MAE']:7.1f} RMSE={agg['RMSE']:7.1f} MAPE={agg['MAPE']:5.2f}% R2={agg['R2']:.4f} MAE_clean={agg['MAE_clean']:6.1f} | "
          f"LAST FOLD MAE={last['MAE']:7.1f} MAPE={last['MAPE']:5.2f}% bias={last['bias']:+6.1f} | {agg['fit_seconds']:.0f}s")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", default=["log_rpm"])
    ap.add_argument("--train-filter", nargs="+", default=["all"])
    ap.add_argument("--feature-sets", nargs="+", default=["full"])
    ap.add_argument("--params", default="{}", help="JSON dict of model params")
    ap.add_argument("--tag", default="")
    ap.add_argument("--level", nargs="+", default=["none"], help="none | fe1 | fe2 | fe3 ... (LevelModel with k last months)")
    args = ap.parse_args()
    params = json.loads(args.params)
    tr = clean(load_train())
    y = tr["posted_rate"]
    for m in args.models:
        for t in args.targets:
            for f in args.train_filter:
                for fs in args.feature_sets:
                    for lv in args.level:
                        run_experiment(tr, y, m, t, f, fs, params=params, tag=args.tag, level=lv)


if __name__ == "__main__":
    main()
