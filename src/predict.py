"""Phase 11 - Final training on all labelled data and prediction of validation.csv + the fixed December lane.

Usage:
  python src/predict.py --model lgbm_l1 --target log_rpm --params '{"n_estimators": 1200}' --seeds 0 1 2

Outputs:
  outputs/validation_predictions.csv   (load_id,predicted_rate - template order preserved)
  outputs/december_chart_inputs.csv    (the 7 original columns with predicted_rate filled)
  outputs/final_model_summary.json     (config, iterations, prediction statistics, feature importances)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import FEATURE_SETS, Target, _cat_frame, make_model, parse_level
from features import (CATEGORICAL_FEATURES, FeaturePipeline, LevelModel, clean, daily_market_index, load_december_inputs, load_train,
                      load_validation)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def city_coordinates(tr: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        tr[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
        tr[["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}),
    ]).drop_duplicates().set_index("city")


def december_frame(tr: pd.DataFrame, va: pd.DataFrame) -> pd.DataFrame:
    """The 31 chart rows, with coordinates from the city table and market_index from validation.csv's daily means."""
    dec = load_december_inputs().drop(columns=["predicted_rate"])
    coords = city_coordinates(tr)
    for side in ["pickup", "delivery"]:
        dec[f"{side}_lat"] = dec[side].map(coords["lat"])
        dec[f"{side}_lon"] = dec[side].map(coords["lon"])
    assert dec[["pickup_lat", "delivery_lat"]].notna().all().all(), "chart cities missing from city table"
    return clean(dec, daily_mi=daily_market_index(va))


def fit_full(model_name: str, target_kind: str, params: dict, seeds: list[int], feature_set: str, tr: pd.DataFrame, y: pd.Series,
             train_filter: str, level: str):
    """Fit the LevelModel (if any) and the member model(s) x seeds on all labelled data. Returns everything needed to predict."""
    features = FEATURE_SETS[feature_set]
    if train_filter == "clean":
        rpm = y / tr["distance"]
        keep = ((rpm > 1.2) & (rpm < 5)).values
        tr, y = tr[keep], y[keep]
    pipe = FeaturePipeline()
    X = pipe.fit_transform(tr, y)[features]
    target = Target(target_kind)
    z = target.forward(y, X)
    lm = None
    if level != "none":
        assert target_kind in ("log", "log_rpm")
        lm = parse_level(level).fit(tr, y)
        z = z - lm.offset(tr)
    models = []  # list of (member_name, fitted model)
    for member in model_name.split("+"):
        for s in seeds:
            p = dict(params, seed=s)
            model, info = make_model(member, features, p)
            # final fit: fixed iteration budget (from the backtest), no early stopping, all data used
            if info.get("early_stopping") == "xgb":
                model.set_params(early_stopping_rounds=None)
                model.fit(X, z, verbose=False)
            elif info.get("early_stopping") == "cat":
                model.set_params(od_type=None, od_wait=None)
                model.fit(_cat_frame(X), z)
            else:
                model.fit(X, z)
            models.append((member, model))
    return pipe, models, target, features, lm


def predict(models, pipe, target, features, lm, df):
    X = pipe.transform(df)[features]
    zs = []
    for member, m in models:
        Xm = _cat_frame(X) if member.startswith("cat") else X
        zs.append(np.asarray(m.predict(Xm), float))
    z = np.mean(zs, axis=0)
    if lm is not None:
        z = z + lm.offset(df)
    return target.inverse(z, X)


def importances(models, features):
    try:
        imp = np.mean([getattr(m, "feature_importances_") for _, m in models if hasattr(m, "feature_importances_")], axis=0)
        return dict(sorted(zip(features, map(float, imp)), key=lambda kv: -kv[1]))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb_gpu+ridge")
    ap.add_argument("--target", default="log_rpm")
    ap.add_argument("--params", default='{"max_depth": 4, "min_child_weight": 50, "n_estimators": 2500}')
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--feature-set", default="level")
    ap.add_argument("--train-filter", default="clean")
    ap.add_argument("--level", default="fe3d5", help="none | fe<k> | fe<k>d<x>: LevelModel using the last k month effects, plus x/10 of the recent drift")
    args = ap.parse_args()
    params = json.loads(args.params)

    tr, va = clean(load_train()), clean(load_validation())
    y = tr["posted_rate"]
    pipe, models, target, features, lm = fit_full(args.model, args.target, params, args.seeds, args.feature_set, tr, y, args.train_filter, args.level)

    # ---- validation.csv -> template order
    pred_va = predict(models, pipe, target, features, lm, va)
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    assert list(template["load_id"]) == list(va["load_id"]), "template order differs from validation.csv"
    sub = pd.DataFrame({"load_id": template["load_id"], "predicted_rate": np.round(pred_va, 2)})

    # ---- December fixed lane
    dec = december_frame(tr, va)
    pred_dec = predict(models, pipe, target, features, lm, dec)
    dec_out = load_december_inputs()
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
    dec_out["predicted_rate"] = np.round(pred_dec, 2)

    # ---- checks that mirror score.py; nothing is written unless all pass
    assert len(sub) == 12_000 and list(sub.columns) == ["load_id", "predicted_rate"]
    assert sub["load_id"].is_unique and sub["predicted_rate"].notna().all() and np.isfinite(sub["predicted_rate"]).all()
    assert (sub["predicted_rate"] > 0).all() and (dec_out["predicted_rate"] > 0).all()
    assert len(dec_out) == 31 and dec_out["date"].is_unique and np.isfinite(dec_out["predicted_rate"]).all()

    OUT.mkdir(exist_ok=True)
    sub.to_csv(OUT / "validation_predictions.csv", index=False)
    dec_out.to_csv(OUT / "december_chart_inputs.csv", index=False)
    # also write both files where the README's score.py command expects them
    sub.to_csv(ROOT / "validation_predictions.csv", index=False)
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)
    rpm_pred = sub["predicted_rate"] / va["distance"]
    train_rpm = (y / tr["distance"])
    summary = {
        "config": vars(args),
        "n_models": len(models),
        "level_model": lm.describe() if lm is not None else None,
        "december_market_index": {str(k.date()): round(float(v), 4) for k, v in daily_market_index(va).items() if k.month == 12},
        "iterations": [int(getattr(m, "best_iteration_", 0) or getattr(m, "n_estimators", 0) or getattr(m, "tree_count_", 0) or 0) for _, m in models],
        "validation_pred": {"min": float(sub.predicted_rate.min()), "p01": float(sub.predicted_rate.quantile(0.01)), "median": float(sub.predicted_rate.median()),
                            "mean": float(sub.predicted_rate.mean()), "p99": float(sub.predicted_rate.quantile(0.99)), "max": float(sub.predicted_rate.max())},
        "validation_pred_rpm": {"p01": float(rpm_pred.quantile(0.01)), "median": float(rpm_pred.median()), "p99": float(rpm_pred.quantile(0.99))},
        "train_label_rpm": {"p01": float(train_rpm.quantile(0.01)), "median": float(train_rpm.median()), "p99": float(train_rpm.quantile(0.99))},
        "train_label": {"median": float(y.median()), "mean": float(y.mean())},
        "december": {"min": float(dec_out.predicted_rate.min()), "max": float(dec_out.predicted_rate.max()), "mean": float(dec_out.predicted_rate.mean()),
                     "by_date": dict(zip(dec_out["date"], dec_out["predicted_rate"].astype(float)))},
        "feature_importances": importances(models, features),
    }
    (OUT / "final_model_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "december"}, indent=2))
    print("December predictions:", dec_out[["date", "predicted_rate"]].to_string(index=False))
    print("\nwrote outputs/validation_predictions.csv, outputs/december_chart_inputs.csv, outputs/final_model_summary.json")
    print("also wrote validation_predictions.csv and data/december_chart_inputs.csv where score.py expects them")


if __name__ == "__main__":
    main()
