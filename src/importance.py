"""Feature importance for the selected configuration on the last backtest fold (train Jan-Aug -> Sep-Oct).

Three views:
  1. gain importance of the XGBoost model (what the trees split on),
  2. permutation importance on the Sep-Oct fold, measured in MAE dollars on clean rows (what actually matters out of time),
  3. SHAP summary (direction and magnitude per feature) on a 3,000-row sample.

Run:  python src/importance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import FEATURE_SETS, Target, fit_predict, make_model, parse_level
from features import FeaturePipeline, LevelModel, clean, load_train
from validation import fold_indices

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "outputs"


def main():
    cfg = json.loads((OUT / "final_config.json").read_text()) if (OUT / "final_config.json").exists() else {"model": "xgb_gpu", "params": {}, "feature_set": "level", "level": "fe3"}
    features = FEATURE_SETS[cfg["feature_set"]]
    tr = clean(load_train()); y = tr["posted_rate"]
    *_, (name, itr, iva) = fold_indices(tr["date"])
    dtr, dva, ytr, yva = tr.iloc[itr], tr.iloc[iva], y.iloc[itr], y.iloc[iva]
    rpm = ytr / dtr["distance"]; keep = ((rpm > 1.2) & (rpm < 5)).values
    dtr, ytr = dtr[keep], ytr[keep]
    pipe = FeaturePipeline(); Xtr, Xva = pipe.fit_transform(dtr, ytr), pipe.transform(dva)
    target = Target("log_rpm"); ztr = target.forward(ytr, Xtr)
    lm = parse_level(cfg["level"]).fit(dtr, ytr); ztr = ztr - lm.offset(dtr); off_va = lm.offset(dva)
    model, info = make_model(cfg["model"].split("+")[0], features, cfg["params"])
    zhat, best = fit_predict(cfg["model"], model, info, Xtr, ztr, Xva, features)
    base_pred = target.inverse(np.asarray(zhat) + off_va, Xva)
    clean_va = ((yva / dva["distance"]) > 1.2) & ((yva / dva["distance"]) < 5)
    base_mae = np.mean(np.abs(base_pred - yva)[clean_va])
    print(f"fold: {name} | best_iter={best} | clean-row MAE baseline = {base_mae:.2f}")

    # 1. gain importance
    gain = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nXGBoost gain importance (share):"); print((gain / gain.sum()).round(4).to_string())

    # 2. permutation importance in MAE dollars (clean rows), 3 repeats
    rng = np.random.default_rng(0); rows = []
    Xv = Xva[features].copy()
    for f in features:
        deltas = []
        for r in range(3):
            Xp = Xv.copy(); Xp[f] = Xp[f].values[rng.permutation(len(Xp))]
            zp = model.predict(Xp) if not cfg["model"].startswith("cat") else model.predict(Xp)
            pp = target.inverse(np.asarray(zp) + off_va, Xva)
            deltas.append(np.mean(np.abs(pp - yva)[clean_va]) - base_mae)
        rows.append({"feature": f, "MAE_increase_$": np.mean(deltas)})
    perm = pd.DataFrame(rows).sort_values("MAE_increase_$", ascending=False).set_index("feature")
    print("\npermutation importance (increase in clean-row MAE, $):"); print(perm.round(2).to_string())
    # level model contribution: replace offset by its training-window mean
    pp = target.inverse(np.asarray(zhat) + np.full(len(off_va), lm.offset(dtr).mean()), Xva)
    print(f"\nreplacing the LevelModel forecast by the training-window mean level: clean-row MAE {np.mean(np.abs(pp - yva)[clean_va]):.2f} (vs {base_mae:.2f})")

    # 3. SHAP
    try:
        import shap
        samp = Xv.sample(3000, random_state=0)
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(samp)
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False)
        print("\nSHAP mean |value| (log $/mile units):"); print(mean_abs.round(4).to_string())
        plt.figure(figsize=(9, 7))
        shap.summary_plot(sv, samp, show=False, max_display=18, plot_size=None)
        plt.title("SHAP summary - XGBoost on level-adjusted log($/mile), Sep-Oct fold", loc="left", fontweight="bold")
        plt.tight_layout(); plt.savefig(FIG / "15_shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()
        # dependence for distance and abs_weight
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for a, f in zip(ax, ["log_distance", "abs_weight", "lane_effect"]):
            j = features.index(f); a.scatter(samp[f], sv[:, j], s=5, alpha=0.4, color="#0B6E7A"); a.set_xlabel(f); a.set_ylabel("SHAP (log $/mile)"); a.set_title(f"effect of {f}")
        fig.tight_layout(); fig.savefig(FIG / "16_shap_dependence.png", dpi=130); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("SHAP skipped:", e)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    (gain / gain.sum()).sort_values().plot.barh(ax=ax[0], color="#0B6E7A"); ax[0].set_title("XGBoost gain importance (share)")
    perm["MAE_increase_$"].sort_values().plot.barh(ax=ax[1], color="#D9822B"); ax[1].set_title("Permutation importance: increase in clean-row MAE ($), Sep-Oct fold")
    fig.tight_layout(); fig.savefig(FIG / "17_feature_importance.png", dpi=130); plt.close(fig)
    perm.to_csv(OUT / "permutation_importance.csv")


if __name__ == "__main__":
    main()
