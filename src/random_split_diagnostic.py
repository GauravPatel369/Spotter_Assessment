"""Diagnostic only: how optimistic would a random K-fold split be compared with the chronological backtest?

Runs the same feature pipeline + HistGradientBoosting (log $/mile target) under
  (a) random 5-fold CV over all 48,000 rows, and
  (b) the rolling-origin backtest from validation.py,
and prints both. Nothing here is used for model selection.

Run:  python src/random_split_diagnostic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from experiments import FEATURE_SETS, Target
from features import FeaturePipeline, clean, load_train
from validation import fold_indices, metrics


def fit_eval(tr, y, itr, iva):
    dtr, dva, ytr, yva = tr.iloc[itr], tr.iloc[iva], y.iloc[itr], y.iloc[iva]
    pipe = FeaturePipeline().fit(dtr, ytr)
    Xtr, Xva = pipe.fit_transform(dtr, ytr)[FEATURE_SETS["full"]], pipe.transform(dva)[FEATURE_SETS["full"]]
    t = Target("log_rpm")
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40,
                                      categorical_features="from_dtype", early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=0)
    m.fit(Xtr, t.forward(ytr, Xtr))
    return metrics(yva, t.inverse(m.predict(Xva), Xva))


def main():
    tr = clean(load_train()); y = tr["posted_rate"]
    rows = []
    for k, (itr, iva) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
        m = fit_eval(tr, y, itr, iva); m["scheme"] = "random 5-fold"; m["fold"] = k; rows.append(m)
    for name, itr, iva in fold_indices(tr["date"]):
        m = fit_eval(tr, y, itr, iva); m["scheme"] = "chronological backtest"; m["fold"] = name; rows.append(m)
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    print("\nmean by scheme:")
    print(df.groupby("scheme")[["MAE", "RMSE", "MAPE", "R2", "bias"]].mean().round(3).to_string())
    print("\nWith quote_signal INCLUDED (to show what a naive random-split model would report):")
    from features import NUMERIC_FEATURES
    tr2 = tr.copy()
    rows = []
    for k, (itr, iva) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr2)):
        dtr, dva, ytr, yva = tr2.iloc[itr], tr2.iloc[iva], y.iloc[itr], y.iloc[iva]
        pipe = FeaturePipeline().fit(dtr, ytr)
        Xtr, Xva = pipe.fit_transform(dtr, ytr)[FEATURE_SETS["full"]].copy(), pipe.transform(dva)[FEATURE_SETS["full"]].copy()
        Xtr["quote_signal"], Xva["quote_signal"] = dtr["quote_signal"].values, dva["quote_signal"].values
        Xtr["month"], Xva["month"] = dtr["date"].dt.month.values, dva["date"].dt.month.values
        t = Target("log_rpm")
        m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=600, categorical_features="from_dtype", random_state=0)
        m.fit(Xtr, t.forward(ytr, Xtr))
        mm = metrics(yva, t.inverse(m.predict(Xva), Xva)); mm["fold"] = k; rows.append(mm)
    print(pd.DataFrame(rows)[["MAE", "RMSE", "MAPE", "R2"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
