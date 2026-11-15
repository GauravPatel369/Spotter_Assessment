"""Backtest folds and metrics.

Rolling-origin, two-month-horizon folds that mirror the real task (train Jan..M, predict M+1..M+2).
The final production fit is train Jan-Oct -> predict Nov-Dec, so the last fold (train Jan-Aug ->
predict Sep-Oct) is the closest analogue and is reported separately as well as in the average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FOLDS = [  # (train_end_month_inclusive, valid_months)
    ("2025-04", ["2025-05", "2025-06"]),
    ("2025-06", ["2025-07", "2025-08"]),
    ("2025-07", ["2025-08", "2025-09"]),
    ("2025-08", ["2025-09", "2025-10"]),
]


def fold_indices(dates: pd.Series):
    """Yield (name, train_idx, valid_idx) as positional integer arrays."""
    per = dates.dt.to_period("M").astype(str).values
    for train_end, valid_months in FOLDS:
        tr = np.where(per <= train_end)[0]
        va = np.where(np.isin(per, valid_months))[0]
        yield f"train<= {train_end} -> {valid_months[0]}..{valid_months[1][-2:]}", tr, va


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    rpm = y_true / 1.0  # placeholder to keep signature simple
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "MAPE": float(np.mean(np.abs(err) / y_true) * 100),
        "MedAE": float(np.median(np.abs(err))),
        "bias": float(np.mean(err)),
    }
    return out


def metrics_clean(y_true, y_pred, distance) -> dict[str, float]:
    """Same metrics restricted to the 'clean' label band (1.2 < $/mile < 5), i.e. excluding injected extremes."""
    rpm = np.asarray(y_true) / np.asarray(distance)
    ok = (rpm > 1.2) & (rpm < 5)
    m = metrics(np.asarray(y_true)[ok], np.asarray(y_pred)[ok])
    return {f"{k}_clean": v for k, v in m.items()}
