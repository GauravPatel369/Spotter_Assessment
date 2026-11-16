# Freight Rate Prediction

Predicts the posted rate of truckload freight loads for the Spotter Machine Learning Engineer assessment. The model trains on 48,000 labelled loads from January to October 2025 and predicts 12,000 loads from November and December 2025.

| Deliverable | Location |
|---|---|
| Final predictions, 12,000 loads | [`validation_predictions.csv`](validation_predictions.csv) |
| Filled December chart inputs | [`data/december_chart_inputs.csv`](data/december_chart_inputs.csv) |
| Official December chart from `score.py` | [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) |
| Report | [`reports/assessment_report.pdf`](reports/assessment_report.pdf) and [`.docx`](reports/assessment_report.docx) |
| Full record of the work and every result | [`WORK_SUMMARY.md`](WORK_SUMMARY.md) |
| Original assessment instructions | [`docs/`](docs/) |

## Approach in brief

- **Validation:** a rolling-origin backtest. Each fold trains on the months up to a cut-off and predicts the next two, the same shape as the real task. A random split was rejected because it overstated accuracy by about 31%.
- **Removed feature:** `quote_signal`. It equals the label in half the training months and is random noise in `validation.csv`, so it was dropped.
- **Corrupted labels:** 1.4% of labels are prices at roughly 3.4 or 0.29 times normal. These rows are excluded from training but kept in every evaluation.
- **Market level:** the daily price level is modelled explicitly, as `market_index` plus recent monthly effects. Tree models alone cannot separate the market from the upward drift through the year.
- **Model:** XGBoost averaged with Ridge regression, both predicting log rate per mile after the market level is removed.

| Backtest, mean of 4 folds | MAE | MAPE | MAE on clean-label rows |
|---|---|---|---|
| Final model | $96.5 | 4.15% | $45.7 |
| Best model without the market-level step | $113.6 | 4.89% | $63.1 |
| Predict the median rate per mile | $252.3 | 11.54% | $203.7 |

Even a perfect model would score an MAE of about $48, because of the corrupted labels. `score.py` itself computes no metric, so these numbers come from the backtest.

## Setup

Python 3.11 was used. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

XGBoost runs on an NVIDIA GPU by default. Without one, set `XGB_DEVICE=cpu`. The backtest gives the same result on CPU: MAE $96.5.

```bash
export XGB_DEVICE=cpu          # bash
$env:XGB_DEVICE = "cpu"        # PowerShell
```

On Windows, also set `PYTHONIOENCODING=utf-8` so console output prints correctly.

## Data

Place the provided files in `data/` with these names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Reproduce the submission

Train on all labelled data and write both prediction files:

```bash
python src/predict.py
```

This uses the final configuration by default. It writes `validation_predictions.csv` and `data/december_chart_inputs.csv` where the scorer expects them. Copies also go to `outputs/`, together with `outputs/final_model_summary.json`.

Then run the official scorer exactly as the assessment README instructs:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Expected output:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

Running `predict.py` twice gives identical predictions.

## Evaluate the final configuration

Run the four-fold backtest:

```bash
python src/experiments.py --models "xgb_gpu+ridge" --targets log_rpm --train-filter clean \
  --feature-sets level --level fe3d5 --params '{"max_depth": 4, "min_child_weight": 50}'
```

Each run appends one row per fold, plus a mean row, to `outputs/experiment_log.csv`. It saves the last fold's predictions to `outputs/oof/`.

Analyse the errors of a saved run by segment:

```bash
python src/evaluate.py "outputs/oof/xgb_gpu+ridge__log_rpm__clean__level__lvlfe3d5.csv" --label final
```

## Analysis scripts

Each script prints its findings, which are saved in `outputs/*.txt`, and writes charts to `reports/figures/`.

| Script | Purpose |
|---|---|
| `src/eda.py` | Exploratory analysis: target, features, data quality, time patterns |
| `src/leakage_audit.py` | Leakage audit, including the `quote_signal` investigation |
| `src/drift_analysis.py` | Training versus validation comparison, with PSI and unseen cities |
| `src/check_features.py` | Self-check of the feature pipeline and fold boundaries |
| `src/temporal_analysis.py` | Study of the market and time component of prices |
| `src/random_split_diagnostic.py` | Random K-fold versus time-based backtest |
| `src/importance.py` | Gain, permutation and SHAP feature importance |
| `reports/build_report.py` | Rebuilds `reports/assessment_report.docx` from saved results |

## Code layout

```
src/
  features.py      cleaning, features, out-of-fold lane encoder, market-level model (LevelModel)
  validation.py    backtest folds and metrics
  experiments.py   model definitions, target transforms, backtest harness
  predict.py       final training and prediction files
outputs/           script outputs, experiment logs, final predictions
reports/           report, report builder, figures
scorer_results/    official chart
score.py           provided scorer, unchanged
```

## Main options

| Option | Values |
|---|---|
| `--models` | `xgb_gpu`, `lgbm_l2`, `lgbm_l1`, `lgbm_huber`, `cat_gpu`, `hgb_l2`, `hgb_l1`, `linear`, `ridge`, `dummy_median`, `dummy_mean`. Join with `+` to average models, for example `xgb_gpu+ridge`. |
| `--targets` | `raw`, `log`, `rpm`, `log_rpm` (log of rate per mile) |
| `--train-filter` | `all`, or `clean` to drop rows outside $1.20 to $5.00 per mile from training |
| `--feature-sets` | `full`, `level`, `lean`, `no_lane`, `no_cities`, and others listed in `experiments.py` |
| `--level` | `none`, `fe3` for the last 3 month effects, `fe3d5` to add half the recent drift |

`cat_gpu` needs a GPU. Set `CATBOOST_TASK=CPU` to run it on CPU.
