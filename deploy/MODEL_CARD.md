---
license: other
license_name: cc-by-igo-3.0
library_name: sklearn
tags:
  - tabular-classification
  - sklearn
  - food-prices
  - pakistan
  - time-series
metrics:
  - roc_auc
  - f1
pipeline_tag: tabular-classification
---

# Kiran Bazaar: next-month food price spike classifier (Pakistan)

Predicts the probability that the **retail price of a food staple in a Pakistani city rises by more than 5 % next month**.
CT-378 Machine Learning semester project.

## Model details
| | |
|---|---|
| Type | scikit-learn `Pipeline`: `DerivedFeatures` → `ColumnTransformer` (median imputation + missing indicators + `StandardScaler` / one-hot) → `RandomForestClassifier` (300 trees, `max_depth=16`, `min_samples_leaf=25`, `max_features="sqrt"`, `class_weight="balanced_subsample"`), wrapped in `CalibratedClassifierCV` (sigmoid / Platt scaling, 5-fold `TimeSeriesSplit`) |
| Files | `spike_classifier.joblib` (calibrated classifier), `price_change_regressor.joblib` (companion `HistGradientBoostingRegressor`, expected % change), `metadata.json` |
| Decision threshold | **0.22** (F1-optimal on calibrated validation probabilities; use it instead of 0.5) |
| Selection | 9 algorithms tuned with `GridSearchCV` / `RandomizedSearchCV` on a 5-fold `TimeSeriesSplit`; best validation ROC-AUC selected (Random Forest 0.751, Bagging 0.750) |
| Calibration (validation) | mean predicted spike chance 44 % → 18 % (actual 17.8 %); Brier score 0.204 → 0.132 |
| Versions | Python 3.14, scikit-learn 1.9.0, pandas 3.0.2, numpy 2.4.4 |

## Intended use
An early-warning signal for households, NGOs and journalists ("atta in Lahore: 23 % chance of a jump next month").
**Not** for trading or for price-setting decisions. Covers only Karachi, Lahore, Multan, Peshawar and Quetta and 13 staples.
The forecast is for the month after the latest published data; WFP publishes about two months late.

## Training data
[WFP Food Prices for Pakistan](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan) (HDX, CC BY-IGO; sources: FAO GIEWS / FPMA,
Pakistan Bureau of Statistics) and the WFP markets table. One row per (city, staple, month), 2014-01 to 2026-09: **9,633 rows**,
26 input features (price momentum and volatility, the gap vs other cities, national commodity momentum, diesel/petrol/wage
changes, implied USD/PKR changes, seasonality, city/province/category).

Chronological split: train 2014-01 to 2022-12 (6,711 rows) · validation 2023-01 to 2024-10 (1,427) · test 2024-11 to 2026-09 (1,495).

## Evaluation (held-out test set, reported once)
| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Random Forest, calibrated (final)** | **0.829** | **0.414** | **0.481** | 0.377 | 0.665 |
| Persistence rule ("it jumped last month") | 0.614 | 0.193 | 0.327 | 0.330 | 0.325 |
| Class prior | 0.500 | 0.128 | 0.000 | – | – |

Confusion matrix at threshold 0.22: TN 1094 · FP 210 · FN 64 · TP 127.

## Limitations and biases
* Much of the ranking skill comes from separating volatile items (poultry, eggs, wheat) from stable ones. **Within** one commodity, the test ROC-AUC ranges from 0.50 (eggs) to 0.86 (moong daal); masoor daal had only one test spike.
* Exogenous shocks (floods, support-price announcements, import bans, bird flu) are not in the features.
* Fuel and FX features did not help on the 2024-26 test period, when the rupee was stable.
* The spike base rate moves between periods (15 % train, 18 % validation, 13 % test), so calibration should be refreshed when the model is retrained.

## How to use
```python
import joblib, pandas as pd, sys
sys.path.insert(0, "src")            # the pipeline uses kiranbazaar.transformers.DerivedFeatures
clf = joblib.load("spike_classifier.joblib")
row = pd.read_csv("features.csv").query("market == 'Lahore' and commodity == 'Wheat flour'").tail(1)
p = clf.predict_proba(row)[:, 1]     # calibrated P(spike next month); alert if p >= 0.22
```

## Citation
World Food Programme (2026). *Pakistan – Food Prices*. Humanitarian Data Exchange. https://data.humdata.org/dataset/wfp-food-prices-for-pakistan (CC BY-IGO).
