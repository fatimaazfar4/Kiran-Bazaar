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
datasets:
  - wfp-food-prices-for-pakistan
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
| Type | scikit-learn `Pipeline`: `DerivedFeatures` → `ColumnTransformer` (median imputation + missing indicators + `StandardScaler` / one-hot) → `BaggingClassifier` (100 decision trees, `min_samples_leaf=20`, `max_samples=0.3`, `max_features=0.5`) |
| Files | `spike_classifier.joblib` (classifier), `price_change_regressor.joblib` (companion `HistGradientBoostingRegressor`, expected % change), `metadata.json` |
| Decision threshold | **0.266** (chosen on validation data to maximise F1; use it instead of 0.5) |
| Selection | 9 algorithms tuned with `GridSearchCV` / `RandomizedSearchCV` on a 5-fold `TimeSeriesSplit`; best validation ROC-AUC selected |
| Versions | Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3 |

## Intended use
An early-warning signal for households, NGOs and journalists ("atta in Lahore: 16 % chance of a jump next month").
**Not** for trading or for price-setting decisions. Covers only Karachi, Lahore, Multan, Peshawar and Quetta and 13 staples.

## Training data
[WFP Food Prices for Pakistan](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan) (HDX, CC BY-IGO; sources: FAO GIEWS / FPMA,
Pakistan Bureau of Statistics) and the WFP markets table. One row per (city, staple, month), 2014-01 to 2026-08: **9,568 rows**,
26 input features (price momentum and volatility, the gap vs other cities, national commodity momentum, diesel/petrol/wage
changes, implied USD/PKR changes, seasonality, city/province/category).

Chronological split: train 2014-01 to 2022-11 (6,646 rows) · validation 2022-12 to 2024-09 (1,427) · test 2024-10 to 2026-08 (1,495).

## Evaluation (held-out test set, reported once)
| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Bagging (final)** | **0.818** | **0.395** | **0.487** | 0.417 | 0.585 |
| Persistence rule ("it jumped last month") | 0.619 | 0.194 | 0.332 | 0.330 | 0.335 |
| Class prior | 0.500 | 0.126 | 0.000 | – | – |

Confusion matrix at threshold 0.266: TN 1153 · FP 154 · FN 78 · TP 110.

## Limitations and biases
* Much of the ranking skill comes from separating volatile items (poultry, eggs, wheat) from stable ones. **Within** one commodity, the test ROC-AUC ranges from 0.50 to 0.88; it is near chance for rice, mash daal and eggs.
* Exogenous shocks (floods, support-price announcements, import bans, bird flu) are not in the features.
* Fuel and FX features did not help on the 2024-26 test period, when the rupee was stable. They may matter more in devaluation periods, but this is untested.
* The probabilities are not calibrated against later regimes: the spike base rate moved from 15 % (train) to 19 % (validation) to 13 % (test).

## How to use
```python
import joblib, pandas as pd, sys
sys.path.insert(0, "src")            # the pipeline uses kiranbazaar.transformers.DerivedFeatures
clf = joblib.load("spike_classifier.joblib")
row = pd.read_csv("features.csv").query("market == 'Lahore' and commodity == 'Wheat flour'").tail(1)
p = clf.predict_proba(row)[:, 1]     # P(spike next month)
```

## Citation
World Food Programme (2026). *Pakistan – Food Prices*. Humanitarian Data Exchange. https://data.humdata.org/dataset/wfp-food-prices-for-pakistan (CC BY-IGO).
