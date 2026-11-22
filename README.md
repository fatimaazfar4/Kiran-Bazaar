# 🛒 Kiran Bazaar: will a staple food's price spike next month?

**CT-378 Machine Learning semester project.** Given a city and a staple (atta, rice, sugar, oil, ghee, milk, eggs,
chicken, daals), predict the probability that its retail price **rises by more than 5 % next month**, plus the expected
price, using 12 years of WFP / Pakistan Bureau of Statistics market data.

> *"Wheat flour (atta) in Lahore: 23 % chance of a >5 % price jump in November 2026."*

| Deployment | Link |
|---|---|
| Streamlit Community Cloud | `https://<your-app>.streamlit.app` ← *add after deploying* |
| Hugging Face Space (Gradio) | `https://huggingface.co/spaces/<user>/kiran-bazaar` ← *add after deploying* |
| Hugging Face model repo | `https://huggingface.co/<user>/kiran-bazaar-spike-classifier` ← *add after deploying* |
| Local Flask app | `python app/flask_app.py` → http://localhost:5000 |

## Results (held-out test set, Nov 2024 to Sep 2026)
| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Random Forest, calibrated** (final) | **0.829** | **0.414** | **0.481** | 0.377 | 0.665 |
| Baseline: persistence rule | 0.614 | 0.193 | 0.327 | 0.330 | 0.325 |
| Baseline: class prior | 0.500 | 0.128 | 0.000 | – | – |

9 course algorithms were compared on validation (Logistic Regression, PCA+LR, KNN, Naive Bayes, SVM, Decision Tree,
Random Forest, Bagging, Gradient Boosting); see `reports/classification_comparison_validation.csv`. Random Forest (0.751)
and Bagging (0.750) were essentially tied; Random Forest won by the pre-defined rule. Its probabilities are calibrated with
Platt scaling (`CalibratedClassifierCV`), and the alert threshold is 22 % (F1-optimal on validation).
Companion regression (expected % change): Gradient Boosting regressor, test MAE 3.46 pp vs 3.56 pp for "no change".

## Repository layout
```
data/            raw/ (WFP CSVs), processed/features.csv, download_data.py, README.md (sources, licence, dates)
notebooks/       01_eda · 02_modelling · 03_unsupervised_pca_kmeans_apriori   (executed, with outputs)
src/kiranbazaar/ config · features (feature building + time split) · transformers · modeling · inference
scripts/         train.py (full lifecycle) · build_notebooks.py · build_hf.py · make_report.py · make_explainer.py
models/          spike_classifier.joblib · price_change_regressor.joblib · metadata.json
app/             streamlit_app.py · gradio_app.py · flask_app.py + templates/ + static/
deploy/          MODEL_CARD.md; build_hf.py writes hf_model/ and hf_space/ here
reports/         figures/, screenshots/, result tables (CSV), report.pdf
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt   # tested on Python 3.14
python scripts/train.py          # ~6 min: builds features, tunes 9 models, calibrates, saves models/ and reports/
```
**Windows notes.** If `pip`, `streamlit` or `jupyter` are blocked ("An Application Control policy has blocked this file"), run them through
Python: `python -m pip …`, `python -m streamlit run …`, `python -m jupyter …`. If scikit-learn's `.pyd` files are blocked, Windows
*Smart App Control* is on: turn it off (Windows Security → App & browser control) or run the project in WSL.

To refresh the data from HDX (WFP publishes about two months late): `python data/download_data.py`, then `python scripts/train.py`.
To re-run the notebooks: `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.

## Running and deploying the three apps
**1. Local Flask** (`/predict` endpoint + HTML front end, input validation and error messages)
```bash
python app/flask_app.py
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
     -d '{"market": "Lahore", "commodity": "Wheat flour", "overrides": {"diesel_ret_1": 8}}'
```
**2. Streamlit Community Cloud**: `streamlit run app/streamlit_app.py` locally. To deploy: push this repo to GitHub →
share.streamlit.io → *New app* → main file `app/streamlit_app.py`, Python 3.13 (or 3.14 if offered) → Deploy. Paste the URL above.

**3. Hugging Face (model card + Space)**
```bash
pip install huggingface_hub && huggingface-cli login     # needs a write token
python scripts/build_hf.py --push --user <your-hf-username>
```
This uploads `deploy/hf_model/` (model card + pipelines) and `deploy/hf_space/` (self-contained Gradio app).
Test the Space locally first with `python app/gradio_app.py`.

## Method in one paragraph
Raw long-format prices → complete monthly grid per (city, item) → features computed from **past months only**
(1/2/3/6/12-month changes, 6-month volatility, moving-average ratios, spike count, gap vs other cities, national momentum,
city food-basket momentum, diesel/petrol/wage changes, implied USD/PKR changes) → **chronological 70/15/15 split** →
scikit-learn Pipeline (derived features → imputation → scaling/one-hot → model) fitted on training data only →
5-fold `TimeSeriesSplit` tuning (ROC-AUC) → model selection and F1-optimal threshold on validation → refit on
train+validation → **single** test evaluation → `joblib`. See `notebooks/02_modelling.ipynb`.

## Data & licence
WFP *Food Prices for Pakistan* and *Markets for Pakistan*, Humanitarian Data Exchange, CC BY-IGO. Prices retrieved 2026-11-22 (data to Oct 2026).
Details in [`data/README.md`](data/README.md).

## Team & contributions
| Member | Roll no. | Contribution |
|---|---|---|
| *name* | | |
| *name* | | |
| *name* | | |
