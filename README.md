# 🛒 Kiran Bazaar: will a staple food's price spike next month?

**CT-378 Machine Learning semester project.** Given a city and a staple (atta, rice, sugar, oil, ghee, milk, eggs,
chicken, daals), predict the probability that its retail price **rises by more than 5 % next month**, plus the expected
price, using 12 years of WFP / Pakistan Bureau of Statistics market data.

> *"Wheat flour (atta) in Lahore: 16 % chance of a >5 % price jump in October 2026."*

| Deployment | Link |
|---|---|
| Streamlit Community Cloud | `https://<your-app>.streamlit.app` ← *add after deploying* |
| Hugging Face Space (Gradio) | `https://huggingface.co/spaces/<user>/kiran-bazaar` ← *add after deploying* |
| Hugging Face model repo | `https://huggingface.co/<user>/kiran-bazaar-spike-classifier` ← *add after deploying* |
| Local Flask app | `python app/flask_app.py` → http://localhost:5000 |

## Results (held-out test set, Oct 2024 to Aug 2026)
| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Bagging (Decision Trees)**, final | **0.818** | **0.395** | **0.487** | 0.417 | 0.585 |
| Baseline: persistence rule | 0.619 | 0.194 | 0.332 | 0.330 | 0.335 |
| Baseline: class prior | 0.500 | 0.126 | 0.000 | – | – |

9 course algorithms were compared on validation (Logistic Regression, PCA+LR, KNN, Naive Bayes, SVM, Decision Tree,
Random Forest, Bagging, Gradient Boosting); see `reports/classification_comparison_validation.csv`.
Companion regression (expected % change): Gradient Boosting regressor, test MAE 3.49 pp vs 3.60 pp for "no change".

## Repository layout
```
data/            raw/ (WFP CSVs), processed/features.csv, download_data.py, README.md (sources, licence, dates)
notebooks/       01_eda · 02_modelling · 03_unsupervised_pca_kmeans_apriori   (executed, with outputs)
src/kiranbazaar/ config · features (feature building + time split) · transformers · modeling · inference
scripts/         train.py (full lifecycle) · build_notebooks.py · build_hf.py · make_report.py
models/          spike_classifier.joblib · price_change_regressor.joblib · metadata.json
app/             streamlit_app.py · gradio_app.py · flask_app.py + templates/ + static/
deploy/          MODEL_CARD.md; build_hf.py writes hf_model/ and hf_space/ here
reports/         figures/, screenshots/, result tables (CSV), report.pdf
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py          # ~5 min: builds features, tunes 9 models, saves models/ and reports/
```
To refresh the data from HDX first: `python data/download_data.py`.
To re-run the notebooks: `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.

## Running and deploying the three apps
**1. Local Flask** (`/predict` endpoint + HTML front end, input validation and error messages)
```bash
python app/flask_app.py
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
     -d '{"market": "Lahore", "commodity": "Wheat flour", "overrides": {"diesel_ret_1": 8}}'
```
**2. Streamlit Community Cloud**: `streamlit run app/streamlit_app.py` locally. To deploy: push this repo to GitHub →
share.streamlit.io → *New app* → main file `app/streamlit_app.py`, Python 3.13 → Deploy. Paste the URL above.

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
WFP *Food Prices for Pakistan* and *Markets for Pakistan*, Humanitarian Data Exchange, CC BY-IGO. Retrieved 2026-11-18.
Details in [`data/README.md`](data/README.md).

## Team & contributions
| Member | Roll no. | Contribution |
|---|---|---|
| *name* | | |
| *name* | | |
| *name* | | |
