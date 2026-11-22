"""Builds reports/report.html from saved results and prints it to reports/report.pdf (needs Playwright + Chromium).
    python scripts/make_report.py
"""
import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
meta = json.loads((ROOT / "models" / "metadata.json").read_text())
val = pd.read_csv(R / "classification_comparison_validation.csv")
test = pd.read_csv(R / "classification_test.csv")
reg = pd.read_csv(R / "regression_test.csv")
regv = pd.read_csv(R / "regression_comparison_validation.csv")
byc = pd.read_csv(R / "test_by_commodity.csv")


def img(path, width="100%"):
    b = base64.b64encode((ROOT / path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'


def table(df, cols, fmt=None):
    fmt = fmt or {}
    head = "".join(f"<th>{c}</th>" for c in cols.values())
    body = ""
    for _, r in df.iterrows():
        tds = ""
        for c in cols:
            v = r[c]
            tds += f"<td>{fmt[c](v) if c in fmt and pd.notna(v) else ('–' if pd.isna(v) else v)}</td>"
        body += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


f3 = lambda v: f"{v:.3f}"  # noqa: E731
cal = meta["calibration"]
_feat = pd.read_csv(ROOT / "data" / "processed" / "features.csv", parse_dates=["date"])
len_lab = int((_feat["spike_next"].notna() & (_feat["date"] >= "2014-01-01")).sum())
len_raw = len(pd.read_csv(ROOT / "data" / "raw" / "wfp_food_prices_pak.csv"))
s, m, cm = meta["split"], meta["test_metrics_classifier"], meta["test_confusion_matrix"]
v_sorted = val.sort_values("val_roc_auc", ascending=False)
best_per = byc.dropna().sort_values("roc_auc")
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Kiran Bazaar report</title><style>
@page {{ size: A4; margin: 14mm 14mm 14mm 14mm; }}
body {{ font: 9.6pt/1.38 "Times New Roman", Georgia, serif; color:#111; }}
h1 {{ font-size: 16pt; margin: 0 0 2px; text-align:center; }} .sub {{ text-align:center; color:#444; margin-bottom:8px; font-size:9pt }}
h2 {{ font-size: 11.5pt; margin: 9px 0 3px; border-bottom: 1px solid #999; }}
p {{ margin: 3px 0 5px; text-align: justify; }} ul {{ margin: 2px 0 5px 16px; padding:0 }} li {{ margin: 1px 0 }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 4px 0 6px; }}
th, td {{ border: 1px solid #bbb; padding: 2px 4px; text-align: center; }} th {{ background: #eef1f6; }}
td:first-child {{ text-align:left }}
.row {{ display:flex; gap:8px; align-items:flex-start }} .row > div {{ flex:1 }}
.cap {{ font-size: 8pt; color:#444; text-align:center; margin: 0 0 4px }} .abs {{ background:#f5f6f8; padding:6px 9px; border-left:3px solid #2b5797 }}
.pb {{ page-break-before: always }} code {{ font-size: 8.3pt }}
</style></head><body>
<h1>Kiran Bazaar: Predicting Next-Month Food Price Spikes in Pakistan</h1>
<div class="sub">CT-378 Machine Learning · Semester Project · Group: <i>[names / roll numbers]</i> · {pd.Timestamp.now():%B %Y}</div>

<h2>Abstract</h2>
<p class="abs">Food inflation is one of the most pressing household problems in Pakistan. We frame it as a binary classification
task: <i>will the retail price of a staple in a given city rise by more than 5 % next month?</i> Using WFP / Pakistan Bureau of Statistics
monthly prices for 13 staples in 5 cities (2014-2026, {len_lab:,} city-item-months), combined with fuel, labour-wage and implied exchange-rate
series, we engineered 26 leakage-free features and compared nine course algorithms under a chronological split with 5-fold time-series
cross-validation. A Random Forest, selected on validation and probability-calibrated (Platt scaling), reached <b>ROC-AUC {m['roc_auc']:.2f} and
F1 {m['f1']:.2f}</b> on the held-out 2024-26 test period, against {test.iloc[1].roc_auc:.2f} / {test.iloc[1].f1:.2f} for a persistence baseline. The model is deployed as a Streamlit app,
a Hugging Face Space with model card, and a Flask REST API with an HTML front end.</p>

<h2>1. Problem and motivation</h2>
<p>Between 2022 and 2024 Pakistan's food prices rose faster than at any time in the dataset: atta in Lahore roughly tripled from its 2021 level at the 2023-24 peak. A household,
NGO or ration-distribution programme that knew a month ahead that <i>"atta in Lahore has a high chance of jumping next month"</i> could
buy early or pre-position stock. Most prior ML work on this data predicts the price <i>level</i> (regression). We instead predict the
<i>event</i> that matters to users: a spike above 5 % (≈ 15 % of city-item-months). This gives a clear target, meaningful metrics
(F1, ROC-AUC) and an actionable output. A companion regression model supplies the expected next-month price.</p>

<h2>2. Data collection</h2>
<table><tr><th>Source</th><th>Content</th><th>Licence</th><th>Date</th></tr>
<tr><td>WFP Food Prices for Pakistan (HDX)</td><td>{len_raw:,} monthly retail prices, 5 cities, 18 series, 2004-01 to {meta['data_last_month'][:7]}; sources FAO GIEWS/FPMA, PBS</td><td>CC BY-IGO</td><td>retrieved 2026-11-22</td></tr>
<tr><td>WFP Markets for Pakistan (HDX)</td><td>market → province, district, latitude, longitude</td><td>CC BY-IGO</td><td>retrieved 2026-11-18</td></tr></table>
<p>We combine (i) the 13 food series (targets), (ii) the diesel, petrol and unskilled-wage series as cost-push drivers joined by city and month,
(iii) a monthly <b>USD/PKR rate implied by WFP's own conversion</b> (Σprice / Σusdprice) and (iv) the market table. Data are market-level aggregates with no
personal identifiers. The modelling window starts in 2014-01: before that only rice and wheat exist and the fuel/wage series are empty.
Final table: <b>{len_lab:,} labelled rows × 26 features</b> (4 categorical, 22 numeric) + 5 derived in-pipeline.</p>

<h2>3. Exploratory data analysis</h2>
<div class="row"><div>{img("reports/figures/03_trends_fx.png")}<div class="cap">Fig. 1 - National price index of staples vs implied USD/PKR.</div></div>
<div>{img("reports/figures/05_class_balance.png")}<div class="cap">Fig. 2 - Spike rate by commodity and by year (class balance).</div></div></div>
<p>Seven EDA plots are in <code>notebooks/01_eda.ipynb</code>: coverage/missingness heatmap, distributions, trends, outliers, class balance,
correlations, seasonality. Key findings: (1) prices are right-skewed and trend strongly with the rupee, so we use <i>percentage changes</i>
rather than levels; (2) 40-50 % of months are missing for staples that start in 2013-15, and internal gaps are imputed inside the pipeline;
(3) poultry and eggs spike 35-40 % of months while oil, ghee and milk rarely do; extreme changes are genuine market events and are kept;
(4) the target is imbalanced (15 % positive) and its rate shifts over time (18 % in the 2023-24 validation period, 13 % in test); (5) the best single correlates are recent
volatility and national momentum (Spearman ρ ≈ 0.29), so signals must be combined; (6) poultry and eggs are clearly seasonal.</p>

<h2>4. Methodology</h2>
<ul>
<li><b>Features (past-only):</b> 1/2/3/6/12-month changes, 6-month volatility, price vs 3/12-month mean and 12-month high, spikes in last 12 months,
gap vs other cities, national momentum of the item, city food-basket momentum, diesel/petrol/wage changes, USD/PKR 1/3-month changes, city, province, category,
month. Level features (log-price, PKR level) were tried and <b>removed</b>: they trend for 20 years and made linear models extrapolate
(validation ROC-AUC of logistic regression rose from 0.51 to 0.71 after removal).</li>
<li><b>Pipeline</b> (fitted on training data only): <code>DerivedFeatures</code> (cyclical month, relative momentum, fuel pressure, acceleration) →
<code>ColumnTransformer</code> [median imputation + missing indicators + StandardScaler | most-frequent imputation + one-hot] → model.</li>
<li><b>Split:</b> chronological 70/15/15 on whole months. Train {s['train']['start'][:7]} to {s['train']['end'][:7]} ({s['train']['rows']:,} rows),
validation {s['validation']['start'][:7]} to {s['validation']['end'][:7]} ({s['validation']['rows']:,}), test {s['test']['start'][:7]} to {s['test']['end'][:7]} ({s['test']['rows']:,}).</li>
<li><b>Models:</b> baselines (class prior; persistence rule "jumped last month") + Logistic Regression, PCA+LR, KNN, Gaussian Naive Bayes, SVM (RBF),
Decision Tree, Random Forest, Bagging, Gradient Boosting. Regression: Linear, Ridge, RF and Gradient Boosting regressors.</li>
<li><b>Tuning:</b> <code>GridSearchCV</code> / <code>RandomizedSearchCV</code> with 5-fold <code>TimeSeriesSplit</code>, scored on ROC-AUC;
<code>class_weight="balanced"</code> for imbalance. The best model on validation ROC-AUC is chosen.</li>
<li><b>Calibration:</b> class weighting inflated the forest's probabilities (validation: mean prediction {cal['val_mean_prob_raw']:.0%} vs actual spike rate
{cal['val_actual_spike_rate']:.0%}), so the selected model is wrapped in <code>CalibratedClassifierCV</code> (sigmoid / Platt, 5-fold <code>TimeSeriesSplit</code>):
validation Brier score {cal['val_brier_raw']:.3f} → {cal['val_brier_calibrated']:.3f}, mean prediction {cal['val_mean_prob_calibrated']:.0%}. The decision threshold is set on the
calibrated validation probabilities to maximise F1 ({meta['decision_threshold']:.2f}, not 0.5); the model is refit on train+validation and evaluated once on test.</li>
<li><b>Unsupervised (notebook 03):</b> PCA (10 components for 90 % variance), K-Means on per-series volatility profiles (k = 4) and Apriori
association rules on monthly co-spike baskets.</li></ul>

<h2>5. Results</h2>
<p><b>Model comparison (validation set; used for selection).</b></p>
{table(v_sorted, {"model": "Model", "cv_roc_auc": "CV ROC-AUC", "train_roc_auc": "Train ROC-AUC", "val_roc_auc": "Val ROC-AUC",
                  "val_pr_auc": "Val PR-AUC", "val_f1": "Val F1*", "val_precision": "Val Prec.", "val_recall": "Val Recall"},
       {k: f3 for k in ["cv_roc_auc", "train_roc_auc", "val_roc_auc", "val_pr_auc", "val_f1", "val_precision", "val_recall"]})}
<div class="cap" style="text-align:left">*F1 at each model's own F1-optimal validation threshold; baselines at 0.5.</div>
<p><b>Final evaluation on the held-out test set (reported once).</b></p>
{table(test, {"model": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "precision": "Precision", "recall": "Recall"},
       {k: f3 for k in ["roc_auc", "pr_auc", "f1", "precision", "recall"]})}
<div class="row"><div style="flex:2.2">{img("reports/figures/10_roc_pr_validation.png")}<div class="cap">Fig. 3 - ROC and precision-recall curves on validation.</div></div>
<div style="flex:1">{img("reports/figures/16_calibration_validation.png")}<div class="cap">Fig. 4 - Calibration on validation.</div></div></div>
<div class="row"><div>{img("reports/figures/11_confusion_matrix_test.png", "88%")}<div class="cap">Fig. 5 - Test confusion matrix
(TN {cm[0][0]}, FP {cm[0][1]}, FN {cm[1][0]}, TP {cm[1][1]}).</div></div>
<div>{img("reports/figures/12_train_vs_val_auc.png")}<div class="cap">Fig. 6 - Train vs validation ROC-AUC.</div></div>
<div>{img("reports/figures/13_permutation_importance.png", "92%")}<div class="cap">Fig. 7 - Permutation importance on test.</div></div></div>
<p><b>Over/under-fitting.</b> KNN (train ROC-AUC 1.00 vs val 0.70) and Gradient Boosting (0.96 vs 0.74) over-fit. Logistic Regression and Naive
Bayes show small gaps but low training scores (high bias, under-fitting). Random Forest and Bagging (both 0.93 vs 0.75) are essentially tied; by the
pre-defined rule (highest validation ROC-AUC) Random Forest is selected (0.751 vs 0.750). Its remaining over-fit is limited by <code>min_samples_leaf=25</code>
and <code>max_depth=16</code>. The higher spike rate in validation (18 % vs 15 %) explains part of the gap between CV and validation scores.</p>
<p><b>Error analysis.</b> The final model catches {m['recall']:.0%} of test spikes with {m['precision']:.0%} precision, versus {test.iloc[1].recall:.0%}/{test.iloc[1].precision:.0%} for persistence.
Its strongest signals are the national momentum of the item, its own last-month change and its recent spike history. Per commodity, test ROC-AUC is best for
{", ".join(f"{r.commodity} ({r.roc_auc:.2f})" for r in best_per.tail(3).itertuples())} and near chance for
{", ".join(f"{r.commodity} ({r.roc_auc:.2f})" for r in best_per.head(3).itertuples())}.
(Masoor daal's high score rests on a single test spike.) Part of the pooled {m['roc_auc']:.2f} therefore comes from separating volatile from stable items.
Fuel and FX features had near-zero or slightly negative importance on 2024-26, when the rupee was stable.
<b>Regression:</b> the Gradient Boosting regressor reaches test MAE {reg.iloc[0].mae * 100:.2f} pp (R² {reg.iloc[0].r2:.3f}) vs {reg.iloc[2].mae * 100:.2f} pp for "no change".
Monthly changes are close to a random walk, which supports the classification framing.</p>

<h2>6. Deployment</h2>
<div class="row"><div>{img("reports/screenshots/streamlit_app.png")}<div class="cap">Fig. 8 - Streamlit app: probability, verdict, expected price, history and "why" charts.</div></div>
<div>{img("reports/screenshots/gradio_app.png")}<div class="cap">Fig. 9 - Hugging Face Space (Gradio).</div></div></div>
<div class="row"><div style="flex:1">{img("reports/screenshots/flask_app.png")}<div class="cap">Fig. 10 - Flask app (<code>POST /predict</code> + HTML front end).</div></div>
<div style="flex:1"><p>All three apps share <code>src/kiranbazaar/inference.py</code> and load the same <code>joblib</code> pipelines. The user picks a city, a staple and
an "as-of" month, and can optionally override this month's price change, diesel change and USD/PKR change for what-if analysis. Each app returns the
spike probability, a verdict at the tuned threshold, the expected next-month price and a local explanation: the change in probability when each
feature is reset to its median. The Flask API validates inputs (unknown city/commodity, bad month format, out-of-range or non-numeric overrides,
non-JSON body) and returns HTTP 400 with a readable message. The front end also validates fields before sending.</p>
<p><b>Links:</b> Streamlit: <i>[URL]</i> · HF Space: <i>[URL]</i> · HF model: <i>[URL]</i> · GitHub: <i>[URL]</i></p></div></div>

<h2>7. Limitations and future work</h2>
<ul><li>Only five large cities and monthly granularity. Weekly PBS SPI data would allow earlier warnings and more cities.</li>
<li>No exogenous shock features (floods, wheat support-price announcements, import duties, bird-flu outbreaks, Ramadan/Eid shifting through the calendar).</li>
<li>Regime shifts change the base rate (15 % → 18 % → 13 %), so even calibrated probabilities drift: the model should be retrained and re-calibrated monthly as new data arrives.</li>
<li>Within-commodity skill is uneven. Per-commodity models or commodity-specific features (international palm-oil and wheat prices) are the next step.</li></ul>

<h2>8. References</h2>
<ol style="margin:2px 0 4px 16px;padding:0;font-size:8.8pt">
<li>World Food Programme (2026). <i>Pakistan - Food Prices</i> and <i>Pakistan - Markets</i>. Humanitarian Data Exchange. https://data.humdata.org/dataset/wfp-food-prices-for-pakistan. CC BY-IGO.</li>
<li><i>[Add the SSRN paper on ML forecasting of Pakistani retail food prices (20,160 observations) cited in the proposal: authors, year, title, URL.]</i></li>
<li>Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i> 12, 2825-2830.</li>
<li>Breiman, L. (1996). Bagging predictors. <i>Machine Learning</i> 24(2), 123-140.</li>
<li>Breiman, L. (2001). Random forests. <i>Machine Learning</i> 45(1), 5-32.</li>
<li>Raschka, S. (2018). MLxtend: providing machine learning and data science utilities. <i>JOSS</i> 3(24), 638.</li></ol>

<h2>9. Individual contribution</h2>
<table><tr><th>Member</th><th>Roll no.</th><th>Contribution</th></tr>
<tr><td><i>[Name 1]</i></td><td></td><td><i>e.g. data pipeline, feature engineering, EDA notebook</i></td></tr>
<tr><td><i>[Name 2]</i></td><td></td><td><i>e.g. model training and tuning, evaluation, unsupervised analysis</i></td></tr>
<tr><td><i>[Name 3]</i></td><td></td><td><i>e.g. Streamlit / HF / Flask deployment, report</i></td></tr></table>
</body></html>"""

(R / "report.html").write_text(html, encoding="utf-8")
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto((R / "report.html").as_uri())
        pg.pdf(path=str(R / "report.pdf"), format="A4", print_background=True, prefer_css_page_size=True)
        b.close()
    print("wrote", R / "report.pdf")
except ImportError:
    print("Python Playwright not installed: run `node scripts/report_pdf.js`, or open reports/report.html in a browser and print to PDF.")
