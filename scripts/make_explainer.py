"""Builds reports/project_explainer.html (plain-language study guide). Print to PDF with scripts/report_pdf.js-style Playwright."""
import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
meta = json.loads((ROOT / "models" / "metadata.json").read_text())
val = pd.read_csv(R / "classification_comparison_validation.csv").sort_values("val_roc_auc", ascending=False)
test = pd.read_csv(R / "classification_test.csv")
reg_val = pd.read_csv(R / "regression_comparison_validation.csv")
reg_test = pd.read_csv(R / "regression_test.csv")
byc = pd.read_csv(R / "test_by_commodity.csv")
m, cm, s = meta["test_metrics_classifier"], meta["test_confusion_matrix"], meta["split"]
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
acc = (tn + tp) / (tn + fp + fn + tp)


def img(p, w="100%"):
    return f'<img src="data:image/png;base64,{base64.b64encode((ROOT / p).read_bytes()).decode()}" style="width:{w}">'


role = {
    "Logistic Regression": "Linear model; draws one straight boundary",
    "PCA + Logistic Regression": "Compresses features to 10 components, then logistic regression",
    "KNN": "Looks at the 61 most similar past months",
    "Naive Bayes": "Probability model assuming features are independent",
    "SVM (RBF)": "Finds a curved boundary with maximum margin",
    "Decision Tree": "One tree of if/else questions (depth 5)",
    "Random Forest": "Many trees, each on random rows and features",
    "Bagging (Decision Trees)": "100 trees on random 30% samples; votes averaged",
    "Gradient Boosting": "Trees built one after another, each fixing the last one's errors",
    "Baseline: class prior": "Always predicts the average spike rate (no skill)",
    "Baseline: persistence rule": "\"It jumped last month, so it will jump again\"",
}
rows = ""
for r in val.itertuples():
    sel = r.model.startswith("Bagging")
    rows += (f"<tr{' class=sel' if sel else ''}><td>{r.model}{' ✅ <b>chosen</b>' if sel else ''}</td><td class=l>{role.get(r.model, '')}</td>"
             f"<td>{r.val_roc_auc:.3f}</td><td>{r.val_f1:.3f}</td><td>{r.train_roc_auc:.3f}</td></tr>")
reg_rows = "".join(
    f"<tr{' class=sel' if r.model.startswith('Gradient') else ''}><td>{r.model}</td><td>{r.val_mae * 100:.2f}</td><td>{r.val_r2:.3f}</td></tr>"
    for r in reg_val.itertuples())
byc_rows = "".join(
    f"<tr><td>{r.commodity}</td><td>{r.spike_rate:.0%}</td><td>{'–' if pd.isna(r.roc_auc) else f'{r.roc_auc:.2f}'}</td><td>{r.f1:.2f}</td></tr>"
    for r in byc.sort_values("roc_auc", ascending=False).itertuples())

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Kiran Bazaar explainer</title><style>
@page {{ size:A4; margin:14mm 15mm; }}
body {{ font: 10pt/1.45 "Segoe UI", system-ui, Arial, sans-serif; color:#1b1f27; }}
h1 {{ font-size:20pt; margin:0; color:#1f3f75 }} .sub {{ color:#555; margin:2px 0 10px }}
h2 {{ font-size:13pt; color:#1f3f75; margin:14px 0 5px; padding-bottom:2px; border-bottom:2px solid #1f3f75 }}
h3 {{ font-size:10.5pt; margin:8px 0 3px }}
p {{ margin:3px 0 6px }} ul,ol {{ margin:3px 0 6px 18px; padding:0 }} li {{ margin:2px 0 }}
table {{ border-collapse:collapse; width:100%; font-size:8.8pt; margin:5px 0 8px }}
th,td {{ border:1px solid #c8ccd4; padding:3px 6px; text-align:center }} th {{ background:#e8edf5 }}
td:first-child, td.l {{ text-align:left }} tr.sel td {{ background:#fff6d6; font-weight:600 }}
.box {{ background:#f2f5fa; border-left:4px solid #1f3f75; padding:7px 10px; margin:6px 0 }}
.warn {{ background:#fff4ec; border-left:4px solid #d9822b; padding:7px 10px; margin:6px 0 }}
.big {{ display:flex; gap:8px; margin:6px 0 }} .big div {{ flex:1; border:1px solid #c8ccd4; border-radius:6px; padding:6px; text-align:center }}
.big b {{ display:block; font-size:16pt; color:#1f3f75 }} .big span {{ font-size:8pt; color:#555 }}
.row {{ display:flex; gap:10px; align-items:flex-start }} .row > div {{ flex:1 }} .cap {{ font-size:8pt; color:#555; text-align:center }}
.pb {{ page-break-before:always }} code {{ background:#eef0f4; padding:0 3px; border-radius:3px; font-size:8.8pt }}
.qa p {{ margin:1px 0 7px }} .qa b {{ color:#1f3f75 }}
</style></head><body>

<h1>🛒 Kiran Bazaar: Project Explainer</h1>
<div class="sub">CT-378 Machine Learning · plain-language guide to what the project does, for the group and the viva</div>

<div class="box"><b>In one sentence:</b> given a <b>city</b> (Karachi, Lahore, Multan, Peshawar, Quetta) and a <b>food staple</b>
(atta, rice, sugar, oil, ghee, milk, eggs, chicken, daals), the system predicts the <b>probability that its price will rise by more than 5 % next month</b>,
e.g. <i>"Wheat flour (atta) in Lahore: 16 % chance of a &gt;5 % jump in October 2026"</i>.</div>

<h2>1. What problem are we solving? Classification or regression?</h2>
<p><b>Main task = binary CLASSIFICATION.</b> For every city + staple + month, the answer is <b>Yes (spike)</b> or <b>No (no spike)</b>:</p>
<ul><li><b>Spike = 1</b> if next month's price is more than 5 % higher than this month's price; otherwise <b>0</b>.</li>
<li>The model outputs a <b>probability</b> (0-100 %). If it is above <b>{meta['decision_threshold']:.0%}</b> (our tuned alert threshold), the app says "spike likely".</li></ul>
<p><b>Extra task = REGRESSION</b> (a smaller side model): predict the <i>expected % change</i> in price, which the app turns into "expected price next month: Rs 111.9".</p>
<p><b>Why classification as the main task?</b> Month-to-month price changes are close to random, so predicting the exact number is very hard (our regression
only barely beats "no change"). Predicting the <i>event</i> users care about ("will it jump?") is learnable and gives clean metrics (F1, ROC-AUC).
It also differs from the published work on this dataset, which used regression.</p>

<h2>2. Where does the data come from?</h2>
<table><tr><th>Dataset</th><th>Publisher / source</th><th>What it contains</th><th>Licence</th></tr>
<tr><td class=l><b>WFP Food Prices for Pakistan</b> (<code>wfp_food_prices_pak.csv</code>)</td><td class=l>World Food Programme on the <b>Humanitarian Data Exchange (HDX)</b>,
data.humdata.org. Original sources: Pakistan Bureau of Statistics and FAO GIEWS/FPMA</td>
<td class=l>14,345 rows, 16 columns: monthly <b>retail prices in PKR</b> for 18 items in 5 cities, Jan 2004 to Sep 2026</td><td>CC BY-IGO (free with citation)</td></tr>
<tr><td class=l><b>WFP Markets for Pakistan</b> (<code>wfp_markets_pak.csv</code>)</td><td class=l>Same HDX dataset</td>
<td class=l>Market list: city → province, district, latitude, longitude</td><td>CC BY-IGO</td></tr></table>
<p><b>How the sources are combined</b> (the guideline asks for at least two sources):</p>
<ol><li><b>13 food items</b> → what we predict (targets).</li>
<li><b>Diesel, petrol and daily labour wage</b> prices (same table) → used as <i>cost drivers</i> (if fuel jumps, transport costs rise and food may follow).</li>
<li><b>USD/PKR exchange rate</b> → recovered from the dataset as <code>price ÷ usdprice</code> (WFP converts every price to dollars, so dividing gives the rupee rate).</li>
<li><b>Markets file</b> → joined on city to add province and location.</li></ol>
<p>No personal data: all values are city-level average prices.</p>

<h2>3. How the raw data becomes a training table</h2>
<ul><li>Each <b>row</b> = one city × one staple × one month, e.g. "Lahore, Sugar, March 2023".</li>
<li><b>Time window used: Jan 2014 to Aug 2026</b> (before 2014 only rice and wheat were recorded, and fuel/wage data did not exist).
Result: <b>9,568 labelled rows</b>; ~15 % of them are spikes.</li>
<li><b>26 input features</b>, all computed using <b>only the current and past months</b> (never the future), plus 5 more created inside the pipeline:</li></ul>
<table><tr><th>Group</th><th>Features (plain words)</th></tr>
<tr><td>Price momentum</td><td class=l>% change over the last 1, 2, 3, 6 and 12 months; price vs its 3- and 12-month average; price vs its 12-month high</td></tr>
<tr><td>Volatility history</td><td class=l>how jumpy the price was over 6 months; how many spikes it had in the last 12 months</td></tr>
<tr><td>Other cities / other items</td><td class=l>price gap vs the same item in other cities; national % change of the item; average change of all food in this city</td></tr>
<tr><td>Cost drivers</td><td class=l>diesel change (1 and 3 months), petrol change, labour wage change, USD/PKR change (1 and 3 months)</td></tr>
<tr><td>Context</td><td class=l>city, province, item, item category, calendar month (as sin/cos so Dec and Jan are neighbours), latitude/longitude</td></tr></table>

<h2>4. How we trained and tested (the ML lifecycle)</h2>
<ol><li><b>EDA</b> (<code>notebooks/01_eda.ipynb</code>): 7 plots covering data coverage/missing values, distributions, price vs rupee trend, outliers, class balance, correlations, seasonality.</li>
<li><b>Pre-processing inside a scikit-learn Pipeline</b>: fill missing values (median, plus missing-value flags) → scale numbers (StandardScaler) → one-hot encode text columns. Fitted on training data only, so there is <b>no data leakage</b>.</li>
<li><b>Time-ordered split (70/15/15):</b> train = {s['train']['start'][:7]} to {s['train']['end'][:7]} ({s['train']['rows']:,} rows) · validation = {s['validation']['start'][:7]} to {s['validation']['end'][:7]} ({s['validation']['rows']:,}) ·
test = {s['test']['start'][:7]} to {s['test']['end'][:7]} ({s['test']['rows']:,}). We never train on the future and test on the past.</li>
<li><b>Tuning:</b> GridSearchCV / RandomizedSearchCV with <b>5-fold time-series cross-validation</b>, optimising ROC-AUC.</li>
<li><b>Model selection on validation</b> → best model picked; alert threshold tuned for best F1 → retrained on train+validation → <b>tested once</b> on the test set.</li>
<li><b>Saved</b> with joblib in <code>models/</code>; versions pinned in <code>requirements.txt</code>.</li></ol>

<h2 class="pb">5. Which models did we use?</h2>
<p>We trained <b>2 baselines + 9 course algorithms</b> for the classification task and compared them on the <b>validation set</b>.
Higher ROC-AUC = better at ranking risky months above safe ones (0.5 = random guessing, 1.0 = perfect).</p>
<table><tr><th>Model</th><th>What it does</th><th>Val ROC-AUC</th><th>Val F1</th><th>Train ROC-AUC</th></tr>{rows}</table>
<div class="box"><b>Final deployed model = Bagging Classifier</b>: 100 decision trees, each trained on a random 30 % of rows and 50 % of features; their votes are averaged.
It had the <b>best validation ROC-AUC (0.755) and PR-AUC</b>. Averaging many trees reduces the over-fitting a single tree suffers from.</div>
<h3>Regression side-model (expected % change): validation results</h3>
<table><tr><th>Model</th><th>Val MAE (percentage points, lower is better)</th><th>Val R²</th></tr>{reg_rows}</table>
<p>The <b>Gradient Boosting Regressor</b> is deployed for the "expected price" number. Plain <b>Linear Regression was worse than just predicting the average</b>,
because price changes are not linear in these features.</p>
<h3>Unsupervised algorithms (notebook 03, analysis only)</h3>
<ul><li><b>PCA</b>: 10 components keep 90 % of the information; the first component is a "price momentum" direction.</li>
<li><b>K-Means (k = 4)</b>: groups the 65 city-item series into volatile perishables (chicken, eggs), rupee-sensitive items (daals, rice, wheat), rare-jump items (ghee, oil) and stable items (milk, sugar).</li>
<li><b>Apriori</b>: finds items that spike together. <b>Ghee ↔ cooking oil</b> (lift ≈ 7, both use imported palm oil), the three daals, and <b>wheat ↔ atta</b>.</li></ul>

<h2>6. How accurate is it? (held-out test set, Oct 2024 to Aug 2026)</h2>
<div class="big"><div><b>{acc:.0%}</b><span>Accuracy</span></div><div><b>{m['roc_auc']:.2f}</b><span>ROC-AUC</span></div><div><b>{m['f1']:.2f}</b><span>F1 (spike class)</span></div>
<div><b>{m['recall']:.0%}</b><span>Recall: real spikes caught</span></div><div><b>{m['precision']:.0%}</b><span>Precision: alerts that were right</span></div></div>
<table><tr><th>Model (test set)</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th><th>Precision</th><th>Recall</th></tr>
{''.join(f"<tr{' class=sel' if 'final' in r.model else ''}><td>{r.model}</td><td>{r.roc_auc:.3f}</td><td>{r.pr_auc:.3f}</td><td>{r.f1:.3f}</td><td>{r.precision:.3f}</td><td>{r.recall:.3f}</td></tr>" for r in test.itertuples())}</table>
<div class="row"><div>{img("reports/figures/11_confusion_matrix_test.png", "85%")}<div class="cap">Confusion matrix on the 1,495 test rows</div></div>
<div><h3>Reading the confusion matrix</h3><ul>
<li><b>{tp}</b> spikes correctly predicted (true positives)</li><li><b>{fn}</b> spikes missed (false negatives)</li>
<li><b>{fp}</b> false alarms (false positives)</li><li><b>{tn}</b> calm months correctly predicted (true negatives)</li></ul>
<div class="warn"><b>Why we don't lead with accuracy:</b> only ~13 % of test months are spikes, so a useless model that <i>always</i> says "no spike"
would score ~87 % accuracy. That's why the guideline asks for <b>F1 / ROC-AUC / confusion matrix</b>, and why we report those.</div></div></div>
<p><b>Is it good enough?</b> Yes for the project. The guidelines mark the <i>quality of the lifecycle</i>, not a minimum score. The model clearly beats both baselines
(ROC-AUC 0.82 vs 0.62; F1 0.49 vs 0.33). Food prices one month ahead are partly random, so 0.82 is a solid result. We set the alert threshold to
{meta['decision_threshold']:.0%} (not 50 %) on purpose: for an early-warning tool it is better to catch most spikes and accept some false alarms.</p>
<p><b>Regression on test:</b> MAE {reg_test.iloc[0].mae * 100:.2f} percentage points vs {reg_test.iloc[2].mae * 100:.2f} for "no change", R² {reg_test.iloc[0].r2:.2f}. That is only a small improvement,
which confirms that classification was the right main framing.</p>

<h2 class="pb">7. Over-fitting / under-fitting</h2>
<div class="row"><div>{img("reports/figures/12_train_vs_val_auc.png")}<div class="cap">Train vs validation ROC-AUC</div></div>
<div><ul><li><b>Over-fitting:</b> KNN (train 1.00 → val 0.69) memorises the training data; Gradient Boosting (0.96 → 0.73) also over-fits.</li>
<li><b>Under-fitting:</b> Logistic Regression and Naive Bayes have small gaps but low scores; they are too simple for the non-linear pattern.</li>
<li><b>Bagging (0.89 → 0.76)</b> is the best balance.</li>
<li>Validation (Dec 2022 to Sep 2024) is harder than training because of the <b>2022-24 inflation shock</b>: spike rate 19 % vs 15 %.</li></ul></div></div>

<h2>8. What drives the predictions? Where does it fail?</h2>
<div class="row"><div>{img("reports/figures/13_permutation_importance.png")}<div class="cap">Most important features (drop in test ROC-AUC when shuffled)</div></div>
<div><table><tr><th>Item (test)</th><th>Spike rate</th><th>ROC-AUC</th><th>F1</th></tr>{byc_rows}</table></div></div>
<p>Top signals: <b>the item's national price momentum</b>, <b>its own change this month</b>, and <b>how many spikes it had in the last year</b>.
Fuel and exchange-rate features did <b>not</b> help in 2024-26 because the rupee was stable then.</p>

<h2>9. Limitations (be ready to say these yourselves)</h2>
<ol><li><b>Only 5 big cities</b> and monthly data. Small towns and weekly movements are not covered.</li>
<li><b>Uneven skill across items:</b> good for chicken, moong daal, cooking oil and milk; close to random for rice, mash daal and eggs. Part of the overall 0.82 comes from simply telling volatile items from stable ones.</li>
<li><b>No shock information:</b> floods, government wheat support-price announcements, import bans, bird flu and Ramadan/Eid timing are not in the features.</li>
<li><b>Regime shifts:</b> the spike rate changes over time (15 % → 19 % → 13 %), so probabilities can drift; the model should be retrained regularly.</li>
<li><b>Fuel / exchange-rate features</b> were untested in a devaluation period inside the test window.</li>
<li><b>Regression is weak</b> (R² ≈ 0.05): the "expected price" is a rough guide only.</li>
<li><b>Data is secondary</b> (collected by PBS/WFP), so we inherit any recording errors and gaps (40-50 % of months are missing for items that start in 2013-15).</li></ol>
<p><b>Future work:</b> add weekly PBS SPI data and more cities, international wheat and palm-oil prices, a calendar of Eid/Ramadan and policy events, per-item models, and probability calibration.</p>

<h2>10. The three deployed apps</h2>
<table><tr><th>App</th><th>File</th><th>How to run / where</th></tr>
<tr><td>Flask web app + REST API (<code>/predict</code>)</td><td><code>app/flask_app.py</code></td><td class=l><code>python app/flask_app.py</code> → localhost:5000. Has input validation and error messages.</td></tr>
<tr><td>Streamlit</td><td><code>app/streamlit_app.py</code></td><td class=l>Locally: <code>python -m streamlit run app/streamlit_app.py</code>; deploy on Streamlit Community Cloud</td></tr>
<tr><td>Hugging Face Space (Gradio) + model card</td><td><code>app/gradio_app.py</code>, <code>deploy/MODEL_CARD.md</code></td><td class=l><code>python scripts/build_hf.py --push --user &lt;name&gt;</code></td></tr></table>
<p>All three show: spike probability, verdict, current and expected price, a price-history chart, and a <b>"Why this prediction?"</b> chart (how much each input pushes the risk up or down).
Users can also try <b>what-if</b> values for this month's price change, diesel change and USD/PKR change.</p>

<h2>11. Where everything is in the project folder</h2>
<table><tr><th>Folder / file</th><th>Contents</th></tr>
<tr><td><code>data/raw/</code>, <code>data/README.md</code></td><td class=l>Original CSVs + source, licence and date documentation</td></tr>
<tr><td><code>src/kiranbazaar/features.py</code></td><td class=l>Builds the features and the target; time split</td></tr>
<tr><td><code>src/kiranbazaar/modeling.py</code></td><td class=l>Pipeline + all algorithms and their search grids</td></tr>
<tr><td><code>scripts/train.py</code></td><td class=l>Runs everything: train, tune, select, test, save models and figures</td></tr>
<tr><td><code>notebooks/01, 02, 03</code></td><td class=l>EDA · modelling and results · PCA/K-Means/Apriori</td></tr>
<tr><td><code>models/</code></td><td class=l>Saved classifier, regressor, metadata.json (all scores)</td></tr>
<tr><td><code>reports/report.pdf</code></td><td class=l>The formal 4-page report to submit</td></tr></table>

<h2 class="pb">12. Likely viva questions and short answers</h2>
<div class="qa">
<p><b>Q: Is this classification or regression?</b><br>Classification (spike yes/no, &gt;5 % rise next month). A small regression model gives the expected price as an extra.</p>
<p><b>Q: Why 5 %?</b><br>It marks a real jump that hurts households, yet happens often enough to learn from (~15 % of months). At 10 % only ~4 % of months would be positive.</p>
<p><b>Q: Why not a random train/test split?</b><br>Neighbouring months are almost identical, so a random split leaks future information into training. We split by time and use TimeSeriesSplit for cross-validation.</p>
<p><b>Q: How did you avoid data leakage?</b><br>(1) Features use only current and past months. (2) All pre-processing sits inside the Pipeline, fitted on training data only. (3) Time-ordered split. (4) The test set was used exactly once.</p>
<p><b>Q: Why is accuracy 84 % but you say F1 0.49?</b><br>The classes are imbalanced (≈13 % spikes in test), so accuracy is misleading. F1 and ROC-AUC measure performance on the rare spike class.</p>
<p><b>Q: Why Bagging and not Random Forest?</b><br>Both are tree ensembles. Bagging had higher validation ROC-AUC (0.755 vs 0.739) and PR-AUC. We selected on validation, not on test.</p>
<p><b>Q: Why is the threshold 27 % and not 50 %?</b><br>With imbalanced classes the 0.5 default misses most spikes. We chose the threshold that maximised F1 on the validation set.</p>
<p><b>Q: How did you handle class imbalance?</b><br><code>class_weight="balanced"</code>, tuning on ROC-AUC instead of accuracy, and the F1-optimal threshold.</p>
<p><b>Q: What was your biggest mistake and how did you fix it?</b><br>The first version used price and exchange-rate <i>levels</i>. These rise for 20 years, so models had to extrapolate (Logistic Regression val ROC-AUC 0.51). Switching to % changes raised it to 0.71.</p>
<p><b>Q: What are the two data sources?</b><br>WFP food prices (with fuel, wage and implied USD/PKR series) and the WFP markets table, both from HDX under CC BY-IGO.</p>
<p><b>Q: What do PCA, K-Means and Apriori add?</b><br>PCA shows the signal is spread over ~10 directions; K-Means groups items by price behaviour; Apriori shows which items spike together (ghee and oil, wheat and atta).</p>
</div>
</body></html>"""
(R / "project_explainer.html").write_text(html, encoding="utf-8")
print("ok")
