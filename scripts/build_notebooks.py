"""Generates the three notebooks in notebooks/. Execute them afterwards with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = """import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from kiranbazaar import config as C, features as F
FIG = C.FIG_DIR; FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep"); plt.rcParams["figure.dpi"] = 110
pd.set_option("display.width", 160, "display.max_columns", 40)"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    return n


eda = [
    """md:# 01 · Exploratory Data Analysis: Kiran Bazaar
**Question:** will the retail price of a food staple in a Pakistani city jump by more than **5 %** next month?

Data sources (see `data/README.md`):
1. **WFP Food Prices for Pakistan** (HDX, CC BY-IGO): monthly retail prices, 5 cities, 18 series, 2004-2026.
2. **WFP Markets for Pakistan** (HDX, CC BY-IGO): market metadata (province, latitude/longitude).
3. Inside (1), we use the **fuel (diesel, petrol)** and **labour wage** series and the **USD/PKR exchange rate** implied by `price / usdprice` as *external drivers* for food prices.""",
    SETUP,
    """prices, markets = F.load_raw()
print(prices.shape); prices.head()""",
    """prices.info()""",
    """md:## 1. Overview: what is in the raw table""",
    """display(prices.describe(include="all").T[["count", "unique", "top", "mean", "min", "max"]])
print("markets with prices:", prices.market.unique())
print("duplicate (date, market, commodity) rows:", prices.duplicated(["date", "market", "commodity"]).sum())
print("constant columns:", [c for c in prices if prices[c].nunique() == 1])""",
    """md:`priceflag`, `pricetype` and `currency` are constant, so they are dropped. No personal identifiers exist (market-level aggregates only).
## 2. Missing values: coverage of each series over time""",
    """p = prices.assign(ym=prices.date.dt.to_period("M"))
cov = p.pivot_table(index="commodity", columns="ym", values="price", aggfunc="count")
full = pd.period_range(p.ym.min(), p.ym.max(), freq="M")
cov = cov.reindex(columns=full).fillna(0)
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(cov, cmap="Blues", cbar_kws={"label": "# markets reporting"}, ax=ax,
            xticklabels=24)
ax.set_title("Plot 1 - Data coverage: number of markets reporting each commodity per month")
ax.set_xlabel("month"); fig.tight_layout(); fig.savefig(FIG / "01_coverage_heatmap.png")""",
    """miss = 1 - cov.astype(bool).sum(axis=1) / len(full)
print("share of months with NO observation, per commodity"); miss.sort_values().round(3)""",
    """md:Rice and wheat start in 2004/2006; most other staples begin in 2013-2015. *Salt* stops in 2018 and *Lentils (gram)* only has 9 months, so both are dropped as targets. Gaps inside a series become NaN in the monthly grid and are imputed (median + missing-indicator) **inside** the Pipeline.
## 3. Distributions""",
    """fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(data=prices[prices.commodity.isin(C.TARGET_COMMODITIES)], y="commodity", x="price", ax=ax[0],
            fliersize=2, order=C.TARGET_COMMODITIES)
ax[0].set_xscale("log"); ax[0].set_title("Plot 2a - Price distribution per staple (PKR, log scale)")
feats = F.build_features(prices, markets)
sns.histplot(feats.ret_1.dropna() * 100, bins=120, ax=ax[1], color="#2b5797")
ax[1].axvline(5, color="crimson", ls="--", label="spike threshold +5%"); ax[1].set_xlim(-25, 40)
ax[1].set_title("Plot 2b - Month-on-month price change (%)"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "02_distributions.png")""",
    """md:Prices are right-skewed and span two orders of magnitude, so the model uses `log_price` and *percentage changes* instead of raw PKR. The monthly change is centred near 0 % with heavy tails: spikes are rare events.
## 4. Price trends and the exchange rate""",
    """staples = ["Wheat flour", "Rice (basmati, broken)", "Sugar", "Oil (cooking)", "Poultry"]
nat = feats[feats.commodity.isin(staples)].groupby(["date", "commodity"]).price.mean().unstack()
fx = feats.groupby("date").fx_level.first()
fig, ax = plt.subplots(figsize=(12, 4.5))
(nat / nat.bfill().iloc[0] * 100).plot(ax=ax, lw=1.3)
ax2 = ax.twinx(); ax2.plot(fx.index, fx, "k--", lw=1, label="PKR per USD (implied)"); ax2.set_ylabel("PKR / USD")
ax2.grid(False); ax2.legend(loc="upper left")
ax.set(title="Plot 3 - National average price index (first obs = 100) vs USD/PKR", ylabel="index")
fig.tight_layout(); fig.savefig(FIG / "03_trends_fx.png")""",
    """md:Food prices follow the rupee: the sharp devaluations of 2018-19 and 2022-23 line up with steep rises in staples. This motivates the `fx_ret_1/3` features and warns us that the **2022-24 inflation shock** sits in the validation period (a regime shift).
## 5. Outliers""",
    """d = feats[["commodity", "ret_1"]].dropna()
q1, q3 = d.groupby("commodity").ret_1.transform(lambda s: s.quantile(.25)), d.groupby("commodity").ret_1.transform(lambda s: s.quantile(.75))
iqr = q3 - q1
d["outlier"] = (d.ret_1 < q1 - 3 * iqr) | (d.ret_1 > q3 + 3 * iqr)
print("extreme (3xIQR) monthly changes per commodity:"); display(d.groupby("commodity").outlier.agg(["sum", "mean"]).round(3))
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=d, x=d.ret_1 * 100, y="commodity", fliersize=2, ax=ax, order=C.TARGET_COMMODITIES)
ax.set(xlim=(-40, 60), xlabel="month-on-month change (%)", title="Plot 4 - Outliers in monthly price changes")
fig.tight_layout(); fig.savefig(FIG / "04_outliers.png")""",
    """md:Two patterns stand out. **Poultry and eggs** routinely move ±20-40 % in a month, so their IQR is wide and almost nothing counts as an outlier. **Rice, ghee, cooking oil and milk** are flat most months (IQR close to 0), so each occasional step change shows up as an "outlier". Both kinds are **real market events, not data errors**, and they are exactly what we want to predict, so we keep them. We use percentage changes (scale-free) and `StandardScaler`; the tree ensembles are insensitive to extreme values anyway.
## 6. Class balance""",
    """lab = F.labelled(feats)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
lab.groupby("commodity")[C.TARGET_CLF].mean().sort_values().plot.barh(ax=ax[0], color="#2b5797")
ax[0].set(title="Plot 5a - Spike rate (>5% next month) by commodity", xlabel="share of months")
yr = lab.groupby(lab.date.dt.year)[C.TARGET_CLF].mean()
yr.plot.bar(ax=ax[1], color="#c0504d"); ax[1].set(title="Plot 5b - Spike rate by year", ylabel="share", xlabel="")
fig.tight_layout(); fig.savefig(FIG / "05_class_balance.png")
print(lab[C.TARGET_CLF].value_counts(), "\\npositive rate:", round(lab[C.TARGET_CLF].mean(), 3))""",
    """md:~15 % positives: imbalanced but learnable. We handle it with `class_weight="balanced"`, by tuning on ROC-AUC and by choosing the decision threshold on validation to maximise F1 (not the default 0.5). Accuracy would be misleading here: always saying "no spike" is already 85 % accurate.
## 7. Correlations""",
    """num = lab[C.NUMERIC_FEATURES + [C.TARGET_REG, C.TARGET_CLF]]
corr = num.corr(method="spearman")
fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, annot=False, square=True)
ax.set_title("Plot 6 - Spearman correlation of numeric features and targets")
fig.tight_layout(); fig.savefig(FIG / "06_correlation_heatmap.png")
corr[C.TARGET_CLF].drop([C.TARGET_CLF, C.TARGET_REG]).sort_values(key=abs, ascending=False).head(10).round(3)""",
    """md:The return features (`ret_1/2/3`, `ma_ratio_3`) are strongly inter-correlated, which suits regularised Logistic Regression and tree ensembles; Naive Bayes is expected to suffer because it assumes independence. The strongest single correlates of a spike are **recent volatility** (`spikes_12`, `vol_6`, ρ ≈ 0.29) and the **national momentum** of that commodity (`nat_ret_1`). Every correlation is weak to moderate, so a model has to combine many signals.
## 8. Seasonality""",
    """season = lab.groupby(["month", "commodity"])[C.TARGET_CLF].mean().unstack()[["Poultry", "Eggs", "Wheat flour", "Sugar", "Milk"]]
fig, ax = plt.subplots(figsize=(10, 4)); season.plot(ax=ax, marker="o")
ax.set(title="Plot 7 - Spike probability by calendar month", xlabel="month", ylabel="P(spike next month)", xticks=range(1, 13))
fig.tight_layout(); fig.savefig(FIG / "07_seasonality.png")""",
    """md:Poultry and eggs show clear seasonality (winter demand, summer heat losses), so the cyclical `month_sin/cos` features are built inside the pipeline.
## 9. Final modelling table""",
    """print("labelled rows:", len(lab), "| raw features:", len(C.CATEGORICAL_FEATURES + C.NUMERIC_FEATURES),
      "(+5 derived inside the pipeline)")
lab[C.CATEGORICAL_FEATURES + C.NUMERIC_FEATURES + [C.TARGET_CLF]].head()""",
]

model_nb = [
    """md:# 02 · Modelling, tuning and evaluation
Lifecycle steps 2-7. All of the logic lives in `scripts/train.py` and `src/kiranbazaar/` so that the notebook, the
apps and the saved model share **one** code path. This notebook runs the script and explains each step.""",
    SETUP,
    """md:## Pre-processing Pipeline (fitted on training data only)
`DerivedFeatures` (cyclical month, relative momentum, fuel pressure, acceleration) → `ColumnTransformer`
[numeric: median imputation + missing indicators + StandardScaler | categorical: most-frequent imputation + One-Hot] → model.

Lag/rolling features are computed per series from **past months only**, so they hold no future information.""",
    """from kiranbazaar import modeling as M
M.make_pipeline(M.LogisticRegression())""",
    """md:## Train / validation / test split: chronological (time series)
70 / 15 / 15 by month: training ends Jan 2022, validation covers the 2022-24 inflation shock, test is the most recent ~2 years.
A random split would leak the future into training (neighbouring months are near-identical).""",
    """lab = F.labelled(F.build_features(*F.load_raw()))
tr, va, te = F.time_split(lab)
pd.DataFrame({n: {"rows": len(d), "from": d.date.min().date(), "to": d.date.max().date(), "spike rate": round(d[C.TARGET_CLF].mean(), 3)}
              for n, d in [("train", tr), ("validation", va), ("test", te)]}).T""",
    """md:## Training all models
Baselines: (a) class prior, (b) persistence rule ("jumped last month, so jumps again").
Course algorithms: Logistic Regression, PCA + Logistic Regression, KNN, Gaussian Naive Bayes, SVM (RBF), Decision Tree, Random Forest, Bagging, Gradient Boosting. Plus Linear/Ridge regression for the companion "expected % change" task.

Tuning: **5-fold `TimeSeriesSplit`** cross-validation (forward-chaining, so each fold validates on later months) with `GridSearchCV` / `RandomizedSearchCV`, scored on ROC-AUC.""",
    """%run ../scripts/train.py""",
    """md:## Model comparison (validation set: used for model selection)""",
    """R = C.ROOT / "reports"
cmp = pd.read_csv(R / "classification_comparison_validation.csv")
cmp.drop(columns="best_params").sort_values("val_roc_auc", ascending=False).round(3)""",
    """cmp[["model", "best_params"]]""",
    """from IPython.display import Image, display
display(Image(FIG / "10_roc_pr_validation.png")); display(Image(FIG / "12_train_vs_val_auc.png"))""",
    """md:**Over/under-fitting (read the train vs validation chart).**
* **KNN** reaches train ROC-AUC 1.00 against 0.70 on validation: distance weighting makes every training point its own nearest neighbour, so it memorises. That is clear over-fitting.
* **Gradient Boosting** (0.96 → 0.74) also over-fits: CV chose unlimited tree depth.
* **Logistic Regression, PCA + LR and Naive Bayes** show small gaps (≈ 0.04-0.08) and low training scores: they are *high-bias* models that under-fit the non-linear signal.
* **Random Forest** (0.93 → 0.75) and **Bagging** (0.93 → 0.75) are the two best models and are essentially tied. Both average many trees, which reduces variance. Random Forest also draws a random subset of features at every split, which de-correlates the trees. By our pre-defined rule (highest validation ROC-AUC) **Random Forest is selected** (0.751 vs 0.750). It still over-fits somewhat; `min_samples_leaf=25` and `max_depth=16` limit this.

Validation is also harder than training because of the **2022-24 inflation shock** (spike rate 18 % vs 15 %).
## Probability calibration
`class_weight="balanced_subsample"` helps the forest notice the rare spikes, but it **inflates the predicted probabilities**: on validation
the raw forest predicts a 44 % average spike chance when the real rate is 18 %. An app that says "59 % chance" when the truth is closer to 20 %
would mislead users. We therefore wrap the selected model in `CalibratedClassifierCV(method="sigmoid")` (Platt scaling) with a 5-fold
`TimeSeriesSplit`. This is decided and checked on **validation** data, and the F1-optimal decision threshold is chosen on the calibrated validation probabilities.
## Final evaluation on the held-out test set (reported once)""",
    """import json
meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
print("final model:", meta["classifier"], "| decision threshold:", round(meta["decision_threshold"], 3))
display(pd.read_csv(R / "classification_test.csv").round(3))
print((R / "classification_report_test.txt").read_text())
display(Image(FIG / "11_confusion_matrix_test.png"))
print("calibration (validation):", meta["calibration"])
display(Image(FIG / "16_calibration_validation.png"))""",
    """md:## Error analysis""",
    """display(pd.read_csv(R / "test_by_commodity.csv").round(3).sort_values("roc_auc"))
display(Image(FIG / "14_test_auc_by_commodity.png")); display(Image(FIG / "13_permutation_importance.png"))""",
    """md:## Companion regression: expected next-month % change (MAE / RMSE / R²)""",
    """display(pd.read_csv(R / "regression_comparison_validation.csv").drop(columns="best_params").round(4))
display(pd.read_csv(R / "regression_test.csv").round(4)); display(Image(FIG / "15_regression_test_scatter.png"))""",
    """md:Monthly price changes are close to a random walk, so R² is low and the regression only slightly beats "no change". This is the
main argument for framing the problem as **classification** (is a spike coming?), where the models clearly beat the baselines.
## Saved artefacts""",
    """import joblib
clf = joblib.load(C.MODELS_DIR / "spike_classifier.joblib")
latest = F.build_features(*F.load_raw()); latest = latest[latest.date == latest.date.max()]
latest.assign(p_spike=clf.predict_proba(latest[M.FEATURES])[:, 1].round(3))[["date", "market", "commodity", "price", "p_spike"]] \\
      .sort_values("p_spike", ascending=False).head(10)""",
]

unsup = [
    """md:# 03 · Unsupervised analysis: PCA, K-Means clustering and Apriori
These complement the supervised model and cover the remaining course algorithms.""",
    SETUP,
    """from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
lab = F.labelled(F.build_features(*F.load_raw()))
tr, va, te = F.time_split(lab)""",
    """md:## 1. PCA on the numeric features (fitted on the training set only)""",
    """num = [c for c in C.NUMERIC_FEATURES if c not in ("latitude", "longitude")]
imp, sc = SimpleImputer(strategy="median").fit(tr[num]), None
Xtr = imp.transform(tr[num]); sc = StandardScaler().fit(Xtr); Z = sc.transform(Xtr)
pca = PCA().fit(Z)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(np.arange(1, len(num) + 1), np.cumsum(pca.explained_variance_ratio_), marker="o")
ax[0].axhline(0.9, color="crimson", ls="--"); ax[0].set(title="PCA - cumulative explained variance", xlabel="# components", ylabel="variance explained")
P = pca.transform(Z)[:, :2]
s = ax[1].scatter(P[:, 0], P[:, 1], c=tr[C.TARGET_CLF], cmap="coolwarm", s=4, alpha=0.5)
ax[1].set(title="Training rows on PC1-PC2 (red = spike next month)", xlabel="PC1", ylabel="PC2", xlim=(-6, 8), ylim=(-6, 8))
fig.tight_layout(); fig.savefig(FIG / "20_pca.png")
print("components for 90% variance:", int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1))
pd.DataFrame(pca.components_[:3].T, index=num, columns=["PC1", "PC2", "PC3"]).round(2).sort_values("PC1", key=abs, ascending=False).head(8)""",
    """md:PC1 is a "momentum" axis: all the return and moving-average-ratio features load on it with similar weights. It takes 10 components to keep 90 % of the variance,
and spikes and non-spikes overlap heavily in the PC1-PC2 plane, so the class signal is not concentrated in a few linear directions.
Compare `PCA + Logistic Regression` with plain Logistic Regression in notebook 02.
## 2. K-Means: which market-commodity series behave alike?
Each of the 65 series (5 cities × 13 staples) is described by its **volatility profile** on training data.""",
    """g = tr.groupby(["market", "commodity"])
prof = pd.DataFrame({"mean_change": g.ret_1.mean(), "volatility": g.ret_1.std(), "spike_rate": g[C.TARGET_CLF].mean(),
                     "skew": g.ret_1.skew(), "mean_nat_gap": g.nat_gap.mean(), "fx_corr": g.apply(lambda d: d.ret_1.corr(d.fx_ret_1))}).dropna()
Zp = StandardScaler().fit_transform(prof)
ks = range(2, 9); inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=20, random_state=42).fit(Zp); inertia.append(km.inertia_); sil.append(silhouette_score(Zp, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(ks, inertia, marker="o"); ax[0].set(title="Elbow method", xlabel="k", ylabel="inertia")
ax[1].plot(ks, sil, marker="o", color="#c0504d"); ax[1].set(title="Silhouette score", xlabel="k")
fig.tight_layout(); fig.savefig(FIG / "21_kmeans_k.png")""",
    """k = 4
km = KMeans(k, n_init=20, random_state=42).fit(Zp)
prof["cluster"] = km.labels_
display(prof.groupby("cluster").mean().round(3))
for c in range(k):
    print(f"cluster {c}:", sorted(set(prof[prof.cluster == c].index.get_level_values("commodity"))))
P2 = PCA(2).fit_transform(Zp)
fig, ax = plt.subplots(figsize=(8, 5.5))
for c in range(k):
    m = prof.cluster.values == c; ax.scatter(P2[m, 0], P2[m, 1], s=40, label=f"cluster {c}")
for (mk, cm), (x, y) in zip(prof.index, P2):
    if mk == "Lahore": ax.annotate(cm, (x, y), fontsize=7)
ax.set(title="K-Means clusters of market x commodity price behaviour (labels: Lahore)", xlabel="PC1", ylabel="PC2"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "22_kmeans_clusters.png")""",
    """md:Interpretation (read the cluster means table above):
* **Volatile perishables** (Eggs, Poultry in every city): ~3× higher volatility and spike rate than anything else.
* **FX-sensitive series** (highest correlation with the rupee): daals, rice and wheat in most cities.
* **Rare-jump series** (high positive skew): flat most months, then a big step. Ghee and cooking oil (administered / import-priced) fall here, plus wheat and milk in some cities.
* **Stable series** (lowest skew): milk, sugar and some daal/rice series.

Apart from poultry and eggs, the same staple falls into **different clusters in different cities** (e.g. coarse rice appears in three clusters).
Local market behaviour matters, which is why `market` and the city-vs-national gap `nat_gap` are model features.
## 3. Apriori: which staples spike together?
Each (market, month) is a "basket" whose items are the staples that spiked (>5 %) that month. Association rules show co-movement.""",
    """from mlxtend.frequent_patterns import apriori, association_rules
feats = F.build_features(*F.load_raw())
basket = (feats.assign(spike=feats.ret_1 > C.SPIKE_THRESHOLD).pivot_table(index=["market", "date"], columns="commodity", values="spike", aggfunc="max")
          .dropna(thresh=10).fillna(False).astype(bool))
basket = basket[basket.any(axis=1)]
print("baskets (market-months with >=1 spike):", len(basket))
freq = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.2)
rules = rules[(rules.confidence >= 0.4)].sort_values("lift", ascending=False)
rules["antecedents"] = rules.antecedents.apply(lambda s: ", ".join(sorted(s))); rules["consequents"] = rules.consequents.apply(lambda s: ", ".join(sorted(s)))
rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(12).round(3)""",
    """top = rules.head(12)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(top.antecedents + "  →  " + top.consequents, top.lift, color="#2b5797"); ax.invert_yaxis()
ax.set(title="Apriori - strongest co-spike rules (lift)", xlabel="lift"); fig.tight_layout(); fig.savefig(FIG / "23_apriori_rules.png")""",
    """md:The strongest rule is **ghee ↔ cooking oil** (lift ≈ 7: both depend on imported palm oil), then the **three daals** (masoor, moong, mash) and
**wheat ↔ wheat flour** (same supply chain). That supports the cross-commodity `basket_ret_1` feature and is useful in itself: when wheat
jumps, households should expect atta to follow.""",
]

for name, cells in [("01_eda.ipynb", eda), ("02_modelling.ipynb", model_nb), ("03_unsupervised_pca_kmeans_apriori.ipynb", unsup)]:
    nbf.write(nb(cells), NB / name)
    print("wrote", NB / name)
