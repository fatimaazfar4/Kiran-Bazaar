"""Pipelines, model zoo and search spaces (all algorithms from CT-378)."""
from __future__ import annotations

from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (BaggingClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .transformers import DERIVED_NUMERIC, DerivedFeatures

FEATURES = C.CATEGORICAL_FEATURES + C.NUMERIC_FEATURES
RS = C.RANDOM_STATE


def make_preprocessor():
    num_cols = [c for c in C.NUMERIC_FEATURES if c != "month"] + DERIVED_NUMERIC
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, C.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("derive", DerivedFeatures()), ("columns", ct)])


def make_pipeline(model, pca: bool = False):
    steps = [("prep", make_preprocessor())]
    if pca:
        steps.append(("pca", PCA(random_state=RS)))
    steps.append(("model", model))
    return Pipeline(steps)


def classifier_zoo():
    """name -> (pipeline, search space, search type). 'grid' = GridSearchCV."""
    return {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced")),
            {"model__C": [0.01, 0.1, 1, 10]}, "grid"),
        "PCA + Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced"), pca=True),
            {"pca__n_components": [5, 10, 20], "model__C": [0.1, 1]}, "grid"),
        "KNN": (
            make_pipeline(KNeighborsClassifier()),
            {"model__n_neighbors": [15, 31, 61, 101], "model__weights": ["uniform", "distance"]}, "grid"),
        "Naive Bayes": (
            make_pipeline(GaussianNB()),
            {"model__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}, "grid"),
        "SVM (RBF)": (
            make_pipeline(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RS)),
            {"model__C": [0.3, 1, 3], "model__gamma": ["scale", 0.01]}, "grid"),
        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=RS)),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [20, 50, 100]}, "grid"),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=RS)),
            {"model__max_depth": [6, 10, 16, None], "model__min_samples_leaf": randint(5, 60),
             "model__max_features": ["sqrt", 0.3, 0.5]}, "random"),
        "Bagging (Decision Trees)": (
            make_pipeline(BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=20), n_estimators=100,
                                            n_jobs=-1, random_state=RS)),
            {"model__max_samples": [0.3, 0.6, 1.0], "model__max_features": [0.5, 0.8, 1.0]}, "grid"),
        "Gradient Boosting": (
            make_pipeline(HistGradientBoostingClassifier(class_weight="balanced", random_state=RS)),
            {"model__learning_rate": loguniform(0.01, 0.2), "model__max_depth": [3, 5, 8, None],
             "model__max_iter": randint(100, 500), "model__min_samples_leaf": randint(20, 120),
             "model__l2_regularization": loguniform(1e-3, 10)}, "random"),
    }


def baseline_classifier():
    return make_pipeline(DummyClassifier(strategy="prior"))


def regressor_zoo():
    return {
        "Baseline (train mean)": (make_pipeline(DummyRegressor()), None, None),
        "Linear Regression": (make_pipeline(LinearRegression()), None, None),
        "Ridge Regression": (make_pipeline(Ridge()), {"model__alpha": [0.1, 1, 10, 100]}, "grid"),
        "Random Forest Regressor": (
            make_pipeline(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RS)),
            {"model__max_depth": [6, 10, 16], "model__min_samples_leaf": randint(5, 60),
             "model__max_features": ["sqrt", 0.3, 0.5]}, "random"),
        "Gradient Boosting Regressor": (
            make_pipeline(HistGradientBoostingRegressor(loss="absolute_error", random_state=RS)),
            {"model__learning_rate": loguniform(0.01, 0.2), "model__max_depth": [3, 5, 8],
             "model__max_iter": randint(100, 400), "model__min_samples_leaf": randint(20, 120)}, "random"),
    }
