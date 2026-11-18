"""Custom scikit-learn transformers used inside the saved Pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Stateless feature engineering that runs inside the Pipeline.

    * cyclical month encoding (Dec and Jan are neighbours)
    * relative momentum: this market's change minus the national change
    * fuel pressure: diesel change scaled by recent volatility

    It learns nothing in fit(), so it cannot leak test information.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        m = X["month"].astype(float)
        X["month_sin"] = np.sin(2 * np.pi * m / 12)
        X["month_cos"] = np.cos(2 * np.pi * m / 12)
        X["rel_momentum"] = X["ret_1"] - X["nat_ret_1"]
        X["fuel_pressure"] = X["diesel_ret_1"] / (X["vol_6"].abs() + 0.01)
        X["accel"] = X["ret_1"] - (X["ret_3"] / 3)
        return X.drop(columns=["month"])

    def get_feature_names_out(self, input_features=None):
        cols = [c for c in input_features if c != "month"]
        return np.array(cols + ["month_sin", "month_cos", "rel_momentum", "fuel_pressure", "accel"])


DERIVED_NUMERIC = ["month_sin", "month_cos", "rel_momentum", "fuel_pressure", "accel"]
