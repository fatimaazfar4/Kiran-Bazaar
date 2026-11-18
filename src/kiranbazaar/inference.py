"""Shared inference logic for the Streamlit, Gradio and Flask apps."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as C

FEATURES = C.CATEGORICAL_FEATURES + C.NUMERIC_FEATURES

# Inputs a user may override in "what-if" mode: feature -> (label, min %, max %)
OVERRIDABLE = {
    "ret_1": ("This item's price change this month (%)", -50.0, 100.0),
    "diesel_ret_1": ("Diesel price change this month (%)", -50.0, 100.0),
    "fx_ret_1": ("USD/PKR change this month (%)", -30.0, 50.0),
}

FEATURE_LABELS = {
    "ret_1": "price change this month", "ret_2": "2-month change", "ret_3": "3-month change",
    "ret_6": "6-month change", "ret_12": "12-month change", "vol_6": "6-month volatility",
    "ma_ratio_3": "price vs 3-month avg", "ma_ratio_12": "price vs 12-month avg",
    "max12_ratio": "price vs 12-month high", "spikes_12": "# spikes in last 12 months",
    "nat_gap": "price gap vs other cities", "nat_ret_1": "national change of this item",
    "basket_ret_1": "city food-basket change", "diesel_ret_1": "diesel change (1m)",
    "diesel_ret_3": "diesel change (3m)", "petrol_ret_1": "petrol change (1m)",
    "wage_ret_3": "labour wage change (3m)", "fx_ret_1": "USD/PKR change (1m)", "fx_ret_3": "USD/PKR change (3m)",
    "month": "calendar month",
}


class InputError(ValueError):
    """Raised for invalid user input; message is safe to show to the user."""


@dataclass
class Prediction:
    market: str
    commodity: str
    as_of: str
    target_month: str
    current_price: float
    unit: str
    probability: float
    spike: bool
    decision_threshold: float
    expected_change: float
    expected_price: float
    risk: str

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@lru_cache(maxsize=1)
def artifacts():
    clf = joblib.load(C.MODELS_DIR / "spike_classifier.joblib")
    reg = joblib.load(C.MODELS_DIR / "price_change_regressor.joblib")
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    return clf, reg, meta


@lru_cache(maxsize=1)
def feature_table() -> pd.DataFrame:
    df = pd.read_csv(C.FEATURES_CSV, parse_dates=["date"])
    return df[df["date"] >= C.MODEL_START].reset_index(drop=True)


@lru_cache(maxsize=1)
def units() -> dict:
    p = pd.read_csv(C.PRICES_CSV, usecols=["commodity", "unit"]).drop_duplicates("commodity")
    return dict(zip(p.commodity, p.unit))


def options():
    df = feature_table()
    return {"markets": sorted(df.market.unique()), "commodities": [c for c in C.TARGET_COMMODITIES if c in set(df.commodity)],
            "latest": str(df.date.max().date())}


def available_months(market: str, commodity: str) -> list[str]:
    df = feature_table()
    d = df[(df.market == market) & (df.commodity == commodity)].date.sort_values(ascending=False)
    return [x.strftime("%Y-%m") for x in d]


def _risk(p, thr):
    if p >= max(thr, 0.5):
        return "HIGH"
    if p >= thr:
        return "ELEVATED"
    if p >= thr / 2:
        return "MODERATE"
    return "LOW"


def get_row(market: str, commodity: str, month: str | None = None) -> pd.DataFrame:
    opts = options()
    if market not in opts["markets"]:
        raise InputError(f"Unknown city '{market}'. Choose one of: {', '.join(opts['markets'])}.")
    if commodity not in opts["commodities"]:
        raise InputError(f"Unknown commodity '{commodity}'. Choose one of: {', '.join(opts['commodities'])}.")
    df = feature_table()
    rows = df[(df.market == market) & (df.commodity == commodity)]
    if month:
        try:
            ym = pd.Period(str(month)[:7], freq="M")
        except Exception:
            raise InputError(f"Invalid month '{month}'. Use the format YYYY-MM, e.g. 2026-09.")
        rows = rows[rows.date.dt.to_period("M") == ym]
        if rows.empty:
            raise InputError(f"No {commodity} price recorded for {market} in {ym}. "
                             f"Available range: {available_months(market, commodity)[-1]} to {available_months(market, commodity)[0]}.")
    else:
        rows = rows[rows.date == rows.date.max()]
    return rows.tail(1).copy()


def apply_overrides(row: pd.DataFrame, overrides: dict | None) -> pd.DataFrame:
    row = row.copy()
    for k, v in (overrides or {}).items():
        if v is None or v == "":
            continue
        if k not in OVERRIDABLE:
            raise InputError(f"'{k}' cannot be overridden. Allowed: {', '.join(OVERRIDABLE)}.")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise InputError(f"'{OVERRIDABLE[k][0]}' must be a number.")
        lo, hi = OVERRIDABLE[k][1], OVERRIDABLE[k][2]
        if not lo <= v <= hi:
            raise InputError(f"'{OVERRIDABLE[k][0]}' must be between {lo:g} and {hi:g}.")
        row[k] = v / 100.0
    return row


def predict(market: str, commodity: str, month: str | None = None, overrides: dict | None = None) -> Prediction:
    clf, reg, meta = artifacts()
    row = apply_overrides(get_row(market, commodity, month), overrides)
    p = float(clf.predict_proba(row[FEATURES])[:, 1][0])
    chg = float(reg.predict(row[FEATURES])[0])
    thr = float(meta["decision_threshold"])
    d = row.date.iloc[0]
    price = float(row.price.iloc[0])
    return Prediction(market=market, commodity=commodity, as_of=d.strftime("%Y-%m"),
                      target_month=(d.to_period("M") + 1).strftime("%Y-%m"), current_price=price,
                      unit=units().get(commodity, ""), probability=p, spike=p >= thr, decision_threshold=thr,
                      expected_change=chg, expected_price=price * (1 + chg), risk=_risk(p, thr))


def explain(market: str, commodity: str, month: str | None = None, overrides: dict | None = None, top: int = 8):
    """Local, model-agnostic explanation: how much does P(spike) change if one feature is set to its
    typical (training-median) value? Positive = this feature pushes the risk UP for this prediction."""
    clf, _, _ = artifacts()
    row = apply_overrides(get_row(market, commodity, month), overrides)
    base = float(clf.predict_proba(row[FEATURES])[:, 1][0])
    med = feature_table()[C.NUMERIC_FEATURES].median()
    out = []
    for f in C.NUMERIC_FEATURES:
        if f in ("latitude", "longitude"):
            continue
        alt = row.copy()
        alt[f] = med[f]
        out.append((f, base - float(clf.predict_proba(alt[FEATURES])[:, 1][0]), row[f].iloc[0]))
    res = pd.DataFrame(out, columns=["feature", "effect", "value"])
    res["label"] = res.feature.map(FEATURE_LABELS).fillna(res.feature)
    return res.reindex(res.effect.abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def history(market: str, commodity: str) -> pd.DataFrame:
    """Price history plus the model's spike probability at each past month (for charts)."""
    clf, _, _ = artifacts()
    df = feature_table()
    h = df[(df.market == market) & (df.commodity == commodity)].sort_values("date").copy()
    h["probability"] = clf.predict_proba(h[FEATURES])[:, 1]
    return h[["date", "price", "probability", "spike_next", "ret_next"]]


def national_board(month: str | None = None) -> pd.DataFrame:
    """P(spike) for every city x commodity at the latest (or given) month."""
    clf, _, _ = artifacts()
    df = feature_table()
    d = df.date.max() if not month else df[df.date.dt.to_period("M") == pd.Period(month, "M")].date.max()
    cur = df[df.date == d].copy()
    cur["probability"] = clf.predict_proba(cur[FEATURES])[:, 1]
    return cur.pivot_table(index="commodity", columns="market", values="probability").reindex(
        [c for c in C.TARGET_COMMODITIES if c in set(cur.commodity)])
