"""Turn the raw WFP long-format price table into a monthly modelling table.

Every feature for month t uses ONLY information available at the end of
month t (current and past prices). The target looks one month ahead (t+1).
That is what makes the time-ordered split leakage-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_raw(prices_csv=C.PRICES_CSV, markets_csv=C.MARKETS_CSV):
    prices = pd.read_csv(prices_csv, parse_dates=["date"])
    markets = pd.read_csv(markets_csv)
    return prices, markets


def _monthly_wide(prices: pd.DataFrame) -> pd.DataFrame:
    """Price matrix: index = month (complete monthly range), columns = (market, commodity)."""
    p = prices.copy()
    p["ym"] = p["date"].dt.to_period("M")
    wide = p.pivot_table(index="ym", columns=["market", "commodity"], values="price", aggfunc="mean")
    full = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    return wide.reindex(full)  # missing months become NaN -> handled by imputation later


def implied_fx(prices: pd.DataFrame) -> pd.Series:
    """Monthly PKR per USD implied by WFP's own conversion (price / usdprice).

    A price-weighted ratio (sum price / sum usdprice) is used because usdprice is
    rounded to 2 decimals, which makes the ratio noisy for cheap items.
    """
    p = prices.assign(ym=prices["date"].dt.to_period("M"))
    g = p.groupby("ym")
    fx = g["price"].sum() / g["usdprice"].sum()
    full = pd.period_range(fx.index.min(), fx.index.max(), freq="M")
    return fx.reindex(full).ffill()


def build_features(prices: pd.DataFrame, markets: pd.DataFrame, threshold: float = C.SPIKE_THRESHOLD) -> pd.DataFrame:
    wide = _monthly_wide(prices)
    ret1 = wide.pct_change(fill_method=None)

    fx = implied_fx(prices).reindex(wide.index)
    fx_ret1 = fx.pct_change(fill_method=None)
    fx_ret3 = fx.pct_change(3, fill_method=None)

    cat_of = prices.drop_duplicates("commodity").set_index("commodity")["category"]
    mk = markets.set_index("market")

    # National (cross-market) aggregates per commodity, computed per month.
    nat_mean = wide.T.groupby(level="commodity").mean().T
    nat_ret1 = ret1.T.groupby(level="commodity").mean().T

    frames = []
    for market in wide.columns.get_level_values("market").unique():
        # Market-level food basket momentum: mean monthly change over all staples here
        food_cols = [(market, c) for c in C.TARGET_COMMODITIES if (market, c) in wide.columns]
        basket = ret1[food_cols].mean(axis=1)

        def series(name):
            return wide[(market, name)] if (market, name) in wide.columns else pd.Series(np.nan, index=wide.index)

        diesel, petrol, wage = series(C.DIESEL), series(C.PETROL), series(C.WAGE)

        for commodity in C.TARGET_COMMODITIES:
            if (market, commodity) not in wide.columns:
                continue
            s = wide[(market, commodity)]
            r = s.pct_change(fill_method=None)
            spike_hist = (r > threshold).astype(float).where(r.notna())
            df = pd.DataFrame({
                "ym": wide.index,
                "market": market,
                "commodity": commodity,
                "category": cat_of.get(commodity),
                "province": mk.loc[market, "admin1"] if market in mk.index else np.nan,
                "latitude": mk.loc[market, "latitude"] if market in mk.index else np.nan,
                "longitude": mk.loc[market, "longitude"] if market in mk.index else np.nan,
                "price": s.values,
                "log_price": np.log(s.values),
                "ret_1": r.values,
                "ret_2": s.pct_change(2, fill_method=None).values,
                "ret_3": s.pct_change(3, fill_method=None).values,
                "ret_6": s.pct_change(6, fill_method=None).values,
                "ret_12": s.pct_change(12, fill_method=None).values,
                "vol_6": r.rolling(6, min_periods=3).std().values,
                "ma_ratio_3": (s / s.rolling(3, min_periods=2).mean() - 1).values,
                "ma_ratio_12": (s / s.rolling(12, min_periods=6).mean() - 1).values,
                "max12_ratio": (s / s.rolling(12, min_periods=6).max() - 1).values,
                "spikes_12": spike_hist.rolling(12, min_periods=6).sum().values,
                "nat_gap": (s / nat_mean[commodity] - 1).values,
                "nat_ret_1": nat_ret1[commodity].values,
                "basket_ret_1": basket.values,
                "diesel_ret_1": diesel.pct_change(fill_method=None).values,
                "diesel_ret_3": diesel.pct_change(3, fill_method=None).values,
                "petrol_ret_1": petrol.pct_change(fill_method=None).values,
                "wage_ret_3": wage.pct_change(3, fill_method=None).values,
                "fx_ret_1": fx_ret1.values,
                "fx_ret_3": fx_ret3.values,
                "fx_level": fx.values,
                # ---- targets (one month ahead) ----
                "price_next": s.shift(-1).values,
            })
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["ret_next"] = out["price_next"] / out["price"] - 1
    out[C.TARGET_CLF] = np.where(out["ret_next"].notna(), (out["ret_next"] > threshold).astype(float), np.nan)
    out["date"] = out["ym"].dt.to_timestamp() + pd.Timedelta(days=14)  # WFP stamps the 15th
    out["month"] = out["ym"].dt.month
    out = out.drop(columns="ym")
    # Keep rows where the item was actually observed this month.
    out = out[out["price"].notna()].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["date", "market", "commodity"]).reset_index(drop=True)


def labelled(df: pd.DataFrame, start: str = C.MODEL_START) -> pd.DataFrame:
    """Rows inside the modelling window with a known next-month outcome."""
    return df[df[C.TARGET_CLF].notna() & (df["date"] >= start)].copy()


def time_split(df: pd.DataFrame, train=0.70, val=0.15):
    """Chronological 70/15/15 split on whole months (no month straddles two sets)."""
    df = df.sort_values("date")
    dates = df["date"].sort_values().values
    d1 = dates[int(len(dates) * train)]
    d2 = dates[int(len(dates) * (train + val))]
    tr = df[df["date"] < d1]
    va = df[(df["date"] >= d1) & (df["date"] < d2)]
    te = df[df["date"] >= d2]
    return tr, va, te
