"""Kiran Bazaar - Streamlit app.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from kiranbazaar import config as C  # noqa: E402
from kiranbazaar import inference as I  # noqa: E402

st.set_page_config(page_title="Kiran Bazaar - food price spike forecast", page_icon="🛒", layout="wide")
clf, reg, meta = I.artifacts()
opts = I.options()
name = lambda c: C.DISPLAY_NAMES.get(c, c)  # noqa: E731

st.title("🛒 Kiran Bazaar")
st.caption("Will a staple food's price in your city jump by more than "
           f"**{int(C.SPIKE_THRESHOLD * 100)}%** next month? Trained on WFP / Pakistan Bureau of Statistics retail "
           f"prices, 2014–{opts['latest'][:4]}. Model: **{meta['classifier']}**.")

with st.sidebar:
    st.header("Inputs")
    with st.form("inputs"):
        market = st.selectbox("City", opts["markets"], index=opts["markets"].index("Lahore"))
        commodity = st.selectbox("Commodity", opts["commodities"], format_func=name)
        months = I.available_months(market, commodity)
        month = st.selectbox("Data as of month", months, index=0,
                             help="The model uses prices up to the end of this month and forecasts the next month.")
        st.markdown("**What-if (optional)**: leave blank to use the recorded value")
        overrides = {}
        for key, (label, lo, hi) in I.OVERRIDABLE.items():
            txt = st.text_input(label, value="", placeholder=f"{lo:g} … {hi:g}")
            overrides[key] = txt.strip() or None
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

try:
    pred = I.predict(market, commodity, month, overrides)
    expl = I.explain(market, commodity, month, overrides)
except I.InputError as e:
    st.error(f"⚠️ {e}")
    st.stop()

colour = {"HIGH": "🔴", "ELEVATED": "🟠", "MODERATE": "🟡", "LOW": "🟢"}[pred.risk]
st.subheader(f"{colour} {name(commodity)} in {market}: **{pred.probability:.0%} chance** of a >"
             f"{int(C.SPIKE_THRESHOLD * 100)}% price jump in {pd.Period(pred.target_month).strftime('%B %Y')}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spike probability", f"{pred.probability:.0%}", help="Model's predicted probability")
c2.metric("Verdict", "Spike likely" if pred.spike else "No spike",
          help=f"Alert raised when probability ≥ {pred.decision_threshold:.0%} (threshold tuned on validation for best F1)")
c3.metric(f"Price in {pd.Period(pred.as_of).strftime('%b %Y')}", f"Rs {pred.current_price:,.1f} / {pred.unit}")
c4.metric("Expected next month", f"Rs {pred.expected_price:,.1f}", f"{pred.expected_change:+.1%}",
          delta_color="inverse", help=f"From the companion regression model ({meta['regressor']})")
st.progress(min(pred.probability, 1.0))

left, right = st.columns([3, 2])
with left:
    st.markdown("#### Price history and past spike warnings")
    h = I.history(market, commodity)
    base = alt.Chart(h).encode(x=alt.X("date:T", title=None))
    price = base.mark_line(color="#2b5797").encode(y=alt.Y("price:Q", title=f"Price (Rs / {pred.unit})"),
                                                  tooltip=["date:T", alt.Tooltip("price:Q", format=",.1f"),
                                                           alt.Tooltip("probability:Q", format=".0%")])
    prob = base.mark_area(color="#e8743b", opacity=0.30).encode(
        y=alt.Y("probability:Q", title="P(spike next month)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])))
    spikes = alt.Chart(h[h.spike_next == 1]).mark_point(color="crimson", size=25, filled=True).encode(
        x="date:T", y="price:Q", tooltip=["date:T", alt.Tooltip("ret_next:Q", format="+.1%", title="next-month change")])
    st.altair_chart(alt.layer(prob, price + spikes).resolve_scale(y="independent").properties(height=330),
                    use_container_width=True)
    st.caption("Blue: price. Orange area: the model's spike probability each month. Red dots: months that were actually followed by a >5% jump.")
with right:
    st.markdown("#### Why this prediction?")
    e = expl.assign(direction=lambda d: d.effect.map(lambda v: "raises risk" if v > 0 else "lowers risk"))
    chart = alt.Chart(e).mark_bar().encode(
        x=alt.X("effect:Q", title="change in probability vs a typical value", axis=alt.Axis(format="+%")),
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"], range=["#c0392b", "#2b8a3e"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["label", alt.Tooltip("value:Q", format=".3f", title="actual value"), alt.Tooltip("effect:Q", format="+.1%")])
    st.altair_chart(chart.properties(height=330), use_container_width=True)
    st.caption("Each bar: how much the probability would change if that one input were replaced by its typical (median) value.")

with st.expander("📋 National board: spike probability for every city and staple (latest month)"):
    board = I.national_board()
    board.index = [name(c) for c in board.index]
    st.dataframe(board.style.format("{:.0%}").background_gradient(cmap="OrRd", vmin=0, vmax=1), use_container_width=True)

with st.expander("ℹ️ About the model"):
    m = meta["test_metrics_classifier"]
    st.markdown(f"""
* **Data**: WFP Food Prices for Pakistan (HDX, CC BY-IGO), 5 cities × 13 staples, monthly. Fuel, wage and the
  implied USD/PKR rate are used as drivers.
* **Model**: `{meta['classifier']}` inside a scikit-learn Pipeline (imputation → scaling/one-hot → model), tuned with
  5-fold time-series cross-validation.
* **Held-out test ({meta['split']['test']['start'][:7]} to {meta['split']['test']['end'][:7]})**: ROC-AUC **{m['roc_auc']:.2f}**,
  F1 **{m['f1']:.2f}**, precision {m['precision']:.2f}, recall {m['recall']:.2f}.
* **Limitations**: only 5 large cities; shocks such as floods or policy changes (e.g. a sudden wheat support-price
  change) are not in the features; probabilities are estimates, not guarantees.
""")
