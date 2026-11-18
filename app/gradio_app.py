"""Kiran Bazaar - Gradio app (used for the Hugging Face Space).  Run:  python app/gradio_app.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works both inside the repo (app/../src) and inside the flattened HF Space folder (./src)
for p in (HERE / "src", HERE.parent / "src"):
    if p.exists():
        sys.path.insert(0, str(p))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from kiranbazaar import config as C  # noqa: E402
from kiranbazaar import inference as I  # noqa: E402

_, _, META = I.artifacts()
OPTS = I.options()
CHOICES = [(C.DISPLAY_NAMES.get(c, c), c) for c in OPTS["commodities"]]


def months_for(market, commodity):
    m = I.available_months(market, commodity)
    return gr.update(choices=m, value=m[0] if m else None)


def run(market, commodity, month, ret_1, diesel, fx):
    overrides = {"ret_1": (ret_1 or "").strip() or None, "diesel_ret_1": (diesel or "").strip() or None,
                 "fx_ret_1": (fx or "").strip() or None}
    try:
        p = I.predict(market, commodity, month, overrides)
        e = I.explain(market, commodity, month, overrides)
    except I.InputError as err:
        raise gr.Error(str(err))
    target = pd.Period(p.target_month).strftime("%B %Y")
    name = C.DISPLAY_NAMES.get(commodity, commodity)
    md = (f"## {name} in {market}: **{p.probability:.0%} chance** of a >{int(C.SPIKE_THRESHOLD * 100)}% jump in {target}\n"
          f"**Risk: {p.risk}** · Verdict: **{'SPIKE LIKELY' if p.spike else 'no spike expected'}** "
          f"(alert threshold {p.decision_threshold:.0%})\n\n"
          f"Price in {p.as_of}: **Rs {p.current_price:,.1f} / {p.unit}** → expected next month **Rs {p.expected_price:,.1f}** "
          f"({p.expected_change:+.1%})")
    h = I.history(market, commodity)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 2]})
    ax[0].plot(h.date, h.price, color="#2b5797", lw=1.5)
    ax[0].set_ylabel(f"Rs / {p.unit}"); ax[0].set_title("Price history and model spike probability")
    a2 = ax[0].twinx(); a2.fill_between(h.date, h.probability, color="#e8743b", alpha=0.25); a2.set_ylim(0, 1)
    a2.set_ylabel("P(spike next month)")
    ax[1].barh(e.label[::-1], e.effect[::-1] * 100, color=["#c0392b" if v > 0 else "#2b8a3e" for v in e.effect[::-1]])
    ax[1].set_xlabel("percentage-point change vs typical value"); ax[1].set_title("Why this prediction?")
    fig.tight_layout()
    return {"spike": p.probability, "no spike": 1 - p.probability}, md, fig


with gr.Blocks(title="Kiran Bazaar") as demo:
    gr.Markdown(f"# 🛒 Kiran Bazaar: food price spike forecast (Pakistan)\nPredicts whether a staple's retail price "
                f"will rise by more than {int(C.SPIKE_THRESHOLD * 100)}% next month. Model: **{META['classifier']}**, "
                f"test ROC-AUC {META['test_metrics_classifier']['roc_auc']:.2f}. Data: WFP via HDX (CC BY-IGO).")
    with gr.Row():
        with gr.Column(scale=1):
            market = gr.Dropdown(OPTS["markets"], value="Lahore", label="City")
            commodity = gr.Dropdown(CHOICES, value="Wheat flour", label="Commodity")
            m0 = I.available_months("Lahore", "Wheat flour")
            month = gr.Dropdown(m0, value=m0[0], label="Data as of month")
            gr.Markdown("**What-if overrides** (leave empty to use recorded values)")
            ret_1 = gr.Textbox(label=I.OVERRIDABLE["ret_1"][0], placeholder="-50 … 100  (blank = recorded value)")
            diesel = gr.Textbox(label=I.OVERRIDABLE["diesel_ret_1"][0], placeholder="-50 … 100  (blank = recorded value)")
            fx = gr.Textbox(label=I.OVERRIDABLE["fx_ret_1"][0], placeholder="-30 … 50  (blank = recorded value)")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column(scale=2):
            label = gr.Label(label="Probability", num_top_classes=2)
            text = gr.Markdown()
            plot = gr.Plot()
    market.change(months_for, [market, commodity], month)
    commodity.change(months_for, [market, commodity], month)
    btn.click(run, [market, commodity, month, ret_1, diesel, fx], [label, text, plot])
    demo.load(run, [market, commodity, month, ret_1, diesel, fx], [label, text, plot])

if __name__ == "__main__":
    demo.launch()
