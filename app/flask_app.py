"""Kiran Bazaar - Flask backend + HTML front end.

Run:   python app/flask_app.py        then open http://localhost:5000
API:   POST /predict  {"market": "Lahore", "commodity": "Wheat flour", "month": "2026-09",
                       "overrides": {"ret_1": 3, "diesel_ret_1": 5, "fx_ret_1": 1}}   (month/overrides optional)
       GET  /api/options     GET /api/months?market=..&commodity=..     GET /health
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from kiranbazaar import config as C  # noqa: E402
from kiranbazaar import inference as I  # noqa: E402

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
I.artifacts()  # load the model once at start-up (fail fast if missing)


def error(msg, status=400):
    return jsonify({"ok": False, "error": msg}), status


@app.get("/")
def index():
    _, _, meta = I.artifacts()
    return render_template("index.html", opts=I.options(), names=C.DISPLAY_NAMES, meta=meta,
                           overridable=I.OVERRIDABLE, threshold=int(C.SPIKE_THRESHOLD * 100))


@app.get("/health")
def health():
    _, _, meta = I.artifacts()
    return jsonify({"ok": True, "model": meta["classifier"], "data_last_month": meta["data_last_month"]})


@app.get("/api/options")
def api_options():
    return jsonify({"ok": True, **I.options(), "overridable": {k: v[0] for k, v in I.OVERRIDABLE.items()}})


@app.get("/api/months")
def api_months():
    market, commodity = request.args.get("market", ""), request.args.get("commodity", "")
    if market not in I.options()["markets"] or commodity not in I.options()["commodities"]:
        return error("Unknown market or commodity.")
    return jsonify({"ok": True, "months": I.available_months(market, commodity)})


@app.post("/predict")
def predict():
    if not request.is_json:
        return error("Request body must be JSON (Content-Type: application/json).", 415)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return error("Malformed JSON body.")
    missing = [k for k in ("market", "commodity") if not body.get(k)]
    if missing:
        return error(f"Missing required field(s): {', '.join(missing)}.")
    overrides = body.get("overrides") or {}
    if not isinstance(overrides, dict):
        return error("'overrides' must be an object, e.g. {\"ret_1\": 4.5}.")
    try:
        pred = I.predict(str(body["market"]), str(body["commodity"]), body.get("month"), overrides)
        expl = I.explain(str(body["market"]), str(body["commodity"]), body.get("month"), overrides)
    except I.InputError as e:
        return error(str(e))
    hist = I.history(pred.market, pred.commodity).tail(36)
    return jsonify({
        "ok": True, "prediction": pred.to_dict(),
        "explanation": [{"feature": r.label, "effect": round(r.effect, 4)} for r in expl.itertuples()],
        "history": {"dates": hist.date.dt.strftime("%Y-%m").tolist(), "price": hist.price.round(2).tolist(),
                    "probability": hist.probability.round(4).tolist()},
    })


@app.errorhandler(404)
def not_found(_):
    return error("Not found. Endpoints: GET /, POST /predict, GET /api/options, GET /health", 404)


@app.errorhandler(Exception)
def server_error(e):
    app.logger.exception(e)
    return error("Internal server error. Check the server log.", 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
