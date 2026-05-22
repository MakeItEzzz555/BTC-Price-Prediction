#!/usr/bin/env python3
"""Minimal web UI for the rule-based BTC predictor."""

from __future__ import annotations

from datetime import datetime

from flask import Flask, render_template, request

from script import (
    build_accuracy_summary,
    collect_all_factors,
    compute_prediction,
    load_config,
    load_prediction_entries,
)

app = Flask(__name__)

VALID_HOURS = {1.0, 4.0}


def _web_config() -> dict:
    """Load config for the website while disabling shadow inference."""
    config = load_config()
    config["ml_shadow_model"] = None
    return config


def _serialize_prediction(prediction):
    return {
        "timestamp": prediction.timestamp,
        "signal": prediction.signal,
        "confidence": prediction.confidence,
        "composite_score": prediction.composite_score,
        "current_price": prediction.current_price,
        "price_low": prediction.price_low,
        "price_high": prediction.price_high,
        "horizon_hours": prediction.horizon_hours,
        "warnings": prediction.warnings,
        "conflicts": prediction.conflicts,
        "factors": [
            {
                "name": factor.name,
                "display_name": factor.display_name,
                "score": factor.score,
                "weight": factor.weight,
                "available": factor.available,
                "signals": factor.key_signals[:3],
            }
            for factor in sorted(prediction.factors, key=lambda item: abs(item.score), reverse=True)
        ],
    }


def _recent_prediction_rows(limit: int = 8) -> list[dict]:
    entries = load_prediction_entries()
    predictions = [entry for entry in entries if entry.get("entry_type", "prediction") == "prediction"]
    rows = []
    for entry in reversed(predictions[-limit:]):
        rows.append({
            "timestamp": entry.get("timestamp"),
            "regime_tag": entry.get("regime_tag"),
            "horizon_hours": entry.get("horizon_hours"),
            "signal": entry.get("signal"),
            "confidence": entry.get("confidence"),
            "price": entry.get("price"),
            "range": entry.get("range") or [0, 0],
        })
    return rows


@app.route("/", methods=["GET", "POST"])
def index():
    requested_hours = request.values.get("hours", "4")
    try:
        horizon_hours = float(requested_hours)
    except (TypeError, ValueError):
        horizon_hours = 4.0
    if horizon_hours not in VALID_HOURS:
        horizon_hours = 4.0

    prediction = None
    error = None
    generated_at = None

    if request.method == "POST":
        config = _web_config()
        config["_active_horizon_hours"] = horizon_hours
        try:
            factors = collect_all_factors(config)
            prediction = _serialize_prediction(compute_prediction(factors, horizon_hours, config))
            generated_at = datetime.utcnow()
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        selected_hours=horizon_hours,
        prediction=prediction,
        error=error,
        generated_at=generated_at,
        recent_predictions=_recent_prediction_rows(),
        summary_1h=build_accuracy_summary(horizon_hours=1),
        summary_4h=build_accuracy_summary(horizon_hours=4),
        shadow_disabled=True,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
