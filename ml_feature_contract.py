"""Shared ML feature contract for training exports and runtime shadow inference."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Optional

import numpy as np

MODEL_FEATURE_SCHEMA_VERSION = "train_model.extract_features.v2"
MODEL_FACTOR_NAMES = [
    "crypto_market",
    "funding_rate",
    "us_policy",
    "geopolitics",
    "ai_chip_stocks",
    "oil_gold",
    "open_interest",
]
MODEL_EXCLUDED_FACTORS = {"twitter_social"}
MODEL_INFERENCE_BLOCKED_FACTORS = frozenset({"open_interest"})
KNOWN_REGIME_COLUMNS = (
    "regime_core_v1",
    "regime_core_v2",
    "regime_core_v3",
    "regime_legacy_unversioned",
    "regime_unknown",
)


def _safe_float(val) -> float:
    if val is None:
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _normalize_regime_bucket(regime_tag: Optional[str]) -> str:
    if not regime_tag:
        return "legacy_unversioned"
    regime = str(regime_tag)
    if "core-v1" in regime:
        return "core_v1"
    if "core-v2" in regime:
        return "core_v2"
    if "core-v3" in regime:
        return "core_v3"
    return "unknown"


def build_prediction_features(
    prediction: dict[str, Any],
    *,
    blocked_factors: Optional[Iterable[str]] = None,
) -> dict[str, Any] | None:
    """Build model features from a prediction row without training targets."""
    factor_scores = prediction.get("factor_scores", {})
    factors_unavailable = set(prediction.get("factors_unavailable") or [])
    factor_weights = prediction.get("factor_weights", {})
    blocked = set(blocked_factors or ())

    ts_str = prediction.get("timestamp")
    try:
        ts = datetime.fromisoformat(ts_str)
    except (TypeError, ValueError):
        return None

    features: dict[str, Any] = {}
    available_scores = []

    for factor_name in MODEL_FACTOR_NAMES:
        score = factor_scores.get(factor_name)
        is_blocked = factor_name in blocked
        is_unavailable = factor_name in factors_unavailable

        if not is_blocked and not is_unavailable and score is not None and factor_name not in MODEL_EXCLUDED_FACTORS:
            score_value = float(score)
            features[f"score_{factor_name}"] = score_value
            available_scores.append(score_value)
        else:
            features[f"score_{factor_name}"] = np.nan

    for factor_name in MODEL_FACTOR_NAMES:
        raw = features.get(f"score_{factor_name}", np.nan)
        features[f"abs_score_{factor_name}"] = abs(raw) if not np.isnan(raw) else np.nan

    for factor_name in MODEL_FACTOR_NAMES:
        available = 1.0
        if factor_name in blocked or factor_name in factors_unavailable:
            available = 0.0
        elif factor_scores.get(factor_name) is None:
            contribs = prediction.get("factor_contributions", {})
            if factor_name in contribs and isinstance(contribs[factor_name], dict):
                if contribs[factor_name].get("available") is False:
                    available = 0.0
            else:
                available = 0.0
        features[f"avail_{factor_name}"] = available

    for factor_name in MODEL_FACTOR_NAMES:
        if factor_name in blocked:
            features[f"weight_{factor_name}"] = 0.0
        else:
            features[f"weight_{factor_name}"] = _safe_float(factor_weights.get(factor_name))

    composite = float(prediction.get("composite_score", 0))
    features["composite_score"] = composite
    features["abs_composite"] = abs(composite)

    signal = prediction.get("signal", "NEUTRAL")
    features["pred_signal_up"] = 1.0 if signal == "UP" else 0.0
    features["pred_signal_down"] = 1.0 if signal == "DOWN" else 0.0

    conf = prediction.get("confidence", "Low Confidence")
    conf_map = {
        "Low Confidence": 0.0,
        "Slightly Likely": 1.0,
        "Moderately Likely": 2.0,
        "Likely": 3.0,
        "Very Likely": 4.0,
    }
    features["confidence_level"] = conf_map.get(conf, 0.0)

    price = float(prediction.get("price", 0))
    features["price"] = price

    horizon = float(prediction.get("horizon_hours", 1.0))
    features["horizon_hours"] = horizon
    features["is_4h"] = 1.0 if horizon >= 3.5 else 0.0

    hour = ts.hour + ts.minute / 60.0
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    features["day_of_week"] = ts.weekday()
    features["is_weekend"] = 1.0 if ts.weekday() >= 5 else 0.0

    pred_range = prediction.get("range", [0, 0])
    if isinstance(pred_range, list) and len(pred_range) == 2 and price > 0:
        range_width = pred_range[1] - pred_range[0]
        features["range_width_pct"] = (range_width / price) * 100
    else:
        features["range_width_pct"] = np.nan

    regime_bucket = _normalize_regime_bucket(prediction.get("regime_tag"))
    for column_name in KNOWN_REGIME_COLUMNS:
        features[column_name] = 1.0 if column_name == f"regime_{regime_bucket}" else 0.0

    n_avail = sum(1 for factor_name in MODEL_FACTOR_NAMES if features.get(f"avail_{factor_name}", 0) > 0.5)
    features["n_factors_available"] = float(n_avail)

    if available_scores:
        features["sentiment_mean"] = np.mean(available_scores)
        features["sentiment_std"] = np.std(available_scores)
        features["sentiment_min"] = min(available_scores)
        features["sentiment_max"] = max(available_scores)
        features["sentiment_range"] = max(available_scores) - min(available_scores)
        features["n_bearish_factors"] = sum(1.0 for score_value in available_scores if score_value < -10)
        features["n_bullish_factors"] = sum(1.0 for score_value in available_scores if score_value > 10)
        features["factor_agreement"] = (
            1.0
            if all(score_value > 0 for score_value in available_scores)
            or all(score_value < 0 for score_value in available_scores)
            else 0.0
        )
    else:
        features["sentiment_mean"] = np.nan
        features["sentiment_std"] = np.nan
        features["sentiment_min"] = np.nan
        features["sentiment_max"] = np.nan
        features["sentiment_range"] = np.nan
        features["n_bearish_factors"] = 0.0
        features["n_bullish_factors"] = 0.0
        features["factor_agreement"] = 0.0

    cm = features.get("score_crypto_market", np.nan)
    fr = features.get("score_funding_rate", np.nan)
    geo = features.get("score_geopolitics", np.nan)
    usp = features.get("score_us_policy", np.nan)

    if not np.isnan(cm) and not np.isnan(fr):
        features["crypto_x_funding"] = cm * fr / 100
    else:
        features["crypto_x_funding"] = np.nan

    if not np.isnan(geo) and not np.isnan(usp):
        features["macro_score"] = (geo + usp) / 2
    else:
        features["macro_score"] = np.nan

    return features
