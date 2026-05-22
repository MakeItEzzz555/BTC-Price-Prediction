#!/usr/bin/env python3
"""BTC Price Prediction System — Multi-signal analysis for Bitcoin price direction."""

import os
import sys
import json
import time
import math
import pickle
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import pandas as pd

from ml_feature_contract import (
    MODEL_FEATURE_SCHEMA_VERSION,
    MODEL_INFERENCE_BLOCKED_FACTORS,
    build_prediction_features,
)

load_dotenv()
console = Console()
vader = SentimentIntensityAnalyzer()

PREDICTIONS_LOG_FILE = "predictions.log"
COINGECKO_RESOLUTION_WINDOW_SECONDS = 12 * 3600
DEFAULT_ACCURACY_HOURS = 4.0
NEUTRAL_RETURN_TOLERANCE_PCT = 0.75
DEFAULT_BACKFILL_LIMIT_PER_RUN = 10
MAX_SHORT_HORIZON_RANGE_DEVIATION_PCT = 25.0
SHORT_HORIZON_STALE_MARKET_DATA_HOURS = 30.0
MIN_NEWS_SENTIMENT_MAGNITUDE = 0.05
MAX_RESOLUTION_OBSERVED_DELTA_SECONDS = 3600
OPEN_INTEREST_CACHE_FILE = "open_interest_cache.json"
MIN_OPEN_INTEREST_BASELINE_MAX_AGE_HOURS = 2.0
ML_SHADOW_CACHE: dict[str, dict] = {}


# ─── Data Structures ────────────────────────────────────────────────────────


@dataclass
class CollectorResult:
    """Result from a single data collector."""
    name: str
    display_name: str
    score: float
    weight: float
    available: bool = True
    key_signals: list = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Final prediction output."""
    timestamp: datetime
    composite_score: float
    signal: str
    confidence: str
    current_price: float
    price_low: float
    price_high: float
    horizon_hours: float
    factors: list
    conflicts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def format_horizon_tag(horizon_hours: float) -> str:
    """Return a compact horizon tag such as 4h or 1.5h."""
    if float(horizon_hours).is_integer():
        return f"{int(horizon_hours)}h"
    return f"{horizon_hours:g}h"


def build_regime_tag(horizon_hours: float) -> str:
    """Return the current production regime tag for new prediction rows."""
    return f"{format_horizon_tag(horizon_hours)}-core-v3"


# ─── Configuration ──────────────────────────────────────────────────────────


DEFAULT_CONFIG = {
    "prediction_hours": 4,
    "accuracy_neutral_band_pct": NEUTRAL_RETURN_TOLERANCE_PCT,
    "backfill_limit_per_run": DEFAULT_BACKFILL_LIMIT_PER_RUN,
    "news_min_sentiment_magnitude": MIN_NEWS_SENTIMENT_MAGNITUDE,
    "news_vader_baseline_correction": -0.15,
    "short_horizon_stale_market_data_hours": SHORT_HORIZON_STALE_MARKET_DATA_HOURS,
    "weights": {
        "crypto_market": 30,
        "funding_rate": 20,
        "us_policy": 18,
        "geopolitics": 15,
        "ai_chip_stocks": 10,
        "oil_gold": 7,
        "open_interest": 0,
    },
    "binance_futures_symbol": "BTCUSDT",
    "stock_tickers": ["NVDA", "AMD", "TSM", "AAPL", "INTC", "MSFT", "GOOGL"],
    "commodity_tickers": {"gold": "GC=F", "oil": "CL=F"},
    "monitor_interval_minutes": 30,
    "ml_shadow_model": None,
    "news_keywords": {
        "us_policy": [
            "federal reserve", "interest rate", "crypto regulation",
            "SEC crypto", "executive order", "trade war", "tariff",
            "bitcoin ban", "crypto bill", "stablecoin",
        ],
        "geopolitics": [
            "war", "military", "invasion", "sanctions", "NATO",
            "missile", "conflict", "ceasefire", "nuclear", "troops",
        ],
    },
}
SHADOW_PROMOTION_FACTOR_NAMES = {"funding_rate", "open_interest", "exchange_flow", "vix_proxy"}

PREDICTIONS_LOG_FILE = "predictions.log"


def load_config(config_path: str = "config.json") -> dict:
    """Load config from file, falling back to defaults."""
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path) as f:
            user_config = json.load(f)
        if "weights" in user_config:
            config["weights"].update(user_config["weights"])
            user_config_copy = {k: v for k, v in user_config.items() if k != "weights"}
            config.update(user_config_copy)
        else:
            config.update(user_config)
    config["weights"] = {
        key: value for key, value in config["weights"].items()
        if key in DEFAULT_CONFIG["weights"]
    }
    return config


def normalize_weights(weights: dict) -> dict:
    """Ensure weights sum to 100, normalizing if needed."""
    total = sum(weights.values())
    live_shadow_factors = [
        factor_name for factor_name in SHADOW_PROMOTION_FACTOR_NAMES
        if float(weights.get(factor_name, 0) or 0) > 0
    ]
    if live_shadow_factors and abs(total - 100) > 0.01:
        raise ValueError(
            "Promoting a shadow factor requires manual rebalance; weights must already sum to 100. "
            f"Live shadow factor(s): {', '.join(sorted(live_shadow_factors))}"
        )
    if abs(total - 100) > 0.01:
        console.print(f"[yellow]Warning: weights sum to {total}, normalizing to 100%[/yellow]")
        factor = 100.0 / total
        weights = {k: round(v * factor, 1) for k, v in weights.items()}
    return weights


def validate_shadow_factor_weights(weights: dict):
    """Allow at most one promoted shadow factor at a time."""
    live_shadow_factors = [
        factor_name
        for factor_name in SHADOW_PROMOTION_FACTOR_NAMES
        if float(weights.get(factor_name, 0) or 0) > 0
    ]
    if len(live_shadow_factors) > 1:
        raise ValueError(
            "Only one shadow factor may have a live weight at a time: "
            + ", ".join(sorted(live_shadow_factors))
        )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def active_horizon_hours(config: dict) -> float:
    """Return the currently requested prediction horizon in hours."""
    try:
        horizon_hours = float(config.get("_active_horizon_hours", config.get("prediction_hours", 4)))
    except (TypeError, ValueError):
        return 4.0
    return horizon_hours if horizon_hours > 0 else 4.0


def news_sentiment_floor(config: dict) -> float:
    """Return the minimum absolute sentiment required for a news article to count."""
    try:
        floor = float(config.get("news_min_sentiment_magnitude", MIN_NEWS_SENTIMENT_MAGNITUDE))
    except (TypeError, ValueError):
        return MIN_NEWS_SENTIMENT_MAGNITUDE
    if floor < 0:
        return MIN_NEWS_SENTIMENT_MAGNITUDE
    return floor


def news_vader_baseline_correction(config: dict) -> float:
    """Return the configured VADER baseline correction."""
    try:
        return float(config.get("news_vader_baseline_correction", -0.15))
    except (TypeError, ValueError):
        return -0.15


def stale_market_data_max_age(config: dict) -> float:
    """Return the configured stale-data cutoff for short-horizon daily closes."""
    try:
        max_age = float(config.get("short_horizon_stale_market_data_hours", SHORT_HORIZON_STALE_MARKET_DATA_HOURS))
    except (TypeError, ValueError):
        return SHORT_HORIZON_STALE_MARKET_DATA_HOURS
    if max_age <= 0:
        return SHORT_HORIZON_STALE_MARKET_DATA_HOURS
    return max_age


def open_interest_baseline_max_age_hours(config: dict) -> float:
    """Return the maximum age for an open-interest baseline snapshot."""
    return max(MIN_OPEN_INTEREST_BASELINE_MAX_AGE_HOURS, active_horizon_hours(config) / 2.0)


def _normalize_index_timestamp(index_value) -> Optional[datetime]:
    """Convert a pandas-like index value into a timezone-aware UTC timestamp."""
    to_pydatetime = getattr(index_value, "to_pydatetime", None)
    if callable(to_pydatetime):
        index_value = to_pydatetime()

    if isinstance(index_value, datetime):
        if index_value.tzinfo is None:
            return datetime(index_value.year, index_value.month, index_value.day, 21, tzinfo=timezone.utc)
        return index_value.astimezone(timezone.utc)

    date_method = getattr(index_value, "date", None)
    if callable(date_method):
        date_value = date_method()
        return datetime(date_value.year, date_value.month, date_value.day, 21, tzinfo=timezone.utc)
    return None


def market_data_staleness_hours(index, now: Optional[datetime] = None) -> Optional[float]:
    """Return the age in hours of the most recent daily market datapoint."""
    if index is None:
        return None
    try:
        if len(index) == 0:
            return None
        latest_value = index[-1]
    except Exception:
        return None

    latest_timestamp = _normalize_index_timestamp(latest_value)
    if latest_timestamp is None:
        return None
    now = now or datetime.now(timezone.utc)
    age_hours = (now - latest_timestamp).total_seconds() / 3600.0
    return max(age_hours, 0.0)


def short_horizon_market_data_is_stale(index, horizon_hours: float,
                                       max_age_hours: float = SHORT_HORIZON_STALE_MARKET_DATA_HOURS) -> bool:
    """Return True when daily market closes are too stale to help a short-horizon forecast."""
    if horizon_hours > 6:
        return False
    age_hours = market_data_staleness_hours(index)
    return age_hours is None or age_hours > max_age_hours


def classify_prediction_signal(composite: float) -> tuple[str, str]:
    """Map a composite score to signal and confidence labels."""
    if composite >= 60:
        return "UP", "Very Likely"
    if composite >= 30:
        return "UP", "Likely"
    if composite >= 15:
        return "UP", "Slightly Likely"
    if composite > -15:
        return "NEUTRAL", "Low Confidence"
    if composite > -30:
        return "DOWN", "Slightly Likely"
    if composite > -60:
        return "DOWN", "Likely"
    return "DOWN", "Very Likely"


def ml_shadow_decision_contract() -> dict:
    """Return the rule-system contract that shadow artifacts must record."""
    return {
        "neutral_return_tolerance_pct": NEUTRAL_RETURN_TOLERANCE_PCT,
        "composite_signal_thresholds": [
            {"min_inclusive": 60.0, "signal": "UP", "confidence": "Very Likely"},
            {"min_inclusive": 30.0, "signal": "UP", "confidence": "Likely"},
            {"min_inclusive": 15.0, "signal": "UP", "confidence": "Slightly Likely"},
            {"min_exclusive": -15.0, "signal": "NEUTRAL", "confidence": "Low Confidence"},
            {"min_exclusive": -30.0, "signal": "DOWN", "confidence": "Slightly Likely"},
            {"min_exclusive": -60.0, "signal": "DOWN", "confidence": "Likely"},
            {"else_signal": "DOWN", "else_confidence": "Very Likely"},
        ],
    }

# ─── Collectors ─────────────────────────────────────────────────────────────


def _coingecko_get(url: str, params: dict, timeout: int = 15, max_retries: int = 3) -> requests.Response:
    """GET with retry+backoff for CoinGecko 429 rate limits."""
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        if attempt < max_retries - 1:
            wait = 2 ** attempt  # 1s, 2s
            console.print(f"[yellow]CoinGecko 429, retrying in {wait}s ({attempt + 1}/{max_retries})[/yellow]")
            time.sleep(wait)
    r.raise_for_status()  # raise on final 429
    return r  # unreachable, but keeps type checkers happy


def collect_crypto_market(config: dict) -> CollectorResult:
    """Fetch BTC price, volume, trend, and Fear & Greed Index."""
    name = "crypto_market"
    display = "Crypto Market Data"
    weight = config["weights"].get(name, 30)
    signals = []
    raw = {}

    try:
        r = _coingecko_get(
            "https://api.coingecko.com/api/v3/coins/bitcoin",
            params={"localization": "false", "tickers": "false",
                    "community_data": "false", "developer_data": "false"},
        )
        data = r.json()
        market = data["market_data"]

        price = market["current_price"]["usd"]
        change_24h = market.get("price_change_percentage_24h") or 0
        volume = market["total_volume"]["usd"]
        raw["price"] = price
        raw["change_24h_pct"] = change_24h
        raw["volume_usd"] = volume

        # Hourly price history for trend + volatility
        r2 = _coingecko_get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": "1"},
        )
        prices_hist = r2.json().get("prices", [])

        hourly_returns = []
        if len(prices_hist) >= 2:
            for i in range(1, len(prices_hist)):
                prev_price = prices_hist[i - 1][1]
                curr_price = prices_hist[i][1]
                if prev_price > 0:
                    hourly_returns.append((curr_price - prev_price) / prev_price * 100)

        hourly_trend = hourly_returns[-1] if hourly_returns else 0
        raw["hourly_trend_pct"] = hourly_trend
        raw["hourly_returns"] = hourly_returns
        raw["prices_history"] = prices_hist

        # Fear & Greed Index
        r3 = requests.get("https://api.alternative.me/fng/?limit=2", timeout=10)
        r3.raise_for_status()
        fng_data = r3.json().get("data", [])
        fng_today = int(fng_data[0]["value"]) if len(fng_data) > 0 else 50
        fng_yesterday = int(fng_data[1]["value"]) if len(fng_data) > 1 else fng_today
        fng_label = fng_data[0].get("value_classification", "") if fng_data else ""
        raw["fear_greed"] = fng_today
        raw["fear_greed_yesterday"] = fng_yesterday
        raw["fear_greed_label"] = fng_label

        # ── Scoring (piecewise-linear, no dead zones) ──
        score_price = clamp(change_24h * 15, -30, 30)

        # Volume heuristic: higher 24h change with volume = confirmation
        vol_change_est = abs(change_24h) * 5 if change_24h > 0 else abs(change_24h) * -3
        vol_score = clamp(vol_change_est, -15, 20)

        score_trend = clamp(hourly_trend * 10, -15, 15)

        # Fear & Greed with contrarian flip
        if fng_today > 80:
            score_fng = -10
        elif fng_today >= 75:
            score_fng = 15
        elif fng_today >= 50:
            score_fng = 5 + (fng_today - 50) * 0.4
        elif fng_today >= 25:
            score_fng = -10 + (fng_today - 25) * 0.6
        else:
            score_fng = -20 + fng_today * 0.4

        fng_direction = clamp((fng_today - fng_yesterday) * 0.5, -10, 10)

        pre_flip_total = score_price + vol_score + score_trend + score_fng + fng_direction
        total_score = clamp(-pre_flip_total, -100, 100)
        raw["pre_flip_total"] = round(pre_flip_total, 4)
        raw["post_flip_total"] = round(total_score, 4)

        signals.append(f"Fear & Greed Index: {fng_today} ({fng_label})")
        if abs(change_24h) > 0.5:
            direction = "up" if change_24h > 0 else "down"
            signals.append(f"BTC 24h: {change_24h:+.1f}% ({direction})")

        return CollectorResult(name=name, display_name=display, score=total_score,
                               weight=weight, key_signals=signals, raw_data=raw)

    except Exception as e:
        console.print(f"[red]Crypto collector error: {e}[/red]")
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False,
                               key_signals=[f"Error: {e}"])

def collect_funding_rate(config: dict) -> CollectorResult:
    """Fetch recent Binance BTCUSDT funding rates for shadow evaluation."""
    name = "funding_rate"
    display = "Binance Funding Rate"
    weight = config["weights"].get(name, 0)
    raw = {}
    symbol = config.get("binance_futures_symbol", "BTCUSDT")

    try:
        response = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 3},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise ValueError("No funding rate data returned")

        latest = payload[-1]
        latest_rate = float(latest["fundingRate"])
        previous_rate = float(payload[-2]["fundingRate"]) if len(payload) >= 2 else latest_rate
        rate_delta = latest_rate - previous_rate
        latest_mark_price = float(latest.get("markPrice") or 0.0)
        funding_time = latest.get("fundingTime")

        # Positive funding implies crowded longs; treat extreme positive rates as bearish pressure.
        score = clamp(-latest_rate * 100000, -100, 100)
        raw.update({
            "symbol": symbol,
            "latest_funding_rate": latest_rate,
            "previous_funding_rate": previous_rate,
            "funding_rate_delta": rate_delta,
            "mark_price": latest_mark_price,
            "funding_time": funding_time,
            "source": "binance_fapi_fundingRate",
            "shadow_mode": weight == 0,
        })

        signals = [f"{symbol} funding {latest_rate:+.5%}"]
        if abs(rate_delta) > 0:
            signals.append(f"Funding delta {rate_delta:+.5%} vs previous print")
        if weight == 0:
            signals.append("Shadow factor only; not affecting live prediction")

        return CollectorResult(name=name, display_name=display, score=score,
                               weight=weight, key_signals=signals, raw_data=raw)
    except Exception as e:
        console.print(f"[red]Funding rate collector error: {e}[/red]")
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False, key_signals=[f"Error: {e}"])


def collect_open_interest(config: dict) -> CollectorResult:
    """Fetch Binance BTCUSDT open interest and score it against the prior local snapshot."""
    name = "open_interest"
    display = "Binance Open Interest"
    weight = config["weights"].get(name, 0)
    raw = {}
    symbol = config.get("binance_futures_symbol", "BTCUSDT")
    horizon_hours = active_horizon_hours(config)

    try:
        response = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        open_interest = float(payload["openInterest"])
        observed_time_ms = int(payload["time"])
        observed_at = datetime.fromtimestamp(observed_time_ms / 1000, tz=timezone.utc)

        cache = _load_open_interest_cache()
        cache_key = f"{symbol}"
        previous_snapshot = cache.get(cache_key, {})
        previous_open_interest = previous_snapshot.get("open_interest")
        previous_observed_at = previous_snapshot.get("observed_at")
        oi_change_pct = 0.0
        baseline_captured = previous_open_interest is None
        stale_baseline = False
        snapshot_age_hours = None
        if previous_observed_at:
            try:
                previous_observed_dt = datetime.fromisoformat(previous_observed_at)
                snapshot_age_hours = (observed_at - previous_observed_dt).total_seconds() / 3600.0
                stale_baseline = snapshot_age_hours > open_interest_baseline_max_age_hours(config)
            except ValueError:
                stale_baseline = True
        if previous_open_interest is not None and float(previous_open_interest) > 0:
            if stale_baseline:
                baseline_captured = True
            else:
                oi_change_pct = ((open_interest - float(previous_open_interest)) / float(previous_open_interest)) * 100.0

        cache[cache_key] = {
            "open_interest": open_interest,
            "observed_at": observed_at.isoformat(),
        }
        _save_open_interest_cache(cache)

        score = clamp(oi_change_pct * 2.0, -100, 100)
        raw.update({
            "symbol": symbol,
            "open_interest": open_interest,
            "previous_open_interest": previous_open_interest,
            "previous_observed_at": previous_observed_at,
            "open_interest_change_pct": oi_change_pct,
            "observed_at": observed_at.isoformat(),
            "baseline_age_hours": round(snapshot_age_hours, 2) if snapshot_age_hours is not None else None,
            "baseline_stale": stale_baseline,
            "directional_assumption": "rising_open_interest_treated_as_bullish_in_shadow_mode_without_price_context",
            "horizon_hours": horizon_hours,
            "source": "binance_fapi_openInterest",
            "shadow_mode": weight == 0,
        })

        signals = [f"{symbol} OI {open_interest:,.0f}"]
        if baseline_captured:
            if stale_baseline:
                signals.append("Stale OI baseline reset; score held at 0 until a fresh comparison is available")
            else:
                signals.append("Baseline snapshot captured for future OI change scoring")
        else:
            signals.append(f"OI change {oi_change_pct:+.2f}% vs previous snapshot")
        signals.append("Shadow OI assumes rising OI is bullish until price-context validation proves otherwise")
        if weight == 0:
            signals.append("Shadow factor only; not affecting live prediction")

        return CollectorResult(name=name, display_name=display, score=score,
                               weight=weight, key_signals=signals, raw_data=raw)
    except Exception as e:
        console.print(f"[red]Open interest collector error: {e}[/red]")
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False, key_signals=[f"Error: {e}"])


# ─── News Cache & Collectors ───────────────────────────────────────────────


NEWS_CACHE_FILE = "news_cache.json"
NEWS_DAILY_LIMIT = 90


def _load_news_cache() -> dict:
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"queries": {}, "daily_count": 0, "daily_reset": ""}


def _save_news_cache(cache: dict):
    with open(NEWS_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _load_open_interest_cache(cache_path: str = OPEN_INTEREST_CACHE_FILE) -> dict:
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_open_interest_cache(cache: dict, cache_path: str = OPEN_INTEREST_CACHE_FILE):
    Path(cache_path).write_text(json.dumps(cache))


def _fetch_news(query: str, api_key: str) -> list:
    """Fetch news from NewsAPI with file-based caching and daily rate limit tracking."""
    cache = _load_news_cache()
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    if cache.get("daily_reset") != today_str:
        cache["daily_count"] = 0
        cache["daily_reset"] = today_str

    cache_key = query[:50]
    cached = cache.get("queries", {}).get(cache_key, {})

    # Check cache freshness (60 min TTL)
    if cached:
        cached_time = cached.get("timestamp", "")
        if cached_time:
            try:
                cache_dt = datetime.fromisoformat(cached_time)
                if (now - cache_dt).total_seconds() < 3600:
                    return cached.get("articles", [])
            except (ValueError, TypeError):
                pass

    # Check daily limit
    if cache["daily_count"] >= NEWS_DAILY_LIMIT:
        console.print("[yellow]NewsAPI daily limit approached, using cached data[/yellow]")
        return cached.get("articles", [])

    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 20},
            headers={"X-Api-Key": api_key},
            timeout=15,
        )
        r.raise_for_status()
        articles = r.json().get("articles", [])

        cache["daily_count"] = cache.get("daily_count", 0) + 1
        cache.setdefault("queries", {})[cache_key] = {
            "timestamp": now.isoformat(),
            "articles": articles,
        }
        _save_news_cache(cache)
        return articles

    except Exception as e:
        console.print(f"[yellow]NewsAPI error: {e}, using cached data[/yellow]")
        return cached.get("articles", [])


def _score_news_articles(articles: list, positive_kw: list, negative_kw: list,
                         min_sentiment_magnitude: float = MIN_NEWS_SENTIMENT_MAGNITUDE,
                         baseline_correction: float = -0.15) -> tuple:
    """Score articles with VADER + keyword boosters + recency. Returns (score, signals, raw)."""
    now = datetime.now(timezone.utc)
    weighted_sum = 0.0
    total_weight = 0.0
    signals = []
    kept_articles = 0
    filtered_low_signal = 0
    raw_sentiment_sum = 0.0
    corrected_sentiment_sum = 0.0
    final_sentiment_sum = 0.0
    sentiment_samples = 0

    for article in articles[:20]:
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        text = f"{title}. {description}"
        published = article.get("publishedAt", "")

        raw_sentiment = vader.polarity_scores(text)["compound"]
        corrected_sentiment = raw_sentiment - baseline_correction
        sentiment = corrected_sentiment

        text_lower = text.lower()
        for kw in positive_kw:
            if kw in text_lower:
                sentiment += 0.3
                break
        for kw in negative_kw:
            if kw in text_lower:
                sentiment -= 0.3
                break
        sentiment = clamp(sentiment, -1.0, 1.0)
        raw_sentiment_sum += raw_sentiment
        corrected_sentiment_sum += corrected_sentiment
        final_sentiment_sum += sentiment
        sentiment_samples += 1
        if abs(sentiment) < min_sentiment_magnitude:
            filtered_low_signal += 1
            continue

        recency = 1.0
        if published:
            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                age_hours = (now - pub_dt).total_seconds() / 3600
                if age_hours < 2:
                    recency = 2.0
                elif age_hours < 6:
                    recency = 1.5
            except (ValueError, TypeError):
                pass

        weighted_sum += sentiment * recency
        total_weight += recency
        kept_articles += 1

        if abs(sentiment) > 0.4:
            direction = "+" if sentiment > 0 else "-"
            signals.append(f"[{direction}] {title[:70]}")

    if total_weight > 0:
        avg = weighted_sum / total_weight
        score = clamp(avg * 100, -100, 100)
    else:
        avg = 0
        score = 0

    return score, signals[:3], {
        "articles_analyzed": len(articles),
        "kept_articles": kept_articles,
        "filtered_low_signal_articles": filtered_low_signal,
        "avg_sentiment": avg,
        "avg_raw_sentiment": (raw_sentiment_sum / sentiment_samples) if sentiment_samples else 0.0,
        "avg_baseline_corrected_sentiment": (corrected_sentiment_sum / sentiment_samples) if sentiment_samples else 0.0,
        "avg_final_article_sentiment": (final_sentiment_sum / sentiment_samples) if sentiment_samples else 0.0,
        "baseline_correction": baseline_correction,
        "min_sentiment_magnitude": min_sentiment_magnitude,
    }


def collect_us_policy(config: dict) -> CollectorResult:
    """Fetch and score U.S. policy news."""
    name = "us_policy"
    display = "U.S. Policy & Regulation"
    weight = config["weights"].get(name, 18)

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False,
                               key_signals=["NEWS_API_KEY not set in .env"])
    try:
        keywords = config.get("news_keywords", {}).get("us_policy", [])
        query = " OR ".join(f'"{kw}"' for kw in keywords[:5])
        articles = _fetch_news(query, api_key)

        positive_kw = ["rate cut", "approve", "pro-crypto", "easing", "support", "adopt"]
        negative_kw = ["rate hike", "ban", "crackdown", "restrict", "tighten", "lawsuit"]

        score, signals, raw = _score_news_articles(
            articles,
            positive_kw,
            negative_kw,
            min_sentiment_magnitude=news_sentiment_floor(config),
            baseline_correction=news_vader_baseline_correction(config),
        )

        return CollectorResult(name=name, display_name=display, score=score,
                               weight=weight, key_signals=signals or ["No notable policy news"],
                               raw_data=raw)
    except Exception as e:
        console.print(f"[red]U.S. Policy collector error: {e}[/red]")
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False, key_signals=[f"Error: {e}"])


def collect_geopolitics(config: dict) -> CollectorResult:
    """Fetch and score geopolitics/war news."""
    name = "geopolitics"
    display = "Geopolitics & War"
    weight = config["weights"].get(name, 15)

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False,
                               key_signals=["NEWS_API_KEY not set in .env"])
    try:
        keywords = config.get("news_keywords", {}).get("geopolitics", [])
        query = " OR ".join(f'"{kw}"' for kw in keywords[:5])
        articles = _fetch_news(query, api_key)

        positive_kw = ["ceasefire", "peace talks", "withdrawal", "agreement", "treaty", "de-escalation"]
        negative_kw = ["attack", "troops deployed", "nuclear", "escalation", "invasion", "strike", "bombing"]

        score, signals, raw = _score_news_articles(
            articles,
            positive_kw,
            negative_kw,
            min_sentiment_magnitude=news_sentiment_floor(config),
            baseline_correction=news_vader_baseline_correction(config),
        )

        return CollectorResult(name=name, display_name=display, score=score,
                               weight=weight, key_signals=signals or ["No notable geopolitical news"],
                               raw_data=raw)
    except Exception as e:
        console.print(f"[red]Geopolitics collector error: {e}[/red]")
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False, key_signals=[f"Error: {e}"])


def collect_ai_chip_stocks(config: dict) -> CollectorResult:
    """Fetch AI/chip stock prices and score sector health."""
    name = "ai_chip_stocks"
    display = "AI & Chip Stocks"
    weight = config["weights"].get(name, 10)
    signals = []
    raw = {}
    horizon_hours = active_horizon_hours(config)

    try:
        tickers = config.get("stock_tickers", ["NVDA", "AMD", "TSM", "AAPL", "INTC", "MSFT", "GOOGL"])
        ticker_str = " ".join(tickers)

        data = yf.download(ticker_str, period="2d", interval="1d", progress=False, threads=True)
        if data.empty:
            raise ValueError("No stock data returned from yfinance")
        age_hours = market_data_staleness_hours(data.index)
        raw["data_age_hours"] = round(age_hours, 2) if age_hours is not None else None
        if short_horizon_market_data_is_stale(
            data.index,
            horizon_hours,
            max_age_hours=stale_market_data_max_age(config),
        ):
            return CollectorResult(
                name=name,
                display_name=display,
                score=0,
                weight=weight,
                available=False,
                key_signals=[f"Daily stock closes stale for {horizon_hours:.1f}h horizon"],
                raw_data=raw,
            )

        changes = {}
        close = data["Close"]
        if len(close) >= 2:
            for ticker in tickers:
                if ticker in close.columns:
                    prev = close[ticker].iloc[-2]
                    curr = close[ticker].iloc[-1]
                    try:
                        prev_val = float(prev)
                        curr_val = float(curr)
                        if prev_val > 0 and not (math.isnan(prev_val) or math.isnan(curr_val)):
                            pct = ((curr_val - prev_val) / prev_val) * 100
                            changes[ticker] = round(pct, 2)
                    except (ValueError, TypeError):
                        pass

        raw["changes"] = changes

        if not changes:
            return CollectorResult(
                name=name,
                display_name=display,
                score=0,
                weight=weight,
                available=False,
                key_signals=[f"No fresh stock changes available for {horizon_hours:.1f}h horizon"],
                raw_data=raw,
            )

        avg_change = sum(changes.values()) / len(changes)

        raw["avg_change"] = round(avg_change, 2)

        centered_avg_change = avg_change - 0.75
        raw["centered_avg_change"] = round(centered_avg_change, 4)
        score = clamp(centered_avg_change * 20.0, -60, 60)

        for ticker, pct in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
            arrow = "+" if pct > 0 else ""
            signals.append(f"{ticker} {arrow}{pct:.1f}%")
        direction = "bullish" if avg_change > 0.3 else "bearish" if avg_change < -0.3 else "flat"
        signals.insert(0, f"Sector avg: {avg_change:+.1f}% ({direction})")

        return CollectorResult(name=name, display_name=display, score=score,
                               weight=weight, key_signals=signals, raw_data=raw)

    except Exception as e:
        console.print(f"[red]Stocks collector error: {e}[/red]")
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False, key_signals=[f"Error: {e}"])


def collect_oil_gold(config: dict) -> CollectorResult:
    """Fetch oil and gold prices and score commodity signals."""
    name = "oil_gold"
    display = "Oil & Gold"
    weight = config["weights"].get(name, 7)
    signals = []
    raw = {}
    horizon_hours = active_horizon_hours(config)

    try:
        commodities = config.get("commodity_tickers", {"gold": "GC=F", "oil": "CL=F"})
        ticker_str = " ".join(commodities.values())

        data = yf.download(ticker_str, period="2d", interval="1d", progress=False, threads=True)
        if data.empty:
            raise ValueError("No commodity data returned")
        age_hours = market_data_staleness_hours(data.index)
        raw["data_age_hours"] = round(age_hours, 2) if age_hours is not None else None
        if short_horizon_market_data_is_stale(
            data.index,
            horizon_hours,
            max_age_hours=stale_market_data_max_age(config),
        ):
            return CollectorResult(
                name=name,
                display_name=display,
                score=0,
                weight=weight,
                available=False,
                key_signals=[f"Daily commodity closes stale for {horizon_hours:.1f}h horizon"],
                raw_data=raw,
            )

        close = data["Close"]
        changes = {}
        for label, ticker in commodities.items():
            if ticker in close.columns and len(close) >= 2:
                try:
                    prev = float(close[ticker].iloc[-2])
                    curr = float(close[ticker].iloc[-1])
                    if prev > 0 and not (math.isnan(prev) or math.isnan(curr)):
                        changes[label] = ((curr - prev) / prev) * 100
                except (ValueError, TypeError):
                    pass

        raw["changes"] = changes
        if not changes:
            return CollectorResult(
                name=name,
                display_name=display,
                score=0,
                weight=weight,
                available=False,
                key_signals=[f"No fresh commodity changes available for {horizon_hours:.1f}h horizon"],
                raw_data=raw,
            )
        gold_chg = changes.get("gold", 0)
        oil_chg = changes.get("oil", 0)

        gold_component = -gold_chg * 8
        oil_component = -oil_chg * 4
        raw_score = gold_component + oil_component
        raw["gold_component"] = round(gold_component, 2)
        raw["oil_component"] = round(oil_component, 2)
        raw["raw_score_before_deadband"] = round(raw_score, 4)
        if abs(raw_score) < 8:
            score = 0
        else:
            score = clamp(raw_score, -40, 40)

        gold_note = " (safe-haven demand)" if gold_chg > 1 else ""
        oil_note = " (instability signal)" if oil_chg > 3 else ""
        signals.append(f"Gold {gold_chg:+.1f}%{gold_note}")
        signals.append(f"Oil {oil_chg:+.1f}%{oil_note}")

        return CollectorResult(name=name, display_name=display, score=score,
                               weight=weight, key_signals=signals, raw_data=raw)

    except Exception as e:
        console.print(f"[red]Commodities collector error: {e}[/red]")
        return CollectorResult(name=name, display_name=display, score=0,
                               weight=weight, available=False, key_signals=[f"Error: {e}"])


# ─── Prediction Engine ──────────────────────────────────────────────────────


def compute_prediction(factors: list, horizon_hours: float, config: dict) -> PredictionResult:
    """Compute weighted composite score, signal, and price range."""
    now = datetime.now(timezone.utc)
    warnings_list = []

    if horizon_hours < 0.5:
        warnings_list.append("Low reliability — time horizon below 30-minute model minimum")

    available = [f for f in factors if f.available]
    unavailable = [f for f in factors if not f.available]

    if not available:
        return PredictionResult(
            timestamp=now, composite_score=0, signal="NEUTRAL",
            confidence="Low Confidence", current_price=0,
            price_low=0, price_high=0, horizon_hours=horizon_hours,
            factors=factors, warnings=["All collectors failed"])

    # Redistribute weights from failed collectors
    total_available_weight = sum(f.weight for f in available)
    if total_available_weight > 0 and abs(total_available_weight - 100) > 0.01:
        redistribute = 100.0 / total_available_weight
        for f in available:
            f.weight = round(f.weight * redistribute, 1)
    for f in unavailable:
        f.weight = 0

    composite = sum(f.score * (f.weight / 100.0) for f in available)
    composite = clamp(composite, -100, 100)

    signal, confidence = classify_prediction_signal(composite)

    # Conflict detection (only factors >= 15% weight)
    conflicts = []
    significant = [f for f in available if f.weight >= 15]
    if len(significant) >= 2:
        scores = [(f.display_name, f.score) for f in significant]
        max_f = max(scores, key=lambda x: x[1])
        min_f = min(scores, key=lambda x: x[1])
        if max_f[1] - min_f[1] > 80:
            conflicts.append(f"{min_f[0]} bearish vs {max_f[0]} bullish")
            conf_levels = ["Very Likely", "Likely", "Slightly Likely", "Low Confidence"]
            idx = conf_levels.index(confidence) if confidence in conf_levels else 3
            confidence = conf_levels[min(idx + 1, 3)]

    # Price range
    crypto_result = next((f for f in factors if f.name == "crypto_market"), None)
    current_price = crypto_result.raw_data.get("price", 0) if crypto_result and crypto_result.available else 0

    price_low, price_high = 0, 0
    if current_price > 0:
        hourly_returns = []
        if crypto_result and crypto_result.raw_data.get("hourly_returns"):
            hourly_returns = crypto_result.raw_data["hourly_returns"]

        if len(hourly_returns) >= 2:
            mean_ret = sum(hourly_returns) / len(hourly_returns)
            variance = sum((r - mean_ret) ** 2 for r in hourly_returns) / len(hourly_returns)
            volatility = math.sqrt(variance) / 100
        else:
            volatility = 0.005

        time_factor = math.sqrt(max(horizon_hours, 0.1))
        expected_move = (composite / 100) * volatility * time_factor
        range_width = volatility * time_factor * 1.5

        price_low = current_price * (1 + expected_move - range_width / 2)
        price_high = current_price * (1 + expected_move + range_width / 2)
    elif crypto_result and not crypto_result.available:
        warnings_list.append("BTC spot price unavailable; price range could not be computed")

    if unavailable:
        warnings_list.append(f"Unavailable: {', '.join(f.display_name for f in unavailable)}")

    return PredictionResult(
        timestamp=now, composite_score=round(composite, 1),
        signal=signal, confidence=confidence,
        current_price=current_price,
        price_low=round(price_low, 2), price_high=round(price_high, 2),
        horizon_hours=horizon_hours, factors=factors,
        conflicts=conflicts, warnings=warnings_list,
    )


# ─── Report Generator ───────────────────────────────────────────────────────


def generate_report(prediction: PredictionResult, verbose: bool = False,
                    prev_prediction: Optional[PredictionResult] = None):
    """Print a Rich-formatted prediction report to terminal."""
    p = prediction
    console.clear()

    horizon_str = f"{p.horizon_hours:.1f}h" if p.horizon_hours >= 1 else f"{int(p.horizon_hours * 60)}m"
    ts = p.timestamp.strftime("%Y-%m-%d %H:%M UTC")

    color = "green" if p.signal == "UP" else "red" if p.signal == "DOWN" else "yellow"
    arrow = "▲" if p.signal == "UP" else "▼" if p.signal == "DOWN" else "─"

    # Header
    header = Text()
    header.append("BTC PRICE PREDICTION REPORT\n", style="bold white")
    header.append(f"{ts} | {horizon_str} Horizon", style="cyan")
    console.print(Panel(header, style="bold blue", title="[bold]BTC Predictor[/bold]"))

    # Signal
    signal_text = Text()
    signal_text.append(f"  SIGNAL:  {arrow} {p.signal}", style=f"bold {color}")
    signal_text.append(f"        Confidence: {p.confidence} ({p.composite_score:+.0f}/100)\n", style=color)
    current_price_text = f"${p.current_price:,.0f}" if p.current_price > 0 else "N/A"
    signal_text.append(f"\n  Current Price:   {current_price_text}\n", style="cyan")
    if p.price_low > 0 and p.price_high > 0:
        signal_text.append(
            f"  Predicted Range: ${p.price_low:,.0f} — ${p.price_high:,.0f}  ({horizon_str})\n",
            style="white",
        )
    console.print(Panel(signal_text, title="[bold]Prediction[/bold]"))

    # Factor breakdown table
    table = Table(title="Factor Breakdown", show_header=True, header_style="bold")
    table.add_column("", width=3)
    table.add_column("Factor", min_width=22)
    table.add_column("Score Bar", min_width=12)
    table.add_column("Score", justify="right", min_width=6)
    table.add_column("Weight", justify="right", min_width=6)

    for f in sorted(p.factors, key=lambda x: x.weight, reverse=True):
        if not f.available:
            f_arrow, f_color = "✗", "dim"
            bar = "[dim]unavailable[/dim]"
            score_str = "N/A"
        else:
            f_arrow = "▲" if f.score > 10 else "▼" if f.score < -10 else "─"
            f_color = "green" if f.score > 10 else "red" if f.score < -10 else "yellow"
            filled = min(int(abs(f.score) / 10), 10)
            empty = 10 - filled
            bar = f"[{f_color}]{'█' * filled}{'░' * empty}[/{f_color}]"
            score_str = f"{f.score:+.0f}"

        table.add_row(
            f"[{f_color}]{f_arrow}[/{f_color}]",
            f"[{f_color}]{f.display_name}[/{f_color}]",
            bar, f"[{f_color}]{score_str}[/{f_color}]", f"{f.weight:.0f}%",
        )
    console.print(table)

    # Conflicts & warnings
    for conflict in p.conflicts:
        console.print(f"  [yellow bold]⚠ CONFLICT: {conflict}[/yellow bold]")
    for w in p.warnings:
        console.print(f"  [yellow]⚠ {w}[/yellow]")

    # Key signals
    console.print("\n[bold]Key Signals:[/bold]")
    for f in p.factors:
        if f.available:
            for sig in f.key_signals[:2]:
                console.print(f"  • {sig}")

    # Trend (continuous mode)
    if prev_prediction:
        console.print(
            f"\n  [dim]Trend: was {prev_prediction.signal} "
            f"({prev_prediction.composite_score:+.0f}) → now {p.signal} "
            f"({p.composite_score:+.0f})[/dim]"
        )

    # Verbose: raw data
    if verbose:
        console.print("\n[bold dim]─── Raw Data ───[/bold dim]")
        for f in p.factors:
            if f.raw_data:
                display_raw = {k: v for k, v in f.raw_data.items()
                               if k not in ("prices_history", "hourly_returns")}
                console.print(f"\n[dim]{f.display_name}:[/dim]")
                console.print(f"[dim]{json.dumps(display_raw, indent=2, default=str)}[/dim]")

    console.print()


# ─── Logging ────────────────────────────────────────────────────────────────


def log_prediction(prediction: PredictionResult, regime_tag: Optional[str] = None,
                   ml_shadow: Optional[dict] = None):
    """Append prediction to predictions.log as JSON line."""
    target_timestamp = prediction.timestamp + timedelta(hours=prediction.horizon_hours)
    point_estimate = None
    if prediction.price_low > 0 and prediction.price_high > 0:
        point_estimate = round((prediction.price_low + prediction.price_high) / 2, 2)

    entry = {
        "entry_type": "prediction",
        "prediction_id": str(uuid.uuid4()),
        "timestamp": prediction.timestamp.isoformat(),
        "target_timestamp": target_timestamp.isoformat(),
        "composite_score": prediction.composite_score,
        "signal": prediction.signal,
        "confidence": prediction.confidence,
        "price": prediction.current_price,
        "range": [prediction.price_low, prediction.price_high],
        "point_estimate": point_estimate,
        "horizon_hours": prediction.horizon_hours,
        "regime_tag": regime_tag,
        "factor_scores": {f.name: f.score for f in prediction.factors},
        "factor_weights": {f.name: f.weight for f in prediction.factors},
        "factor_contributions": {
            f.name: {
                "display_name": f.display_name,
                "available": f.available,
                "score": round(f.score, 4),
                "weight_pct": round(f.weight, 4),
                "contribution_score": round(f.score * (f.weight / 100.0), 4) if f.available else 0.0,
                "key_signals": f.key_signals[:2],
            }
            for f in prediction.factors
        },
        "factors_unavailable": [f.name for f in prediction.factors if not f.available],
    }
    if ml_shadow is not None:
        entry["ml_shadow"] = ml_shadow

    with open(PREDICTIONS_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def _load_shadow_artifact(model_path: str) -> tuple[Optional[dict], Optional[str]]:
    """Load and cache one shadow artifact by mtime."""
    try:
        resolved_path = str(Path(model_path).expanduser().resolve())
    except OSError as exc:
        return None, f"bad_path: {exc}"

    artifact_path = Path(resolved_path)
    if not artifact_path.exists():
        return None, "missing_model"

    try:
        mtime_ns = artifact_path.stat().st_mtime_ns
    except OSError as exc:
        return None, f"stat_failed: {exc}"

    cached = ML_SHADOW_CACHE.get(resolved_path)
    if cached and cached.get("mtime_ns") == mtime_ns:
        return cached.get("payload"), None

    try:
        with open(artifact_path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        return None, f"load_failed: {exc}"

    ML_SHADOW_CACHE[resolved_path] = {"mtime_ns": mtime_ns, "payload": payload}
    return payload, None


def run_ml_shadow(prediction: PredictionResult, regime_tag: str, config: dict) -> Optional[dict]:
    """Run optional shadow inference without affecting the live rule-based prediction."""
    model_path = config.get("ml_shadow_model")
    if not model_path:
        return None

    artifact_path = Path(str(model_path)).expanduser()
    payload, load_error = _load_shadow_artifact(str(artifact_path))
    base_entry = {
        "enabled": True,
        "artifact": str(artifact_path),
        "artifact_regime": None,
        "runtime_regime": regime_tag,
        "horizon": format_horizon_tag(prediction.horizon_hours),
        "direction_signal": None,
        "direction_confidence": None,
        "return_estimate_pct": None,
        "feature_version": MODEL_FEATURE_SCHEMA_VERSION,
    }
    if load_error:
        base_entry["status"] = "load_failed"
        base_entry["error"] = load_error
        return base_entry
    if not isinstance(payload, dict):
        base_entry["status"] = "load_failed"
        base_entry["error"] = "artifact payload is not a dict"
        return base_entry

    training_slice = payload.get("training_slice") or {}
    base_entry["artifact"] = str(artifact_path.resolve())
    base_entry["artifact_regime"] = training_slice.get("requested_regime_tag")

    if not payload.get("shadow_only"):
        base_entry["status"] = "artifact_not_shadow_only"
        return base_entry
    if payload.get("feature_schema_version") != MODEL_FEATURE_SCHEMA_VERSION:
        base_entry["status"] = "feature_failed"
        base_entry["error"] = "feature schema version mismatch"
        return base_entry
    if payload.get("decision_contract") != ml_shadow_decision_contract():
        base_entry["status"] = "feature_failed"
        base_entry["error"] = "decision contract mismatch"
        return base_entry

    horizon_key = format_horizon_tag(prediction.horizon_hours)
    horizon_payload = (payload.get("horizons") or {}).get(horizon_key)
    if not horizon_payload:
        base_entry["status"] = "unsupported_horizon"
        return base_entry

    feature_row = build_prediction_features(
        {
            "timestamp": prediction.timestamp.isoformat(),
            "regime_tag": regime_tag,
            "composite_score": prediction.composite_score,
            "signal": prediction.signal,
            "confidence": prediction.confidence,
            "price": prediction.current_price,
            "range": [prediction.price_low, prediction.price_high],
            "horizon_hours": prediction.horizon_hours,
            "factor_scores": {factor.name: factor.score for factor in prediction.factors},
            "factor_weights": {factor.name: factor.weight for factor in prediction.factors},
            "factor_contributions": {
                factor.name: {"available": factor.available}
                for factor in prediction.factors
            },
            "factors_unavailable": [factor.name for factor in prediction.factors if not factor.available],
        },
        blocked_factors=payload.get("inference_blocked_factors", sorted(MODEL_INFERENCE_BLOCKED_FACTORS)),
    )
    if feature_row is None:
        base_entry["status"] = "feature_failed"
        base_entry["error"] = "could not build feature row"
        return base_entry

    feature_columns = payload.get("feature_columns") or []
    if not feature_columns or any(column not in feature_row for column in feature_columns):
        base_entry["status"] = "feature_failed"
        base_entry["error"] = "artifact feature columns do not match runtime features"
        return base_entry

    direction_classifier = horizon_payload.get("direction_classifier")
    label_encoder = horizon_payload.get("direction_label_encoder")
    if direction_classifier is None or label_encoder is None:
        base_entry["status"] = "load_failed"
        base_entry["error"] = "artifact missing direction classifier"
        return base_entry

    try:
        feature_frame = pd.DataFrame([[feature_row[column] for column in feature_columns]], columns=feature_columns)
        encoded_prediction = direction_classifier.predict(feature_frame)[0]
        predicted_signal = label_encoder.inverse_transform([encoded_prediction])[0]
        predicted_confidence = None
        if hasattr(direction_classifier, "predict_proba"):
            probabilities = direction_classifier.predict_proba(feature_frame)[0]
            predicted_confidence = round(float(max(probabilities)), 4)
    except Exception as exc:
        base_entry["status"] = "feature_failed"
        base_entry["error"] = f"inference_failed: {exc}"
        return base_entry

    base_entry["status"] = "ok"
    base_entry["direction_signal"] = str(predicted_signal)
    base_entry["direction_confidence"] = predicted_confidence
    return base_entry


def prediction_key(entry: dict) -> str:
    """Return a stable key for a prediction or resolution event."""
    if entry.get("prediction_id"):
        return str(entry["prediction_id"])
    timestamp = entry.get("timestamp", "")
    horizon = entry.get("horizon_hours", "")
    price = entry.get("price", "")
    composite = entry.get("composite_score", "")
    return f"{timestamp}|{horizon}|{price}|{composite}"


def load_prediction_entries(log_path: str = PREDICTIONS_LOG_FILE) -> list[dict]:
    """Load JSONL entries, skipping blank and malformed lines."""
    path = Path(log_path)
    if not path.exists():
        return []

    entries: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def append_log_entries(entries: list[dict], log_path: str = PREDICTIONS_LOG_FILE):
    """Append JSONL entries to the log file."""
    if not entries:
        return
    with open(log_path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def prediction_target_time(prediction: dict) -> Optional[datetime]:
    """Return the target timestamp for a prediction entry."""
    target_timestamp = prediction.get("target_timestamp")
    if target_timestamp:
        try:
            return datetime.fromisoformat(target_timestamp)
        except ValueError:
            pass

    timestamp = prediction.get("timestamp")
    horizon_hours = prediction.get("horizon_hours")
    if timestamp is None or horizon_hours is None:
        return None

    try:
        base = datetime.fromisoformat(timestamp)
        return base + timedelta(hours=float(horizon_hours))
    except (TypeError, ValueError):
        return None


def parse_prediction_horizon_hours(prediction: dict) -> Optional[float]:
    """Return a finite positive horizon in hours when present."""
    try:
        horizon_hours = float(prediction.get("horizon_hours"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(horizon_hours) or horizon_hours <= 0:
        return None
    return horizon_hours


def config_neutral_tolerance_pct(config: Optional[dict] = None) -> float:
    """Return the configured neutral tolerance percentage."""
    raw_value = (config or {}).get("accuracy_neutral_band_pct", NEUTRAL_RETURN_TOLERANCE_PCT)
    try:
        tolerance_pct = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("accuracy_neutral_band_pct must be a finite non-negative number") from exc
    if not math.isfinite(tolerance_pct) or tolerance_pct < 0:
        raise ValueError("accuracy_neutral_band_pct must be a finite non-negative number")
    return tolerance_pct


def config_backfill_limit(config: Optional[dict] = None, override: Optional[int] = None) -> int:
    """Return the configured per-run resolution cap."""
    raw_value = override if override is not None else (config or {}).get(
        "backfill_limit_per_run",
        DEFAULT_BACKFILL_LIMIT_PER_RUN,
    )
    try:
        backfill_limit = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("backfill limit must be a positive integer") from exc
    if backfill_limit <= 0:
        raise ValueError("backfill limit must be a positive integer")
    return backfill_limit


def classify_realized_signal(actual_return_pct: float,
                             neutral_tolerance_pct: float = NEUTRAL_RETURN_TOLERANCE_PCT) -> str:
    """Map realized return to UP/DOWN/NEUTRAL using a tolerance band."""
    if actual_return_pct > neutral_tolerance_pct:
        return "UP"
    if actual_return_pct < -neutral_tolerance_pct:
        return "DOWN"
    return "NEUTRAL"


def horizon_matches(entry: dict, horizon_hours: Optional[float]) -> bool:
    """Return True when an entry belongs to the requested horizon bucket."""
    if horizon_hours is None:
        return True
    try:
        return math.isclose(float(entry.get("horizon_hours")), float(horizon_hours), abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def validate_prediction_record(prediction: dict) -> Optional[str]:
    """Return an unscorable status when a prediction record is permanently invalid."""
    horizon_hours = parse_prediction_horizon_hours(prediction)
    target_time = prediction_target_time(prediction)
    if target_time is None or horizon_hours is None:
        return "unscorable_invalid_target_time"

    try:
        start_price = float(prediction.get("price", 0) or 0)
    except (TypeError, ValueError):
        return "unscorable_missing_entry_price"
    if not math.isfinite(start_price) or start_price <= 0:
        return "unscorable_missing_entry_price"

    raw_range = prediction.get("range", [])
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        return "unscorable_invalid_prediction_record"

    try:
        range_low = float(raw_range[0])
        range_high = float(raw_range[1])
    except (TypeError, ValueError):
        return "unscorable_invalid_prediction_record"
    if (
        not math.isfinite(range_low)
        or not math.isfinite(range_high)
        or range_low <= 0
        or range_high <= 0
        or range_low > range_high
    ):
        return "unscorable_invalid_prediction_record"

    point_estimate = prediction.get("point_estimate")
    if point_estimate is not None:
        try:
            point_estimate = float(point_estimate)
        except (TypeError, ValueError):
            return "unscorable_invalid_prediction_record"
        if (
            not math.isfinite(point_estimate)
            or point_estimate <= 0
            or point_estimate < range_low
            or point_estimate > range_high
        ):
            return "unscorable_invalid_prediction_record"

    if horizon_hours <= 24:
        max_deviation = max(
            abs(range_low - start_price) / start_price * 100.0,
            abs(range_high - start_price) / start_price * 100.0,
        )
        if max_deviation > MAX_SHORT_HORIZON_RANGE_DEVIATION_PCT:
            return "unscorable_absurd_short_horizon_range"
    return None


def nearest_price_point(prices: list, target_time: datetime) -> tuple[float, datetime, float]:
    """Return the closest price point to a target timestamp."""
    if not prices:
        raise ValueError("no price points returned")

    target_ms = target_time.timestamp() * 1000
    nearest = min(prices, key=lambda point: abs(point[0] - target_ms))
    nearest_timestamp = datetime.fromtimestamp(nearest[0] / 1000, tz=timezone.utc)
    delta_seconds = abs((nearest_timestamp - target_time).total_seconds())
    return float(nearest[1]), nearest_timestamp, delta_seconds


def fetch_btc_price_near(target_time: datetime) -> dict:
    """Fetch BTC/USD price near a target time using CoinGecko range data."""
    start_ts = int(target_time.timestamp() - COINGECKO_RESOLUTION_WINDOW_SECONDS)
    end_ts = int(target_time.timestamp() + COINGECKO_RESOLUTION_WINDOW_SECONDS)

    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range",
        params={
            "vs_currency": "usd",
            "from": start_ts,
            "to": end_ts,
            "precision": "full",
        },
        timeout=20,
    )
    r.raise_for_status()
    prices = r.json().get("prices", [])
    actual_price, observed_at, delta_seconds = nearest_price_point(prices, target_time)
    return {
        "actual_price": actual_price,
        "observed_at": observed_at,
        "observed_delta_seconds": delta_seconds,
        "price_source": "coingecko_market_chart_range",
    }


def resolve_prediction_accuracy(prediction: dict, actual_price: float,
                                observed_at: Optional[datetime] = None,
                                price_source: str = "manual",
                                neutral_tolerance_pct: float = NEUTRAL_RETURN_TOLERANCE_PCT) -> dict:
    """Resolve one logged prediction against an actual BTC price."""
    start_price = float(prediction["price"])
    if start_price <= 0:
        raise ValueError("prediction price must be greater than zero")
    if actual_price <= 0:
        raise ValueError("actual price must be greater than zero")

    raw_range = prediction.get("range", [])
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        raise ValueError("prediction range must contain two values")

    range_low, range_high = sorted((float(raw_range[0]), float(raw_range[1])))
    predicted_signal = str(prediction.get("signal", "NEUTRAL")).upper()
    actual_return_pct = ((actual_price - start_price) / start_price) * 100.0
    actual_signal = classify_realized_signal(
        actual_return_pct,
        neutral_tolerance_pct=neutral_tolerance_pct,
    )
    target_time = prediction_target_time(prediction)
    observed_delta_seconds = (
        abs((observed_at - target_time).total_seconds())
        if observed_at and target_time
        else None
    )
    resolution_status = (
        "approximate"
        if observed_delta_seconds is not None and observed_delta_seconds > MAX_RESOLUTION_OBSERVED_DELTA_SECONDS
        else "resolved"
    )
    predicted_midpoint = (range_low + range_high) / 2.0
    midpoint_abs_error_usd = abs(actual_price - predicted_midpoint)
    midpoint_abs_error_pct = midpoint_abs_error_usd / start_price * 100.0
    hold_abs_error_usd = abs(actual_price - start_price)
    hold_abs_error_pct = hold_abs_error_usd / start_price * 100.0
    direction_hit = predicted_signal == actual_signal
    actionable_direction_hit = predicted_signal in {"UP", "DOWN"} and direction_hit
    neutral_hit = predicted_signal == "NEUTRAL" and direction_hit

    return {
        "entry_type": "resolution",
        "prediction_id": prediction_key(prediction),
        "timestamp": prediction.get("timestamp"),
        "target_timestamp": prediction.get("target_timestamp"),
        "horizon_hours": prediction.get("horizon_hours"),
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "resolution_status": resolution_status,
        "start_price": start_price,
        "actual_price": actual_price,
        "actual_price_timestamp": observed_at.isoformat() if observed_at else None,
        "observed_delta_seconds": observed_delta_seconds,
        "price_source": price_source,
        "predicted_signal": predicted_signal,
        "actual_signal": actual_signal,
        "neutral_tolerance_pct": neutral_tolerance_pct,
        "direction_hit": direction_hit,
        "actionable_direction_hit": actionable_direction_hit,
        "neutral_hit": neutral_hit,
        "range_low": range_low,
        "range_high": range_high,
        "range_hit": range_low <= actual_price <= range_high,
        "predicted_midpoint": predicted_midpoint,
        "actual_return_pct": actual_return_pct,
        "midpoint_abs_error_usd": midpoint_abs_error_usd,
        "midpoint_abs_error_pct": midpoint_abs_error_pct,
        "hold_abs_error_usd": hold_abs_error_usd,
        "hold_abs_error_pct": hold_abs_error_pct,
        "beats_hold_baseline": midpoint_abs_error_pct < hold_abs_error_pct,
    }


def resolution_status_reason(status: str) -> str:
    """Return a stable human-readable reason for an unscorable status."""
    reasons = {
        "unscorable_invalid_target_time": "missing or invalid timestamp/horizon",
        "unscorable_missing_entry_price": "missing or invalid entry price",
        "unscorable_invalid_prediction_record": "missing or invalid prediction range/point estimate",
        "unscorable_absurd_short_horizon_range": "short-horizon range exceeds 25% sanity bound",
    }
    return reasons.get(status, status.replace("unscorable_", "").replace("_", " "))


def resolve_pending_predictions(log_path: str = PREDICTIONS_LOG_FILE,
                                neutral_tolerance_pct: float = NEUTRAL_RETURN_TOLERANCE_PCT,
                                backfill_limit: int = DEFAULT_BACKFILL_LIMIT_PER_RUN) -> dict:
    """Resolve any matured prediction entries that do not yet have resolution events."""
    entries = load_prediction_entries(log_path)
    predictions = [entry for entry in entries if entry.get("entry_type", "prediction") == "prediction"]
    resolved_ids = {
        prediction_key(entry)
        for entry in entries
        if entry.get("entry_type") == "resolution"
    }

    appended_entries = []
    fetched_prices = {}
    stats = {
        "due_predictions": 0,
        "attempted_predictions": 0,
        "resolved_predictions": 0,
        "approximate_predictions": 0,
        "unscorable_predictions": 0,
        "resolution_errors": 0,
        "skipped_due_to_limit": 0,
    }
    now = datetime.now(timezone.utc)
    due_predictions = []

    for prediction in predictions:
        prediction_id = prediction_key(prediction)
        if prediction_id in resolved_ids:
            continue

        validation_error = validate_prediction_record(prediction)
        if validation_error is not None:
            appended_entries.append({
                "entry_type": "resolution",
                "prediction_id": prediction_id,
                "timestamp": prediction.get("timestamp"),
                "target_timestamp": prediction.get("target_timestamp"),
                "horizon_hours": prediction.get("horizon_hours"),
                "resolved_at": now.isoformat(),
                "resolution_status": validation_error,
                "reason": resolution_status_reason(validation_error),
            })
            stats["unscorable_predictions"] += 1
            resolved_ids.add(prediction_id)
            continue

        target_time = prediction_target_time(prediction)
        if target_time is None or target_time > now:
            continue
        due_predictions.append((target_time, prediction))

    due_predictions.sort(key=lambda item: item[0])
    stats["due_predictions"] = len(due_predictions)

    for index, (target_time, prediction) in enumerate(due_predictions):
        if index >= backfill_limit:
            stats["skipped_due_to_limit"] += 1
            continue

        prediction_id = prediction_key(prediction)
        stats["attempted_predictions"] += 1
        target_key = int(target_time.timestamp())
        try:
            price_data = fetched_prices.get(target_key)
            if price_data is None:
                price_data = fetch_btc_price_near(target_time)
                fetched_prices[target_key] = price_data

            resolution_entry = resolve_prediction_accuracy(
                    prediction,
                    actual_price=price_data["actual_price"],
                    observed_at=price_data["observed_at"],
                    price_source=price_data["price_source"],
                    neutral_tolerance_pct=neutral_tolerance_pct,
                )
            appended_entries.append(resolution_entry)
            if resolution_entry.get("resolution_status") == "approximate":
                stats["approximate_predictions"] += 1
            else:
                stats["resolved_predictions"] += 1
            resolved_ids.add(prediction_id)
        except ValueError as e:
            appended_entries.append({
                "entry_type": "resolution",
                "prediction_id": prediction_id,
                "timestamp": prediction.get("timestamp"),
                "target_timestamp": prediction.get("target_timestamp"),
                "resolved_at": now.isoformat(),
                "resolution_status": "unscorable_invalid_prediction_record",
                "reason": str(e),
            })
            stats["unscorable_predictions"] += 1
            resolved_ids.add(prediction_id)
        except Exception:
            stats["resolution_errors"] += 1

    append_log_entries(appended_entries, log_path=log_path)
    stats["appended_entries"] = len(appended_entries)
    return stats


def summarize_accuracy(resolved_predictions: list[dict], pending_predictions: int = 0,
                       total_predictions: Optional[int] = None,
                       unscorable_predictions: int = 0,
                       approximate_predictions: int = 0,
                       outside_filter_predictions: int = 0,
                       horizon_hours: Optional[float] = None,
                       all_horizons: bool = False,
                       horizons_present: Optional[list[float]] = None) -> dict:
    """Summarize accuracy metrics across resolved predictions."""
    normalized_predictions = []
    for prediction in resolved_predictions:
        predicted_signal = prediction.get("predicted_signal", "NEUTRAL")
        direction_hit = prediction.get("direction_hit")
        if direction_hit is None:
            direction_hit = predicted_signal == prediction.get("actual_signal")
        normalized_predictions.append({
            **prediction,
            "predicted_signal": predicted_signal,
            "direction_hit": direction_hit,
            "actionable_direction_hit": prediction.get(
                "actionable_direction_hit",
                predicted_signal in {"UP", "DOWN"} and direction_hit,
            ),
            "neutral_hit": prediction.get(
                "neutral_hit",
                predicted_signal == "NEUTRAL" and direction_hit,
            ),
        })

    resolved_predictions = normalized_predictions
    neutral_tolerance_bands = sorted({
        round(float(prediction.get("neutral_tolerance_pct", NEUTRAL_RETURN_TOLERANCE_PCT)), 8)
        for prediction in resolved_predictions
        if prediction.get("neutral_tolerance_pct") is not None
    })
    total_resolved = len(resolved_predictions)
    actionable = [prediction for prediction in resolved_predictions if prediction["predicted_signal"] in {"UP", "DOWN"}]
    neutral_predictions = [prediction for prediction in resolved_predictions if prediction["predicted_signal"] == "NEUTRAL"]
    range_scored = [prediction for prediction in resolved_predictions if prediction.get("range_hit") is not None]
    baseline_scored = [prediction for prediction in resolved_predictions if prediction.get("beats_hold_baseline") is not None]
    signal_counts = {
        "UP": sum(1 for prediction in resolved_predictions if prediction.get("predicted_signal") == "UP"),
        "DOWN": sum(1 for prediction in resolved_predictions if prediction.get("predicted_signal") == "DOWN"),
        "NEUTRAL": sum(1 for prediction in resolved_predictions if prediction.get("predicted_signal") == "NEUTRAL"),
    }
    sample_status = (
        "minimum tuning sample reached" if total_resolved >= 50
        else "insufficient for tuning" if total_resolved >= 20
        else "exploratory only"
    )

    if total_resolved == 0:
        return {
            "count": 0,
            "total_predictions": total_predictions or 0,
            "pending_predictions": pending_predictions,
            "unscorable_predictions": unscorable_predictions,
            "approximate_predictions": approximate_predictions,
            "outside_filter_predictions": outside_filter_predictions,
            "actionable_count": 0,
            "neutral_count": 0,
            "direction_hit_rate": 0.0,
            "actionable_direction_hit_rate": 0.0,
            "neutral_hit_rate": 0.0,
            "range_hit_rate": 0.0,
            "baseline_beat_rate": 0.0,
            "mean_actual_return_pct": 0.0,
            "mean_midpoint_abs_error_pct": 0.0,
            "mean_hold_abs_error_pct": 0.0,
            "mean_error_improvement_pct": 0.0,
            "mean_observed_delta_minutes": 0.0,
            "signal_counts": signal_counts,
            "sample_status": sample_status,
            "filtered_horizon_hours": horizon_hours,
            "all_horizons": all_horizons,
            "horizons_present": horizons_present or [],
            "neutral_tolerance_bands": neutral_tolerance_bands,
            "recent_resolutions": [],
        }

    direction_hits = sum(1 for prediction in actionable if prediction["direction_hit"])
    overall_direction_hits = sum(1 for prediction in resolved_predictions if prediction["direction_hit"])
    neutral_hits = sum(1 for prediction in neutral_predictions if prediction["neutral_hit"])
    range_hits = sum(1 for prediction in range_scored if prediction["range_hit"])
    baseline_beats = sum(1 for prediction in baseline_scored if prediction["beats_hold_baseline"])
    mean_actual_return_pct = sum(abs(prediction["actual_return_pct"]) for prediction in resolved_predictions) / total_resolved
    mean_midpoint_abs_error_pct = (
        sum(prediction["midpoint_abs_error_pct"] for prediction in baseline_scored) / len(baseline_scored)
        if baseline_scored else 0.0
    )
    mean_hold_abs_error_pct = (
        sum(prediction["hold_abs_error_pct"] for prediction in baseline_scored) / len(baseline_scored)
        if baseline_scored else 0.0
    )
    observed_deltas = [
        prediction["observed_delta_seconds"] / 60.0
        for prediction in resolved_predictions
        if prediction.get("observed_delta_seconds") is not None
    ]
    recent_resolutions = sorted(
        resolved_predictions,
        key=lambda prediction: prediction.get("resolved_at") or prediction.get("timestamp") or "",
        reverse=True,
    )[:10]

    return {
        "count": total_resolved,
        "total_predictions": total_predictions or total_resolved,
        "pending_predictions": pending_predictions,
        "unscorable_predictions": unscorable_predictions,
        "approximate_predictions": approximate_predictions,
        "outside_filter_predictions": outside_filter_predictions,
        "actionable_count": len(actionable),
        "neutral_count": len(neutral_predictions),
        "direction_hit_rate": overall_direction_hits / total_resolved,
        "actionable_direction_hit_rate": (direction_hits / len(actionable)) if actionable else 0.0,
        "neutral_hit_rate": (neutral_hits / len(neutral_predictions)) if neutral_predictions else 0.0,
        "range_hit_rate": (range_hits / len(range_scored)) if range_scored else 0.0,
        "baseline_beat_rate": (baseline_beats / len(baseline_scored)) if baseline_scored else 0.0,
        "mean_actual_return_pct": mean_actual_return_pct,
        "mean_midpoint_abs_error_pct": mean_midpoint_abs_error_pct,
        "mean_hold_abs_error_pct": mean_hold_abs_error_pct,
        "mean_error_improvement_pct": mean_hold_abs_error_pct - mean_midpoint_abs_error_pct,
        "mean_observed_delta_minutes": (
            sum(observed_deltas) / len(observed_deltas) if observed_deltas else 0.0
        ),
        "signal_counts": signal_counts,
        "sample_status": sample_status,
        "filtered_horizon_hours": horizon_hours,
        "all_horizons": all_horizons,
        "horizons_present": horizons_present or [],
        "neutral_tolerance_bands": neutral_tolerance_bands,
        "recent_resolutions": recent_resolutions,
    }


def build_accuracy_summary(log_path: str = PREDICTIONS_LOG_FILE,
                           horizon_hours: float = DEFAULT_ACCURACY_HOURS,
                           all_horizons: bool = False) -> dict:
    """Build a current accuracy summary from prediction and resolution log events."""
    entries = load_prediction_entries(log_path)
    predictions = [entry for entry in entries if entry.get("entry_type", "prediction") == "prediction"]
    resolutions = {}
    now = datetime.now(timezone.utc)
    pending_predictions = 0
    selected_horizon = None if all_horizons else horizon_hours
    horizons_present = sorted({
        round(float(entry["horizon_hours"]), 8)
        for entry in predictions
        if entry.get("horizon_hours") is not None
    })

    for entry in entries:
        if entry.get("entry_type") == "resolution":
            resolutions[prediction_key(entry)] = entry

    for prediction in predictions:
        if not horizon_matches(prediction, selected_horizon):
            continue
        target_time = prediction_target_time(prediction)
        if target_time is None:
            continue
        if target_time <= now and prediction_key(prediction) not in resolutions:
            pending_predictions += 1

    resolved_predictions = [
        entry for entry in resolutions.values()
        if entry.get("resolution_status") == "resolved" and horizon_matches(entry, selected_horizon)
    ]
    approximate_predictions = sum(
        1 for entry in resolutions.values()
        if entry.get("resolution_status") == "approximate" and horizon_matches(entry, selected_horizon)
    )
    unscorable_predictions = sum(
        1 for entry in resolutions.values()
        if entry.get("resolution_status", "").startswith("unscorable") and horizon_matches(entry, selected_horizon)
    )
    filtered_prediction_count = sum(1 for entry in predictions if horizon_matches(entry, selected_horizon))
    outside_filter_predictions = 0 if all_horizons else max(len(predictions) - filtered_prediction_count, 0)
    return summarize_accuracy(
        resolved_predictions,
        pending_predictions=pending_predictions,
        total_predictions=filtered_prediction_count,
        unscorable_predictions=unscorable_predictions,
        approximate_predictions=approximate_predictions,
        outside_filter_predictions=outside_filter_predictions,
        horizon_hours=selected_horizon,
        all_horizons=all_horizons,
        horizons_present=horizons_present,
    )


def print_accuracy_report(summary: dict, resolution_stats: Optional[dict] = None):
    """Render a compact accuracy report."""
    header = Text()
    header.append("BTC PREDICTION ACCURACY\n", style="bold white")
    horizon_label = "All horizons" if summary["all_horizons"] else f"{summary['filtered_horizon_hours']:.1f}h"
    header.append(
        f"Horizon: {horizon_label} | "
        f"Predictions: {summary['total_predictions']} | "
        f"Resolved: {summary['count']} | "
        f"Approximate: {summary['approximate_predictions']} | "
        f"Pending: {summary['pending_predictions']} | "
        f"Unscorable: {summary['unscorable_predictions']}",
        style="cyan",
    )
    console.print(Panel(header, style="bold blue", title="[bold]Accuracy[/bold]"))

    if resolution_stats and resolution_stats.get("appended_entries"):
        console.print(
            f"[dim]Resolved {resolution_stats['resolved_predictions']} matured predictions, "
            f"marked {resolution_stats['approximate_predictions']} approximate, "
            f"marked {resolution_stats['unscorable_predictions']} unscorable, "
            f"attempted {resolution_stats['attempted_predictions']} of {resolution_stats['due_predictions']} due this run.[/dim]"
        )
    if resolution_stats and resolution_stats.get("resolution_errors"):
        console.print(
            f"[yellow]Skipped {resolution_stats['resolution_errors']} matured predictions due to "
            "price-resolution errors; they will be retried later.[/yellow]"
        )
    if resolution_stats and resolution_stats.get("skipped_due_to_limit"):
        console.print(
            f"[dim]Deferred {resolution_stats['skipped_due_to_limit']} matured predictions due to "
            "the per-run backfill cap.[/dim]"
        )
    console.print(f"[dim]Sample status: {summary['sample_status']}[/dim]")
    if summary["all_horizons"] and len(summary["horizons_present"]) > 1:
        console.print("[yellow]Warning: all-horizon accuracy mixes multiple forecast windows.[/yellow]")
    elif summary["outside_filter_predictions"] > 0:
        console.print(
            f"[dim]{summary['outside_filter_predictions']} predictions exist outside the "
            f"{summary['filtered_horizon_hours']:.1f}h filter; use --all-horizons to inspect them.[/dim]"
        )
    if (not summary["all_horizons"]) and math.isclose(summary["filtered_horizon_hours"], DEFAULT_ACCURACY_HOURS, abs_tol=1e-9):
        console.print(
            "[dim]At --interval 30 and --hours 4, expect about 12 resolved predictions per day "
            "and roughly 4-5 days to reach 50 resolved samples.[/dim]"
        )

    metrics = Table(title="Accuracy Summary", show_header=True, header_style="bold")
    metrics.add_column("Metric", min_width=26)
    metrics.add_column("Value", justify="right")
    if summary["neutral_tolerance_bands"]:
        tolerance_label = ", ".join(f"+/-{band:.2f}%" for band in summary["neutral_tolerance_bands"])
    else:
        tolerance_label = f"+/-{NEUTRAL_RETURN_TOLERANCE_PCT:.2f}%"
    metrics.add_row("Neutral tolerance band", tolerance_label)
    metrics.add_row("Outside filter count", str(summary["outside_filter_predictions"]))
    metrics.add_row("Overall direction hit rate", f"{summary['direction_hit_rate']:.1%}")
    metrics.add_row("Actionable direction hit rate", f"{summary['actionable_direction_hit_rate']:.1%}")
    metrics.add_row("Neutral hit rate", f"{summary['neutral_hit_rate']:.1%}")
    metrics.add_row("Range coverage", f"{summary['range_hit_rate']:.1%}")
    metrics.add_row("Beat no-change baseline", f"{summary['baseline_beat_rate']:.1%}")
    metrics.add_row("Mean abs realized move", f"{summary['mean_actual_return_pct']:.2f}%")
    metrics.add_row("Model midpoint error", f"{summary['mean_midpoint_abs_error_pct']:.2f}%")
    metrics.add_row("No-change error", f"{summary['mean_hold_abs_error_pct']:.2f}%")
    metrics.add_row("Error improvement", f"{summary['mean_error_improvement_pct']:.2f}%")
    metrics.add_row("Mean price timing delta", f"{summary['mean_observed_delta_minutes']:.1f} min")
    console.print(metrics)

    signal_mix = Table(title="Resolved Signal Mix", show_header=True, header_style="bold")
    signal_mix.add_column("Signal")
    signal_mix.add_column("Count", justify="right")
    signal_mix.add_row("UP", str(summary["signal_counts"]["UP"]))
    signal_mix.add_row("DOWN", str(summary["signal_counts"]["DOWN"]))
    signal_mix.add_row("NEUTRAL", str(summary["signal_counts"]["NEUTRAL"]))
    console.print(signal_mix)

    if summary["recent_resolutions"]:
        recent = Table(title="Recent Resolutions", show_header=True, header_style="bold")
        recent.add_column("Target", min_width=16)
        recent.add_column("Signal", min_width=7)
        recent.add_column("Move", justify="right")
        recent.add_column("Dir", justify="center")
        recent.add_column("Range", justify="center")
        recent.add_column("Baseline", justify="center")

        for item in summary["recent_resolutions"]:
            target_raw = item.get("target_timestamp") or item.get("timestamp") or ""
            target_time = str(target_raw)[:16].replace("T", " ")
            signal = item.get("predicted_signal", "")
            move = f"{item.get('actual_return_pct', 0):+.2f}%"
            dir_hit = "Y" if item.get("direction_hit") else "N"
            range_hit = "Y" if item.get("range_hit") else "N"
            baseline = "Y" if item.get("beats_hold_baseline") else "N"
            recent.add_row(target_time, signal, move, dir_hit, range_hit, baseline)
        console.print(recent)


# ─── Orchestrator & CLI ─────────────────────────────────────────────────────


def run_prediction(config: dict, horizon_hours: float,
                   verbose: bool = False,
                   prev_prediction: Optional[PredictionResult] = None,
                   neutral_tolerance_pct: float = NEUTRAL_RETURN_TOLERANCE_PCT,
                   backfill_limit: int = DEFAULT_BACKFILL_LIMIT_PER_RUN) -> PredictionResult:
    """Run all collectors in parallel and generate prediction."""
    collector_config = dict(config)
    collector_config["_active_horizon_hours"] = horizon_hours
    resolution_stats = resolve_pending_predictions(
        neutral_tolerance_pct=neutral_tolerance_pct,
        backfill_limit=backfill_limit,
    )
    if resolution_stats.get("appended_entries"):
        console.print(
            f"[dim]Accuracy backfill: resolved {resolution_stats['resolved_predictions']} "
            f"approximate {resolution_stats['approximate_predictions']} "
            f"and marked {resolution_stats['unscorable_predictions']} unscorable "
            f"(attempted {resolution_stats['attempted_predictions']} of "
            f"{resolution_stats['due_predictions']} due).[/dim]"
        )
    if resolution_stats.get("skipped_due_to_limit"):
        console.print(
            f"[dim]Deferred {resolution_stats['skipped_due_to_limit']} matured predictions "
            f"because the backfill cap is {backfill_limit} per run.[/dim]"
        )

    console.print("[dim]Fetching data from all sources...[/dim]")
    results = collect_all_factors(collector_config, emit_status=console.print)

    funding_result = next((result for result in results if result.name == "funding_rate"), None)
    open_interest_result = next((result for result in results if result.name == "open_interest"), None)
    if funding_result and open_interest_result and not funding_result.available and not open_interest_result.available:
        console.print(
            "[yellow]Warning: both Binance shadow collectors are unavailable. "
            "Binance Futures may be geoblocked or unreachable, so shadow data will not accumulate.[/yellow]"
        )

    prediction = compute_prediction(results, horizon_hours, collector_config)
    generate_report(prediction, verbose=verbose, prev_prediction=prev_prediction)
    regime_tag = build_regime_tag(horizon_hours)
    ml_shadow = run_ml_shadow(prediction, regime_tag, collector_config)
    log_prediction(prediction, regime_tag=regime_tag, ml_shadow=ml_shadow)

    return prediction


def collect_all_factors(config: dict, emit_status=None) -> list[CollectorResult]:
    """Run the current collector set in parallel and return their results."""
    collectors = [
        ("crypto_market", collect_crypto_market),
        ("funding_rate", collect_funding_rate),
        ("open_interest", collect_open_interest),
        ("us_policy", collect_us_policy),
        ("geopolitics", collect_geopolitics),
        ("ai_chip_stocks", collect_ai_chip_stocks),
        ("oil_gold", collect_oil_gold),
    ]
    results: list[CollectorResult] = []

    with ThreadPoolExecutor(max_workers=len(collectors)) as executor:
        future_to_name = {}
        completed_futures = set()
        for collector_name, collector_fn in collectors:
            future = executor.submit(collector_fn, config)
            future_to_name[future] = collector_name

        try:
            for future in as_completed(future_to_name, timeout=45):
                completed_futures.add(future)
                collector_name = future_to_name[future]
                try:
                    result = future.result()
                    results.append(result)
                    if emit_status is not None:
                        status = "[green]✓[/green]" if result.available else "[red]✗[/red]"
                        emit_status(f"  {status} {result.display_name}")
                except Exception as exc:
                    if emit_status is not None:
                        emit_status(f"  [red]✗ {collector_name}: {exc}[/red]")
                    weight = config["weights"].get(collector_name, 0)
                    results.append(CollectorResult(
                        name=collector_name,
                        display_name=collector_name,
                        score=0,
                        weight=weight,
                        available=False,
                        key_signals=[f"Timeout/error: {exc}"],
                    ))
        except FuturesTimeoutError:
            for future, collector_name in future_to_name.items():
                if future in completed_futures:
                    continue
                future.cancel()
                if emit_status is not None:
                    emit_status(f"  [red]✗ {collector_name}: overall collector timeout[/red]")
                weight = config["weights"].get(collector_name, 0)
                results.append(CollectorResult(
                    name=collector_name,
                    display_name=collector_name,
                    score=0,
                    weight=weight,
                    available=False,
                    key_signals=["Timeout/error: overall collector timeout"],
                ))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="BTC Price Prediction — Multi-signal analysis for Bitcoin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                          # Default 4h prediction
  python script.py --hours 1                # 1-hour prediction
  python script.py --minutes 30             # 30-minute prediction
  python script.py --monitor --interval 15  # Monitor every 15 min
  python script.py --accuracy               # Resolve matured calls + show hit rates
  python script.py --accuracy --backfill-limit 25
  python script.py --verbose                # Show raw data
        """,
    )
    parser.add_argument("--hours", type=float, default=None, help="Prediction horizon in hours (default: 4)")
    parser.add_argument("--minutes", type=float, default=None, help="Prediction horizon in minutes (overrides --hours)")
    parser.add_argument("--monitor", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=None, help="Minutes between monitor runs (default: 30)")
    parser.add_argument("--weights", type=str, default=None,
                        help='Weight overrides as JSON: \'{"crypto_market": 35}\'')
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Show raw data from collectors")
    parser.add_argument("--accuracy", action="store_true",
                        help="Resolve matured predictions and show accuracy metrics")
    parser.add_argument("--accuracy-hours", type=float, default=DEFAULT_ACCURACY_HOURS,
                        help="Horizon bucket for --accuracy (default: 4h)")
    parser.add_argument("--all-horizons", action="store_true",
                        help="Show --accuracy across all logged horizons")
    parser.add_argument("--backfill-limit", type=int, default=None,
                        help="Max matured predictions to resolve per run (default: 10)")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.weights:
        try:
            overrides = json.loads(args.weights)
            config["weights"].update(overrides)
        except json.JSONDecodeError:
            console.print("[red]Error: --weights must be valid JSON[/red]")
            sys.exit(1)

    try:
        config["weights"] = normalize_weights(config["weights"])
        validate_shadow_factor_weights(config["weights"])
        neutral_tolerance_pct = config_neutral_tolerance_pct(config)
        backfill_limit = config_backfill_limit(config, override=args.backfill_limit)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)

    if args.minutes is not None:
        horizon_hours = args.minutes / 60
    elif args.hours is not None:
        horizon_hours = args.hours
    else:
        horizon_hours = config.get("prediction_hours", 4)

    interval = args.interval or config.get("monitor_interval_minutes", 30)

    if args.accuracy:
        resolution_stats = resolve_pending_predictions(
            neutral_tolerance_pct=neutral_tolerance_pct,
            backfill_limit=backfill_limit,
        )
        summary = build_accuracy_summary(
            horizon_hours=args.accuracy_hours,
            all_horizons=args.all_horizons,
        )
        print_accuracy_report(summary, resolution_stats=resolution_stats)
        return

    if args.monitor:
        console.print(f"[bold cyan]Starting continuous monitoring "
                       f"(every {interval} min, {horizon_hours:.1f}h horizon)[/bold cyan]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        prev = None
        try:
            while True:
                prev = run_prediction(config, horizon_hours,
                                      verbose=args.verbose,
                                      prev_prediction=prev,
                                      neutral_tolerance_pct=neutral_tolerance_pct,
                                      backfill_limit=backfill_limit)
                console.print(f"\n[dim]Next update in {interval} minutes... (Ctrl+C to stop)[/dim]")
                time.sleep(interval * 60)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")
    else:
        run_prediction(config, horizon_hours,
                       verbose=args.verbose,
                       neutral_tolerance_pct=neutral_tolerance_pct,
                       backfill_limit=backfill_limit)


if __name__ == "__main__":
    main()
