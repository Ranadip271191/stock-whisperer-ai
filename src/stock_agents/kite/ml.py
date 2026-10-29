"""Gradient-boosted probability filter for ORB breakout signals.

Not a replacement for the deterministic ORB rule in strategy.py - this model only
answers "given this breakout already fired, how likely is it to be net profitable
after costs?". It is trained offline on past trades, saved to disk, and loaded
read-only at inference time: no network calls and no randomness per prediction,
so the same determinism property as the rest of this package still holds once a
model file exists. Training itself is explicitly out of the live trading path.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestResult, _close_trade, _fetch_candles
from .risk import shares_for_risk
from .strategy import Signal, _session_frame, bar_minutes

MODEL_PATH = Path(__file__).parent / ".orb_filter_model.pkl"

FEATURE_NAMES = [
    "rvol",
    "gap_pct",
    "vwap_dev_pct",
    "rsi",
    "ema_spread_pct",
    "volatility_pct",
    "bars_since_open",
]


def build_features(day: pd.DataFrame, idx: int, rvol: float, gap_pct: float) -> dict[str, float]:
    """Numeric features at bar idx, session-local; nothing here looks past idx."""
    window = day.iloc[: idx + 1]
    closes = window["close"]
    close = float(closes.iloc[-1])

    returns = closes.pct_change().dropna()
    volatility = float(returns.tail(20).std()) if len(returns) >= 2 else 0.0

    typical = (window["high"] + window["low"] + window["close"]) / 3
    volume_sum = float(window["volume"].sum())
    vwap = float((typical * window["volume"]).sum() / volume_sum) if volume_sum else close

    ema_fast = float(closes.ewm(span=12, adjust=False).mean().iloc[-1]) if len(closes) >= 12 else close
    ema_slow = float(closes.ewm(span=26, adjust=False).mean().iloc[-1]) if len(closes) >= 26 else close

    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_series = (100 - 100 / (1 + gain / loss.replace(0, pd.NA))).dropna()
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    return {
        "rvol": rvol,
        "gap_pct": gap_pct,
        "vwap_dev_pct": (close / vwap - 1) * 100 if vwap else 0.0,
        "rsi": rsi,
        "ema_spread_pct": (ema_fast / ema_slow - 1) * 100 if ema_slow else 0.0,
        "volatility_pct": volatility * 100,
        "bars_since_open": float(idx),
    }


@dataclass
class TrainingSample:
    features: dict[str, float]
    label: int  # 1 = that specific trade was net profitable after costs, 0 = not


def generate_orb_training_samples(
    symbol: str,
    from_date: datetime,
    to_date: datetime,
    account_equity: float,
    risk_per_trade_pct: float = 0.5,
    orb_minutes: int = 15,
    rvol_threshold: float = 1.5,
    target_r_multiple: float = 2.0,
    interval: str = "5minute",
    lookback_sessions: int = 10,
) -> list[TrainingSample]:
    """Replays the same ORB+RVOL rule as backtest_orb, but instead of aggregating
    P&L, captures a feature snapshot at every signal bar plus whether that
    specific trade ended up net profitable after costs. Same mechanics, same
    cost model as backtest_orb, so the label matches exactly what the live
    probability filter needs to predict.
    """
    frame = _fetch_candles(symbol, interval, from_date, to_date)
    samples: list[TrainingSample] = []
    if frame.empty:
        return samples

    sessions = sorted(frame["session_date"].unique())
    bars_in_orb = max(1, orb_minutes // bar_minutes(interval))

    for i, session in enumerate(sessions):
        if i < lookback_sessions:
            continue

        prior_sessions = sessions[max(0, i - lookback_sessions) : i]
        prior_volumes = [frame[frame["session_date"] == d]["volume"].sum() for d in prior_sessions]
        avg_prior_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0
        if avg_prior_volume == 0:
            continue

        day = frame[frame["session_date"] == session].reset_index(drop=True)
        if len(day) <= bars_in_orb:
            continue

        orb_high = float(day.iloc[:bars_in_orb]["high"].max())
        orb_low = float(day.iloc[:bars_in_orb]["low"].min())

        gap_pct = 0.0
        if i > 0:
            prior_close = float(frame[frame["session_date"] == sessions[i - 1]]["close"].iloc[-1])
            if prior_close:
                gap_pct = (float(day.iloc[0]["open"]) / prior_close - 1) * 100

        position: dict[str, Any] | None = None

        for idx in range(bars_in_orb, len(day)):
            bar = day.iloc[idx]

            if position is None:
                rvol = float(day.iloc[: idx + 1]["volume"].sum()) / avg_prior_volume
                direction = stop = None
                if bar["close"] > orb_high and rvol >= rvol_threshold:
                    direction, stop = "BUY", orb_low
                elif bar["close"] < orb_low and rvol >= rvol_threshold:
                    direction, stop = "SELL", orb_high

                if direction and idx + 1 < len(day):
                    fill_bar = day.iloc[idx + 1]
                    shares = shares_for_risk(account_equity, risk_per_trade_pct, float(fill_bar["open"]), stop)
                    if shares > 0:
                        risk_per_share = abs(float(fill_bar["open"]) - stop)
                        target = (
                            float(fill_bar["open"]) + risk_per_share * target_r_multiple
                            if direction == "BUY"
                            else float(fill_bar["open"]) - risk_per_share * target_r_multiple
                        )
                        position = {
                            "direction": direction,
                            "entry_time": str(fill_bar["date"]),
                            "entry_price": float(fill_bar["open"]),
                            "stop": stop,
                            "target": target,
                            "shares": shares,
                            "reason": "orb",
                            "features": build_features(day, idx, rvol, gap_pct),
                        }
                continue

            if position["direction"] == "BUY":
                stop_hit = bar["low"] <= position["stop"]
                target_hit = bar["high"] >= position["target"]
            else:
                stop_hit = bar["high"] >= position["stop"]
                target_hit = bar["low"] <= position["target"]

            closed = None
            if stop_hit:
                closed = _close_trade(symbol, session, position, position["stop"], str(bar["date"]), "stop hit")
            elif target_hit:
                closed = _close_trade(symbol, session, position, position["target"], str(bar["date"]), "target hit")

            if closed is not None:
                samples.append(TrainingSample(position["features"], int(closed.net_pnl > 0)))
                position = None

        if position is not None:
            last_bar = day.iloc[-1]
            closed = _close_trade(symbol, session, position, float(last_bar["close"]), str(last_bar["date"]), "EOD square-off")
            samples.append(TrainingSample(position["features"], int(closed.net_pnl > 0)))

    return samples


def train_model(samples: list[TrainingSample]):
    """Trains a gradient-boosted-trees binary classifier on (features, net-profitable?)
    pairs. HistGradientBoostingClassifier, not LightGBM - avoids LightGBM's macOS
    libomp system dependency while staying in the same model family.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    features = pd.DataFrame([s.features for s in samples])[FEATURE_NAMES]
    labels = np.array([s.label for s in samples])
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20)
    model.fit(features, labels)
    return model


def save_model(booster, path: Path = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(booster, f)


def load_model(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_win_probability(booster, features: dict[str, float]) -> float:
    row = pd.DataFrame([features])[FEATURE_NAMES]
    return float(booster.predict_proba(row)[0][1])


def backtest_orb_ml_filtered(
    symbol: str,
    from_date: datetime,
    to_date: datetime,
    account_equity: float,
    booster,
    probability_threshold: float = 0.55,
    risk_per_trade_pct: float = 0.5,
    orb_minutes: int = 15,
    rvol_threshold: float = 1.5,
    target_r_multiple: float = 2.0,
    interval: str = "5minute",
    lookback_sessions: int = 10,
) -> BacktestResult:
    """Same ORB+RVOL walk-forward as backtest_orb, but a breakout only opens a
    position if the trained model predicts P(net-profitable) >= probability_threshold.
    The model is a fixed, pre-trained artifact loaded read-only here - no training,
    no network calls, no randomness happen during this walk-forward.
    """
    frame = _fetch_candles(symbol, interval, from_date, to_date)
    result = BacktestResult(symbol=symbol)
    if frame.empty:
        return result

    sessions = sorted(frame["session_date"].unique())
    bars_in_orb = max(1, orb_minutes // bar_minutes(interval))

    for i, session in enumerate(sessions):
        if i < lookback_sessions:
            continue

        prior_sessions = sessions[max(0, i - lookback_sessions) : i]
        prior_volumes = [frame[frame["session_date"] == d]["volume"].sum() for d in prior_sessions]
        avg_prior_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0
        if avg_prior_volume == 0:
            continue

        day = frame[frame["session_date"] == session].reset_index(drop=True)
        if len(day) <= bars_in_orb:
            continue

        orb_high = float(day.iloc[:bars_in_orb]["high"].max())
        orb_low = float(day.iloc[:bars_in_orb]["low"].min())

        gap_pct = 0.0
        if i > 0:
            prior_close = float(frame[frame["session_date"] == sessions[i - 1]]["close"].iloc[-1])
            if prior_close:
                gap_pct = (float(day.iloc[0]["open"]) / prior_close - 1) * 100

        position: dict[str, Any] | None = None

        for idx in range(bars_in_orb, len(day)):
            bar = day.iloc[idx]

            if position is None:
                rvol = float(day.iloc[: idx + 1]["volume"].sum()) / avg_prior_volume
                direction = stop = reason = None
                if bar["close"] > orb_high and rvol >= rvol_threshold:
                    direction, stop, reason = "BUY", orb_low, f"ORB breakout up, RVOL {rvol:.2f}"
                elif bar["close"] < orb_low and rvol >= rvol_threshold:
                    direction, stop, reason = "SELL", orb_high, f"ORB breakout down, RVOL {rvol:.2f}"

                if direction:
                    win_probability = predict_win_probability(booster, build_features(day, idx, rvol, gap_pct))
                    if win_probability < probability_threshold:
                        direction = None
                    else:
                        reason = f"{reason}, P(win)={win_probability:.2f}"

                if direction and idx + 1 < len(day):
                    fill_bar = day.iloc[idx + 1]
                    shares = shares_for_risk(account_equity, risk_per_trade_pct, float(fill_bar["open"]), stop)
                    if shares > 0:
                        risk_per_share = abs(float(fill_bar["open"]) - stop)
                        target = (
                            float(fill_bar["open"]) + risk_per_share * target_r_multiple
                            if direction == "BUY"
                            else float(fill_bar["open"]) - risk_per_share * target_r_multiple
                        )
                        position = {
                            "direction": direction,
                            "entry_time": str(fill_bar["date"]),
                            "entry_price": float(fill_bar["open"]),
                            "stop": stop,
                            "target": target,
                            "shares": shares,
                            "reason": reason,
                        }
                continue

            if position["direction"] == "BUY":
                stop_hit = bar["low"] <= position["stop"]
                target_hit = bar["high"] >= position["target"]
            else:
                stop_hit = bar["high"] >= position["stop"]
                target_hit = bar["low"] <= position["target"]

            if stop_hit:
                result.trades.append(_close_trade(symbol, session, position, position["stop"], str(bar["date"]), "stop hit"))
                position = None
            elif target_hit:
                result.trades.append(_close_trade(symbol, session, position, position["target"], str(bar["date"]), "target hit"))
                position = None

        if position is not None:
            last_bar = day.iloc[-1]
            result.trades.append(
                _close_trade(symbol, session, position, float(last_bar["close"]), str(last_bar["date"]), "EOD square-off")
            )

    return result


def orb_ml_filtered_signal(
    symbol: str,
    booster,
    probability_threshold: float = 0.55,
    orb_minutes: int = 15,
    rvol_threshold: float = 1.5,
    target_r_multiple: float = 2.0,
    interval: str = "5minute",
) -> Signal:
    """Live, single-bar version of backtest_orb_ml_filtered: same ORB+RVOL rule as
    orb_breakout_signal, gated by the same pre-trained probability model. Used by
    the scan command for dry-run/live trading, not just backtesting.
    """
    frame = _session_frame(symbol, interval, days=10)
    if frame.empty:
        return Signal(symbol, "NONE", None, None, None, "no intraday candles returned")

    sessions = sorted(frame["session_date"].unique())
    if len(sessions) < 2:
        return Signal(symbol, "NONE", None, None, None, "need 2+ sessions of candles")

    today = frame[frame["session_date"] == sessions[-1]].reset_index(drop=True)
    bars_in_orb = max(1, orb_minutes // bar_minutes(interval))
    if len(today) <= bars_in_orb:
        return Signal(symbol, "NONE", None, None, None, "still inside the opening range")

    orb_window = today.iloc[:bars_in_orb]
    orb_high = float(orb_window["high"].max())
    orb_low = float(orb_window["low"].min())

    prior_volumes = [frame[frame["session_date"] == d]["volume"].sum() for d in sessions[:-1]]
    avg_prior_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0
    rvol = float(today["volume"].sum()) / avg_prior_volume if avg_prior_volume else 0.0
    last_price = float(today["close"].iloc[-1])
    idx = len(today) - 1

    direction = None
    if last_price > orb_high and rvol >= rvol_threshold:
        direction = "BUY"
    elif last_price < orb_low and rvol >= rvol_threshold:
        direction = "SELL"

    if direction:
        prior_close = float(frame[frame["session_date"] == sessions[-2]]["close"].iloc[-1])
        gap_pct = (float(today.iloc[0]["open"]) / prior_close - 1) * 100 if prior_close else 0.0
        win_probability = predict_win_probability(booster, build_features(today, idx, rvol, gap_pct))
        if win_probability < probability_threshold:
            return Signal(symbol, "NONE", None, None, None, f"ORB fired but P(win)={win_probability:.2f} below threshold")

        if direction == "BUY":
            risk_per_share = last_price - orb_low
            return Signal(
                symbol, "BUY", last_price, orb_low, last_price + risk_per_share * target_r_multiple,
                f"broke above ORB high {orb_high:.2f}, RVOL {rvol:.2f}, P(win)={win_probability:.2f}",
            )
        risk_per_share = orb_high - last_price
        return Signal(
            symbol, "SELL", last_price, orb_high, last_price - risk_per_share * target_r_multiple,
            f"broke below ORB low {orb_low:.2f}, RVOL {rvol:.2f}, P(win)={win_probability:.2f}",
        )

    return Signal(symbol, "NONE", None, None, None, f"no qualifying breakout (RVOL {rvol:.2f})")
