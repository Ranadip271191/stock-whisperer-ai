"""Deterministic intraday signal rules, built on Kite historical candles.

No LLM anywhere in this file - that is the point. A rule here produces the
same output every time given the same candles, which is what makes it
backtestable before it is ever allowed to place a real order.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .client import historical_candles


def bar_minutes(interval: str) -> int:
    """Minutes per bar for a Kite interval string, e.g. '5minute' -> 5, 'minute' -> 1."""
    digits = "".join(ch for ch in interval if ch.isdigit())
    return int(digits) if digits else 1


def vwap_bias(day: pd.DataFrame, idx: int) -> str | None:
    """Price vs session VWAP, computed only from bars up to and including idx."""
    window = day.iloc[: idx + 1]
    typical = (window["high"] + window["low"] + window["close"]) / 3
    volume_sum = window["volume"].sum()
    if volume_sum == 0:
        return None
    vwap = (typical * window["volume"]).sum() / volume_sum
    close = float(day.iloc[idx]["close"])
    if close == vwap:
        return None
    return "BUY" if close > vwap else "SELL"


def rsi_bias(
    day: pd.DataFrame, idx: int, period: int = 14, oversold: float = 30, overbought: float = 70
) -> str | None:
    """Session-local RSI mean-reversion read: oversold -> BUY, overbought -> SELL."""
    closes = day.iloc[: idx + 1]["close"]
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, pd.NA))).dropna()
    if rsi.empty:
        return None
    value = float(rsi.iloc[-1])
    if value <= oversold:
        return "BUY"
    if value >= overbought:
        return "SELL"
    return None


def ema_bias(day: pd.DataFrame, idx: int, fast: int = 12, slow: int = 26) -> str | None:
    """EMA(fast) vs EMA(slow) crossover read, session-local."""
    closes = day.iloc[: idx + 1]["close"]
    if len(closes) < slow:
        return None
    ema_fast = closes.ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = closes.ewm(span=slow, adjust=False).mean().iloc[-1]
    if ema_fast == ema_slow:
        return None
    return "BUY" if ema_fast > ema_slow else "SELL"


def gap_bias(today_open: float, prior_close: float, threshold_pct: float = 0.3) -> str | None:
    """Classic gap-fill read: fades the gap direction (expects partial reversion),
    not a gap-and-go continuation read. One value per session, not per bar.
    """
    if prior_close == 0:
        return None
    gap_pct = (today_open / prior_close - 1) * 100
    if gap_pct >= threshold_pct:
        return "SELL"
    if gap_pct <= -threshold_pct:
        return "BUY"
    return None


def confirmation_count(direction: str, confirmations: list[str | None]) -> int:
    return sum(1 for c in confirmations if c == direction)


def session_ohlc(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    """One row per session: open (first bar), high, low, close (last bar). Used to compute a
    daily ATR without needing day-level candles from Kite.
    """
    rows = [
        {
            "session_date": s,
            "open": float(day["open"].iloc[0]),
            "high": float(day["high"].max()),
            "low": float(day["low"].min()),
            "close": float(day["close"].iloc[-1]),
        }
        for s in sessions
        for day in [frame[frame["session_date"] == s]]
    ]
    return pd.DataFrame(rows)


def daily_atr(session_table: pd.DataFrame, period: int = 14) -> pd.Series:
    """True-range based ATR over consecutive sessions. Row i's value never uses session i's own
    high/low plus a later roll - it is a trailing rolling mean, so atr.iloc[i] is only safe to use
    as "known going into session i+1", not session i itself.
    """
    prev_close = session_table["close"].shift(1)
    true_range = pd.concat(
        [
            session_table["high"] - session_table["low"],
            (session_table["high"] - prev_close).abs(),
            (session_table["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def bollinger_bands(
    day: pd.DataFrame, idx: int, period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float] | None:
    """Rolling (mean, upper, lower) band from the last `period` closes up to and including idx,
    session-local - resets every session rather than carrying over from the prior day.
    """
    closes = day.iloc[: idx + 1]["close"]
    if len(closes) < period:
        return None
    window = closes.iloc[-period:]
    mean = float(window.mean())
    std = float(window.std(ddof=0))
    return mean, mean + num_std * std, mean - num_std * std


def relative_strength_bias(symbol_day: pd.DataFrame, index_day: pd.DataFrame, idx: int) -> str | None:
    """Symbol's return since session open vs the index's return since session open, read bar-for-bar.
    BUY when the symbol is outperforming the index, SELL when underperforming. Assumes symbol_day and
    index_day share the same session's bars at the same interval - a crude relative-strength read,
    not a weighted sector model.
    """
    if idx >= len(index_day) or idx >= len(symbol_day):
        return None
    symbol_open = float(symbol_day.iloc[0]["open"])
    index_open = float(index_day.iloc[0]["open"])
    if symbol_open == 0 or index_open == 0:
        return None
    symbol_return = float(symbol_day.iloc[idx]["close"]) / symbol_open - 1
    index_return = float(index_day.iloc[idx]["close"]) / index_open - 1
    if symbol_return == index_return:
        return None
    return "BUY" if symbol_return > index_return else "SELL"


def volume_profile_levels(
    day: pd.DataFrame, idx: int, bins: int = 20, value_area_pct: float = 0.7
) -> tuple[float, float, float] | None:
    """POC (point of control), VAH, VAL from the volume-at-price histogram of bars up to and
    including idx, session-local. Buckets typical price into `bins` equal-width bins weighted by
    bar volume; VAH/VAL bound the smallest set of highest-volume bins covering value_area_pct of
    the session's volume so far.
    """
    window = day.iloc[: idx + 1]
    if len(window) < 2:
        return None
    typical = (window["high"] + window["low"] + window["close"]) / 3
    if float(typical.max()) == float(typical.min()):
        return None
    edges = pd.cut(typical, bins=bins, include_lowest=True)
    volume_by_bin = window.groupby(edges, observed=True)["volume"].sum()
    total_volume = float(volume_by_bin.sum())
    if total_volume == 0:
        return None

    poc = float(volume_by_bin.idxmax().mid)
    target_volume = total_volume * value_area_pct
    accumulated = 0.0
    included_mids: list[float] = []
    for interval_key, vol in volume_by_bin.sort_values(ascending=False).items():
        accumulated += float(vol)
        included_mids.append(float(interval_key.mid))
        if accumulated >= target_volume:
            break
    return poc, max(included_mids), min(included_mids)


def volume_profile_bias(day: pd.DataFrame, idx: int, bins: int = 20, value_area_pct: float = 0.7) -> str | None:
    """Fades price outside the session's volume value area back toward it: BUY below VAL, SELL above VAH."""
    levels = volume_profile_levels(day, idx, bins, value_area_pct)
    if levels is None:
        return None
    _, vah, val = levels
    close = float(day.iloc[idx]["close"])
    if close < val:
        return "BUY"
    if close > vah:
        return "SELL"
    return None


@dataclass
class Signal:
    symbol: str
    action: str  # "BUY", "SELL" or "NONE"
    entry: float | None
    stop: float | None
    target: float | None
    reason: str


def _session_frame(symbol: str, interval: str, days: int) -> pd.DataFrame:
    candles = historical_candles(symbol, interval=interval, days=days)
    frame = pd.DataFrame(candles)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame["session_date"] = frame["date"].dt.date
    return frame


def orb_breakout_signal(
    symbol: str,
    orb_minutes: int = 15,
    rvol_threshold: float = 1.5,
    target_r_multiple: float = 2.0,
    interval: str = "5minute",
    min_confirmations: int = 0,
    gap_threshold_pct: float = 0.3,
    use_relative_strength: bool = False,
    use_volume_profile: bool = False,
    index_symbol: str = "NIFTY 50",
) -> Signal:
    """Opening-range breakout, confirmed by relative volume.

    Long on a close above the opening-range high with RVOL >= threshold;
    short on the symmetric break below the opening-range low. No trade
    otherwise, including while still inside the opening range itself.

    With min_confirmations > 0, ORB still decides direction/stop/target but a
    breakout is only accepted if at least that many of {VWAP, RSI
    mean-reversion, EMA crossover, gap-fill, relative strength, volume
    profile} agree with it - confluence filtering, not an independent
    strategy. Relative strength and volume profile are opt-in via their
    use_* flags since the former needs an extra index fetch. Default 0 keeps
    prior behavior.

    The target is a fixed multiple of the stop distance (reward:risk, not a
    chart level) - purely informational here. The live runner does not yet
    place a target order: Kite's regular orders have no built-in OCO, so a
    naive stop+target pair risks an orphaned order once one side fills.
    """
    frame = _session_frame(symbol, interval, days=10)
    if frame.empty:
        return Signal(symbol, "NONE", None, None, None, "no intraday candles returned")

    sessions = sorted(frame["session_date"].unique())
    if len(sessions) < 2:
        return Signal(symbol, "NONE", None, None, None, "need 2+ sessions of candles")

    today = frame[frame["session_date"] == sessions[-1]]
    bars_in_orb = max(1, orb_minutes // bar_minutes(interval))
    if len(today) <= bars_in_orb:
        return Signal(symbol, "NONE", None, None, None, "still inside the opening range")

    orb_window = today.iloc[:bars_in_orb]
    orb_high = float(orb_window["high"].max())
    orb_low = float(orb_window["low"].min())

    prior_volumes = [
        frame[frame["session_date"] == d]["volume"].sum() for d in sessions[:-1]
    ]
    avg_prior_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0
    rvol = float(today["volume"].sum()) / avg_prior_volume if avg_prior_volume else 0.0
    last_price = float(today["close"].iloc[-1])
    idx = len(today) - 1

    direction = None
    if last_price > orb_high and rvol >= rvol_threshold:
        direction = "BUY"
    elif last_price < orb_low and rvol >= rvol_threshold:
        direction = "SELL"

    if direction and min_confirmations > 0:
        prior_close = frame[frame["session_date"] == sessions[-2]]["close"].iloc[-1]
        gap_bias_value = gap_bias(float(today["open"].iloc[0]), float(prior_close), gap_threshold_pct)
        votes = [vwap_bias(today, idx), rsi_bias(today, idx), ema_bias(today, idx), gap_bias_value]
        if use_volume_profile:
            votes.append(volume_profile_bias(today, idx))
        if use_relative_strength:
            index_frame = _session_frame(index_symbol, interval, days=10)
            if not index_frame.empty:
                index_sessions = sorted(index_frame["session_date"].unique())
                if sessions[-1] in index_sessions:
                    index_today = index_frame[index_frame["session_date"] == sessions[-1]]
                    votes.append(relative_strength_bias(today, index_today, idx))
        if confirmation_count(direction, votes) < min_confirmations:
            direction = None

    if direction == "BUY":
        risk_per_share = last_price - orb_low
        return Signal(
            symbol, "BUY", last_price, orb_low, last_price + risk_per_share * target_r_multiple,
            f"broke above ORB high {orb_high:.2f}, RVOL {rvol:.2f}",
        )
    if direction == "SELL":
        risk_per_share = orb_high - last_price
        return Signal(
            symbol, "SELL", last_price, orb_high, last_price - risk_per_share * target_r_multiple,
            f"broke below ORB low {orb_low:.2f}, RVOL {rvol:.2f}",
        )
    return Signal(symbol, "NONE", None, None, None, f"no qualifying breakout (RVOL {rvol:.2f})")


def atr_breakout_signal(
    symbol: str,
    atr_period: int = 14,
    atr_multiple: float = 1.0,
    target_r_multiple: float = 2.0,
    interval: str = "5minute",
) -> Signal:
    """Volatility breakout, independent of ORB: trades the first break of today's open +/-
    atr_multiple * ATR(atr_period prior sessions). No opening-range window and no volume filter -
    the trigger is today's range expansion past a level set from recent daily true range, so this
    fires on genuinely different days than ORB does. Stop is the session open; target is a fixed
    R-multiple of that distance.
    """
    frame = _session_frame(symbol, interval, days=atr_period + 15)
    if frame.empty:
        return Signal(symbol, "NONE", None, None, None, "no intraday candles returned")

    sessions = sorted(frame["session_date"].unique())
    if len(sessions) < atr_period + 2:
        return Signal(symbol, "NONE", None, None, None, f"need {atr_period + 2}+ sessions for ATR")

    session_table = session_ohlc(frame, sessions)
    atr_series = daily_atr(session_table, atr_period)
    atr_today = atr_series.iloc[-2]  # known as of yesterday's close, before today's own range
    if pd.isna(atr_today):
        return Signal(symbol, "NONE", None, None, None, "not enough sessions for ATR yet")

    today = frame[frame["session_date"] == sessions[-1]]
    today_open = float(today["open"].iloc[0])
    last_price = float(today["close"].iloc[-1])
    breakout_long = today_open + atr_multiple * atr_today
    breakout_short = today_open - atr_multiple * atr_today

    if last_price > breakout_long:
        risk_per_share = last_price - today_open
        return Signal(
            symbol, "BUY", last_price, today_open, last_price + risk_per_share * target_r_multiple,
            f"volatility breakout above open+{atr_multiple}xATR ({breakout_long:.2f})",
        )
    if last_price < breakout_short:
        risk_per_share = today_open - last_price
        return Signal(
            symbol, "SELL", last_price, today_open, last_price - risk_per_share * target_r_multiple,
            f"volatility breakdown below open-{atr_multiple}xATR ({breakout_short:.2f})",
        )
    return Signal(symbol, "NONE", None, None, None, f"inside volatility range (ATR {atr_today:.2f})")


def bollinger_reversion_signal(
    symbol: str,
    period: int = 20,
    num_std: float = 2.0,
    interval: str = "5minute",
) -> Signal:
    """Standalone mean-reversion, independent of ORB: fades a close beyond the session's rolling
    Bollinger band back toward the rolling mean. BUY below the lower band, SELL above the upper
    band. Stop is one more band-width past the triggering band; target is the rolling mean itself
    (not an R-multiple) - the mean is the natural target for a mean-reversion trade.
    """
    frame = _session_frame(symbol, interval, days=10)
    if frame.empty:
        return Signal(symbol, "NONE", None, None, None, "no intraday candles returned")

    sessions = sorted(frame["session_date"].unique())
    if not sessions:
        return Signal(symbol, "NONE", None, None, None, "no sessions of candles")

    today = frame[frame["session_date"] == sessions[-1]].reset_index(drop=True)
    idx = len(today) - 1
    bands = bollinger_bands(today, idx, period, num_std)
    if bands is None:
        return Signal(symbol, "NONE", None, None, None, f"need {period}+ bars in session for bands")

    mean, upper, lower = bands
    band_width = upper - mean
    last_price = float(today.iloc[idx]["close"])

    if last_price < lower:
        return Signal(
            symbol, "BUY", last_price, lower - band_width, mean,
            f"closed below lower band {lower:.2f}, target mean {mean:.2f}",
        )
    if last_price > upper:
        return Signal(
            symbol, "SELL", last_price, upper + band_width, mean,
            f"closed above upper band {upper:.2f}, target mean {mean:.2f}",
        )
    return Signal(symbol, "NONE", None, None, None, f"inside bands (mean {mean:.2f})")

