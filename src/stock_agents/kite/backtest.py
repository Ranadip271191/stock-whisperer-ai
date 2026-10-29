"""Walk-forward backtest for the ORB + relative-volume rule in strategy.py.

Replays historical candles bar-by-bar using only data available up to that
bar (no lookahead), sizes with the same risk.py logic the live runner uses,
and nets out an approximate Indian intraday cost stack. Read-only against
Kite's historical data API - never calls the order-placement endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from .client import historical_candles
from .risk import shares_for_risk
from .strategy import (
    bar_minutes,
    bollinger_bands,
    confirmation_count,
    daily_atr,
    ema_bias,
    gap_bias,
    relative_strength_bias,
    rsi_bias,
    session_ohlc,
    volume_profile_bias,
    vwap_bias,
)

# Kite's historical-data API caps how many days you can request per call, and
# the cap varies by interval. Verify current limits in Kite's docs if a
# request fails - these are conservative, possibly stale, assumptions.
_CHUNK_DAYS = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
}


@dataclass
class Trade:
    symbol: str
    session_date: str
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    shares: int
    reason: str
    gross_pnl: float = 0.0
    cost: float = 0.0
    net_pnl: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate_pct(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        return round(100 * wins / len(self.trades), 2)

    @property
    def total_net_pnl(self) -> float:
        return round(sum(t.net_pnl for t in self.trades), 2)

    @property
    def expectancy_per_trade(self) -> float:
        return round(self.total_net_pnl / len(self.trades), 2) if self.trades else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        cumulative = pd.Series([t.net_pnl for t in self.trades]).cumsum()
        return round(float((cumulative - cumulative.cummax()).min()), 2)

    def summary(self) -> str:
        return (
            f"{self.symbol}: {self.trade_count} trades, win rate {self.win_rate_pct}%, "
            f"expectancy/trade \u20b9{self.expectancy_per_trade}, "
            f"total net P&L \u20b9{self.total_net_pnl}, max drawdown \u20b9{self.max_drawdown}"
        )


def intraday_round_trip_cost(entry_value: float, exit_value: float) -> float:
    """Approximate NSE intraday equity round-trip cost. Rates change over time
    and this does not distinguish long vs short for STT/stamp duty purposes -
    verify against your broker's calculator before sizing off this number.
    """
    brokerage = min(entry_value * 0.0003, 20.0) + min(exit_value * 0.0003, 20.0)
    stt = exit_value * 0.00025
    exchange_txn = (entry_value + exit_value) * 0.0000297
    sebi_fee = (entry_value + exit_value) * 0.0000001
    stamp_duty = entry_value * 0.00003
    gst = 0.18 * (brokerage + exchange_txn + sebi_fee)
    return round(brokerage + stt + exchange_txn + sebi_fee + stamp_duty + gst, 2)


def _fetch_candles(symbol: str, interval: str, from_date: datetime, to_date: datetime) -> pd.DataFrame:
    chunk_days = _CHUNK_DAYS.get(interval, 60)
    frames = []
    cursor = from_date
    while cursor < to_date:
        chunk_end = min(cursor + timedelta(days=chunk_days), to_date)
        frames.append(pd.DataFrame(historical_candles(symbol, interval, from_date=cursor, to_date=chunk_end)))
        cursor = chunk_end + timedelta(seconds=1)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset="date")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["session_date"] = frame["date"].dt.date
    return frame.sort_values("date").reset_index(drop=True)


def _close_trade(symbol: str, session_date, position: dict, exit_price: float, exit_time: str, reason: str) -> Trade:
    direction, entry_price, shares = position["direction"], position["entry_price"], position["shares"]
    gross_pnl = (exit_price - entry_price) * shares if direction == "BUY" else (entry_price - exit_price) * shares
    cost = intraday_round_trip_cost(entry_price * shares, exit_price * shares)
    return Trade(
        symbol=symbol,
        session_date=str(session_date),
        direction=direction,
        entry_time=position["entry_time"],
        entry_price=entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        shares=shares,
        reason=f"{position['reason']} -> {reason}",
        gross_pnl=round(gross_pnl, 2),
        cost=cost,
        net_pnl=round(gross_pnl - cost, 2),
    )


def backtest_orb(
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
    min_confirmations: int = 0,
    gap_threshold_pct: float = 0.3,
    use_relative_strength: bool = False,
    use_volume_profile: bool = False,
    index_symbol: str = "NIFTY 50",
) -> BacktestResult:
    """Replays the ORB rule bar-by-bar; re-entry within the same session is
    possible if a position closes (stop/target) and a fresh breakout fires
    later that day - there is no one-trade-per-session cap.

    ORB remains the entry trigger and stop/target source. With
    min_confirmations > 0, a breakout only opens a position if at least that
    many of {VWAP, RSI mean-reversion, EMA crossover, gap-fill, relative
    strength (opt-in), volume profile (opt-in)} agree with ORB's direction at
    the signal bar - confluence filtering, not a second, independent
    strategy. Default 0 disables this and reproduces the original ORB-only
    behavior exactly.

    A signal detected on bar i fills at bar i+1's open, never at bar i's own
    close - that close is exactly what triggered the signal, so trading at it
    would be lookahead. Each bar after entry is checked for the stop, then the
    target; if a single bar's range contains both, the stop is assumed to have
    hit first (the conservative assumption - OHLC bars don't tell you the
    actual intrabar path). If still open at the session's last bar, it is
    squared off there (same as MIS).
    """
    frame = _fetch_candles(symbol, interval, from_date, to_date)
    result = BacktestResult(symbol=symbol)
    if frame.empty:
        return result

    sessions = sorted(frame["session_date"].unique())
    bars_in_orb = max(1, orb_minutes // bar_minutes(interval))

    index_frame = pd.DataFrame()
    if use_relative_strength:
        index_frame = _fetch_candles(index_symbol, interval, from_date, to_date)

    for i, session in enumerate(sessions):
        if i < lookback_sessions:
            continue  # not enough prior sessions yet for a relative-volume baseline

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
        position: dict | None = None

        gap_bias_value = None
        if min_confirmations > 0 and i > 0:
            prior_close = frame[frame["session_date"] == sessions[i - 1]]["close"].iloc[-1]
            gap_bias_value = gap_bias(float(day.iloc[0]["open"]), float(prior_close), gap_threshold_pct)

        index_day = pd.DataFrame()
        if use_relative_strength and not index_frame.empty and session in set(index_frame["session_date"]):
            index_day = index_frame[index_frame["session_date"] == session].reset_index(drop=True)

        for idx in range(bars_in_orb, len(day)):
            bar = day.iloc[idx]

            if position is None:
                rvol = float(day.iloc[: idx + 1]["volume"].sum()) / avg_prior_volume
                direction = stop = reason = None
                if bar["close"] > orb_high and rvol >= rvol_threshold:
                    direction, stop, reason = "BUY", orb_low, f"ORB breakout up, RVOL {rvol:.2f}"
                elif bar["close"] < orb_low and rvol >= rvol_threshold:
                    direction, stop, reason = "SELL", orb_high, f"ORB breakout down, RVOL {rvol:.2f}"

                if direction and min_confirmations > 0:
                    votes = [vwap_bias(day, idx), rsi_bias(day, idx), ema_bias(day, idx), gap_bias_value]
                    if use_volume_profile:
                        votes.append(volume_profile_bias(day, idx))
                    if use_relative_strength and not index_day.empty:
                        votes.append(relative_strength_bias(day, index_day, idx))
                    matches = confirmation_count(direction, votes)
                    if matches < min_confirmations:
                        direction = None  # ORB fired but not enough agreement; no signal this bar
                    else:
                        reason = f"{reason}, {matches}/{len(votes)} confirmations"

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


def backtest_atr_breakout(
    symbol: str,
    from_date: datetime,
    to_date: datetime,
    account_equity: float,
    risk_per_trade_pct: float = 0.5,
    atr_period: int = 14,
    atr_multiple: float = 1.0,
    target_r_multiple: float = 2.0,
    interval: str = "5minute",
) -> BacktestResult:
    """Volatility breakout, independent of ORB: trades the first break of today's open +/-
    atr_multiple * ATR(atr_period prior sessions). No opening-range window and no volume filter -
    the trigger is today's range expansion past a level set from recent daily true range. Stop is
    the session open; target is a fixed R-multiple of that distance. Same no-lookahead fill (signal
    bar i fills at bar i+1's open), same stop-wins-ties and EOD-square-off rules as backtest_orb.
    """
    frame = _fetch_candles(symbol, interval, from_date, to_date)
    result = BacktestResult(symbol=symbol)
    if frame.empty:
        return result

    sessions = sorted(frame["session_date"].unique())
    session_table = session_ohlc(frame, sessions)
    atr_series = daily_atr(session_table, atr_period)

    for i, session in enumerate(sessions):
        if i < atr_period + 1 or pd.isna(atr_series.iloc[i - 1]):
            continue  # not enough prior sessions for ATR yet

        atr_value = float(atr_series.iloc[i - 1])
        day = frame[frame["session_date"] == session].reset_index(drop=True)
        if len(day) < 2:
            continue

        today_open = float(day.iloc[0]["open"])
        breakout_long = today_open + atr_multiple * atr_value
        breakout_short = today_open - atr_multiple * atr_value
        position: dict | None = None

        for idx in range(len(day)):
            bar = day.iloc[idx]

            if position is None:
                direction = stop = reason = None
                if bar["close"] > breakout_long:
                    direction, stop = "BUY", today_open
                    reason = f"volatility breakout above open+{atr_multiple}xATR ({breakout_long:.2f})"
                elif bar["close"] < breakout_short:
                    direction, stop = "SELL", today_open
                    reason = f"volatility breakdown below open-{atr_multiple}xATR ({breakout_short:.2f})"

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


def backtest_bollinger_reversion(
    symbol: str,
    from_date: datetime,
    to_date: datetime,
    account_equity: float,
    risk_per_trade_pct: float = 0.5,
    period: int = 20,
    num_std: float = 2.0,
    interval: str = "5minute",
) -> BacktestResult:
    """Standalone mean-reversion, independent of ORB: fades a close beyond the session's rolling
    Bollinger band back toward the rolling mean. Target is the rolling mean at signal time (not an
    R-multiple); stop is one more band-width past the triggering band. Same no-lookahead fill,
    stop-wins-ties, and EOD-square-off rules as backtest_orb.
    """
    frame = _fetch_candles(symbol, interval, from_date, to_date)
    result = BacktestResult(symbol=symbol)
    if frame.empty:
        return result

    sessions = sorted(frame["session_date"].unique())

    for session in sessions:
        day = frame[frame["session_date"] == session].reset_index(drop=True)
        if len(day) <= period:
            continue

        position: dict | None = None

        for idx in range(period, len(day)):
            bar = day.iloc[idx]

            if position is None:
                bands = bollinger_bands(day, idx, period, num_std)
                if bands is None:
                    continue
                mean, upper, lower = bands
                band_width = upper - mean
                direction = stop = target = reason = None
                if bar["close"] < lower:
                    direction, stop, target = "BUY", lower - band_width, mean
                    reason = f"closed below lower band {lower:.2f}, target mean {mean:.2f}"
                elif bar["close"] > upper:
                    direction, stop, target = "SELL", upper + band_width, mean
                    reason = f"closed above upper band {upper:.2f}, target mean {mean:.2f}"

                if direction and idx + 1 < len(day):
                    fill_bar = day.iloc[idx + 1]
                    shares = shares_for_risk(account_equity, risk_per_trade_pct, float(fill_bar["open"]), stop)
                    if shares > 0:
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

