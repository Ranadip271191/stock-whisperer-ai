"""Data tools specific to the pre-market report: global markets, sector moves,
derivatives positioning and prior-session price action. Company news and NSE
filings reuse the existing tools in tools.py - no need to duplicate those.
"""

from __future__ import annotations

import time
from datetime import datetime, time as dtime, timedelta
from functools import lru_cache
from typing import Annotated, Any

import pandas as pd
from agent_framework import tool
from pydantic import Field

from .tools import _dumps, _history


def _history_with_retry(ticker: str, period: str, interval: str, retries: int = 2, delay: float = 1.5):
    """Yahoo intermittently rate-limits under the burst of calls a full watchlist
    run makes; a short retry avoids misreporting a real symbol as unavailable.
    """
    last_error: ValueError | None = None
    for attempt in range(retries + 1):
        try:
            return _history(ticker, period, interval)
        except ValueError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay)
    raise last_error

GLOBAL_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "FTSE 100": "^FTSE",
    "Crude Oil (WTI)": "CL=F",
    "Gold": "GC=F",
    "US Dollar Index": "DX-Y.NYB",
    "USD/INR": "INR=X",
}

# Best-effort NSE sector index tickers on Yahoo; coverage is patchy, handled gracefully.
# No POWER entry - no working Yahoo ticker found for a Nifty Power sector index after
# checking several candidates; get_sector_snapshot reports it as an unknown sector.
SECTOR_INDEX_TICKERS = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY",
    "REALTY": "^CNXREALTY",
    "INFRA": "^CNXINFRA",
}


def _drop_in_progress_session(frame):
    """yfinance's last daily bar is today's still-forming session while that
    market is open - drop it so 'prior session' always means the last fully
    completed one, not a live partial bar.
    """
    if not frame.empty and frame.index[-1].date() == datetime.now().date():
        return frame.iloc[:-1]
    return frame


def _pct_change_from_history(ticker: str) -> dict[str, Any] | None:
    try:
        frame = _drop_in_progress_session(_history_with_retry(ticker, period="5d", interval="1d"))
    except ValueError:
        return None
    if len(frame) < 2:
        return None
    last_close = float(frame["Close"].iloc[-1])
    prior_close = float(frame["Close"].iloc[-2])
    return {
        "last_close": round(last_close, 2),
        "prior_close": round(prior_close, 2),
        "change_pct": round((last_close / prior_close - 1) * 100, 2),
        "as_of": str(frame.index[-1].date()),
    }


@tool
def get_global_market_snapshot() -> str:
    """Get the most recent close-to-close move for major global indices, crude oil,
    gold, the US dollar index and USD/INR - the overnight global market signal.
    """
    snapshot = {}
    for name, ticker in GLOBAL_TICKERS.items():
        result = _pct_change_from_history(ticker)
        snapshot[name] = result if result is not None else {"error": "no data returned"}
    return _dumps({"global_market_snapshot": snapshot})


@tool
def get_market_wide_news(
    limit: Annotated[int, Field(description="Maximum number of headlines.", ge=1, le=10)] = 6,
) -> str:
    """Get recent broad market-moving headlines - index-level rally/selloff drivers,
    FII/DII flows, RBI/macro catalysts - not tied to any single stock. A per-stock
    news search won't surface 'why did the whole market move today', which often
    explains a stock moving against its own company-specific thesis.
    """
    from .tools import get_recent_news

    return get_recent_news.func("Sensex Nifty stock market today", limit)


@tool
def get_sector_snapshot(
    sector: Annotated[
        str,
        Field(
            description=(
                "Sector keyword: BANK, IT, AUTO, PHARMA, FMCG, METAL, ENERGY, "
                "REALTY, INFRA or POWER."
            )
        ),
    ],
) -> str:
    """Get the most recent close-to-close move for an NSE sector index."""
    ticker = SECTOR_INDEX_TICKERS.get(sector.upper())
    if ticker is None:
        return _dumps(
            {
                "sector": sector.upper(),
                "available": False,
                "note": f"Unknown sector. Choose from: {', '.join(SECTOR_INDEX_TICKERS)}.",
            }
        )
    result = _pct_change_from_history(ticker)
    if result is None:
        return _dumps({"sector": sector.upper(), "available": False, "note": "No data returned for this sector index."})
    return _dumps({"sector": sector.upper(), "available": True, **result})


@tool
def get_prior_session_summary(
    symbol: Annotated[str, Field(description="NSE trading symbol without suffix, e.g. TCS, RELIANCE.")],
) -> str:
    """Get the prior trading session's price action: close vs prior-prior close,
    the day's range, and volume versus its 20-day average.
    """
    ticker = f"{symbol.upper()}.NS"
    try:
        frame = _drop_in_progress_session(_history_with_retry(ticker, period="2mo", interval="1d"))
    except ValueError as exc:
        return _dumps({"symbol": symbol.upper(), "available": False, "note": str(exc)})
    if len(frame) < 2:
        return _dumps({"symbol": symbol.upper(), "available": False, "note": "Not enough completed sessions of history yet."})

    last = frame.iloc[-1]
    prior_close = float(frame["Close"].iloc[-2])
    avg_volume_20d = float(frame["Volume"].tail(21).iloc[:-1].mean())
    return _dumps(
        {
            "symbol": symbol.upper(),
            "available": True,
            "session_date": str(frame.index[-1].date()),
            "open": round(float(last["Open"]), 2),
            "high": round(float(last["High"]), 2),
            "low": round(float(last["Low"]), 2),
            "close": round(float(last["Close"]), 2),
            "change_pct": round((float(last["Close"]) / prior_close - 1) * 100, 2),
            "volume": int(last["Volume"]),
            "avg_volume_20d": round(avg_volume_20d, 0),
            "relative_volume": round(float(last["Volume"]) / avg_volume_20d, 2) if avg_volume_20d else None,
        }
    )


@tool
def get_volatility_snapshot(
    symbol: Annotated[str, Field(description="NSE trading symbol without suffix, e.g. TCS, RELIANCE.")],
    period: Annotated[int, Field(description="Sessions used for the ATR/gap average.", ge=5, le=60)] = 14,
) -> str:
    """Get a volatility basis for projecting tomorrow's likely move: ATR (average
    true range) over the trailing sessions, and the average overnight gap size
    (open vs prior close). Use this - not a guess - to size an expected move,
    a projected opening price, and entry/stop/target levels.
    """
    ticker = f"{symbol.upper()}.NS"
    try:
        frame = _drop_in_progress_session(_history_with_retry(ticker, period="4mo", interval="1d"))
    except ValueError as exc:
        return _dumps({"symbol": symbol.upper(), "available": False, "note": str(exc)})
    if len(frame) < period + 1:
        return _dumps({"symbol": symbol.upper(), "available": False, "note": f"Need {period + 1}+ completed sessions."})

    prev_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.tail(period).mean())

    gap_pct = ((frame["Open"] - prev_close) / prev_close * 100).tail(period)
    last_close = float(frame["Close"].iloc[-1])
    return _dumps(
        {
            "symbol": symbol.upper(),
            "available": True,
            "last_close": round(last_close, 2),
            "atr": round(atr, 2),
            "atr_pct_of_price": round(atr / last_close * 100, 2) if last_close else None,
            "avg_abs_overnight_gap_pct": round(float(gap_pct.abs().mean()), 2),
            "typical_gap_up_pct": round(float(gap_pct[gap_pct > 0].mean()), 2) if (gap_pct > 0).any() else None,
            "typical_gap_down_pct": round(float(gap_pct[gap_pct < 0].mean()), 2) if (gap_pct < 0).any() else None,
            "note": "Historical average, not a forecast of tomorrow's specific gap.",
        }
    )


@lru_cache(maxsize=1)
def _nfo_instruments() -> list[dict[str, Any]]:
    from .kite.client import kite_client

    return kite_client().instruments("NFO")


def _nearest_future_contract(symbol: str) -> dict[str, Any] | None:
    today = datetime.now().date()
    candidates = [
        row
        for row in _nfo_instruments()
        if row.get("name") == symbol.upper()
        and row.get("segment") == "NFO-FUT"
        and row.get("expiry")
        and row["expiry"] >= today
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["expiry"])


@tool
def get_derivatives_snapshot(
    symbol: Annotated[str, Field(description="NSE trading symbol without suffix, e.g. TCS, RELIANCE.")],
) -> str:
    """Get the nearest-month futures contract's prior completed session's close, volume
    and open interest. Deliberately the last finished day's bar, not a live quote - this
    tool is used pre-market (before/around 9:00-9:15 IST) when no trade has happened yet
    today, so a live snapshot would always read a meaningless volume of 0. Kite only
    exposes a point-in-time OI level per bar, not an intraday OI-change series.
    Requires F&O segment access on the connected Kite account.
    """
    from .kite.client import kite_client

    try:
        contract = _nearest_future_contract(symbol.upper())
    except Exception as exc:  # Kite auth/network issues - report, don't crash the agent run
        return _dumps({"symbol": symbol.upper(), "available": False, "note": f"Kite error: {exc}"})

    if contract is None:
        return _dumps(
            {
                "symbol": symbol.upper(),
                "available": False,
                "note": "No active futures contract found - stock may not be in the F&O segment.",
            }
        )

    tradingsymbol = contract["tradingsymbol"]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=10)
    candles = kite_client().historical_data(contract["instrument_token"], from_date, to_date, "day", oi=True)
    if not candles:
        return _dumps(
            {
                "symbol": symbol.upper(),
                "available": False,
                "contract": tradingsymbol,
                "note": "No historical daily candles returned for this contract.",
            }
        )

    last_bar = candles[-1]
    prior_bar = candles[-2] if len(candles) > 1 else None
    oi_change = (
        last_bar.get("oi") - prior_bar.get("oi")
        if prior_bar and last_bar.get("oi") is not None and prior_bar.get("oi") is not None
        else None
    )
    return _dumps(
        {
            "symbol": symbol.upper(),
            "available": True,
            "contract": tradingsymbol,
            "expiry": str(contract["expiry"]),
            "as_of_session": str(last_bar["date"].date()),
            "close": last_bar.get("close"),
            "volume": last_bar.get("volume"),
            "open_interest": last_bar.get("oi"),
            "oi_change_vs_prior_session": oi_change,
            "note": "Prior completed session's figures - today's session hasn't traded yet pre-market.",
        }
    )


# NSE's pre-open session has two sub-phases: 9:00-9:08 is order collection only -
# no price matching has happened yet, so last_price just echoes the prior close.
# The real indicative equilibrium price only appears during 9:08-9:12, the
# order-matching/price-discovery phase.
PREOPEN_WINDOW = (dtime(9, 8), dtime(9, 12))


def get_preopen_quote(symbol: str) -> dict[str, Any]:
    """Plain (non-agent) helper for the deterministic pre-open addendum - not an
    @tool, since that part of the report is rule-based rather than LLM-synthesized.
    """
    from .kite.client import kite_client

    now = datetime.now().time()
    in_window = PREOPEN_WINDOW[0] <= now <= PREOPEN_WINDOW[1]
    quote_key = f"NSE:{symbol.upper()}"
    try:
        quote = kite_client().quote([quote_key]).get(quote_key, {})
    except Exception as exc:
        return {"symbol": symbol.upper(), "available": False, "in_preopen_window": in_window, "note": str(exc)}

    ohlc = quote.get("ohlc", {})
    prior_close = ohlc.get("close")
    last_price = quote.get("last_price")
    change_pct = round((last_price / prior_close - 1) * 100, 2) if prior_close and last_price else None
    return {
        "symbol": symbol.upper(),
        "available": True,
        "in_preopen_window": in_window,
        "indicative_price": last_price,
        "prior_close": prior_close,
        "change_pct": change_pct,
        "note": None if in_window else "Outside the 9:08-9:12 price-discovery window - this is a live/last quote, not the pre-open indicative price.",
    }


PREMARKET_TOOLS = [
    get_global_market_snapshot,
    get_market_wide_news,
    get_sector_snapshot,
    get_prior_session_summary,
    get_volatility_snapshot,
    get_derivatives_snapshot,
]


def compute_price_plan(
    prior_close: float,
    atr: float,
    atr_pct: float,
    typical_gap_up_pct: float | None,
    typical_gap_down_pct: float | None,
    bias: str,
) -> dict[str, float] | None:
    """Deterministic price plan from volatility-snapshot numbers - not an LLM
    estimate, so the projected range always has real, non-degenerate width and
    never silently collapses to a single point. NEUTRAL gets no plan.
    """
    if bias not in ("BULLISH", "BEARISH") or prior_close <= 0 or atr <= 0:
        return None

    if bias == "BULLISH":
        gap_pct = typical_gap_up_pct if typical_gap_up_pct is not None else atr_pct * 0.3
    else:
        gap_pct = typical_gap_down_pct if typical_gap_down_pct is not None else -atr_pct * 0.3
    center = prior_close * (1 + gap_pct / 100)

    # Half-width floored at 0.05% of price so low/high can never collapse to one point.
    half_width = max(atr * 0.15, prior_close * 0.0005)
    low, high = center - half_width, center + half_width
    entry = (low + high) / 2

    if bias == "BULLISH":
        stop, target = entry - atr, entry + 2 * atr
    else:
        stop, target = entry + atr, entry - 2 * atr

    return {
        "projected_open_low": round(min(low, high), 2),
        "projected_open_high": round(max(low, high), 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "expected_move_pct": round((target - prior_close) / prior_close * 100, 2),
    }

