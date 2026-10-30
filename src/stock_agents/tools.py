"""Market-data tools exposed to the agents as callable functions."""

from __future__ import annotations

import difflib
import json
import urllib.parse
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Annotated, Any

import pandas as pd
import requests
import yfinance as yf
from agent_framework import tool
from pydantic import Field

Symbol = Annotated[
    str,
    Field(
        description=(
            "Ticker symbol. Use a Yahoo suffix for non-US listings: "
            "'.NS' for India/NSE (RELIANCE.NS), '.L' for London, '.T' for Tokyo. "
            "Prefer '.NS' over '.BO' for Indian stocks."
        )
    ),
]

CURRENCY_SYMBOLS = {"USD": "$", "INR": "\u20b9", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5"}

# Yahoo serves almost no daily history for BSE ('.BO') tickers; NSE is the usable feed.
_SUFFIX_CURRENCY = {".NS": "INR", ".BO": "INR", ".L": "GBP", ".T": "JPY", ".HK": "HKD"}


@lru_cache(maxsize=128)
def detect_currency(symbol: str) -> str:
    """Resolve the trading currency for a ticker, falling back to the suffix."""
    for suffix, currency in _SUFFIX_CURRENCY.items():
        if symbol.upper().endswith(suffix):
            return currency
    try:
        return _yf_ticker(symbol).info.get("currency") or "USD"
    except Exception:
        return "USD"


def currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOLS.get(code.upper(), code.upper() + " ")


def _dumps(payload: Any) -> str:
    return json.dumps(payload, default=str, indent=2)


@lru_cache(maxsize=1)
def _nse_symbol_universe() -> frozenset[str]:
    """Authoritative NSE equity symbol list, used to auto-correct LLM-typo'd
    tool-call arguments (e.g. 'RELISNCE' for 'RELIANCE') before they ever hit yfinance.
    """
    try:
        from .kite.client import kite_client

        instruments = kite_client().instruments("NSE")
    except Exception:
        return frozenset()
    return frozenset(i["tradingsymbol"] for i in instruments if i.get("segment") == "NSE")


def _autocorrect_ticker(symbol: str) -> str:
    """Corrects a near-miss bare symbol ahead of a '.NS'/'.BO' suffix; leaves
    anything else (other suffixes, already-valid symbols) untouched.
    """
    upper = symbol.upper().strip()
    for suffix in (".NS", ".BO"):
        if upper.endswith(suffix):
            base = upper[: -len(suffix)]
            universe = _nse_symbol_universe()
            if universe and base not in universe:
                match = difflib.get_close_matches(base, universe, n=1, cutoff=0.8)
                if match:
                    return match[0] + suffix
            return upper
    return upper


def _yf_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(_autocorrect_ticker(symbol))


def _history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    frame = _yf_ticker(symbol).history(period=period, interval=interval)
    # Yahoo can emit a trailing bar with volume but NaN prices, which poisons iloc[-1].
    frame = frame[frame["Close"].notna()]
    if frame.empty:
        raise ValueError(f"No price history returned for {symbol!r}.")
    if len(frame) < 2:
        raise ValueError(
            f"Only {len(frame)} bar(s) of history for {symbol!r} - not enough to analyse. "
            "Yahoo serves almost no history for '.BO' (BSE) tickers; try the '.NS' (NSE) "
            "listing instead."
        )
    return frame


@tool
def get_quote(symbol: Symbol) -> str:
    """Get the latest price, day range, volume and market cap for a ticker."""
    info = _yf_ticker(symbol).info
    keys = (
        "shortName",
        "currency",
        "currentPrice",
        "previousClose",
        "open",
        "dayLow",
        "dayHigh",
        "volume",
        "averageVolume",
        "marketCap",
        "fiftyTwoWeekLow",
        "fiftyTwoWeekHigh",
    )
    return _dumps(
        {"symbol": symbol.upper(), "currency": detect_currency(symbol)}
        | {k: info.get(k) for k in keys}
    )


@tool
def get_price_history(
    symbol: Symbol,
    period: Annotated[str, Field(description="1mo, 3mo, 6mo, 1y, 2y, 5y, max.")] = "6mo",
    interval: Annotated[str, Field(description="1d, 1wk or 1mo.")] = "1d",
) -> str:
    """Get OHLCV candles plus summary return statistics over a period."""
    frame = _history(symbol, period, interval)
    close = frame["Close"]
    daily_returns = close.pct_change().dropna()
    return _dumps(
        {
            "symbol": symbol.upper(),
            "currency": detect_currency(symbol),
            "period": period,
            "interval": interval,
            "start": frame.index[0],
            "end": frame.index[-1],
            "first_close": round(float(close.iloc[0]), 4),
            "last_close": round(float(close.iloc[-1]), 4),
            "total_return_pct": round(float(close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
            "annualized_volatility_pct": round(float(daily_returns.std() * (252**0.5)) * 100, 2),
            "max_drawdown_pct": round(float((close / close.cummax() - 1).min()) * 100, 2),
            "recent_candles": [
                {
                    "date": str(idx.date()),
                    "open": round(float(row.Open), 2),
                    "high": round(float(row.High), 2),
                    "low": round(float(row.Low), 2),
                    "close": round(float(row.Close), 2),
                    "volume": int(row.Volume),
                }
                for idx, row in frame.tail(20).iterrows()
            ],
        }
    )


@tool
def get_technical_indicators(
    symbol: Symbol,
    period: Annotated[str, Field(description="Lookback window for the indicators.")] = "1y",
) -> str:
    """Compute SMA, EMA, RSI(14), MACD and Bollinger Bands for a ticker."""
    close = _history(symbol, period, "1d")["Close"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, pd.NA))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    def last(series: pd.Series) -> float | None:
        value = series.dropna()
        return round(float(value.iloc[-1]), 4) if not value.empty else None

    bars = len(close)
    unavailable = [name for window, name in ((20, "sma_20"), (50, "sma_50"), (200, "sma_200")) if bars < window]

    return _dumps(
        {
            "symbol": symbol.upper(),
            "currency": detect_currency(symbol),
            "bars_available": bars,
            "unavailable_indicators": unavailable,
            "last_close": last(close),
            "sma_20": last(sma20),
            "sma_50": last(close.rolling(50).mean()),
            "sma_200": last(close.rolling(200).mean()),
            "ema_12": last(ema12),
            "ema_26": last(ema26),
            "rsi_14": last(rsi),
            "macd": last(macd),
            "macd_signal": last(signal),
            "macd_histogram": last(macd - signal),
            "bollinger_upper": last(sma20 + 2 * std20),
            "bollinger_lower": last(sma20 - 2 * std20),
        }
    )


@tool
def get_fundamentals(symbol: Symbol) -> str:
    """Get valuation multiples, margins, growth and balance-sheet health metrics."""
    info = _yf_ticker(symbol).info
    keys = (
        "sector",
        "industry",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "priceToSalesTrailing12Months",
        "enterpriseToEbitda",
        "pegRatio",
        "profitMargins",
        "operatingMargins",
        "grossMargins",
        "returnOnEquity",
        "returnOnAssets",
        "revenueGrowth",
        "earningsGrowth",
        "debtToEquity",
        "currentRatio",
        "quickRatio",
        "freeCashflow",
        "totalRevenue",
        "dividendYield",
        "payoutRatio",
        "beta",
    )
    values = {k: info.get(k) for k in keys}
    return _dumps(
        {
            "symbol": symbol.upper(),
            "currency": detect_currency(symbol),
            "unavailable_fields": [k for k, v in values.items() if v is None],
        }
        | values
    )


@tool
def get_analyst_view(symbol: Symbol) -> str:
    """Get sell-side analyst recommendations and price targets."""
    info = _yf_ticker(symbol).info
    keys = (
        "recommendationKey",
        "recommendationMean",
        "numberOfAnalystOpinions",
        "targetLowPrice",
        "targetMeanPrice",
        "targetHighPrice",
        "currentPrice",
    )
    values = {k: info.get(k) for k in keys}
    return _dumps(
        {
            "symbol": symbol.upper(),
            "currency": detect_currency(symbol),
            "unavailable_fields": [k for k, v in values.items() if v is None],
        }
        | values
    )


@lru_cache(maxsize=128)
def _company_name(symbol: str) -> str:
    """Plain company name for news search; Yahoo tickers are useless as search terms."""
    try:
        info = _yf_ticker(symbol).info
        name = info.get("longName") or info.get("shortName") or ""
    except Exception:
        name = ""
    for suffix in (" Limited", " Ltd.", " Ltd", " Inc.", " Inc", " Corporation", " plc"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip() or symbol.split(".")[0]


def _yahoo_news(symbol: str, limit: int) -> list[dict[str, Any]]:
    corrected = _autocorrect_ticker(symbol)
    try:
        raw = yf.Search(corrected, news_count=limit).news or yf.Ticker(corrected).news
    except Exception:
        return []

    items = []
    for entry in raw[:limit]:
        content = entry.get("content", entry)
        published = content.get("pubDate") or entry.get("providerPublishTime")
        if isinstance(published, (int, float)):
            published = pd.Timestamp(published, unit="s", tz="UTC").isoformat()
        items.append(
            {
                "title": content.get("title"),
                "publisher": (content.get("provider") or {}).get("displayName")
                or entry.get("publisher"),
                "published": published,
                "summary": content.get("summary"),
            }
        )
    return [item for item in items if item["title"]]


def _google_news(symbol: str, limit: int) -> list[dict[str, Any]]:
    """Google News RSS; the India edition covers NSE/BSE names that Yahoo does not."""
    region = "IN" if symbol.upper().endswith((".NS", ".BO")) else "US"
    locale = f"hl=en-{region}&gl={region}&ceid={region}:en"
    query = urllib.parse.quote(f"{_company_name(symbol)} stock")
    url = f"https://news.google.com/rss/search?q={query}&{locale}"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        entries = ET.fromstring(response.content).findall(".//item")
    except (requests.RequestException, ET.ParseError):
        return []

    return [
        {
            "title": entry.findtext("title"),
            "publisher": entry.findtext("source"),
            "published": entry.findtext("pubDate"),
            "summary": None,
        }
        for entry in entries[:limit]
        if entry.findtext("title")
    ]


@tool
def get_recent_news(
    symbol: Symbol,
    limit: Annotated[int, Field(description="Maximum number of headlines.", ge=1, le=25)] = 10,
) -> str:
    """Get recent news headlines for a ticker to gauge market sentiment.

    Tries Yahoo first, then Google News, which is the only one of the two that
    covers Indian listings.
    """
    items = _yahoo_news(symbol, limit)
    source = "yahoo"
    if not items:
        items = _google_news(symbol, limit)
        source = "google_news"
    return _dumps(
        {
            "symbol": symbol.upper(),
            "company": _company_name(symbol),
            "source": source if items else "none",
            "headline_count": len(items),
            "headlines": items,
        }
    )


@tool
def compare_symbols(
    symbols: Annotated[list[str], Field(description="Two or more ticker symbols.")],
    period: Annotated[str, Field(description="Lookback window.")] = "1y",
) -> str:
    """Compare total return, volatility and drawdown across several tickers."""
    rows = []
    for symbol in symbols:
        try:
            close = _history(symbol, period, "1d")["Close"]
        except ValueError as exc:
            rows.append({"symbol": symbol.upper(), "error": str(exc)})
            continue
        returns = close.pct_change().dropna()
        rows.append(
            {
                "symbol": symbol.upper(),
                "total_return_pct": round(float(close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
                "annualized_volatility_pct": round(float(returns.std() * (252**0.5)) * 100, 2),
                "max_drawdown_pct": round(float((close / close.cummax() - 1).min()) * 100, 2),
            }
        )
    return _dumps({"period": period, "comparison": rows})


@tool
def size_position(
    account_equity: Annotated[float, Field(description="Total account equity.", gt=0)],
    risk_per_trade_pct: Annotated[
        float, Field(description="Percent of equity risked on this trade.", gt=0, le=100)
    ],
    entry_price: Annotated[float, Field(description="Planned entry price.", gt=0)],
    stop_loss_price: Annotated[float, Field(description="Planned stop-loss price.", gt=0)],
    currency: Annotated[
        str, Field(description="Currency code of the prices and equity, e.g. USD or INR.")
    ] = "USD",
) -> str:
    """Compute share count and exposure for a risk-defined trade.

    Equity and prices must be in the same currency; no FX conversion is applied.
    """
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        raise ValueError("Entry price and stop-loss price must differ.")
    risk_budget = account_equity * risk_per_trade_pct / 100
    shares = int(risk_budget // risk_per_share)
    return _dumps(
        {
            "currency": currency.upper(),
            "risk_budget": round(risk_budget, 2),
            "risk_per_share": round(risk_per_share, 4),
            "shares": shares,
            "position_value": round(shares * entry_price, 2),
            "position_pct_of_equity": round(shares * entry_price / account_equity * 100, 2),
        }
    )


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@lru_cache(maxsize=1)
def _nse_session() -> requests.Session:
    """NSE rejects bare API calls; it needs cookies from a prior homepage visit."""
    session = requests.Session()
    session.headers.update({"User-Agent": _BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"})
    session.get("https://www.nseindia.com", timeout=15)
    return session


def _nse_announcements(nse_symbol: str) -> list[dict[str, Any]]:
    url = "https://www.nseindia.com/api/corporate-announcements"
    params = {"index": "equities", "symbol": nse_symbol}
    try:
        response = _nse_session().get(url, params=params, timeout=20)
        response.raise_for_status()
        records = response.json()
    except (requests.RequestException, ValueError):
        return []
    return records if isinstance(records, list) else []


@tool
def get_corporate_announcements(
    symbol: Symbol,
    limit: Annotated[int, Field(description="Maximum number of filings.", ge=1, le=30)] = 15,
) -> str:
    """Get official NSE corporate announcements (filings, results, board meetings).

    India/NSE only - this is exchange-filed primary source material, more
    authoritative than press coverage. Returns an explanatory note for other markets.
    """
    upper = symbol.upper()
    if not upper.endswith(".NS"):
        hint = (
            "NSE announcements cover '.NS' tickers only. "
            "For a BSE ('.BO') listing, try the NSE symbol if the company is dual-listed."
            if upper.endswith(".BO")
            else "NSE announcements are available for Indian ('.NS') listings only."
        )
        return _dumps({"symbol": upper, "supported": False, "announcement_count": 0, "note": hint})

    records = _nse_announcements(upper.removesuffix(".NS"))
    return _dumps(
        {
            "symbol": upper,
            "supported": True,
            "source": "nseindia.com",
            "announcement_count": len(records[:limit]),
            "announcements": [
                {
                    "date": item.get("an_dt"),
                    "category": item.get("desc"),
                    "detail": (item.get("attchmntText") or "").strip()[:500] or None,
                    "attachment": item.get("attchmntFile"),
                }
                for item in records[:limit]
            ],
        }
    )


MARKET_DATA_TOOLS = [get_quote, get_price_history, compare_symbols]
TECHNICAL_TOOLS = [get_technical_indicators, get_price_history, get_quote]
FUNDAMENTAL_TOOLS = [get_fundamentals, get_quote, get_analyst_view]
SENTIMENT_TOOLS = [get_recent_news, get_analyst_view, get_corporate_announcements]
RISK_TOOLS = [size_position, get_price_history, get_technical_indicators]
