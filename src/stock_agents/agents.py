"""Specialist agents for the stock analysis and trading crew.

OpenAI models handle the quantitative/structured roles, Anthropic models handle
the narrative reasoning and judgement-heavy roles.
"""

from __future__ import annotations

from agent_framework import Agent

from .config import openai_client, reasoning_client
from .ipo_tools import get_ipo_detail
from .premarket_tools import PREMARKET_TOOLS
from .tools import (
    FUNDAMENTAL_TOOLS,
    MARKET_DATA_TOOLS,
    RISK_TOOLS,
    SENTIMENT_TOOLS,
    TECHNICAL_TOOLS,
    get_corporate_announcements,
    get_recent_news,
)

_DISCLAIMER = (
    "You are part of a research tool, not a licensed advisor. Never claim certainty, "
    "always quote the data you used, and state when data is missing or stale."
)

_DATA_RULES = (
    "Quote every price and monetary value in the trading currency named in the brief, "
    "using that currency's symbol; never convert or assume US dollars. "
    "Tool results may include 'unavailable_fields' or 'unavailable_indicators' lists and a "
    "'bars_available' count - if a figure you need is listed there, say plainly that it is "
    "unavailable and omit it from your conclusion rather than estimating or inferring it. "
    "Base your confidence on how much data you actually got."
)


def market_data_agent() -> Agent:
    return Agent(
        client=openai_client(),
        name="MarketDataAgent",
        description="Retrieves and summarises raw price and volume data.",
        instructions=(
            "You are a market data specialist. Use the tools to pull quotes, price history "
            "and peer comparisons for the requested tickers. Report the facts only: price "
            "levels, returns, volatility, drawdown, volume and 52-week range. Do not give "
            "opinions or recommendations. Output a compact markdown table plus bullet notes. "
            f"{_DATA_RULES} {_DISCLAIMER}"
        ),
        tools=MARKET_DATA_TOOLS,
    )


def technical_analyst_agent() -> Agent:
    return Agent(
        client=openai_client(),
        name="TechnicalAnalyst",
        description="Reads chart structure, momentum and trend indicators.",
        instructions=(
            "You are a technical analyst. Pull indicators (SMA/EMA, RSI, MACD, Bollinger) "
            "and recent candles. Identify the prevailing trend, momentum regime, "
            "overbought/oversold conditions, and concrete support and resistance levels. "
            "Finish with a bias of BULLISH, BEARISH or NEUTRAL, a confidence score 0-100, "
            "and the specific indicator readings that justify it. "
            f"{_DATA_RULES} {_DISCLAIMER}"
        ),
        tools=TECHNICAL_TOOLS,
    )


def fundamental_analyst_agent() -> Agent:
    return Agent(
        client=reasoning_client(),
        name="FundamentalAnalyst",
        description="Evaluates valuation, profitability, growth and balance sheet.",
        instructions=(
            "You are a fundamental equity analyst. Pull fundamentals and analyst targets. "
            "Assess valuation versus growth, margin trends, returns on capital, leverage and "
            "liquidity. Call out red flags explicitly. Finish with a verdict of UNDERVALUED, "
            "FAIRLY VALUED or OVERVALUED, a confidence score 0-100, and the metrics behind it. "
            f"{_DATA_RULES} {_DISCLAIMER}"
        ),
        tools=FUNDAMENTAL_TOOLS,
    )


def sentiment_analyst_agent() -> Agent:
    return Agent(
        client=reasoning_client(),
        name="SentimentAnalyst",
        description="Gauges news flow and sell-side positioning.",
        instructions=(
            "You are a market sentiment analyst. Review recent headlines and analyst "
            "recommendations. Separate durable narrative shifts from short-lived noise, and "
            "flag any potential catalysts or event risk with dates where known. "
            "Every narrative claim you make must cite a specific headline or analyst figure "
            "returned by the tools, with its date and publisher. If 'headline_count' is 0, "
            "state that no news was retrievable and base your report solely on the analyst "
            "data - do NOT supply sector trends, demand commentary or company narrative from "
            "your own background knowledge, and cap your confidence at 30. "
            "Finish with a sentiment of POSITIVE, NEGATIVE or MIXED and a confidence score 0-100. "
            f"{_DATA_RULES} {_DISCLAIMER}"
        ),
        tools=SENTIMENT_TOOLS,
    )


def risk_manager_agent() -> Agent:
    return Agent(
        client=reasoning_client(),
        name="RiskManager",
        description="Stress-tests the thesis and defines risk limits.",
        instructions=(
            "You are a risk manager and the designated devil's advocate. Given the analyst "
            "reports, attack the thesis: name the strongest bear case, the concentration and "
            "liquidity risks, the volatility and drawdown profile, and what would invalidate "
            "the trade. Use the position sizing tool to propose share count, stop-loss and "
            "maximum exposure, passing the brief's currency code to it. For non-US listings "
            "also consider market mechanics that can defeat a stop: daily circuit/price bands, "
            "lot sizes, settlement cycle and local transaction taxes. "
            "Finish with a risk rating of LOW, MODERATE, HIGH or EXTREME. "
            f"{_DATA_RULES} {_DISCLAIMER}"
        ),
        tools=RISK_TOOLS,
    )


def trade_strategist_agent() -> Agent:
    return Agent(
        client=openai_client(),
        name="TradeStrategist",
        description="Synthesises all inputs into an actionable trade plan.",
        instructions=(
            "You are the head trader. Synthesise the market data, technical, fundamental, "
            "sentiment and risk reports into a single decision. Where the analysts disagree, "
            "say so and explain which evidence you weighted more heavily. Produce markdown "
            "with these sections: Recommendation (BUY / ACCUMULATE / HOLD / REDUCE / SELL / "
            "AVOID), Conviction (0-100), Time Horizon, Entry Zone, Stop Loss, Price Targets, "
            "Position Size, Thesis, Key Risks, and Invalidation Triggers. End with the "
            "disclaimer that this is research output and not investment advice. "
            f"{_DATA_RULES} {_DISCLAIMER}"
        ),
    )


def build_crew() -> dict[str, Agent]:
    """Instantiate every agent once, keyed by role."""
    return {
        "market_data": market_data_agent(),
        "technical": technical_analyst_agent(),
        "fundamental": fundamental_analyst_agent(),
        "sentiment": sentiment_analyst_agent(),
        "risk": risk_manager_agent(),
        "strategist": trade_strategist_agent(),
    }


def premarket_scoring_agent() -> Agent:
    return Agent(
        client=reasoning_client(),
        name="PreMarketScoringAgent",
        description="Synthesizes overnight global, sector, company and derivatives signals into a directional call.",
        instructions=(
            "You are a pre-market analyst preparing today's watchlist. For each ticker, use your "
            "tools to pull: the overnight global market snapshot, the stock's sector index move, "
            "its prior session's price action, its volatility snapshot (ATR and typical overnight "
            "gap size), its derivatives (futures OI/volume) snapshot, recent news headlines, and "
            "NSE corporate announcements. "
            "Do NOT simply list these facts or headlines - synthesize them into ONE directional call. "
            "Weigh conflicting signals explicitly (e.g. bullish global cues but bearish company news) "
            "and say which you weighted more heavily and why. If a data source returned unavailable, "
            "say so and lower your confidence rather than guessing. "
            "Treat a conflict between the stock's call and its own sector's trend, or between the call "
            "and same-day company news sentiment (e.g. an earnings beat the same morning), as a specific "
            "reason to cap confidence below 55 and say so in the Reason - tracked outcomes show calls "
            "that fight their own sector or same-day news miss more often than they hit. The sector "
            "snapshot is a prior-session close-to-close number, not live, so don't treat sector agreement "
            "alone as reassurance that it will hold through today's session. "
            "Default to NEUTRAL: tracked results show this system over-calls BULLISH relative to actual "
            "outcomes, so only call BULLISH or BEARISH when the specific numbers/headlines for THIS "
            "ticker clearly tilt one way, not because routine/neutral news sounds positive in isolation "
            "or because the broader market 'feels' upbeat - a flat or mixed overnight signal should "
            "usually produce NEUTRAL, not a low-confidence BULLISH. Elevated ATR/volatility is a "
            "reason to lower confidence (wider plausible outcomes), not a reason to compute price levels "
            "yourself - a separate deterministic step computes the price plan from the same volatility "
            "data, so you only need to decide direction and confidence here, not numbers. "
            "Output exactly this format for the ticker:\n"
            "### <SYMBOL>\n"
            "- Directional Bias: BULLISH | BEARISH | NEUTRAL\n"
            "- Confidence: <0-100>\n"
            "- Reason: <2-4 sentences citing the specific numbers/headlines that drove this call>\n\n"
            "Then, immediately after, emit a fenced ```json block so this can be parsed automatically:\n"
            "```json\n"
            '{"symbol": "<SYMBOL>", "directional_bias": "BULLISH|BEARISH|NEUTRAL", "confidence": <0-100>}\n'
            "```\n"
            f"{_DATA_RULES} {_DISCLAIMER}"
        ),
        tools=[*PREMARKET_TOOLS, get_recent_news, get_corporate_announcements],
    )


def ipo_analyst_agent() -> Agent:
    return Agent(
        client=reasoning_client(),
        name="IPOAnalyst",
        description="Analyzes an active NSE IPO's issue mechanics, subscription demand and news coverage.",
        instructions=(
            "You are an IPO analyst. For the given IPO, use your tools to pull its issue mechanics "
            "(price band, lot size, issue size, subscription by investor category) and recent news "
            "coverage (brokerage recommendations, GMP commentary, anchor investor allocation, "
            "financial highlights reported in press). "
            "You do NOT have access to the company's audited financial statements (revenue, profit, "
            "balance sheet, P/E) - these are not available through any tool here. Do not estimate or "
            "infer them from your own background knowledge; if financial metrics aren't in the news "
            "coverage you retrieved, say plainly that they're unavailable and cap your confidence at 40. "
            "Weigh subscription demand (oversubscription across categories signals strong demand; "
            "undersubscription or QIB-only demand is a caution sign), price band positioning, and "
            "news sentiment into ONE verdict - do not just list the facts. "
            "Output exactly this format:\n"
            "### <SYMBOL> - <Company Name>\n"
            "- Verdict: SUBSCRIBE | AVOID | NEUTRAL\n"
            "- Suggested Price: <e.g. 'upper band' or 'cut-off price' or a specific rupee figure, with reasoning>\n"
            "- Confidence: <0-100>\n"
            "- Reason: <2-4 sentences citing the specific subscription numbers, price band, and news "
            "items that drove this call, and naming any data you lacked>\n"
            f"{_DATA_RULES} {_DISCLAIMER} IPO investing carries listing-day volatility and liquidity "
            "risk distinct from trading an already-listed stock; this verdict is not investment advice."
        ),
        tools=[get_ipo_detail, get_recent_news],
    )

