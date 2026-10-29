"""Agent tool wrapping NSE's per-IPO detail endpoint: price band, lot size,
issue size, subscription (bidDetails), registrar. No audited financial
statements are available here - the agent must rely on this plus news
coverage, and must say so when financial figures aren't obtainable.
"""

from __future__ import annotations

from typing import Annotated

from agent_framework import tool
from pydantic import Field

from .ipo_tracker import _nse_session
from .tools import _dumps


@tool
def get_ipo_detail(
    symbol: Annotated[str, Field(description="NSE IPO symbol, e.g. BLSPOLY (from the current-issue list).")],
) -> str:
    """Get an NSE IPO's issue mechanics: price band, lot size, issue size, issue
    type, registrar, and subscription (bidDetails - shares offered vs shares bid,
    by investor category). This is issue logistics, not audited financial
    statements (no revenue/profit/balance sheet data is available here).
    """
    response = _nse_session().get(
        "https://www.nseindia.com/api/ipo-detail", params={"symbol": symbol.upper(), "series": "EQ"}, timeout=20
    )
    response.raise_for_status()
    data = response.json()
    info = {item["title"]: item["value"] for item in data.get("issueInfo", {}).get("dataList", []) if item.get("title")}
    return _dumps(
        {
            "symbol": symbol.upper(),
            "price_range": info.get("Price Range"),
            "issue_size": info.get("Issue Size"),
            "issue_type": info.get("Issue Type"),
            "bid_lot": info.get("Bid Lot"),
            "face_value": info.get("Face Value"),
            "issue_period": info.get("Issue Period"),
            "registrar": info.get("Name of the Registrar"),
            "subscription_by_category": data.get("bidDetails", []),
            "note": "Issue mechanics and live subscription only - no audited financial statements available.",
        }
    )
