"""Fetch current/active IPOs listed on NSE.

Not available via Kite Connect's official API - IPO applications in the Zerodha
app go through Zerodha's own internal backend, not the published Kite Connect
trading API this project otherwise uses. This instead reads NSE's public IPO
data (same nseindia.com session-handshake pattern already used in tools.py for
corporate announcements) - informational only, does not place or apply for
anything.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Any

import requests

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


def get_current_ipos() -> list[dict[str, Any]]:
    """Active/current NSE IPO issues (open for bidding right now)."""
    response = _nse_session().get("https://www.nseindia.com/api/ipo-current-issue", timeout=20)
    response.raise_for_status()
    records = response.json()
    return records if isinstance(records, list) else []


def get_upcoming_ipos() -> list[dict[str, Any]]:
    """Upcoming + current NSE IPO issues (broader list, includes not-yet-open ones)."""
    response = _nse_session().get(
        "https://www.nseindia.com/api/all-upcoming-issues", params={"category": "ipo"}, timeout=20
    )
    response.raise_for_status()
    records = response.json()
    return records if isinstance(records, list) else []


def _print_ipo_table(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No IPOs found.")
        return
    for row in records:
        print(
            f"{row.get('symbol', '?'):<12} {row.get('companyName', '?'):<40} "
            f"{row.get('status', '?'):<10} {row.get('issuePrice', '?'):<18} "
            f"{row.get('issueStartDate', '?')} - {row.get('issueEndDate', '?')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NSE IPO data (not available via Kite Connect).")
    parser.add_argument(
        "--mode", choices=["current", "upcoming"], default="current",
        help="current = open for bidding now, upcoming = broader list including not-yet-open issues.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = get_current_ipos() if args.mode == "current" else get_upcoming_ipos()
    _print_ipo_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
