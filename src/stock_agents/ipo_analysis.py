"""IPO analysis report: for every currently-active NSE IPO, synthesizes issue
mechanics, subscription demand and news coverage into a subscribe/avoid verdict
with a suggested price and confidence score.

Important limitation: no audited financial statements are available for a
pre-listing company through any tool here - this is informational, not a
replacement for reading the RHP/DRHP prospectus yourself.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime

from agent_framework import Agent

from .agents import ipo_analyst_agent
from .ipo_tracker import get_current_ipos


def build_ipo_brief(record: dict) -> str:
    return (
        f"NSE IPO symbol: {record.get('symbol')}\n"
        f"Company: {record.get('companyName')}\n"
        f"Price range: {record.get('issuePrice')}\n"
        f"Issue window: {record.get('issueStartDate')} to {record.get('issueEndDate')}\n\n"
        "Use your tools to pull this IPO's issue detail/subscription and recent news coverage, "
        "then produce your verdict in the required format."
    )


async def build_ipo_report(agent: Agent | None = None) -> str:
    agent = agent or ipo_analyst_agent()
    records = get_current_ipos()
    sections = [f"# Active IPO Analysis \u2014 {datetime.now().date()}\n"]
    if not records:
        sections.append("No active IPOs found.")
        return "\n".join(sections)

    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] Analyzing {record.get('symbol')}...", flush=True)
        response = await agent.run(build_ipo_brief(record))
        sections.append(response.text)
        print(response.text, flush=True)

    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based active-IPO subscribe/avoid report.")
    parser.add_argument("--out", help="Write the markdown report to this file instead of just stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = asyncio.run(build_ipo_report())
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nWrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
