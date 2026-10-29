"""CLI entry point for the multi-agent stock analysis crew."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_framework import UsageDetails, add_usage_details  # noqa: E402
from agent_framework.exceptions import ChatClientInvalidAuthException  # noqa: E402

from stock_agents import ALL_ROLES, analyze_portfolio  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent stock analysis and trade planning.",
        epilog="Non-US tickers need a Yahoo suffix, e.g. RELIANCE.NS for India/NSE.",
    )
    parser.add_argument("symbols", nargs="+", help="Ticker symbols, e.g. AAPL or RELIANCE.NS")
    parser.add_argument(
        "--equity", type=float, default=100_000.0, help="Account equity, in the ticker's currency."
    )
    parser.add_argument("--risk", type=float, default=1.0, help="Percent of equity risked.")
    parser.add_argument("--horizon", default="3-6 months", help="Investment horizon.")
    parser.add_argument(
        "--currency", default=None, help="Override the detected currency, e.g. INR."
    )
    parser.add_argument(
        "--agents",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated subset of agents to run. "
            f"Choices: {', '.join(ALL_ROLES)}. Default: all."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("reports"), help="Report output folder.")
    parser.add_argument("--full", action="store_true", help="Print every agent report.")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    try:
        results = await analyze_portfolio(
            [s.upper() for s in args.symbols],
            account_equity=args.equity,
            risk_per_trade_pct=args.risk,
            horizon=args.horizon,
            currency=args.currency,
            roles=args.agents.split(",") if args.agents else None,
            on_progress=lambda stage: print(f"  ... {stage}", file=sys.stderr),
        )
    except ChatClientInvalidAuthException as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr)
        print(
            "Check OPENAI_API_KEY and ANTHROPIC_API_KEY in your .env file.",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = args.out / f"{result.symbol}.md"
        path.write_text(result.to_markdown(), encoding="utf-8")
        print("\n" + "=" * 70)
        # Without the strategist there is no final plan, so fall back to the full report.
        if args.full or not result.final_plan:
            print(result.to_markdown(include_usage=False))
        else:
            print(f"# {result.symbol}\n\n{result.final_plan}")
        print(f"\nFull report saved to {path}")

    print("\n" + "=" * 70)
    grand_total = UsageDetails()
    for result in results:
        print(f"\nToken usage - {result.symbol}\n{result.usage_table()}")
        grand_total = add_usage_details(grand_total, result.total_usage)
    if len(results) > 1:
        print(
            f"\nAll tickers: {grand_total.get('total_token_count') or 0:,} tokens "
            f"({grand_total.get('input_token_count') or 0:,} in / "
            f"{grand_total.get('output_token_count') or 0:,} out, incl. "
            f"{grand_total.get('reasoning_output_token_count') or 0:,} reasoning)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
