"""Pre-market report: synthesizes overnight global/sector/company/derivatives
signals into a scored, directional watchlist (LLM-based, reasoning-heavy), plus
a separate deterministic pre-open price-discovery addendum.

Workflow:
  a. collect overnight information       -> get_global_market_snapshot tool
  b. identify stock-specific catalysts   -> get_recent_news / get_corporate_announcements
  c. calculate directional score         -> premarket_scoring_agent synthesis
  d. observe pre-open price discovery    -> build_preopen_addendum (9:08-9:12 IST only)

This is a research tool (part of the LLM pipeline), not the deterministic Kite
trading engine - it never places orders.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from agent_framework import Agent

from .agents import premarket_scoring_agent
from .premarket_calibration import HISTORY_PATH, SECTOR_MAP, build_calibration_note
from .premarket_tools import (
    compute_price_plan,
    get_global_market_snapshot,
    get_market_wide_news,
    get_preopen_quote,
    get_prior_session_summary,
    get_volatility_snapshot,
)

# Last overnight report's structured calls, read by preopen to check the actual
# gap against what was projected the night before. Lives in reports/ (repo root)
# rather than next to this module, so it ships alongside the markdown reports -
# including as a CI workflow artifact.
STATE_PATH = Path(__file__).resolve().parents[2] / "reports" / ".premarket_latest.json"


def build_premarket_brief(symbol: str, market_context: str | None = None) -> str:
    sector = SECTOR_MAP.get(symbol.upper(), "UNMAPPED")
    brief = (
        f"NSE symbol: {symbol.upper()}\n"
        f"Sector: {sector}\n\n"
        "Gather this stock's sector index move, its prior session's price action, "
        "its derivatives snapshot, and overnight company-specific news and NSE "
        "corporate announcements. Then produce your directional call for today's "
        "session in the required format."
    )
    if market_context:
        brief = f"{brief}\n\n{market_context}"
    note = build_calibration_note(symbol)
    return f"{brief}\n\n{note}" if note else brief


def _build_deterministic_price_plan(symbol: str, bias: str) -> dict | None:
    """Price plan computed in code from get_volatility_snapshot/get_prior_session_summary's
    own numbers - never from the LLM's arithmetic, so the range can't collapse to a
    degenerate single point and the stop/target math is always consistent.
    """
    prior_session = json.loads(get_prior_session_summary.func(symbol))
    volatility = json.loads(get_volatility_snapshot.func(symbol))
    if not prior_session.get("available") or not volatility.get("available"):
        return None
    return compute_price_plan(
        prior_close=prior_session["close"],
        atr=volatility["atr"],
        atr_pct=volatility["atr_pct_of_price"],
        typical_gap_up_pct=volatility.get("typical_gap_up_pct"),
        typical_gap_down_pct=volatility.get("typical_gap_down_pct"),
        bias=bias,
    )


def _format_price_plan(plan: dict) -> str:
    move = plan["expected_move_pct"]
    return (
        f"- Expected Move: {move:+.2f}% (toward target \u20b9{plan['target']})\n"
        f"- Projected Opening Price: \u20b9{plan['projected_open_low']}\u2013\u20b9{plan['projected_open_high']}\n"
        f"- Suggested Entry: \u20b9{plan['entry']}\n"
        f"- Suggested Stop Loss: \u20b9{plan['stop']}\n"
        f"- Suggested Target: \u20b9{plan['target']}\n"
    )


_PRICE_PLAN_NOTE = (
    "(Price plan computed deterministically from ATR/typical-gap history, not estimated by "
    "the LLM - still a volatility-based planning exercise, not a prediction.)"
)


def _insert_price_plan(text: str, plan: dict) -> str:
    """Splices the deterministic plan between Confidence and Reason so field order
    matches Directional Bias / Confidence / Expected Move / ... / Reason.
    """
    block = _format_price_plan(plan)
    match = re.search(r"- Confidence:[^\n]*\n", text)
    if match:
        text = text[: match.end()] + block + text[match.end() :]
    else:
        text = f"{text.rstrip()}\n\n{block}"
    return f"{text.rstrip()}\n\n{_PRICE_PLAN_NOTE}\n"


def _approx_tokens(text: str) -> int:
    """Deterministic, provider-agnostic token size estimate (~4 chars/token).

    Used only to split a single LLM call's *real* token usage (below) into
    segments the provider itself doesn't break down - e.g. how much of one
    call's input was the shared market snapshot vs. everything else. Always
    returns the same number for the same text, so segment proportions are
    reproducible across runs even though the absolute count is an estimate.
    """
    return max(1, len(text) // 4)


def _usage_counts(response) -> dict[str, int]:
    """Extract the provider-reported (exact, billed) token counts from an
    agent.run() response. This is the authoritative number for "how many
    tokens did this call actually use" - not an estimate.
    """
    usage = getattr(response, "usage_details", None) or {}
    input_tokens = usage.get("input_token_count") or 0
    output_tokens = usage.get("output_token_count") or 0
    total_tokens = usage.get("total_token_count") or (input_tokens + output_tokens)
    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}


def _build_token_usage_section(market_context: str, symbol_usage: list[dict]) -> str:
    """Per-overnight-call token report: exact provider-reported totals, plus a
    deterministic estimate of how much of the input was the shared market
    context (fetched once, resent in every symbol's prompt) vs. everything
    else (instructions, sector/news/derivatives tool output, calibration note).
    """
    n = len(symbol_usage)
    if n == 0:
        return ""

    total_input = sum(u["input"] for u in symbol_usage)
    total_output = sum(u["output"] for u in symbol_usage)
    total_tokens = sum(u["total"] for u in symbol_usage)
    have_real_usage = total_tokens > 0

    lines = ["\n## Token Usage\n"]
    if not have_real_usage:
        lines.append(
            "_Provider did not report token usage for this run (usage_details was empty); "
            "no counts available._\n"
        )
        return "\n".join(lines)

    market_context_tokens_est = _approx_tokens(market_context)
    shared_context_total_est = min(market_context_tokens_est * n, total_input)
    other_input_est = max(total_input - shared_context_total_est, 0)

    lines.append(f"- **Total tokens this run**: {total_tokens:,} (input: {total_input:,}, output: {total_output:,})")
    lines.append(f"- **Symbols scored**: {n} (avg {total_tokens // n:,} tokens/symbol)")
    lines.append("- **Segment breakdown** (input tokens; shared-context split is a deterministic size estimate, "
                  "the totals above are exact provider-reported counts):")
    lines.append(
        f"  - Shared market context (global snapshot + market-wide news, repeated in every symbol's prompt): "
        f"~{shared_context_total_est:,} ({shared_context_total_est * 100 // max(total_input, 1)}% of input)"
    )
    lines.append(
        f"  - Per-symbol instructions, tool output (sector/prior session/volatility/derivatives/news) and "
        f"calibration note: ~{other_input_est:,} ({other_input_est * 100 // max(total_input, 1)}% of input)"
    )
    lines.append(f"  - Model output (reasoning + response text): {total_output:,}")

    lines.append("\n| Symbol | Input tokens | Output tokens | Total tokens |")
    lines.append("|---|---|---|---|")
    for u in symbol_usage:
        lines.append(f"| {u['symbol']} | {u['input']:,} | {u['output']:,} | {u['total']:,} |")

    return "\n".join(lines) + "\n"


async def build_overnight_report(
    symbols: list[str], agent: Agent | None = None, on_symbol_done=None
) -> str:
    """Parts a-c of the workflow: overnight catalysts through directional score.
    Runnable any time before market open (or any time at all). Also persists each
    symbol's structured call (plus a deterministically-computed price plan) to
    STATE_PATH so build_preopen_addendum can check the actual gap against it the
    next morning.

    Each symbol takes ~30-90s (multiple tool calls + reasoning); pass on_symbol_done
    to get live progress instead of waiting silently for all symbols to finish.
    """
    agent = agent or premarket_scoring_agent()
    # Fetched once for the whole batch, not per symbol - identical for every symbol
    # in a run, so refetching per symbol would just burn tokens/network for the
    # same answer (see the note in premarket_tools.PREMARKET_TOOLS).
    market_context = (
        "Overnight global market snapshot:\n"
        f"{get_global_market_snapshot.func()}\n\n"
        "Recent broad market-wide news (index-level rally/selloff drivers, FII/DII "
        "flows, macro catalysts):\n"
        f"{get_market_wide_news.func()}"
    )
    sections = [f"# Pre-Market Report \u2014 {datetime.now().date()}\n"]
    calls: dict[str, dict] = {}
    symbol_usage: list[dict] = []
    for i, symbol in enumerate(symbols, 1):
        response = await agent.run(build_premarket_brief(symbol, market_context))
        text = response.text
        parsed = _extract_json_block(text)

        usage = _usage_counts(response)
        symbol_usage.append({"symbol": symbol.upper(), **usage})

        if parsed and parsed.get("directional_bias"):
            plan = _build_deterministic_price_plan(symbol, parsed["directional_bias"])
            if plan:
                text = _insert_price_plan(text, plan)
                parsed.update(plan)

        sections.append(text)
        if parsed:
            calls[symbol.upper()] = parsed
        if on_symbol_done:
            on_symbol_done(i, len(symbols), symbol, text)

    if calls:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({"generated_at": str(datetime.now()), "calls": calls}, indent=2))

    skew_warning = _check_bias_skew(calls)
    if skew_warning:
        sections.append(skew_warning)

    sections.append(_build_token_usage_section(market_context, symbol_usage))

    return "\n".join(sections)


def _check_bias_skew(calls: dict[str, dict]) -> str | None:
    """Logged history shows this agent over-calls BULLISH; flag it in the report itself
    when one direction dominates the batch, instead of relying on a human to notice.
    """
    directional = [c["directional_bias"] for c in calls.values() if c.get("directional_bias") in ("BULLISH", "BEARISH")]
    if len(directional) < 5:
        return None
    bullish_frac = directional.count("BULLISH") / len(directional)
    bearish_frac = directional.count("BEARISH") / len(directional)
    dominant, frac = ("BULLISH", bullish_frac) if bullish_frac >= bearish_frac else ("BEARISH", bearish_frac)
    if frac < 0.65:
        return None
    return (
        f"\n\u26a0 **Skew warning**: {frac * 100:.0f}% of directional calls in this batch are {dominant}. "
        "Logged history (see `premarket stats`) shows this system tends to over-call one direction "
        "relative to realized outcomes - treat a one-sided watchlist like this with extra caution, and "
        "check the actual index/sector move before trusting the batch.\n"
    )


def _extract_json_block(text: str) -> dict | None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _load_latest_calls() -> dict[str, dict]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text()).get("calls", {})
    except (json.JSONDecodeError, OSError):
        return {}


def build_preopen_addendum(symbols: list[str]) -> str:
    """Part d of the workflow: pre-open price discovery. Deterministic, not LLM -
    only meaningful during NSE's 9:08-9:12 IST pre-open order-matching/price-discovery
    sub-phase (9:00-9:08 is order collection only - no real price yet); outside that
    window it still runs but reports a live/last quote and says so explicitly.
    Only symbols with a saved directional (BULLISH/BEARISH) overnight call and a
    projected opening range are included - NEUTRAL calls have no range to check
    the indicative price against.
    """
    saved_calls = _load_latest_calls()
    rows = []
    skipped = []
    for symbol in symbols:
        quote = get_preopen_quote(symbol)
        if not quote.get("available"):
            skipped.append(f"- **{symbol.upper()}**: unavailable ({quote.get('note')})")
            continue

        call = saved_calls.get(symbol.upper())
        low = call.get("projected_open_low") if call else None
        high = call.get("projected_open_high") if call else None
        indicative = quote.get("indicative_price")
        if not call or low is None or high is None or indicative is None:
            continue

        change = quote.get("change_pct")
        change_str = f"{change:+.2f}%" if change is not None else "n/a"
        flag = "" if quote.get("in_preopen_window") else " \u26a0"
        range_check = "within range" if low <= indicative <= high else ("above range" if indicative > high else "below range")
        rows.append(
            (
                symbol.upper(),
                f"{call.get('directional_bias')} ({call.get('confidence')}%)",
                indicative,
                quote.get("prior_close"),
                change_str,
                f"{low}-{high}",
                range_check + flag,
            )
        )

    lines = [f"# Pre-Open Price Discovery \u2014 {datetime.now()}\n"]
    if rows:
        lines.append("| Symbol | Bias (conf) | Indicative | Prior close | Chg | Projected range | Status |")
        lines.append("|---|---|---|---|---|---|---|")
        for symbol, bias, indicative, prior_close, change_str, range_str, status in rows:
            lines.append(f"| {symbol} | {bias} | {indicative} | {prior_close} | {change_str} | {range_str} | {status} |")
    else:
        lines.append("No directional (BULLISH/BEARISH) overnight calls with a projected range to check.")

    if skipped:
        lines.append("\n" + "\n".join(skipped))

    return "\n".join(lines)



def _fetch_actual_outcome(symbol: str) -> dict | None:
    """Today's realized OHLC and last price from Kite - ground truth to score a saved
    call against. actual_change_pct is last-price-based (today's net result so far),
    not open-based, so it reflects the full session rather than just the opening gap.
    """
    from .kite.client import kite_client

    quote_key = f"NSE:{symbol.upper()}"
    try:
        quote = kite_client().quote([quote_key]).get(quote_key, {})
    except Exception:
        return None
    ohlc = quote.get("ohlc", {})
    actual_open, prior_close = ohlc.get("open"), ohlc.get("close")
    last_price = quote.get("last_price")
    if not actual_open or not prior_close or not last_price:
        return None
    return {
        "actual_open": actual_open,
        "actual_high": ohlc.get("high"),
        "actual_low": ohlc.get("low"),
        "prior_close": prior_close,
        "last_price": last_price,
        "actual_change_pct": round((last_price / prior_close - 1) * 100, 2),
    }


def log_outcomes(symbols: list[str] | None = None) -> str:
    """Scores each saved overnight call (direction + projected range + target/stop)
    against today's latest price and appends one record per symbol to HISTORY_PATH.
    Run near/after market close for the final result (direction is based on last
    price vs prior close, not just the opening gap); call summarize_history() later
    (across many logged sessions) to see whether this system actually has directional
    skill, rather than judging it off single misses like HCLTECH.
    """
    saved = _load_latest_calls()
    target_symbols = [s.upper() for s in symbols] if symbols else list(saved.keys())

    records = []
    table_rows = []
    notes = []
    for symbol in target_symbols:
        call = saved.get(symbol)
        if not call:
            notes.append(f"- **{symbol}**: no saved overnight call to compare against")
            continue
        outcome = _fetch_actual_outcome(symbol)
        if not outcome:
            notes.append(f"- **{symbol}**: could not fetch actual outcome")
            continue

        bias = call.get("directional_bias")
        chg = outcome["actual_change_pct"]
        direction_correct = None
        if bias == "BULLISH":
            direction_correct = chg > 0
        elif bias == "BEARISH":
            direction_correct = chg < 0

        low, high = call.get("projected_open_low"), call.get("projected_open_high")
        range_hit = low <= outcome["actual_open"] <= high if low is not None and high is not None else None

        target, stop = call.get("target"), call.get("stop")
        day_high, day_low = outcome.get("actual_high"), outcome.get("actual_low")
        target_hit = stop_hit = None
        if target is not None and day_high is not None and day_low is not None:
            if bias == "BULLISH":
                target_hit, stop_hit = day_high >= target, day_low <= stop
            elif bias == "BEARISH":
                target_hit, stop_hit = day_low <= target, day_high >= stop

        record = {
            "date": str(datetime.now().date()),
            "symbol": symbol,
            "directional_bias": bias,
            "confidence": call.get("confidence"),
            "projected_open_low": low,
            "projected_open_high": high,
            "actual_open": outcome["actual_open"],
            "prior_close": outcome["prior_close"],
            "last_price": outcome["last_price"],
            "actual_change_pct": chg,
            "direction_correct": direction_correct,
            "range_hit": range_hit,
            "target_hit": target_hit,
            "stop_hit": stop_hit,
        }
        records.append(record)

        if bias in ("BULLISH", "BEARISH"):
            table_rows.append((symbol, bias, call.get("confidence"), chg, direction_correct, target_hit, stop_hit))

    if records:
        with open(HISTORY_PATH, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    lines = [f"# Outcome Log \u2014 {datetime.now()}\n"]
    if table_rows:
        scored = [row for row in table_rows if row[4] is not None]
        hits = sum(1 for row in scored if row[4])
        targets_hit = sum(1 for row in table_rows if row[5])
        stops_hit = sum(1 for row in table_rows if row[6])
        lines.append("## Summary")
        lines.append(f"- Directional accuracy: {hits}/{len(scored)} ({100 * hits / len(scored):.0f}%)" if scored else "- Directional accuracy: n/a")
        lines.append(f"- Targets hit: {targets_hit}/{len(table_rows)}")
        lines.append(f"- Stops hit: {stops_hit}/{len(table_rows)}\n")

        lines.append("| Symbol | Bias (conf) | Actual chg | Direction | Target hit | Stop hit |")
        lines.append("|---|---|---|---|---|---|")
        for symbol, bias, conf, chg, direction_correct, target_hit, stop_hit in table_rows:
            direction_str = "-" if direction_correct is None else ("\u2705" if direction_correct else "\u274c")
            target_str = "-" if target_hit is None else ("YES" if target_hit else "no")
            stop_str = "-" if stop_hit is None else ("YES" if stop_hit else "no")
            lines.append(f"| {symbol} | {bias} ({conf}%) | {chg:+.2f}% | {direction_str} | {target_str} | {stop_str} |")
    else:
        lines.append("No directional (BULLISH/BEARISH) calls to score.")

    if notes:
        lines.append("\n" + "\n".join(notes))

    return "\n".join(lines)


def summarize_history() -> str:
    """Aggregates hit-rate across every logged session - the statistic that actually
    matters, as opposed to any single day's miss or hit.
    """
    if not HISTORY_PATH.exists():
        return "No history logged yet. Run `track` after market open to start building a record."

    records = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return "No history logged yet. Run `track` after market open to start building a record."

    directional = [r for r in records if r["direction_correct"] is not None]
    direction_hits = sum(1 for r in directional if r["direction_correct"])
    ranged = [r for r in records if r["range_hit"] is not None]
    range_hits = sum(1 for r in ranged if r["range_hit"])
    n_sessions = len({r["date"] for r in records})

    lines = [f"# Pre-Market Call Accuracy \u2014 {len(records)} logged calls across {n_sessions} session(s)\n"]
    if directional:
        lines.append(f"- Directional accuracy: {direction_hits}/{len(directional)} ({100 * direction_hits / len(directional):.1f}%)")
    else:
        lines.append("- Directional accuracy: n/a (no BULLISH/BEARISH calls logged yet)")
    if ranged:
        lines.append(f"- Projected-range hit rate: {range_hits}/{len(ranged)} ({100 * range_hits / len(ranged):.1f}%)")
    else:
        lines.append("- Projected-range hit rate: n/a")

    by_symbol: dict[str, list[dict]] = {}
    for r in records:
        by_symbol.setdefault(r["symbol"], []).append(r)
    lines.append("\n## By symbol")
    for symbol, recs in sorted(by_symbol.items()):
        hits = sum(1 for r in recs if r["direction_correct"])
        scored = sum(1 for r in recs if r["direction_correct"] is not None)
        lines.append(f"- **{symbol}**: {hits}/{scored} correct across {len(recs)} logged call(s)" if scored else f"- **{symbol}**: no scored calls")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based pre-market watchlist report.")
    sub = parser.add_subparsers(dest="command", required=True)

    overnight = sub.add_parser("overnight", help="Overnight catalysts + directional score (parts a-c).")
    overnight.add_argument("symbols", nargs="+", help="NSE trading symbols, e.g. TCS RELIANCE")
    overnight.add_argument("--out", help="Write the markdown report to this file instead of stdout.")

    preopen = sub.add_parser("preopen", help="Pre-open price discovery addendum (part d), 9:08-9:12 IST only.")
    preopen.add_argument("symbols", nargs="+")
    preopen.add_argument("--out", help="Write the markdown report to this file instead of stdout.")

    track = sub.add_parser("track", help="Score the saved overnight calls against today's actual open and log the outcome.")
    track.add_argument("symbols", nargs="*", help="Optional subset; defaults to every symbol in the last saved report.")
    track.add_argument("--out", help="Write the markdown report to this file instead of stdout.")

    sub.add_parser("stats", help="Aggregate directional/range accuracy across every logged session so far.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "overnight":
        def on_symbol_done(i: int, total: int, symbol: str, text: str) -> None:
            print(f"\n[{i}/{total}] {symbol} done\n{text}", flush=True)

        report = asyncio.run(
            build_overnight_report([s.upper() for s in args.symbols], on_symbol_done=on_symbol_done)
        )
        if args.out:
            with open(args.out, "w") as f:
                f.write(report)
            print(f"\nWrote report to {args.out}")
    elif args.command == "preopen":
        report = build_preopen_addendum([s.upper() for s in args.symbols])
        if args.out:
            with open(args.out, "w") as f:
                f.write(report)
            print(f"Wrote report to {args.out}")
        else:
            print(report)
    elif args.command == "track":
        report = log_outcomes([s.upper() for s in args.symbols] if args.symbols else None)
        if args.out:
            with open(args.out, "w") as f:
                f.write(report)
            print(f"Wrote report to {args.out}")
        else:
            print(report)
    elif args.command == "stats":
        print(summarize_history())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
