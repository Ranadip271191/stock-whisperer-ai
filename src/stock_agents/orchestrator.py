"""Orchestration: fan out to the analysts, fan in to risk, then to the strategist."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from agent_framework import Agent, UsageDetails, add_usage_details

from .agents import build_crew
from .tools import currency_symbol, detect_currency

PARALLEL_PHASE = ("market_data", "technical", "fundamental", "sentiment")
FAN_IN_PHASE = ("risk", "strategist")
ALL_ROLES = (*PARALLEL_PHASE, *FAN_IN_PHASE)

AGENT_TITLES = {
    "market_data": "Market Data",
    "technical": "Technical Analysis",
    "fundamental": "Fundamental Analysis",
    "sentiment": "Sentiment Analysis",
    "risk": "Risk Assessment",
    "strategist": "Trade Plan",
}


@dataclass
class AnalysisResult:
    symbol: str
    currency: str = "USD"
    reports: dict[str, str] = field(default_factory=dict)
    usage: dict[str, UsageDetails] = field(default_factory=dict)

    @property
    def final_plan(self) -> str:
        return self.reports.get("strategist", "")

    @property
    def total_usage(self) -> UsageDetails:
        total = UsageDetails()
        for details in self.usage.values():
            total = add_usage_details(total, details)
        return total

    def usage_table(self) -> str:
        """Per-agent token breakdown; reasoning tokens are billed but absent from the text."""
        rows = [f"{'Agent':<22}{'input':>10}{'output':>10}{'reasoning':>11}{'total':>10}"]
        for role in ALL_ROLES:
            details = self.usage.get(role)
            if details is None:
                continue
            rows.append(
                f"{AGENT_TITLES[role]:<22}"
                f"{details.get('input_token_count') or 0:>10,}"
                f"{details.get('output_token_count') or 0:>10,}"
                f"{details.get('reasoning_output_token_count') or 0:>11,}"
                f"{details.get('total_token_count') or 0:>10,}"
            )
        total = self.total_usage
        rows.append("-" * 63)
        rows.append(
            f"{'TOTAL':<22}"
            f"{total.get('input_token_count') or 0:>10,}"
            f"{total.get('output_token_count') or 0:>10,}"
            f"{total.get('reasoning_output_token_count') or 0:>11,}"
            f"{total.get('total_token_count') or 0:>10,}"
        )
        return "\n".join(rows)

    def to_markdown(self, *, include_usage: bool = True) -> str:
        sections = [f"# {self.symbol} \u2014 Multi-Agent Analysis ({self.currency})\n"]
        for key, title in AGENT_TITLES.items():
            if key in self.reports:
                sections.append(f"## {title}\n\n{self.reports[key]}\n")
        if include_usage and self.usage:
            sections.append(f"## Token Usage\n\n```\n{self.usage_table()}\n```\n")
        return "\n".join(sections)


async def _ask(agent: Agent, prompt: str) -> tuple[str, UsageDetails | None]:
    response = await agent.run(prompt)
    return response.text, response.usage_details


def build_brief(
    symbol: str,
    currency: str,
    *,
    account_equity: float,
    risk_per_trade_pct: float,
    horizon: str,
) -> str:
    """The shared preamble every agent receives, whether run alone or in the full pipeline."""
    sign = currency_symbol(currency)
    return (
        f"Ticker: {symbol}\n"
        f"Trading currency: {currency}\n"
        f"Investment horizon: {horizon}\n"
        f"Account equity: {sign}{account_equity:,.0f} {currency}\n"
        f"Risk budget per trade: {risk_per_trade_pct}% of equity "
        f"({sign}{account_equity * risk_per_trade_pct / 100:,.0f} {currency})\n\n"
        f"All prices, levels and position values are in {currency}. Quote every number in "
        f"{currency} using '{sign}' and never convert to another currency.\n"
        "Produce your specialist report for this ticker."
    )


def resolve_roles(roles: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a requested agent subset, preserving canonical pipeline order."""
    if roles is None:
        return ALL_ROLES
    requested = {role.strip().lower() for role in roles if role.strip()}
    unknown = sorted(requested - set(ALL_ROLES))
    if unknown:
        raise ValueError(
            f"Unknown agent(s): {', '.join(unknown)}. Choose from: {', '.join(ALL_ROLES)}."
        )
    if not requested:
        raise ValueError("No agents selected.")
    return tuple(role for role in ALL_ROLES if role in requested)


async def analyze_symbol(
    symbol: str,
    *,
    account_equity: float = 100_000.0,
    risk_per_trade_pct: float = 1.0,
    horizon: str = "3-6 months",
    currency: str | None = None,
    roles: Sequence[str] | None = None,
    crew: dict[str, Agent] | None = None,
    on_progress=None,
) -> AnalysisResult:
    """Run the six-agent pipeline, or a subset of it, for a single ticker.

    Account equity is assumed to be in the ticker's own trading currency.
    """
    selected = resolve_roles(roles)
    crew = crew or build_crew()
    symbol = symbol.upper()
    currency = (currency or detect_currency(symbol)).upper()
    result = AnalysisResult(symbol=symbol, currency=currency)

    def progress(stage: str) -> None:
        if on_progress:
            on_progress(stage)

    brief = build_brief(
        symbol,
        currency,
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        horizon=horizon,
    )

    parallel = [role for role in PARALLEL_PHASE if role in selected]
    if parallel:
        progress(f"Running {' + '.join(parallel)}")
        outcomes = await asyncio.gather(
            *(_ask(crew[role], brief) for role in parallel),
            return_exceptions=True,
        )
        for role, outcome in zip(parallel, outcomes):
            if isinstance(outcome, BaseException):
                result.reports[role] = f"_Agent failed: {outcome}_"
                continue
            result.reports[role], details = outcome
            if details:
                result.usage[role] = details

    analyst_digest = (
        "\n\n".join(
            f"### {role.replace('_', ' ').title()} report\n{result.reports[role]}"
            for role in PARALLEL_PHASE
            if role in result.reports
        )
        or "_No analyst reports were produced in this run._"
    )

    if "risk" in selected:
        progress("Running risk manager")
        result.reports["risk"], details = await _ask(
            crew["risk"],
            f"{brief}\n\nAnalyst reports to stress-test:\n\n{analyst_digest}",
        )
        if details:
            result.usage["risk"] = details

    if "strategist" in selected:
        progress("Running trade strategist")
        risk_report = result.reports.get("risk", "_No risk report was produced in this run._")
        result.reports["strategist"], details = await _ask(
            crew["strategist"],
            f"{brief}\n\n{analyst_digest}\n\n### Risk report\n{risk_report}",
        )
        if details:
            result.usage["strategist"] = details

    progress("Done")
    return result


async def analyze_portfolio(
    symbols: list[str],
    *,
    max_concurrency: int = 3,
    **kwargs,
) -> list[AnalysisResult]:
    """Run the pipeline across several tickers with bounded concurrency."""
    crew = build_crew()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def guarded(symbol: str) -> AnalysisResult:
        async with semaphore:
            return await analyze_symbol(symbol, crew=crew, **kwargs)

    return await asyncio.gather(*(guarded(s) for s in symbols))
