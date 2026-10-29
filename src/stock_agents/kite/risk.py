"""Hard intraday risk controls: daily loss cap, trade-count cap, risk-based sizing.

State persists to a local JSON file keyed by calendar date, so a crashed and
restarted runner still respects today's limits instead of resetting them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

STATE_PATH = Path(__file__).parent / ".intraday_state.json"


@dataclass
class DayState:
    trading_date: str
    trades_taken: int = 0
    realized_pnl: float = 0.0


def _load() -> DayState:
    today = date.today().isoformat()
    if STATE_PATH.exists():
        raw = json.loads(STATE_PATH.read_text())
        if raw.get("trading_date") == today:
            return DayState(**raw)
    return DayState(trading_date=today)


def _save(state: DayState) -> None:
    STATE_PATH.write_text(json.dumps(asdict(state)))


def can_take_new_trade(
    account_equity: float,
    max_daily_loss_pct: float,
    max_trades_per_day: int,
) -> tuple[bool, str]:
    state = _load()
    max_loss = account_equity * max_daily_loss_pct / 100
    if -state.realized_pnl >= max_loss:
        return False, f"daily loss limit hit: {state.realized_pnl:.2f} <= -{max_loss:.2f}"
    if state.trades_taken >= max_trades_per_day:
        return False, f"max trades per day reached: {state.trades_taken}"
    return True, "ok"


def record_trade(realized_pnl_delta: float = 0.0) -> None:
    state = _load()
    state.trades_taken += 1
    state.realized_pnl += realized_pnl_delta
    _save(state)


def shares_for_risk(
    account_equity: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_price: float,
) -> int:
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0
    risk_budget = account_equity * risk_per_trade_pct / 100
    return int(risk_budget // risk_per_share)
