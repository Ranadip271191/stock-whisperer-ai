"""Thin wrapper around Kite Connect: instrument lookup, candles, quotes, orders."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. See .env.example for Kite credentials.")
    return value


@lru_cache(maxsize=1)
def kite_client() -> KiteConnect:
    """Access token is issued by the daily login flow - see kite/login.py."""
    kite = KiteConnect(api_key=_env("KITE_API_KEY"))
    kite.set_access_token(_env("KITE_ACCESS_TOKEN"))
    return kite


@lru_cache(maxsize=1)
def _instrument_map() -> dict[str, dict[str, Any]]:
    """tradingsymbol -> instrument record, NSE equities only. One network call per process."""
    return {row["tradingsymbol"]: row for row in kite_client().instruments("NSE")}


def _instrument(tradingsymbol: str) -> dict[str, Any]:
    row = _instrument_map().get(tradingsymbol.upper())
    if row is None:
        raise ValueError(f"{tradingsymbol!r} not found in the NSE instrument list.")
    return row


def instrument_token(tradingsymbol: str) -> int:
    return int(_instrument(tradingsymbol)["instrument_token"])


def round_to_tick(price: float, tick: float = 0.05) -> float:
    """NSE equity tick size is 0.05; orders at other prices are rejected."""
    return round(round(price / tick) * tick, 2)


def historical_candles(
    tradingsymbol: str,
    interval: str = "5minute",
    days: int = 10,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """OHLCV candles.

    Pass from_date/to_date for an explicit window (what the backtester's
    chunking needs); otherwise defaults to the trailing `days` calendar days
    ending now (what the live signal needs).
    interval: minute, 3minute, 5minute, 15minute, 30minute, 60minute, day.
    """
    token = instrument_token(tradingsymbol)
    if to_date is None:
        to_date = datetime.now()
    if from_date is None:
        from_date = to_date - timedelta(days=days)
    return kite_client().historical_data(token, from_date, to_date, interval)


def live_quote(tradingsymbol: str) -> dict[str, Any]:
    key = f"NSE:{tradingsymbol.upper()}"
    return kite_client().quote([key])[key]


def place_market_order(
    tradingsymbol: str,
    transaction_type: str,
    quantity: int,
    product: str = "MIS",
) -> str:
    """Place a market order. Returns the Kite order_id."""
    kite = kite_client()
    return kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NSE,
        tradingsymbol=tradingsymbol.upper(),
        transaction_type=transaction_type,
        quantity=quantity,
        product=product,
        order_type=kite.ORDER_TYPE_MARKET,
    )


def place_limit_order(
    tradingsymbol: str,
    transaction_type: str,
    quantity: int,
    price: float,
    product: str = "MIS",
) -> str:
    """Place a limit order at a target price - fills only at that price or better,
    unlike a market order which fills immediately at whatever the prevailing price is.
    """
    kite = kite_client()
    return kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NSE,
        tradingsymbol=tradingsymbol.upper(),
        transaction_type=transaction_type,
        quantity=quantity,
        product=product,
        order_type=kite.ORDER_TYPE_LIMIT,
        price=round_to_tick(price),
    )


def place_stop_loss_order(
    tradingsymbol: str,
    transaction_type: str,
    quantity: int,
    trigger_price: float,
    product: str = "MIS",
) -> str:
    """Place an SL-M order at the exchange so the stop survives a client crash."""
    kite = kite_client()
    return kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NSE,
        tradingsymbol=tradingsymbol.upper(),
        transaction_type=transaction_type,
        quantity=quantity,
        product=product,
        order_type=kite.ORDER_TYPE_SLM,
        trigger_price=round_to_tick(trigger_price),
    )


def open_mis_positions() -> list[dict[str, Any]]:
    return [p for p in kite_client().positions()["net"] if p["product"] == "MIS" and p["quantity"] != 0]


def square_off_all_mis() -> list[str]:
    """Cancel pending MIS orders, then flatten every open MIS position at market."""
    kite = kite_client()
    for order in kite.orders():
        if order["product"] == "MIS" and order["status"] in ("TRIGGER PENDING", "OPEN"):
            kite.cancel_order(variety=order["variety"], order_id=order["order_id"])

    order_ids = []
    for position in open_mis_positions():
        side = "SELL" if position["quantity"] > 0 else "BUY"
        order_ids.append(place_market_order(position["tradingsymbol"], side, abs(position["quantity"])))
    return order_ids
