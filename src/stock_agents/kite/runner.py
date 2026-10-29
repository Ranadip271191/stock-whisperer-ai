"""CLI: scan a watchlist, size by risk, place entry + exchange-side stop.

Defaults to --dry-run. Live order placement requires --live explicitly.
Run `square-off` yourself before your own deadline - do not rely on the
broker's auto square-off, which can be later than you expect and offers
no control over the exit price.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time as dtime, timedelta

from .client import place_limit_order, place_market_order, place_stop_loss_order, square_off_all_mis
from .risk import can_take_new_trade, record_trade, shares_for_risk
from .strategy import atr_breakout_signal, bollinger_reversion_signal, orb_breakout_signal

# Deliberately ahead of the broker's own square-off window, not a replacement for it.
DEFAULT_SQUARE_OFF_BY = dtime(15, 10)


def scan_and_trade(
    symbols: list[str],
    account_equity: float,
    risk_per_trade_pct: float,
    max_daily_loss_pct: float,
    max_trades_per_day: int,
    strategy: str = "orb",
    target_r_multiple: float = 2.0,
    rvol_threshold: float = 1.5,
    orb_minutes: int = 15,
    min_confirmations: int = 0,
    gap_threshold_pct: float = 0.3,
    use_relative_strength: bool = False,
    use_volume_profile: bool = False,
    index_symbol: str = "NIFTY 50",
    atr_period: int = 14,
    atr_multiple: float = 1.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    ml_threshold: float = 0.55,
    square_off_by: dtime = DEFAULT_SQUARE_OFF_BY,
    dry_run: bool = True,
) -> None:
    if datetime.now().time() >= square_off_by:
        print(f"Past {square_off_by}; not opening new positions.")
        return

    booster = None
    if strategy == "ml":
        from .ml import MODEL_PATH, load_model

        if not MODEL_PATH.exists():
            print(f"No trained model at {MODEL_PATH} - run `train-ml` first.")
            return
        booster = load_model()

    for symbol in symbols:
        allowed, reason = can_take_new_trade(account_equity, max_daily_loss_pct, max_trades_per_day)
        if not allowed:
            print(f"Stopping scan: {reason}")
            return

        if strategy == "atr":
            signal = atr_breakout_signal(
                symbol, atr_period=atr_period, atr_multiple=atr_multiple, target_r_multiple=target_r_multiple
            )
        elif strategy == "bollinger":
            signal = bollinger_reversion_signal(symbol, period=bb_period, num_std=bb_std)
        elif strategy == "ml":
            from .ml import orb_ml_filtered_signal

            signal = orb_ml_filtered_signal(
                symbol,
                booster=booster,
                probability_threshold=ml_threshold,
                orb_minutes=orb_minutes,
                rvol_threshold=rvol_threshold,
                target_r_multiple=target_r_multiple,
            )
        else:
            signal = orb_breakout_signal(
                symbol,
                orb_minutes=orb_minutes,
                rvol_threshold=rvol_threshold,
                target_r_multiple=target_r_multiple,
                min_confirmations=min_confirmations,
                gap_threshold_pct=gap_threshold_pct,
                use_relative_strength=use_relative_strength,
                use_volume_profile=use_volume_profile,
                index_symbol=index_symbol,
            )
        if signal.action == "NONE":
            print(f"{symbol}: no signal ({signal.reason})")
            continue

        shares = shares_for_risk(account_equity, risk_per_trade_pct, signal.entry, signal.stop)
        if shares <= 0:
            print(f"{symbol}: {signal.action} signal but sizing rounds to 0 shares - skipped")
            continue

        print(
            f"{symbol}: {signal.action} {shares} shares, entry~{signal.entry:.2f}, "
            f"stop {signal.stop:.2f}, target {signal.target:.2f} ({signal.reason})"
        )
        if dry_run:
            continue

        entry_side = "BUY" if signal.action == "BUY" else "SELL"
        exit_side = "SELL" if signal.action == "BUY" else "BUY"
        entry_order_id = place_market_order(symbol, entry_side, shares)
        # Target is informational only: Kite's regular orders have no built-in
        # OCO, so a target order would risk being orphaned if the stop fills first.
        stop_order_id = place_stop_loss_order(symbol, exit_side, shares, signal.stop)
        record_trade()
        print(f"  placed entry {entry_order_id}, stop {stop_order_id}")


def square_off() -> None:
    order_ids = square_off_all_mis()
    print(f"Squared off {len(order_ids)} position(s): {order_ids}")


def place_manual_order(
    symbol: str,
    side: str,
    quantity: int,
    price: float | None,
    stop: float | None,
    product: str,
    account_equity: float | None,
    max_daily_loss_pct: float,
    max_trades_per_day: int,
    dry_run: bool = True,
) -> None:
    """One-off manual order outside any strategy signal, with a quantity you choose
    directly instead of risk-based sizing. Entry is a market order unless --price is
    given, in which case it's a limit order that only fills at that price or better.
    Daily loss/trade caps still apply if --equity is given; omit it to skip that check.
    """
    if account_equity is not None:
        allowed, reason = can_take_new_trade(account_equity, max_daily_loss_pct, max_trades_per_day)
        if not allowed:
            print(f"Not placing order: {reason}")
            return

    entry_note = f", limit {price:.2f}" if price is not None else " (market)"
    stop_note = f", stop {stop:.2f}" if stop is not None else ""
    print(f"{symbol}: {side} {quantity} shares{entry_note}{stop_note}")
    if dry_run:
        print("Dry run - no order placed. Pass --live to place a real order.")
        return

    if price is not None:
        entry_order_id = place_limit_order(symbol, side, quantity, price, product=product)
    else:
        entry_order_id = place_market_order(symbol, side, quantity, product=product)

    stop_order_id = None
    if stop is not None:
        exit_side = "SELL" if side == "BUY" else "BUY"
        stop_order_id = place_stop_loss_order(symbol, exit_side, quantity, stop, product=product)
    record_trade()
    stop_note = f", stop {stop_order_id}" if stop_order_id else ""
    print(f"  placed entry {entry_order_id}{stop_note}")


def backtest(
    symbols: list[str],
    account_equity: float,
    risk_per_trade_pct: float,
    days: int,
    interval: str,
    strategy: str = "orb",
    target_r_multiple: float = 2.0,
    rvol_threshold: float = 1.5,
    orb_minutes: int = 15,
    min_confirmations: int = 0,
    gap_threshold_pct: float = 0.3,
    use_relative_strength: bool = False,
    use_volume_profile: bool = False,
    index_symbol: str = "NIFTY 50",
    atr_period: int = 14,
    atr_multiple: float = 1.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    ml_threshold: float = 0.55,
) -> None:
    # local import: pulls in pandas-heavy code only when needed
    from .backtest import backtest_atr_breakout, backtest_bollinger_reversion, backtest_orb

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    booster = None
    if strategy == "ml":
        from .ml import MODEL_PATH, load_model

        if not MODEL_PATH.exists():
            print(f"No trained model at {MODEL_PATH} - run `train-ml` first.")
            return
        booster = load_model()

    for symbol in symbols:
        if strategy == "atr":
            result = backtest_atr_breakout(
                symbol,
                from_date,
                to_date,
                account_equity=account_equity,
                risk_per_trade_pct=risk_per_trade_pct,
                atr_period=atr_period,
                atr_multiple=atr_multiple,
                target_r_multiple=target_r_multiple,
                interval=interval,
            )
        elif strategy == "bollinger":
            result = backtest_bollinger_reversion(
                symbol,
                from_date,
                to_date,
                account_equity=account_equity,
                risk_per_trade_pct=risk_per_trade_pct,
                period=bb_period,
                num_std=bb_std,
                interval=interval,
            )
        elif strategy == "ml":
            from .ml import backtest_orb_ml_filtered

            result = backtest_orb_ml_filtered(
                symbol,
                from_date,
                to_date,
                account_equity=account_equity,
                booster=booster,
                probability_threshold=ml_threshold,
                risk_per_trade_pct=risk_per_trade_pct,
                orb_minutes=orb_minutes,
                rvol_threshold=rvol_threshold,
                target_r_multiple=target_r_multiple,
                interval=interval,
            )
        else:
            result = backtest_orb(
                symbol,
                from_date,
                to_date,
                account_equity=account_equity,
                risk_per_trade_pct=risk_per_trade_pct,
                target_r_multiple=target_r_multiple,
                rvol_threshold=rvol_threshold,
                orb_minutes=orb_minutes,
                interval=interval,
                min_confirmations=min_confirmations,
                gap_threshold_pct=gap_threshold_pct,
                use_relative_strength=use_relative_strength,
                use_volume_profile=use_volume_profile,
                index_symbol=index_symbol,
            )
        print(result.summary())


def train_ml(
    symbols: list[str],
    account_equity: float,
    risk_per_trade_pct: float,
    days: int,
    interval: str,
    orb_minutes: int = 15,
    rvol_threshold: float = 1.5,
    target_r_multiple: float = 2.0,
) -> None:
    """Generates labeled (feature, net-profitable?) samples from ORB trades across
    symbols/days, trains a LightGBM classifier, and saves it to disk. Training only -
    never touches the order-placement endpoints.
    """
    from .ml import MODEL_PATH, generate_orb_training_samples, save_model, train_model

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    samples = []
    for symbol in symbols:
        symbol_samples = generate_orb_training_samples(
            symbol,
            from_date,
            to_date,
            account_equity=account_equity,
            risk_per_trade_pct=risk_per_trade_pct,
            orb_minutes=orb_minutes,
            rvol_threshold=rvol_threshold,
            target_r_multiple=target_r_multiple,
            interval=interval,
        )
        print(f"{symbol}: {len(symbol_samples)} training samples")
        samples.extend(symbol_samples)

    if not samples:
        print("No training samples generated - nothing to train on.")
        return

    wins = sum(s.label for s in samples)
    print(f"Total: {len(samples)} samples, {wins} net-profitable ({100 * wins / len(samples):.1f}%)")
    booster = train_model(samples)
    save_model(booster)
    print(f"Saved model to {MODEL_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic Kite intraday runner. No LLM is in this path."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan a watchlist; place trades for qualifying signals.")
    scan.add_argument("symbols", nargs="+", help="NSE trading symbols, e.g. TCS RELIANCE")
    scan.add_argument("--equity", type=float, required=True, help="Account equity in INR.")
    scan.add_argument("--risk", type=float, default=0.5, help="Percent of equity risked per trade.")
    scan.add_argument("--max-daily-loss", type=float, default=2.0, help="Percent of equity.")
    scan.add_argument("--max-trades", type=int, default=5)
    scan.add_argument(
        "--strategy",
        choices=["orb", "atr", "bollinger", "ml"],
        default="orb",
        help="orb/atr/bollinger as before; ml = ORB gated by a pre-trained probability filter (run train-ml first).",
    )
    scan.add_argument("--target-r", type=float, default=2.0, help="Profit target as a multiple of the stop distance (orb/atr only).")
    scan.add_argument("--rvol-threshold", type=float, default=1.5, help="Minimum relative volume to confirm a breakout (orb only).")
    scan.add_argument("--orb-minutes", type=int, default=15, help="Opening-range window in minutes (orb only).")
    scan.add_argument(
        "--min-confirmations",
        type=int,
        default=0,
        help="orb only: require this many of {VWAP, RSI, EMA, gap-fill, +opt-in} to agree. 0 disables.",
    )
    scan.add_argument("--gap-threshold", type=float, default=0.3, help="Percent gap required for the gap-fill bias to fire (orb only).")
    scan.add_argument("--use-relative-strength", action="store_true", help="orb only: add relative-strength-vs-index as a confirmation vote.")
    scan.add_argument("--use-volume-profile", action="store_true", help="orb only: add volume-profile value-area fade as a confirmation vote.")
    scan.add_argument("--index-symbol", default="NIFTY 50", help="Index tradingsymbol for --use-relative-strength.")
    scan.add_argument("--atr-period", type=int, default=14, help="atr only: sessions used for the ATR average.")
    scan.add_argument("--atr-multiple", type=float, default=1.0, help="atr only: breakout distance as a multiple of ATR.")
    scan.add_argument("--bb-period", type=int, default=20, help="bollinger only: bars in the rolling window.")
    scan.add_argument("--bb-std", type=float, default=2.0, help="bollinger only: band width in standard deviations.")
    scan.add_argument("--ml-threshold", type=float, default=0.55, help="ml only: minimum predicted P(net-profitable) to take the trade.")
    scan.add_argument("--live", action="store_true", help="Place real orders. Default is dry-run.")

    sub.add_parser("square-off", help="Flatten all open MIS positions, cancel pending orders.")

    order = sub.add_parser("order", help="Place a single manual order (market or limit entry + optional SL-M stop) with a quantity you choose.")
    order.add_argument("symbol", help="NSE trading symbol, e.g. TCS")
    order.add_argument("side", choices=["BUY", "SELL"])
    order.add_argument("quantity", type=int, help="Number of shares.")
    order.add_argument("--price", type=float, help="Limit price for entry; omit for a market order.")
    order.add_argument("--stop", type=float, help="Optional stop-loss price; places an SL-M order after the entry if given.")
    order.add_argument("--product", default="MIS", help="MIS (intraday) or CNC (delivery).")
    order.add_argument("--equity", type=float, help="Account equity in INR - if given, enforces the daily loss/trade caps.")
    order.add_argument("--max-daily-loss", type=float, default=2.0, help="Percent of equity.")
    order.add_argument("--max-trades", type=int, default=5)
    order.add_argument("--live", action="store_true", help="Place a real order. Default is dry-run.")

    bt = sub.add_parser("backtest", help="Walk-forward backtest a strategy against historical candles.")
    bt.add_argument("symbols", nargs="+")
    bt.add_argument("--equity", type=float, required=True)
    bt.add_argument("--risk", type=float, default=0.5, help="Percent of equity risked per trade.")
    bt.add_argument("--days", type=int, default=120, help="Lookback window in calendar days.")
    bt.add_argument("--interval", default="5minute")
    bt.add_argument(
        "--strategy",
        choices=["orb", "atr", "bollinger", "ml"],
        default="orb",
        help="orb/atr/bollinger as before; ml = ORB gated by a pre-trained LightGBM probability filter (run train-ml first).",
    )
    bt.add_argument("--target-r", type=float, default=2.0, help="Profit target as a multiple of the stop distance (orb/atr only).")
    bt.add_argument("--rvol-threshold", type=float, default=1.5, help="Minimum relative volume to confirm a breakout (orb/ml only).")
    bt.add_argument("--orb-minutes", type=int, default=15, help="Opening-range window in minutes (orb/ml only).")
    bt.add_argument(
        "--min-confirmations",
        type=int,
        default=0,
        help="orb only: require this many of {VWAP, RSI, EMA, gap-fill, +opt-in} to agree. 0 disables.",
    )
    bt.add_argument("--gap-threshold", type=float, default=0.3, help="Percent gap required for the gap-fill bias to fire (orb only).")
    bt.add_argument("--use-relative-strength", action="store_true", help="orb only: add relative-strength-vs-index as a confirmation vote.")
    bt.add_argument("--use-volume-profile", action="store_true", help="orb only: add volume-profile value-area fade as a confirmation vote.")
    bt.add_argument("--index-symbol", default="NIFTY 50", help="Index tradingsymbol for --use-relative-strength.")
    bt.add_argument("--atr-period", type=int, default=14, help="atr only: sessions used for the ATR average.")
    bt.add_argument("--atr-multiple", type=float, default=1.0, help="atr only: breakout distance as a multiple of ATR.")
    bt.add_argument("--bb-period", type=int, default=20, help="bollinger only: bars in the rolling window.")
    bt.add_argument("--bb-std", type=float, default=2.0, help="bollinger only: band width in standard deviations.")
    bt.add_argument("--ml-threshold", type=float, default=0.55, help="ml only: minimum predicted P(net-profitable) to take the trade.")

    train_ml_parser = sub.add_parser("train-ml", help="Train the ORB probability filter model from historical trades.")
    train_ml_parser.add_argument("symbols", nargs="+")
    train_ml_parser.add_argument("--equity", type=float, required=True)
    train_ml_parser.add_argument("--risk", type=float, default=0.5, help="Percent of equity risked per trade.")
    train_ml_parser.add_argument("--days", type=int, default=180, help="Training lookback window in calendar days.")
    train_ml_parser.add_argument("--interval", default="5minute")
    train_ml_parser.add_argument("--rvol-threshold", type=float, default=1.5)
    train_ml_parser.add_argument("--orb-minutes", type=int, default=15)
    train_ml_parser.add_argument("--target-r", type=float, default=2.0)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "scan":
        scan_and_trade(
            [s.upper() for s in args.symbols],
            account_equity=args.equity,
            risk_per_trade_pct=args.risk,
            max_daily_loss_pct=args.max_daily_loss,
            max_trades_per_day=args.max_trades,
            strategy=args.strategy,
            target_r_multiple=args.target_r,
            rvol_threshold=args.rvol_threshold,
            orb_minutes=args.orb_minutes,
            min_confirmations=args.min_confirmations,
            gap_threshold_pct=args.gap_threshold,
            use_relative_strength=args.use_relative_strength,
            use_volume_profile=args.use_volume_profile,
            index_symbol=args.index_symbol,
            atr_period=args.atr_period,
            atr_multiple=args.atr_multiple,
            bb_period=args.bb_period,
            bb_std=args.bb_std,
            ml_threshold=args.ml_threshold,
            dry_run=not args.live,
        )
    elif args.command == "square-off":
        square_off()
    elif args.command == "order":
        place_manual_order(
            args.symbol.upper(),
            args.side,
            quantity=args.quantity,
            price=args.price,
            stop=args.stop,
            product=args.product,
            account_equity=args.equity,
            max_daily_loss_pct=args.max_daily_loss,
            max_trades_per_day=args.max_trades,
            dry_run=not args.live,
        )
    elif args.command == "backtest":
        backtest(
            [s.upper() for s in args.symbols],
            account_equity=args.equity,
            risk_per_trade_pct=args.risk,
            days=args.days,
            interval=args.interval,
            strategy=args.strategy,
            target_r_multiple=args.target_r,
            rvol_threshold=args.rvol_threshold,
            orb_minutes=args.orb_minutes,
            min_confirmations=args.min_confirmations,
            gap_threshold_pct=args.gap_threshold,
            use_relative_strength=args.use_relative_strength,
            use_volume_profile=args.use_volume_profile,
            index_symbol=args.index_symbol,
            atr_period=args.atr_period,
            atr_multiple=args.atr_multiple,
            bb_period=args.bb_period,
            bb_std=args.bb_std,
            ml_threshold=args.ml_threshold,
        )
    elif args.command == "train-ml":
        train_ml(
            [s.upper() for s in args.symbols],
            account_equity=args.equity,
            risk_per_trade_pct=args.risk,
            days=args.days,
            interval=args.interval,
            orb_minutes=args.orb_minutes,
            rvol_threshold=args.rvol_threshold,
            target_r_multiple=args.target_r,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
