"""Turns logged pre-market outcomes (.premarket_history.jsonl) into a calibration
note fed into the next session's brief, so each day's realized hit/miss record
informs tomorrow's confidence instead of being re-learned from scratch by the LLM.

Below MIN_ML_SAMPLES logged directional calls, reports plain historical hit-rates
(not enough data to fit a model without overfitting). At or above that threshold,
fits a small logistic-regression calibrator on (confidence, bias, sector) ->
direction_correct and reports its probability estimate instead. Retrained from the
full log on each call since the dataset stays tiny - never trained mid-session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HISTORY_PATH = Path(__file__).parent / ".premarket_history.jsonl"

# Best-effort sector tagging for the watchlist this report is intended for; unmapped
# symbols still work, just without a sector-index cross-check.
SECTOR_MAP = {
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK", "KOTAKBANK": "BANK",
    "AXISBANK": "BANK", "BAJFINANCE": "BANK",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY",
    "COALINDIA": "METAL", "TATASTEEL": "METAL",
    "NTPC": "POWER", "POWERGRID": "POWER",
    "MARUTI": "AUTO", "TMCV": "AUTO", "BAJAJ-AUTO": "AUTO",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "TITAN": "FMCG",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "LT": "INFRA", "ULTRACEMCO": "INFRA",
}

# Below this many logged directional calls, a fitted model would just memorise
# noise - report raw hit-rates instead until there's enough history.
MIN_ML_SAMPLES = 40
MIN_SAMPLES_FOR_ANY_NOTE = 5


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One outcome per (date, symbol), keeping the newest. Re-running `track` for a
    day used to append a fresh copy of every row, so 2026-10-28 ended up in the log
    seven times - and because the scoring basis changed from actual_open to
    last_price midway, those copies carried contradictory labels for identical
    features. Collapsing here keeps a re-run a correction rather than a duplicate.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        latest[(record.get("date", ""), record.get("symbol", ""))] = record
    return list(latest.values())


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    records = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return _dedupe(records)


def _hit_rate(records: list[dict[str, Any]]) -> tuple[int, int] | None:
    scored = [r for r in records if r.get("direction_correct") is not None]
    if not scored:
        return None
    hits = sum(1 for r in scored if r["direction_correct"])
    return hits, len(scored)


def _train_calibrator(records: list[dict[str, Any]]):
    """Logistic regression over (confidence, bias, sector) -> direction_correct.
    Returns None if there aren't both outcome classes logged yet to fit against.
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    scored = [r for r in records if r.get("direction_correct") is not None]
    if len({r["direction_correct"] for r in scored}) < 2:
        return None

    frame = pd.DataFrame(scored)
    frame["bias_bullish"] = (frame["directional_bias"] == "BULLISH").astype(int)
    frame["sector"] = frame["symbol"].map(SECTOR_MAP).fillna("UNMAPPED")
    features = pd.get_dummies(frame[["confidence", "bias_bullish", "sector"]], columns=["sector"])
    model = LogisticRegression(max_iter=1000)
    model.fit(features, frame["direction_correct"].astype(int))
    return model, features.columns.tolist()


def _predict(bundle, bias_bullish: int, sector: str, confidence: float = 55.0) -> float:
    import pandas as pd

    model, columns = bundle
    row = {col: 0 for col in columns}
    row["confidence"] = confidence
    row["bias_bullish"] = bias_bullish
    sector_col = f"sector_{sector}"
    if sector_col in row:
        row[sector_col] = 1
    frame = pd.DataFrame([row])[columns]
    return float(model.predict_proba(frame)[0][1])


def calibration_status() -> str:
    """Whether the fitted calibrator is actually active - it trains silently once the
    history crosses MIN_ML_SAMPLES, so without this there's no way to tell from a
    report whether a model is influencing the briefs or not.
    """
    records = _load_history()
    scored = [r for r in records if r.get("direction_correct") is not None]
    n = len(scored)
    sessions = len({r.get("date") for r in scored})
    if n < MIN_SAMPLES_FOR_ANY_NOTE:
        return f"Calibration: INACTIVE - {n} scored call(s), need {MIN_SAMPLES_FOR_ANY_NOTE} for even raw hit-rates."
    if n < MIN_ML_SAMPLES:
        return (
            f"Calibration: RAW HIT-RATES only - {n} unique scored call(s) across {sessions} session(s); "
            f"the fitted model activates at {MIN_ML_SAMPLES}."
        )
    bundle = _train_calibrator(records)
    if bundle is None:
        return f"Calibration: RAW HIT-RATES only - {n} scored call(s) but only one outcome class logged."
    return f"Calibration: MODEL ACTIVE - logistic regression fitted on {n} unique scored call(s) across {sessions} session(s)."


def build_calibration_note(symbol: str) -> str | None:
    """Historical hit-rate context to splice into this symbol's pre-market brief."""
    records = _load_history()
    hit = _hit_rate(records)
    if hit is None or hit[1] < MIN_SAMPLES_FOR_ANY_NOTE:
        return None
    hits, total = hit
    sector = SECTOR_MAP.get(symbol.upper(), "UNMAPPED")

    lines = [
        f"Historical calibration from {total} logged directional call(s) so far "
        f"(see `premarket stats`): overall directional accuracy {100 * hits / total:.0f}%."
    ]

    sector_records = [r for r in records if SECTOR_MAP.get(r["symbol"], "UNMAPPED") == sector]
    sector_hit = _hit_rate(sector_records)
    if sector_hit and sector_hit[1] >= 3:
        lines.append(
            f"{sector} sector-specific accuracy so far: {100 * sector_hit[0] / sector_hit[1]:.0f}% "
            f"(n={sector_hit[1]})."
        )

    if total >= MIN_ML_SAMPLES:
        bundle = _train_calibrator(records)
        if bundle:
            bull_p = _predict(bundle, 1, sector)
            bear_p = _predict(bundle, 0, sector)
            lines.append(
                f"Model-calibrated hit-rate estimate for this sector at ~55% stated confidence: "
                f"BULLISH {bull_p * 100:.0f}%, BEARISH {bear_p * 100:.0f}%."
            )
    else:
        lines.append(
            f"(Below {MIN_ML_SAMPLES} logged directional calls - using raw hit-rates above; a fitted "
            "calibration model will activate automatically once enough history accumulates.)"
        )

    return "\n".join(lines)
