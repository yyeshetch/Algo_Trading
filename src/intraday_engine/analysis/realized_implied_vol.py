"""
Realized vs implied volatility proxies for intraday long ATM straddle context.

Implied (IV proxy): ATM straddle / spot — market-priced expected move as % of spot.
Realized (RV proxy):
  - Session range % from day open (cumulative high-low)
  - Rolling 30-min realized vol scaled to a full-session move %

No Black–Scholes IV — uses straddle premium directly (same basis as expected-move chart).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

BARS_PER_SESSION = 75  # 5-min bars in a 6h15m NSE session
ROLLING_BARS = 6       # 30 minutes of 5-min returns


def _rolling_realized_move_pct(spots: list[float], idx: int, *, window: int = ROLLING_BARS) -> float | None:
    """Std of 5-min log returns, scaled to session-equivalent move %."""
    if idx < 1:
        return None
    start = max(1, idx - window + 1)
    rets: list[float] = []
    for j in range(start, idx + 1):
        prev, cur = spots[j - 1], spots[j]
        if prev > 0 and cur > 0:
            rets.append(math.log(cur / prev))
    if len(rets) < 2:
        return None
    std = statistics.pstdev(rets)
    return round(std * math.sqrt(BARS_PER_SESSION) * 100, 3)


def _vol_regime(
    *,
    realized_range_pct: float | None,
    implied_open_pct: float | None,
    rolling_realized_pct: float | None,
    implied_move_pct: float | None,
    straddle_pnl_pct: float | None,
) -> str:
    """Classify day for a long ATM straddle holder."""
    rv = rolling_realized_pct or realized_range_pct
    iv = implied_move_pct or implied_open_pct
    if rv is not None and iv is not None and rv > iv * 1.05:
        return "expansion"
    if straddle_pnl_pct is not None and straddle_pnl_pct < -4:
        if (realized_range_pct or 0) < (implied_open_pct or 999) * 0.45:
            return "contraction"
    if realized_range_pct is not None and implied_open_pct is not None:
        if realized_range_pct >= implied_open_pct * 0.85:
            return "expansion"
        if realized_range_pct < implied_open_pct * 0.35:
            return "contraction"
    return "neutral"


def attach_volatility_compare(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add vol_compare block to each analysis summary."""
    if not summaries:
        return summaries

    spots = [float(s.get("price_action", {}).get("spot") or 0) for s in summaries]

    for i, s in enumerate(summaries):
        opt = s.get("options") or {}
        em = s.get("expected_move") or {}
        spot = spots[i]
        straddle = float(opt.get("atm_straddle") or em.get("straddle_now") or 0)
        straddle_open = float(em.get("straddle_open") or 0)
        day_open = float(em.get("day_open") or 0) or spot

        implied_move_pct = round(straddle / spot * 100, 3) if spot > 0 and straddle > 0 else None
        implied_open_pct = round(straddle_open / day_open * 100, 3) if day_open > 0 and straddle_open > 0 else None

        session_range_pts = float(em.get("session_range_pts") or 0)
        realized_range_pct = round(session_range_pts / day_open * 100, 3) if day_open > 0 else None

        rolling_realized_pct = _rolling_realized_move_pct(spots, i)

        straddle_pnl_pct = (
            round((straddle / straddle_open - 1) * 100, 2) if straddle_open > 0 else None
        )

        rv_iv_ratio = None
        if realized_range_pct is not None and implied_open_pct and implied_open_pct > 0:
            rv_iv_ratio = round(realized_range_pct / implied_open_pct, 2)

        rolling_vs_implied = None
        if rolling_realized_pct is not None and implied_move_pct and implied_move_pct > 0:
            rolling_vs_implied = round(rolling_realized_pct / implied_move_pct, 2)

        regime = _vol_regime(
            realized_range_pct=realized_range_pct,
            implied_open_pct=implied_open_pct,
            rolling_realized_pct=rolling_realized_pct,
            implied_move_pct=implied_move_pct,
            straddle_pnl_pct=straddle_pnl_pct,
        )

        s["vol_compare"] = {
            "implied_move_pct": implied_move_pct,
            "implied_open_pct": implied_open_pct,
            "realized_range_pct": realized_range_pct,
            "rolling_realized_pct": rolling_realized_pct,
            "straddle_pnl_pct": straddle_pnl_pct,
            "rv_iv_ratio": rv_iv_ratio,
            "rolling_vs_implied": rolling_vs_implied,
            "regime": regime,
            "rv_gt_iv": bool(
                rolling_realized_pct is not None
                and implied_move_pct is not None
                and rolling_realized_pct > implied_move_pct
            ),
        }

    return summaries
