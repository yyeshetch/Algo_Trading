from __future__ import annotations

from intraday_engine.core.config import Settings
from intraday_engine.core.models import ScoreBreakdown, TradePlan


def build_trade_plan(
    spot: float,
    support: float,
    resistance: float,
    score: ScoreBreakdown,
    bias: str,
    momentum: str,
    settings: Settings,
    atm_strike: int = 0,
    call_ltp: float = 0.0,
    put_ltp: float = 0.0,
) -> TradePlan:
    min_conf = settings.min_confidence_neutral_day if bias == "NEUTRAL_DAY" else settings.min_confidence
    long_signal = score.final_score > 0 and score.confidence >= min_conf
    short_signal = score.final_score < 0 and score.confidence >= min_conf

    if long_signal:
        entry = spot
        stop = support
        risk = entry - stop
        target = resistance if resistance > entry else entry + (risk * settings.min_rr)
        rr = _rr(entry, stop, target)
        if not _valid_trade(entry, stop, rr, settings, is_short=False):
            return _no_trade(score, bias, momentum, support, resistance, "Long setup rejected by risk filters.")
        opt_entry, opt_sl, opt_tgt = _option_levels(call_ltp, settings.option_stop_ratio, settings.min_rr)
        return TradePlan(
            signal="BUY",
            entry=round(entry, 2),
            target=round(target, 2),
            stop_loss=round(stop, 2),
            rr=round(rr, 2),
            strike_price=atm_strike if atm_strike else None,
            option_type="CE" if call_ltp > 0 else None,
            option_entry=round(opt_entry, 2) if opt_entry else None,
            option_sl=round(opt_sl, 2) if opt_sl else None,
            option_target=round(opt_tgt, 2) if opt_tgt else None,
            confidence=score.confidence,
            bias=bias,
            momentum=momentum,
            support=round(support, 2),
            resistance=round(resistance, 2),
            score=score,
            notes=["High-confidence bullish setup with structure-based stop."],
        )

    if short_signal:
        entry = spot
        stop = resistance
        risk = stop - entry
        target = support if support < entry else entry - (risk * settings.min_rr)
        rr = _rr(entry, stop, target)
        if not _valid_trade(entry, stop, rr, settings, is_short=True):
            return _no_trade(score, bias, momentum, support, resistance, "Short setup rejected by risk filters.")
        opt_entry, opt_sl, opt_tgt = _option_levels(put_ltp, settings.option_stop_ratio, settings.min_rr)
        return TradePlan(
            signal="SELL",
            entry=round(entry, 2),
            target=round(target, 2),
            stop_loss=round(stop, 2),
            rr=round(rr, 2),
            strike_price=atm_strike if atm_strike else None,
            option_type="PE" if put_ltp > 0 else None,
            option_entry=round(opt_entry, 2) if opt_entry else None,
            option_sl=round(opt_sl, 2) if opt_sl else None,
            option_target=round(opt_tgt, 2) if opt_tgt else None,
            confidence=score.confidence,
            bias=bias,
            momentum=momentum,
            support=round(support, 2),
            resistance=round(resistance, 2),
            score=score,
            notes=["High-confidence bearish setup with structure-based stop."],
        )

    return _no_trade(score, bias, momentum, support, resistance, "No high-confidence directional edge.")


def _option_levels(premium: float, stop_ratio: float, min_rr: float) -> tuple[float, float, float]:
    if premium <= 0:
        return 0.0, 0.0, 0.0
    opt_entry = premium
    opt_sl = premium * (1 - stop_ratio)
    opt_tgt = premium * (1 + stop_ratio * min_rr)
    return opt_entry, opt_sl, opt_tgt


def _rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0:
        return 0.0
    return reward / risk


def _valid_trade(entry: float, stop: float, rr: float, settings: Settings, is_short: bool = False) -> bool:
    stop_pct = abs(entry - stop) / max(entry, 1.0) * 100
    max_pct = settings.max_stop_pct_short if is_short else settings.max_stop_pct
    return rr >= settings.min_rr and stop_pct <= max_pct


def _no_trade(
    score: ScoreBreakdown,
    bias: str,
    momentum: str,
    support: float,
    resistance: float,
    reason: str,
) -> TradePlan:
    return TradePlan(
        signal="NO_TRADE",
        entry=None,
        target=None,
        stop_loss=None,
        rr=None,
        strike_price=None,
        option_type=None,
        option_entry=None,
        option_sl=None,
        option_target=None,
        confidence=score.confidence,
        bias=bias,
        momentum=momentum,
        support=round(support, 2),
        resistance=round(resistance, 2),
        score=score,
        notes=[reason],
    )

