from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class QuotePoint:
    symbol: str
    last_price: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    avg_price: float


@dataclass
class MarketSnapshot:
    timestamp: datetime
    spot: QuotePoint
    future: QuotePoint
    call: QuotePoint
    put: QuotePoint
    atm_strike: int
    ce_symbol: str
    pe_symbol: str
    fut_symbol: str

    def to_record(self) -> Dict[str, float | str]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "spot_symbol": self.spot.symbol,
            "spot_ltp": self.spot.last_price,
            "spot_open": self.spot.open_price,
            "spot_high": self.spot.high_price,
            "spot_low": self.spot.low_price,
            "spot_close": self.spot.close_price,
            "spot_vwap": self.spot.avg_price,
            "future_symbol": self.future.symbol,
            "future_ltp": self.future.last_price,
            "future_open": self.future.open_price,
            "future_high": self.future.high_price,
            "future_low": self.future.low_price,
            "future_close": self.future.close_price,
            "future_vwap": self.future.avg_price,
            "call_symbol": self.call.symbol,
            "call_ltp": self.call.last_price,
            "put_symbol": self.put.symbol,
            "put_ltp": self.put.last_price,
            "atm_strike": self.atm_strike,
            "ce_symbol": self.ce_symbol,
            "pe_symbol": self.pe_symbol,
            "fut_symbol": self.fut_symbol,
        }


@dataclass
class ScoreBreakdown:
    bullish: float
    bearish: float
    no_trade_penalty: float
    final_score: float
    confidence: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class TradePlan:
    signal: str
    entry: float | None
    target: float | None
    stop_loss: float | None
    rr: float | None
    strike_price: int | None
    option_type: str | None
    option_entry: float | None
    option_sl: float | None
    option_target: float | None
    confidence: float
    bias: str
    momentum: str
    support: float
    resistance: float
    score: ScoreBreakdown
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["score"] = self.score.to_dict()
        return payload
