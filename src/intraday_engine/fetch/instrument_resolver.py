from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient


@dataclass
class DerivativeSymbols:
    fut_symbol: str
    ce_symbol: str
    pe_symbol: str
    atm_strike: int
    lot_size: int | None = None  # From instrument when available (stocks)


class InstrumentResolver:
    def __init__(self, client: ZerodhaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def resolve(self, spot_price: float) -> DerivativeSymbols:
        return self.resolve_for_date(spot_price, date.today())

    def resolve_for_date(self, spot_price: float, trade_date: date) -> DerivativeSymbols:
        instruments = self.client.nfo_instruments()
        underlying = self.settings.underlying
        atm_strike = self._to_atm_strike(spot_price, self.settings.option_strike_step)

        fut = self._nearest_future(instruments, underlying, trade_date)
        ce = self._nearest_option(instruments, underlying, trade_date, atm_strike, "CE")
        pe = self._nearest_option(instruments, underlying, trade_date, atm_strike, "PE")

        if not fut or not ce or not pe:
            raise RuntimeError("Unable to resolve required derivatives symbols from NFO instruments.")

        lot_size = int(fut.get("lot_size", 0) or ce.get("lot_size", 0) or 0) or None

        return DerivativeSymbols(
            fut_symbol=f"NFO:{fut['tradingsymbol']}",
            ce_symbol=f"NFO:{ce['tradingsymbol']}",
            pe_symbol=f"NFO:{pe['tradingsymbol']}",
            atm_strike=atm_strike,
            lot_size=lot_size,
        )

    @staticmethod
    def _to_atm_strike(spot_price: float, step: int) -> int:
        return int(round(spot_price / step) * step)

    @staticmethod
    def _matches_underlying(r: Dict[str, object], underlying: str) -> bool:
        """Match by name or tradingsymbol prefix (NFO format: UNDERLYING+EXPIRY+FUT/CE/PE)."""
        if r.get("name") == underlying:
            return True
        ts = str(r.get("tradingsymbol") or "")
        if not ts.startswith(underlying):
            return False
        # Ensure we don't match "LT" to "LTI" - next char should be digit (expiry start)
        if len(ts) <= len(underlying):
            return False
        return ts[len(underlying) : len(underlying) + 1].isdigit()

    @staticmethod
    def _nearest_future(
        records: List[Dict[str, object]],
        underlying: str,
        today: date,
    ) -> Dict[str, object] | None:
        futs = [
            r
            for r in records
            if InstrumentResolver._matches_underlying(r, underlying)
            and r.get("instrument_type") == "FUT"
            and r.get("expiry")
            and r["expiry"] >= today
        ]
        futs.sort(key=lambda x: x["expiry"])
        return futs[0] if futs else None

    @staticmethod
    def _nearest_option(
        records: List[Dict[str, object]],
        underlying: str,
        today: date,
        strike: int,
        option_type: str,
    ) -> Dict[str, object] | None:
        opts = [
            r
            for r in records
            if InstrumentResolver._matches_underlying(r, underlying)
            and r.get("instrument_type") == option_type
            and int(float(r.get("strike", 0))) == strike
            and r.get("expiry")
            and r["expiry"] >= today
        ]
        opts.sort(key=lambda x: x["expiry"])
        return opts[0] if opts else None

