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

        fut = self._nearest_future(instruments, underlying, trade_date)
        ce, pe, atm_strike = self._nearest_option_pair(instruments, underlying, trade_date, spot_price)

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

    @staticmethod
    def _nearest_option_pair(
        records: List[Dict[str, object]],
        underlying: str,
        today: date,
        spot_price: float,
    ) -> tuple[Dict[str, object] | None, Dict[str, object] | None, int]:
        """
        Choose the nearest available CE/PE strike from the live option chain.

        For stock options, listed strikes are often sparse (e.g. 20/40-point steps),
        so rounding by a configured step can point to a strike that does not exist.
        """
        eligible = [
            r
            for r in records
            if InstrumentResolver._matches_underlying(r, underlying)
            and r.get("instrument_type") in ("CE", "PE")
            and r.get("expiry")
            and r["expiry"] >= today
        ]
        if not eligible:
            return None, None, 0

        expiries = sorted({r["expiry"] for r in eligible})
        for expiry in expiries:
            by_strike: Dict[int, Dict[str, Dict[str, object]]] = {}
            for row in eligible:
                if row["expiry"] != expiry:
                    continue
                strike = int(float(row.get("strike", 0) or 0))
                if strike <= 0:
                    continue
                strike_bucket = by_strike.setdefault(strike, {})
                strike_bucket[str(row.get("instrument_type"))] = row

            common_strikes = [strike for strike, legs in by_strike.items() if "CE" in legs and "PE" in legs]
            if not common_strikes:
                continue

            nearest_strike = min(common_strikes, key=lambda strike: (abs(strike - spot_price), strike))
            legs = by_strike[nearest_strike]
            return legs["CE"], legs["PE"], nearest_strike

        return None, None, 0

