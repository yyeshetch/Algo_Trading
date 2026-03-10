"""
Gamma blast detection for expiry-day options trading.

Detects conditions favorable for gamma blast trades:
- Expiry day (Tuesday for Nifty weekly)
- High OI at ATM/OTM strikes
- Low put-call ratio (<0.7 bullish, >1.3 bearish)
- Volume breakout vs recent average
- Best timing: after 1:45 PM
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import List

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.gamma.expiry_utils import is_expiry_day
from intraday_engine.gamma.option_chain_fetcher import (
    OptionChainSnapshot,
    OptionStrikeData,
    fetch_expiry_day_option_chain,
)


@dataclass
class GammaBlastSignal:
    """Detected gamma blast setup."""

    trade_date: date
    spot_price: float
    atm_strike: int
    direction: str  # "BULLISH" | "BEARISH" | "NEUTRAL"
    pcr: float
    total_ce_oi: float
    total_pe_oi: float
    total_ce_volume: float
    total_pe_volume: float
    is_after_1345: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    suggested_strike: int


class GammaBlastDetector:
    """Identify gamma blast setups on expiry day using Kite data."""

    def __init__(self, client: ZerodhaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def scan(self, trade_date: date | None = None, spot_price: float | None = None) -> GammaBlastSignal | None:
        """
        Scan for gamma blast conditions on expiry day.
        Returns None if not expiry day or data unavailable.
        """
        if trade_date is None:
            trade_date = date.today()

        if not is_expiry_day(trade_date, self.settings.underlying):
            return None

        if spot_price is None:
            q = self.client.quote([self.settings.spot_symbol])
            spot_price = float(q.get(self.settings.spot_symbol, {}).get("last_price", 0) or 0)
        if spot_price <= 0:
            return None

        snapshot = fetch_expiry_day_option_chain(
            self.client,
            self.settings,
            trade_date,
            spot_price,
            num_strikes_each_side=3,
        )
        if not snapshot or not snapshot.strikes:
            return None

        return self._evaluate(snapshot)

    def _evaluate(self, snapshot: OptionChainSnapshot) -> GammaBlastSignal:
        now = datetime.now().time()
        cutoff = time(13, 45)
        is_after_1345 = now >= cutoff

        ce_strikes = [s for s in snapshot.strikes if s.option_type == "CE"]
        pe_strikes = [s for s in snapshot.strikes if s.option_type == "PE"]

        total_ce_oi = sum(s.oi for s in ce_strikes)
        total_pe_oi = sum(s.oi for s in pe_strikes)
        total_ce_vol = sum(s.volume for s in ce_strikes)
        total_pe_vol = sum(s.volume for s in pe_strikes)

        # Put-call ratio (OI-based): PE_OI / CE_OI
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0

        # Direction from PCR
        # PCR < 0.7: bullish (more CE OI, call buying pressure)
        # PCR > 1.3: bearish (more PE OI, put buying pressure)
        # 0.7–1.3: neutral
        if pcr < 0.7:
            direction = "BULLISH"
            suggested = self._best_otm_ce(ce_strikes, snapshot.spot_price)
        elif pcr > 1.3:
            direction = "BEARISH"
            suggested = self._best_otm_pe(pe_strikes, snapshot.spot_price)
        else:
            direction = "NEUTRAL"
            suggested = snapshot.atm_strike

        # Confidence: stronger PCR deviation = higher confidence
        if direction == "BULLISH":
            pcr_dev = min(1.0, (0.7 - pcr) / 0.7) if pcr < 0.7 else 0.0
        elif direction == "BEARISH":
            pcr_dev = min(1.0, (pcr - 1.3) / 1.3) if pcr > 1.3 else 0.0
        else:
            pcr_dev = 0.0

        time_factor = 0.7 if is_after_1345 else 0.4  # Higher confidence after 1:45 PM
        confidence = min(1.0, (pcr_dev * 0.6 + time_factor * 0.4))

        if direction == "BULLISH":
            reason = f"PCR {pcr:.2f} < 0.7 (CE-heavy) | CE OI {total_ce_oi:,.0f} vs PE OI {total_pe_oi:,.0f}"
        elif direction == "BEARISH":
            reason = f"PCR {pcr:.2f} > 1.3 (PE-heavy) | PE OI {total_pe_oi:,.0f} vs CE OI {total_ce_oi:,.0f}"
        else:
            reason = f"PCR {pcr:.2f} in neutral range 0.7–1.3"

        if is_after_1345:
            reason += " | Best window (post 1:45 PM on expiry day)"

        return GammaBlastSignal(
            trade_date=snapshot.trade_date,
            spot_price=snapshot.spot_price,
            atm_strike=snapshot.atm_strike,
            direction=direction,
            pcr=pcr,
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            total_ce_volume=total_ce_vol,
            total_pe_volume=total_pe_vol,
            is_after_1345=is_after_1345,
            confidence=confidence,
            reason=reason,
            suggested_strike=suggested,
        )

    def _best_otm_ce(self, ce_strikes: List[OptionStrikeData], spot: float) -> int:
        """OTM CE with strike just above spot (high gamma)."""
        otm = [s for s in ce_strikes if s.strike > spot]
        if not otm:
            return max(s.strike for s in ce_strikes)
        return min(otm, key=lambda s: s.strike - spot).strike

    def _best_otm_pe(self, pe_strikes: List[OptionStrikeData], spot: float) -> int:
        """OTM PE with strike just below spot (high gamma)."""
        otm = [s for s in pe_strikes if s.strike < spot]
        if not otm:
            return min(s.strike for s in pe_strikes)
        return max(otm, key=lambda s: spot - s.strike).strike
