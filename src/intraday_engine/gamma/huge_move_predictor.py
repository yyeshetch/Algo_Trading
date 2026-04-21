"""
Huge move predictor: probability of 100+ point spot/option moves.

Uses option chain (5-10 strikes near spot):
- PCR (OI and volume)
- Max Pain deviation
- OI build-up at strikes
- Premium expansion
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.gamma.option_chain_fetcher import (
    OptionChainSnapshot,
    OptionStrikeData,
    fetch_option_chain_near_spot,
    load_option_chain_snapshots,
    save_option_chain_snapshot,
)


def _opt_strike(s: dict) -> OptionStrikeData:
    return OptionStrikeData(
        tradingsymbol=s.get("tradingsymbol", ""),
        strike=int(s.get("strike", 0)),
        option_type=s.get("option_type", ""),
        instrument_token=0,
        oi=float(s.get("oi", 0)),
        volume=float(s.get("volume", 0)),
        ltp=float(s.get("ltp", 0)),
        open_=float(s.get("open", 0)),
        high=float(s.get("high", 0)),
        low=float(s.get("low", 0)),
        close=float(s.get("close", 0)),
    )


@dataclass
class HugeMovePrediction:
    direction: str  # BULLISH | BEARISH | NEUTRAL
    prob_huge_up: float
    prob_huge_down: float
    prob_no_move: float
    pcr_oi: float
    pcr_volume: float
    max_pain: int | None
    spot_vs_max_pain: float  # positive = above max pain
    reasons: List[str]
    suggested_strike: int | None
    confidence: float


def _compute_pcr(snapshot: OptionChainSnapshot) -> tuple[float, float]:
    """Return (pcr_oi, pcr_volume)."""
    ce_strikes = [s for s in snapshot.strikes if s.option_type == "CE"]
    pe_strikes = [s for s in snapshot.strikes if s.option_type == "PE"]
    total_ce_oi = sum(s.oi for s in ce_strikes)
    total_pe_oi = sum(s.oi for s in pe_strikes)
    total_ce_vol = sum(s.volume for s in ce_strikes)
    total_pe_vol = sum(s.volume for s in pe_strikes)
    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0
    pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0.0
    return pcr_oi, pcr_vol


def _compute_max_pain(strikes: List[OptionStrikeData], spot: float, lot_size: int = 65) -> int | None:
    """
    Max Pain = strike where total option writer payout is minimized.
    Simplified: sum of (intrinsic at expiry) * OI * lot_size for each strike.
    """
    strikes_set = sorted({s.strike for s in strikes})
    if not strikes_set:
        return None

    best_strike = None
    best_payout = float("inf")

    for expiry_price in strikes_set:
        total_payout = 0.0
        for s in strikes:
            oi = s.oi * lot_size
            if s.option_type == "CE":
                # Call writer pays max(0, spot - strike) at expiry
                payout = max(0, expiry_price - s.strike) * oi
            else:
                # Put writer pays max(0, strike - spot) at expiry
                payout = max(0, s.strike - expiry_price) * oi
            total_payout += payout
        if total_payout < best_payout:
            best_payout = total_payout
            best_strike = expiry_price

    return best_strike


def _atm_option_changes(snapshot: OptionChainSnapshot) -> tuple[float, float, float, float]:
    """Return (call_prem_chg, put_prem_chg, ce_oi_chg, pe_oi_chg) for ATM."""
    atm = snapshot.atm_strike
    ce = [s for s in snapshot.strikes if s.option_type == "CE" and s.strike == atm]
    pe = [s for s in snapshot.strikes if s.option_type == "PE" and s.strike == atm]
    call_chg = 0.0
    put_chg = 0.0
    if ce and ce[0].open_ > 0:
        call_chg = ((ce[0].ltp - ce[0].open_) / ce[0].open_) * 100
    if pe and pe[0].open_ > 0:
        put_chg = ((pe[0].ltp - pe[0].open_) / pe[0].open_) * 100
    # OI change needs history; use 0 for single snapshot
    return call_chg, put_chg, 0.0, 0.0


def predict_huge_move_from_snapshot(
    snapshot: OptionChainSnapshot,
    lot_size: int = 65,
    huge_move_points: float = 100,
) -> HugeMovePrediction:
    """
    Predict probability of huge move (default 100+ points) from option chain snapshot.
    """
    from intraday_engine.core.tunables import get_float

    def _hmf(key: str, default: float) -> float:
        return get_float("huge_move_predictor", key, default)

    reasons: List[str] = []
    pcr_oi, pcr_vol = _compute_pcr(snapshot)
    max_pain = _compute_max_pain(snapshot.strikes, snapshot.spot_price, lot_size)
    spot_vs_mp = (snapshot.spot_price - max_pain) if max_pain else 0.0

    call_chg, put_chg, _, _ = _atm_option_changes(snapshot)

    prob_up = 0.0
    prob_down = 0.0

    p_oi_eb = _hmf("PCR_OI_EXTREME_BULL", 0.6)
    p_oi_b = _hmf("PCR_OI_BULL", 0.7)
    p_oi_ee = _hmf("PCR_OI_EXTREME_BEAR", 1.5)
    p_oi_br = _hmf("PCR_OI_BEAR", 1.3)
    p_v_b = _hmf("PCR_VOL_BULL", 0.5)
    p_v_br = _hmf("PCR_VOL_BEAR", 2.0)
    prem_th = _hmf("PREMIUM_CHG_THRESHOLD", 15.0)
    mp_dev = _hmf("MAX_PAIN_DEV_POINTS", 50.0)
    norm_off = _hmf("NORM_TOTAL_OFFSET", 0.3)
    prob_cap = _hmf("PROB_CAP", 0.7)
    prob_no_fl = _hmf("PROB_NO_FLOOR", 0.1)
    dir_th = _hmf("DIRECTION_CONF_THRESHOLD", 0.4)

    if pcr_oi < p_oi_eb:
        prob_up += 0.25
        reasons.append(f"PCR(OI) {pcr_oi:.2f} extreme call heavy")
    elif pcr_oi < p_oi_b:
        prob_up += 0.15
        reasons.append(f"PCR(OI) {pcr_oi:.2f} bullish")
    elif pcr_oi > p_oi_ee:
        prob_down += 0.25
        reasons.append(f"PCR(OI) {pcr_oi:.2f} extreme put heavy")
    elif pcr_oi > p_oi_br:
        prob_down += 0.15
        reasons.append(f"PCR(OI) {pcr_oi:.2f} bearish")

    if pcr_vol < p_v_b:
        prob_up += 0.2
        reasons.append(f"PCR(Vol) {pcr_vol:.2f} call buying")
    elif pcr_vol > p_v_br:
        prob_down += 0.2
        reasons.append(f"PCR(Vol) {pcr_vol:.2f} put buying")

    if call_chg > prem_th:
        prob_up += 0.15
    elif call_chg < -prem_th:
        prob_down += 0.15
    if put_chg > prem_th:
        prob_down += 0.15
    elif put_chg < -prem_th:
        prob_up += 0.15

    if max_pain and abs(spot_vs_mp) > mp_dev:
        if spot_vs_mp > mp_dev:
            prob_down += 0.1
            reasons.append(f"Spot {spot_vs_mp:.0f} pts above Max Pain {max_pain}")
        else:
            prob_up += 0.1
            reasons.append(f"Spot {abs(spot_vs_mp):.0f} pts below Max Pain {max_pain}")

    total = prob_up + prob_down
    if total > 0:
        prob_up = min(prob_cap, prob_up / (total + norm_off))
        prob_down = min(prob_cap, prob_down / (total + norm_off))
    prob_no = max(prob_no_fl, 1.0 - prob_up - prob_down)

    if prob_up > prob_down and prob_up > dir_th:
        direction = "BULLISH"
        confidence = prob_up
        suggested = _best_otm_ce([s for s in snapshot.strikes if s.option_type == "CE"], snapshot.spot_price)
    elif prob_down > prob_up and prob_down > dir_th:
        direction = "BEARISH"
        confidence = prob_down
        suggested = _best_otm_pe([s for s in snapshot.strikes if s.option_type == "PE"], snapshot.spot_price)
    else:
        direction = "NEUTRAL"
        confidence = prob_no
        suggested = snapshot.atm_strike

    return HugeMovePrediction(
        direction=direction,
        prob_huge_up=round(prob_up, 2),
        prob_huge_down=round(prob_down, 2),
        prob_no_move=round(prob_no, 2),
        pcr_oi=round(pcr_oi, 3),
        pcr_volume=round(pcr_vol, 3),
        max_pain=max_pain,
        spot_vs_max_pain=round(spot_vs_mp, 1),
        reasons=reasons,
        suggested_strike=suggested,
        confidence=round(confidence, 2),
    )


def _best_otm_ce(ce_strikes: List[OptionStrikeData], spot: float) -> int:
    otm = [s for s in ce_strikes if s.strike > spot]
    if not otm:
        return max(s.strike for s in ce_strikes) if ce_strikes else 0
    return min(otm, key=lambda s: s.strike - spot).strike


def _best_otm_pe(pe_strikes: List[OptionStrikeData], spot: float) -> int:
    otm = [s for s in pe_strikes if s.strike < spot]
    if not otm:
        return min(s.strike for s in pe_strikes) if pe_strikes else 0
    return max(otm, key=lambda s: spot - s.strike).strike


class HugeMovePredictor:
    """Fetch, store, and predict huge moves from option chain."""

    def __init__(self, client: ZerodhaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.data_dir = settings.data_dir

    def capture_and_store(
        self,
        trade_date: date | None = None,
        spot_price: float | None = None,
        num_strikes: int | None = None,
    ) -> OptionChainSnapshot | None:
        """
        Fetch option chain (5-10 strikes), save to JSONL, return snapshot.
        """
        from datetime import date as date_type

        from intraday_engine.core.tunables import get_int

        td = trade_date or date_type.today()
        ns = num_strikes if num_strikes is not None else get_int("huge_move_predictor", "DEFAULT_NUM_STRIKES", 5)
        if spot_price is None:
            q = self.client.quote([self.settings.spot_symbol])
            spot_price = float(q.get(self.settings.spot_symbol, {}).get("last_price", 0) or 0)
        if spot_price <= 0:
            return None

        snapshot = fetch_option_chain_near_spot(
            self.client,
            self.settings,
            td,
            spot_price,
            num_strikes_each_side=ns,
            use_expiry_day_only=False,
        )
        if snapshot:
            save_option_chain_snapshot(
                snapshot,
                self.data_dir,
                self.settings.underlying,
            )
        return snapshot

    def predict(
        self,
        trade_date: date | None = None,
        spot_price: float | None = None,
        num_strikes: int | None = None,
        use_stored: bool = False,
    ) -> HugeMovePrediction | None:
        """
        Predict huge move probability. If use_stored=True, uses latest stored snapshot for date.
        """
        from datetime import date as date_type

        td = trade_date or date_type.today()
        snapshot = None

        if use_stored:
            records = load_option_chain_snapshots(self.data_dir, td, self.settings.underlying)
            if records:
                snapshot = _snapshot_from_record(records[-1], td)
        if not snapshot:
            snapshot = self.capture_and_store(td, spot_price, num_strikes)
        if not snapshot:
            return None

        from intraday_engine.core.tunables import get_float

        return predict_huge_move_from_snapshot(
            snapshot,
            lot_size=self.settings.lot_size,
            huge_move_points=get_float("huge_move_predictor", "HUGE_MOVE_POINTS", 100.0),
        )


def _snapshot_from_record(record: dict, trade_date: date) -> OptionChainSnapshot | None:
    """Reconstruct OptionChainSnapshot from stored JSONL record."""
    strikes_raw = record.get("strikes", [])
    if not strikes_raw:
        return None
    strikes = [_opt_strike(s) for s in strikes_raw]
    ts = record.get("timestamp", "")
    try:
        timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        timestamp = datetime.now()
    exp = record.get("expiry")
    expiry = None
    if exp:
        try:
            expiry = date.fromisoformat(exp[:10])
        except (ValueError, TypeError):
            pass

    return OptionChainSnapshot(
        trade_date=trade_date,
        spot_price=float(record.get("spot_price", 0)),
        atm_strike=int(record.get("atm_strike", 0)),
        strikes=strikes,
        spot_volume=float(record.get("spot_volume", 0)),
        timestamp=timestamp,
        expiry=expiry,
    )
