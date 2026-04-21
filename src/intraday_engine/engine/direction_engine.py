from __future__ import annotations

import logging
from datetime import date
from typing import Dict

from intraday_engine.analysis.day_bias import probable_day_bias
from intraday_engine.analysis.momentum import momentum_direction
from intraday_engine.analysis.scoring import score_signal
from intraday_engine.analysis.sideways import is_sideways_day
from intraday_engine.analysis.support_resistance import calculate_support_resistance
from intraday_engine.analysis.trade_plan import build_trade_plan
from intraday_engine.core.config import Settings
from intraday_engine.features.feature_engineering import compute_features
from intraday_engine.fetch.market_data import MarketDataFetcher
from intraday_engine.storage import DataStore
from intraday_engine.utils.output import print_signal

logger = logging.getLogger(__name__)


class DirectionEngine:
    def __init__(self, fetcher: MarketDataFetcher, store: DataStore, settings: Settings) -> None:
        self.fetcher = fetcher
        self.store = store
        self.settings = settings

    def run_cycle(self, trade_date: date | None = None) -> Dict[str, object]:
        merged = self.fetcher.fetch_intraday_frame_for_date(trade_date)
        merged = self.store.save_snapshots(merged)
        existing_timestamps = self.store.load_signal_timestamps(trade_date=trade_date)

        latest_payload: Dict[str, object] | None = None
        emitted = 0
        for idx in range(len(merged)):
            candle_ts = str(merged.iloc[idx]["timestamp"])
            if candle_ts in existing_timestamps:
                continue

            frame_till_candle = merged.iloc[: idx + 1].reset_index(drop=True)
            payload = self._analyze_frame(frame_till_candle)
            self.store.append_signal(payload)
            print_signal(payload)
            latest_payload = payload
            emitted += 1

        if latest_payload is None:
            latest_payload = self._analyze_frame(merged)
            logger.info("No new completed candle signal to append.")
            return latest_payload

        logger.info("Signals emitted this cycle: %s", emitted)
        return latest_payload

    def _analyze_frame(self, frame: "pd.DataFrame") -> Dict[str, object]:
        if len(frame) < 3:
            last = frame.iloc[-1]
            return {
                "timestamp": str(last["timestamp"]),
                "signal": "NO_TRADE",
                "entry": None,
                "target": None,
                "stop_loss": None,
                "rr": None,
                "strike_price": None,
                "option_type": None,
                "option_entry": None,
                "option_sl": None,
                "option_target": None,
                "option_symbol": None,
                "confidence": 0.0,
                "bias": "NEUTRAL_DAY",
                "momentum": "NEUTRAL",
                "support": float(last["spot_low"]),
                "resistance": float(last["spot_high"]),
                "score": {
                    "bullish": 0.0,
                    "bearish": 0.0,
                    "no_trade_penalty": 0.0,
                    "final_score": 0.0,
                    "confidence": 0.0,
                    "reasons": ["Insufficient bars for signal generation."],
                },
                "notes": ["Waiting for enough 5-minute bars."],
            }

        features = compute_features(frame)
        support, resistance = calculate_support_resistance(frame, self.settings.lookback_bars)
        momentum = momentum_direction(frame)
        bias = probable_day_bias(features)

        spot = float(frame.iloc[-1]["spot_ltp"])
        is_breakout = spot > resistance
        is_breakdown = spot < support
        follow_through = abs(features["spot_change_pct"]) > 0.06
        range_size = max(resistance - support, 1e-6)
        distance_from_mid = abs(spot - (support + resistance) / 2.0)
        is_mid_range = distance_from_mid < 0.18 * range_size

        long_stop_pct = abs((spot - support) / max(spot, 1.0) * 100)
        short_stop_pct = abs((resistance - spot) / max(spot, 1.0) * 100)
        stop_too_wide = min(long_stop_pct, short_stop_pct) > self.settings.max_stop_pct

        range_pct = range_size / max(spot, 1.0) * 100
        sideways, sideways_reason = is_sideways_day(
            range_pct=range_pct,
            bias=bias,
            momentum=momentum,
            follow_through=follow_through,
            is_breakout=is_breakout,
            is_breakdown=is_breakdown,
            features=features,
            min_range_pct=self.settings.min_day_range_pct,
        )
        if sideways:
            return self._no_trade_payload(
                frame, support, resistance, sideways_reason, bias, momentum
            )

        score = score_signal(
            features=features,
            momentum=momentum,
            bias=bias,
            is_breakout=is_breakout,
            is_breakdown=is_breakdown,
            follow_through=follow_through,
            stop_too_wide=stop_too_wide,
            is_mid_range=is_mid_range,
        )
        last_row = frame.iloc[-1]
        atm_strike = int(last_row.get("atm_strike", 0))
        call_ltp = float(last_row.get("call_ltp", 0.0))
        put_ltp = float(last_row.get("put_ltp", 0.0))
        plan = build_trade_plan(
            spot=spot,
            support=support,
            resistance=resistance,
            score=score,
            bias=bias,
            momentum=momentum,
            settings=self.settings,
            atm_strike=atm_strike,
            call_ltp=call_ltp,
            put_ltp=put_ltp,
        )
        payload = {"timestamp": str(frame.iloc[-1]["timestamp"]), **plan.to_dict()}
        ce_symbol = str(last_row.get("ce_symbol", ""))
        pe_symbol = str(last_row.get("pe_symbol", ""))
        if plan.signal == "BUY" and ce_symbol:
            payload["option_symbol"] = ce_symbol
        elif plan.signal == "SELL" and pe_symbol:
            payload["option_symbol"] = pe_symbol
        else:
            payload["option_symbol"] = None
        return payload

    def _no_trade_payload(
        self,
        frame: "pd.DataFrame",
        support: float,
        resistance: float,
        reason: str,
        bias: str,
        momentum: str,
    ) -> Dict[str, object]:
        last = frame.iloc[-1]
        return {
            "timestamp": str(last["timestamp"]),
            "signal": "NO_TRADE",
            "entry": None,
            "target": None,
            "stop_loss": None,
            "rr": None,
            "strike_price": None,
            "option_type": None,
            "option_entry": None,
            "option_sl": None,
            "option_target": None,
            "option_symbol": None,
            "confidence": 0.0,
            "bias": bias,
            "momentum": momentum,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "score": {
                "bullish": 0.0,
                "bearish": 0.0,
                "no_trade_penalty": 0.0,
                "final_score": 0.0,
                "confidence": 0.0,
                "reasons": [reason],
            },
            "notes": [reason],
        }
