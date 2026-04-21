"""
Position monitor script with BTST and trail modes.

BTST mode: Exit all open positions using limit orders based on LTP.
    - Starts trying from NSE 9:14:50 AM IST
    - Places limit orders at LTP, then steps by 50 points each attempt until executed
    - For SELL (long exit): LTP, LTP-50, LTP-100, ...
    - For BUY (short exit): LTP, LTP+50, LTP+100, ...

Trail mode: Monitor positions every 5 minutes, trail SL by 5 points for NIFTY options.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage import DataStore
from intraday_engine.storage.position_sl_store import (
    get_position_key,
    get_sl as get_sl_record,
    update_sl_trigger as store_update_sl,
)
from intraday_engine.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# NSE times (IST)
BTST_START_TIME = dt_time(9, 14, 50)  # Start trying from 9:14:50 AM
MARKET_OPEN_TIME = dt_time(9, 15, 0)
MARKET_CLOSE_TIME = dt_time(15, 30, 0)


def _pm_i(key: str, default: int) -> int:
    from intraday_engine.core.tunables import get_int

    return get_int("position_monitor", key, default)


def _now_ist() -> datetime:
    """Current time in IST."""
    return datetime.now(IST)


def _is_market_open() -> bool:
    """Check if NSE market is open (IST)."""
    now = _now_ist().time()
    if now < MARKET_OPEN_TIME or now > MARKET_CLOSE_TIME:
        return False
    return True


def _wait_until_btst_start() -> None:
    """Block until NSE 9:14:50 AM IST."""
    while True:
        now = _now_ist()
        target = now.replace(hour=9, minute=14, second=50, microsecond=0)
        if now >= target:
            break
        diff = (target - now).total_seconds()
        if diff > 60:
            logger.info("BTST: Waiting until 9:14:50 AM IST (%.0f min)", diff / 60)
        time.sleep(min(30, max(1, diff)))
    logger.info("BTST: Reached 9:14:50 AM IST. Starting exit attempts.")


def _is_nifty_option(tradingsymbol: str) -> bool:
    """Check if tradingsymbol is a NIFTY option (CE/PE)."""
    s = str(tradingsymbol).upper()
    return s.startswith("NIFTY") and ("CE" in s or "PE" in s)


def _infer_underlying(tradingsymbol: str) -> str:
    """Infer underlying from tradingsymbol."""
    s = str(tradingsymbol).upper()
    if s.startswith("BANKNIFTY"):
        return "BANKNIFTY"
    if s.startswith("NIFTY"):
        return "NIFTY"
    for i, c in enumerate(s):
        if c.isdigit():
            return s[:i] if s[:i] else "NIFTY"
    return s or "NIFTY"


def _get_ltp(client: ZerodhaClient, tradingsymbol: str, exchange: str) -> float | None:
    """Fetch LTP for tradingsymbol. Returns None if quote fails."""
    try:
        exch = (exchange or "NFO").upper()
        key = f"{exch}:{tradingsymbol}" if ":" not in tradingsymbol else tradingsymbol
        quote = client.quote([key])
        data = (quote or {}).get(key, {})
        if isinstance(data, dict):
            return float(data.get("last_price", 0) or 0)
    except Exception:
        pass
    return None


def run_btst_mode(client: ZerodhaClient, settings: Settings) -> None:
    """
    BTST mode: Exit all positions using limit orders at LTP, stepping by 50 points
    each attempt until executed.
    """
    _wait_until_btst_start()

    # Track step and pending order_id per position
    position_state: dict[str, dict] = {}  # key -> {step, order_id}

    while True:
        positions = client.get_positions()
        positions = [p for p in positions if isinstance(p, dict) and int(p.get("quantity", 0)) != 0]

        if not positions:
            logger.info("BTST: All positions closed. Done.")
            return

        now_t = _now_ist().time()
        if now_t < BTST_START_TIME or now_t > MARKET_CLOSE_TIME:
            logger.debug("BTST: Outside trading window, waiting 60s.")
            time.sleep(60)
            continue

        for pos in positions:
            qty = int(pos.get("quantity", 0))
            tradingsymbol = str(pos.get("tradingsymbol", "")).replace("NFO:", "").strip()
            product = str(pos.get("product", "MIS")).upper()
            exchange = str(pos.get("exchange", "NFO")).upper()

            if not tradingsymbol or not qty:
                continue

            pos_key = get_position_key(tradingsymbol, qty)
            state = position_state.setdefault(pos_key, {"step": 0, "order_id": None})

            # Cancel previous pending order before placing new one
            if state.get("order_id"):
                try:
                    client.cancel_order(state["order_id"])
                except Exception:
                    pass
                state["order_id"] = None

            ltp = _get_ltp(client, tradingsymbol, exchange)
            if ltp is None or ltp <= 0:
                logger.debug("BTST: No LTP for %s, skipping this attempt", tradingsymbol)
                continue

            step = state["step"]
            if qty > 0:
                transaction_type = "SELL"
                exit_qty = qty
                price = round(ltp - step * _pm_i("BTST_STEP_POINTS", 50), 2)
            else:
                transaction_type = "BUY"
                exit_qty = abs(qty)
                price = round(ltp + step * _pm_i("BTST_STEP_POINTS", 50), 2)

            price = max(0.05, price)  # Floor for options

            try:
                order_id = client.place_order(
                    tradingsymbol=tradingsymbol,
                    exchange=exchange,
                    transaction_type=transaction_type,
                    quantity=exit_qty,
                    product=product if product in ("CNC", "NRML", "MIS") else "MIS",
                    order_type="LIMIT",
                    price=price,
                )
                state["order_id"] = order_id
                logger.info(
                    "BTST: Placed %s limit %s qty=%s price=%.2f (LTP=%.2f step=%d) order_id=%s",
                    transaction_type,
                    tradingsymbol,
                    exit_qty,
                    price,
                    ltp,
                    step,
                    order_id,
                )
            except Exception as e:
                logger.warning("BTST: Order failed for %s: %s", tradingsymbol, e)
                continue

            time.sleep(_pm_i("BTST_WAIT_SECONDS", 2))

            # Refresh positions to check if filled
            fresh_positions = client.get_positions()
            still_open = any(
                isinstance(p, dict)
                and str(p.get("tradingsymbol", "")).replace("NFO:", "").strip() == tradingsymbol
                and int(p.get("quantity", 0)) != 0
                for p in fresh_positions
            )
            if not still_open:
                logger.info("BTST: Position %s closed.", tradingsymbol)
                position_state.pop(pos_key, None)
            else:
                state["step"] = step + 1

        time.sleep(1)  # Brief pause before next cycle


def run_trail_mode(client: ZerodhaClient, settings: Settings) -> None:
    """
    Trail mode: Every 5 minutes, trail SL by 5 points for NIFTY option positions.
    """
    logger.info(
        "Trail mode: Monitoring every %d min, trailing NIFTY option SL by %d points.",
        _pm_i("TRAIL_INTERVAL_MINUTES", 5),
        _pm_i("TRAIL_POINTS", 5),
    )

    while True:
        try:
            now = _now_ist()
            if not _is_market_open():
                logger.debug("Trail: Market closed, skipping cycle.")
                time.sleep(60)
                continue

            positions = client.get_positions()
            nifty_option_positions = [
                p
                for p in positions
                if isinstance(p, dict)
                and int(p.get("quantity", 0)) != 0
                and _is_nifty_option(str(p.get("tradingsymbol", "")))
            ]

            for pos in nifty_option_positions:
                qty = int(pos.get("quantity", 0))
                tradingsymbol = str(pos.get("tradingsymbol", "")).replace("NFO:", "").strip()
                underlying = _infer_underlying(tradingsymbol)
                store = DataStore(settings.data_dir, underlying=underlying)
                rec = get_sl_record(store.data_dir, tradingsymbol, qty)

                if not rec or not rec.get("sl_order_id"):
                    continue

                sl_order_id = rec["sl_order_id"]
                current_sl = rec.get("sl_trigger")
                if current_sl is None:
                    continue

                # Fetch LTP
                try:
                    quote = client.quote([f"NFO:{tradingsymbol}"])
                    ltp = 0.0
                    for k, v in (quote or {}).items():
                        if isinstance(v, dict) and "last_price" in v:
                            ltp = float(v.get("last_price", 0) or 0)
                            break
                    if ltp <= 0:
                        continue
                except Exception as e:
                    logger.debug("Trail: Could not fetch quote for %s: %s", tradingsymbol, e)
                    continue

                is_long = qty > 0
                new_sl = None

                if is_long:
                    # Long option: trail up when LTP rises 5+ above current SL
                    tp = _pm_i("TRAIL_POINTS", 5)
                    if ltp >= current_sl + tp:
                        new_sl = round(current_sl + tp, 2)
                else:
                    # Short option: trail down when LTP falls 5+ below current SL
                    tp = _pm_i("TRAIL_POINTS", 5)
                    if ltp <= current_sl - tp:
                        new_sl = round(current_sl - tp, 2)

                if new_sl is not None:
                    try:
                        client.modify_sl_order(sl_order_id, new_sl)
                        store_update_sl(store.data_dir, tradingsymbol, qty, new_sl)
                        logger.info("Trail: %s SL %.2f -> %.2f (LTP %.2f)", tradingsymbol, current_sl, new_sl, ltp)
                    except Exception as e:
                        logger.warning("Trail: Modify SL failed for %s: %s", tradingsymbol, e)

        except Exception as e:
            logger.exception("Trail cycle error: %s", e)

        time.sleep(_pm_i("TRAIL_INTERVAL_MINUTES", 5) * 60)


def run(mode: str, underlying: str | None = None) -> None:
    """Run position monitor in given mode. mode: 'btst' or 'trail'."""
    settings = Settings.from_env(underlying=underlying)
    setup_logging(settings.log_level, settings.data_dir)
    client = ZerodhaClient(settings)
    if mode == "btst":
        run_btst_mode(client, settings)
    else:
        run_trail_mode(client, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Position monitor: BTST exit or trail SL.")
    parser.add_argument(
        "mode",
        choices=["btst", "trail"],
        help="btst: Exit at market open (9:14:50 AM IST). trail: Trail SL by 5 pts every 5 min.",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default=None,
        help="Underlying (NIFTY, BANKNIFTY). Default from env.",
    )
    args = parser.parse_args()
    run(args.mode, args.underlying)


if __name__ == "__main__":
    main()
