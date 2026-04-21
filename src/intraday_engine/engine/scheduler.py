from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

from intraday_engine.engine.direction_engine import DirectionEngine

logger = logging.getLogger(__name__)


def _should_capture_option_chain() -> bool:
    val = os.getenv("CAPTURE_OPTION_CHAIN", "0").strip().lower()
    return val in ("1", "true", "yes")


def run_every_five_minutes(engine: DirectionEngine, interval_minutes: int) -> None:
    logger.info("Starting scheduler with %s-minute interval.", interval_minutes)
    capture_chain = _should_capture_option_chain()
    if capture_chain:
        logger.info("Option chain capture enabled (CAPTURE_OPTION_CHAIN=1).")

    while True:
        try:
            engine.run_cycle()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Cycle failed: %s", exc)

        if capture_chain:
            try:
                _capture_option_chain(engine)
            except Exception as exc:  # pragma: no cover
                logger.warning("Option chain capture failed: %s", exc)

        sleep_seconds = _seconds_to_next_boundary(interval_minutes)
        logger.info("Sleeping for %s seconds until next cycle.", sleep_seconds)
        time.sleep(sleep_seconds)


def _capture_option_chain(engine: DirectionEngine) -> None:
    """Capture 5-10 strikes option chain and save to JSONL (for indices only)."""
    from intraday_engine.core.underlyings import list_index_underlyings
    from intraday_engine.gamma.huge_move_predictor import HugeMovePredictor

    if engine.settings.underlying not in list_index_underlyings():
        return
    client = engine.fetcher.resolver.client
    predictor = HugeMovePredictor(client, engine.settings)
    num_strikes = int(os.getenv("OPTION_STRIKES", "5"))
    snapshot = predictor.capture_and_store(num_strikes=num_strikes)
    if snapshot:
        logger.debug("Option chain captured: %d strikes", len(snapshot.strikes))


def _seconds_to_next_boundary(interval_minutes: int) -> int:
    now = datetime.now()
    block = (now.minute // interval_minutes) * interval_minutes
    current = now.replace(minute=block, second=0, microsecond=0)
    nxt = current + timedelta(minutes=interval_minutes)
    return max(1, int((nxt - now).total_seconds()))
