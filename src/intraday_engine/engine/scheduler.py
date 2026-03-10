from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from intraday_engine.engine.direction_engine import DirectionEngine

logger = logging.getLogger(__name__)


def run_every_five_minutes(engine: DirectionEngine, interval_minutes: int) -> None:
    logger.info("Starting scheduler with %s-minute interval.", interval_minutes)
    while True:
        try:
            engine.run_cycle()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Cycle failed: %s", exc)
        sleep_seconds = _seconds_to_next_boundary(interval_minutes)
        logger.info("Sleeping for %s seconds until next cycle.", sleep_seconds)
        time.sleep(sleep_seconds)


def _seconds_to_next_boundary(interval_minutes: int) -> int:
    now = datetime.now()
    block = (now.minute // interval_minutes) * interval_minutes
    current = now.replace(minute=block, second=0, microsecond=0)
    nxt = current + timedelta(minutes=interval_minutes)
    return max(1, int((nxt - now).total_seconds()))
