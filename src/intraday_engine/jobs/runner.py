"""Background asyncio loop for configured dashboard jobs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from intraday_engine.jobs.option_chain_scraper import run_option_chain_scraper_job
from intraday_engine.jobs.registry import CONFIGURED_JOBS
from intraday_engine.utils.nse_session import (
    is_nse_session_active,
    seconds_to_next_interval_within_session,
    seconds_until_next_session_start,
    session_label,
)

logger = logging.getLogger(__name__)


async def run_configured_jobs_loop(data_dir: Path) -> None:
    """Run enabled configured jobs on their intervals (NSE session hours only)."""
    logger.info("Configured jobs loop started (%s).", session_label())
    loop = asyncio.get_event_loop()

    while True:
        if not is_nse_session_active():
            sleep_s = seconds_until_next_session_start()
            logger.info(
                "Outside NSE session (%s); sleeping %.0f min until next open.",
                session_label(),
                sleep_s / 60,
            )
            await asyncio.sleep(min(sleep_s, 3600))
            continue

        for job in CONFIGURED_JOBS:
            if not job.get("enabled", True):
                continue
            job_id = job.get("id")
            interval = int(job.get("interval_minutes", 5) or 5)
            if job_id == "option_chain_scraper":
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: run_option_chain_scraper_job(data_dir, respect_market_hours=True),
                    )
                except Exception as exc:
                    logger.exception("Job %s failed: %s", job_id, exc)

        interval = min(
            int(j.get("interval_minutes", 5) or 5)
            for j in CONFIGURED_JOBS
            if j.get("enabled", True)
        )
        sleep_s = seconds_to_next_interval_within_session(interval)
        await asyncio.sleep(sleep_s)
