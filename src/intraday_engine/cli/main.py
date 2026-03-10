from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import list_underlyings
from intraday_engine.engine import DirectionEngine, run_every_five_minutes
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.market_data import MarketDataFetcher
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.gamma import GammaBlastDetector
from intraday_engine.gamma.expiry_utils import is_expiry_day
from intraday_engine.storage import DataStore
from intraday_engine.utils.logging_setup import setup_logging


def build_engine(underlying: str | None = None) -> DirectionEngine:
    settings = Settings.from_env(underlying=underlying)
    setup_logging(settings.log_level, settings.data_dir)
    logger = logging.getLogger(__name__)
    logger.info("Bootstrapping intraday direction engine for %s.", settings.underlying)

    client = ZerodhaClient(settings)
    resolver = InstrumentResolver(client, settings)
    fetcher = MarketDataFetcher(client, resolver, settings)
    store = DataStore(settings.data_dir, underlying=settings.underlying)
    return DirectionEngine(fetcher, store, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Production intraday direction engine.")
    parser.add_argument("--once", action="store_true", help="Run one 5-minute cycle and exit.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Generate intraday signals for a specific date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--gamma-blast",
        action="store_true",
        help="Scan for gamma blast trades on expiry day (Nifty: Tuesday).",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default=None,
        choices=list_underlyings(),
        help="Underlying index: NIFTY, BANKNIFTY. Default from UNDERLYING env.",
    )
    args = parser.parse_args()
    selected_date = _parse_date(args.date) if args.date else None
    underlying = args.underlying or None

    if args.gamma_blast:
        _run_gamma_blast_scan(selected_date, underlying)
        return

    engine = build_engine(underlying=underlying)
    if args.once or selected_date is not None:
        engine.run_cycle(trade_date=selected_date)
        return

    settings = Settings.from_env()
    run_every_five_minutes(engine, settings.poll_interval_minutes)


def _run_gamma_blast_scan(trade_date: date | None, underlying: str | None = None) -> None:
    """Run gamma blast detection for expiry day."""
    settings = Settings.from_env(underlying=underlying)
    setup_logging(settings.log_level, settings.data_dir)
    logger = logging.getLogger(__name__)

    if trade_date is None:
        trade_date = date.today()

    if not is_expiry_day(trade_date, settings.underlying):
        logger.info(
            "Not expiry day for %s (expires Tuesday). Today: %s %s",
            settings.underlying,
            trade_date,
            trade_date.strftime("%A"),
        )
        return

    client = ZerodhaClient(settings)
    detector = GammaBlastDetector(client, settings)
    signal = detector.scan(trade_date=trade_date)

    if signal is None:
        logger.warning("Could not fetch option chain for gamma blast scan.")
        return

    logger.info("=== Gamma Blast Scan (%s) ===", trade_date)
    logger.info("Spot: %.2f | ATM: %d | Direction: %s", signal.spot_price, signal.atm_strike, signal.direction)
    logger.info("PCR: %.2f | CE OI: %.0f | PE OI: %.0f", signal.pcr, signal.total_ce_oi, signal.total_pe_oi)
    logger.info("Confidence: %.0f%% | After 1:45 PM: %s", signal.confidence * 100, signal.is_after_1345)
    logger.info("Suggested strike: %d | %s", signal.suggested_strike, signal.reason)


def _parse_date(value: str) -> "datetime.date":
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Invalid --date format. Use YYYY-MM-DD.") from exc


if __name__ == "__main__":
    main()
