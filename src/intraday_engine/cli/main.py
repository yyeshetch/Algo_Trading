from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import list_index_underlyings
from intraday_engine.engine import DirectionEngine, run_every_five_minutes
from intraday_engine.engine.stock_cycle_runner import run_every_15_minutes, run_stocks_15min_cycle
from intraday_engine.orb.orb_scanner import run_orb_scan, run_pinbar_scan
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.market_data import MarketDataFetcher
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.gamma import GammaBlastDetector
from intraday_engine.gamma.expiry_utils import is_expiry_day
from intraday_engine.gamma.huge_move_predictor import HugeMovePredictor
from intraday_engine.position_monitor import run as run_position_monitor
from intraday_engine.research.nifty500_accumulation_scanner import print_scan_report, run_nifty500_accumulation_scan
from intraday_engine.research.tomorrow_watchlist_scanner import run_tomorrow_watchlist_scan
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
        help="Underlying: NIFTY, BANKNIFTY, or F&O stock (e.g. RELIANCE, TCS). Default from UNDERLYING env.",
    )
    parser.add_argument(
        "--stocks-15min",
        action="store_true",
        help="Run 15-min F&O stocks cycle (fetch data + generate signals for all stocks).",
    )
    parser.add_argument(
        "--stocks-15min-scheduler",
        action="store_true",
        help="Run F&O stocks 15-min scheduler (continuous, every 15 min).",
    )
    parser.add_argument(
        "--stocks-limit",
        type=int,
        default=50,
        help="Max F&O stocks to process (with --stocks-15min). Default 50.",
    )
    parser.add_argument(
        "--orb",
        action="store_true",
        help="Run 15-min ORB scan (0.2%% variation, bulk quote).",
    )
    parser.add_argument(
        "--orb-limit",
        type=int,
        default=200,
        help="Max stocks for ORB scan. Default 200.",
    )
    parser.add_argument(
        "--pinbar",
        action="store_true",
        help="Scan for bullish/bearish pinbars on 15-min.",
    )
    parser.add_argument(
        "--pinbar-limit",
        type=int,
        default=200,
        help="Max stocks for pinbar scan. Default 200.",
    )
    parser.add_argument(
        "--huge-move",
        action="store_true",
        help="Capture option chain (5-10 strikes) and predict probability of 100+ point move.",
    )
    parser.add_argument(
        "--capture-option-chain",
        action="store_true",
        help="Capture and store option chain (5-10 strikes near spot) to JSONL.",
    )
    parser.add_argument(
        "--option-strikes",
        type=int,
        default=5,
        help="Number of strikes each side of ATM for option chain (default 5).",
    )
    parser.add_argument(
        "--btst",
        action="store_true",
        help="BTST mode: Exit all positions at market when it opens (9:14:50 AM IST).",
    )
    parser.add_argument(
        "--trail",
        action="store_true",
        help="Trail mode: Monitor positions every 5 min, trail NIFTY option SL by 5 points.",
    )
    parser.add_argument(
        "--nifty500-accumulation",
        action="store_true",
        help="Scan NIFTY 500 (1h): accumulation + rising trendline + bullish breakout readiness.",
    )
    parser.add_argument(
        "--nifty500-top",
        type=int,
        default=15,
        help="Max names to print (with --nifty500-accumulation). Default 15.",
    )
    parser.add_argument(
        "--nifty500-workers",
        type=int,
        default=4,
        help="Parallel historical fetches (with --nifty500-accumulation). Default 4.",
    )
    parser.add_argument(
        "--nifty500-symbols-file",
        type=str,
        default=None,
        help="Optional path to txt/csv (one symbol per line) instead of downloading NIFTY 500.",
    )
    parser.add_argument(
        "--nifty500-out",
        type=str,
        default=None,
        help="Optional JSON output path for top scan rows.",
    )
    parser.add_argument(
        "--tomorrow-watchlist",
        action="store_true",
        help="Scan NIFTY 500 (day + 1h + 15m) for next-session watchlist; saves JSON under data_dir.",
    )
    parser.add_argument(
        "--tw-top",
        type=int,
        default=20,
        help="Max picks to keep (with --tomorrow-watchlist). Default 20.",
    )
    parser.add_argument(
        "--tw-workers",
        type=int,
        default=4,
        help="Parallel workers (with --tomorrow-watchlist). Default 4.",
    )
    parser.add_argument(
        "--tw-limit",
        type=int,
        default=None,
        help="Optional max symbols to scan (with --tomorrow-watchlist), for testing.",
    )
    args = parser.parse_args()
    selected_date = _parse_date(args.date) if args.date else None
    underlying = args.underlying or None

    if args.btst:
        setup_logging(Settings.from_env(underlying=underlying).log_level, Settings.from_env().data_dir)
        run_position_monitor("btst", underlying)
        return

    if args.trail:
        setup_logging(Settings.from_env(underlying=underlying).log_level, Settings.from_env().data_dir)
        run_position_monitor("trail", underlying)
        return

    if args.nifty500_accumulation:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        sym_path = Path(args.nifty500_symbols_file) if args.nifty500_symbols_file else None
        out_path = Path(args.nifty500_out) if args.nifty500_out else None
        rows = run_nifty500_accumulation_scan(
            settings=settings,
            symbols_file=sym_path,
            top_n=args.nifty500_top,
            max_workers=args.nifty500_workers,
            out_json=out_path,
            trade_date=selected_date or date.today(),
        )
        print_scan_report(rows)
        return

    if args.tomorrow_watchlist:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        payload = run_tomorrow_watchlist_scan(
            settings=settings,
            trade_date=selected_date or date.today(),
            top_n=args.tw_top,
            max_workers=args.tw_workers,
            stock_limit=args.tw_limit,
        )
        picks = payload.get("picks") or []
        print(f"Scanned {payload.get('scanned')} symbols, {len(picks)} picks (failed {payload.get('failed_count', 0)}).")
        for p in picks[:15]:
            print(
                f"{p.get('stock'):12} {p.get('next_day_bias'):8} {p.get('setup_type'):12} "
                f"conf={p.get('confidence_score')} vol={p.get('volume_profile')} {p.get('reason', '')[:80]}"
            )
        return

    if args.gamma_blast:
        _run_gamma_blast_scan(selected_date, underlying)
        return

    if args.stocks_15min_scheduler:
        setup_logging(Settings.from_env().log_level, Settings.from_env().data_dir)
        run_every_15_minutes(stock_limit=args.stocks_limit)
        return

    if args.stocks_15min:
        setup_logging(Settings.from_env().log_level, Settings.from_env().data_dir)
        n = run_stocks_15min_cycle(trade_date=selected_date or date.today(), stock_limit=args.stocks_limit)
        print(f"Processed {n} F&O stocks.")
        return

    if args.orb:
        setup_logging(Settings.from_env().log_level, Settings.from_env().data_dir)
        signals = run_orb_scan(trade_date=selected_date or date.today(), stock_limit=args.orb_limit)
        buy = [s for s in signals if s["signal"] == "BUY"]
        sell = [s for s in signals if s["signal"] == "SELL"]
        print(f"ORB: {len(buy)} BUY, {len(sell)} SELL")
        for s in buy[:10]:
            print(f"  BUY  {s['stock']} @ {s['price']} (OR {s['or_low']}-{s['or_high']})")
        for s in sell[:10]:
            print(f"  SELL {s['stock']} @ {s['price']} (OR {s['or_low']}-{s['or_high']})")
        return

    if args.pinbar:
        setup_logging(Settings.from_env().log_level, Settings.from_env().data_dir)
        signals = run_pinbar_scan(trade_date=selected_date or date.today(), stock_limit=args.pinbar_limit)
        bull = [s for s in signals if s.get("pattern") == "BULLISH_PINBAR"]
        bear = [s for s in signals if s.get("pattern") == "BEARISH_PINBAR"]
        print(f"Pinbar: {len(bull)} bullish, {len(bear)} bearish")
        for s in bull[:10]:
            print(f"  BULL  {s['stock']} O:{s['open']} H:{s['high']} L:{s['low']} C:{s['close']}")
        for s in bear[:10]:
            print(f"  BEAR  {s['stock']} O:{s['open']} H:{s['high']} L:{s['low']} C:{s['close']}")
        return

    if args.huge_move or args.capture_option_chain:
        _run_huge_move_or_capture(
            capture_only=args.capture_option_chain,
            trade_date=selected_date or date.today(),
            underlying=underlying,
            num_strikes=args.option_strikes,
        )
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


def _run_huge_move_or_capture(
    capture_only: bool,
    trade_date: date,
    underlying: str | None,
    num_strikes: int = 5,
) -> None:
    """Capture option chain and optionally run huge move prediction."""
    settings = Settings.from_env(underlying=underlying)
    setup_logging(settings.log_level, settings.data_dir)
    logger = logging.getLogger(__name__)

    client = ZerodhaClient(settings)
    predictor = HugeMovePredictor(client, settings)

    if capture_only:
        snapshot = predictor.capture_and_store(trade_date=trade_date, num_strikes=num_strikes)
        if snapshot:
            logger.info(
                "Option chain captured: %d strikes, spot %.2f, saved to %s",
                len(snapshot.strikes),
                snapshot.spot_price,
                settings.data_dir,
            )
        else:
            logger.warning("Could not fetch option chain.")
        return

    pred = predictor.predict(trade_date=trade_date, num_strikes=num_strikes, use_stored=False)
    if pred is None:
        logger.warning("Could not run huge move prediction.")
        return

    logger.info("=== Huge Move Prediction (%s) ===", trade_date)
    logger.info("Direction: %s | Confidence: %.0f%%", pred.direction, pred.confidence * 100)
    logger.info("P(huge UP): %.0f%% | P(huge DOWN): %.0f%% | P(no move): %.0f%%",
                pred.prob_huge_up * 100, pred.prob_huge_down * 100, pred.prob_no_move * 100)
    logger.info("PCR(OI): %.2f | PCR(Vol): %.2f | Max Pain: %s | Spot vs MP: %.1f pts",
                pred.pcr_oi, pred.pcr_volume, pred.max_pain or "—", pred.spot_vs_max_pain)
    if pred.suggested_strike:
        logger.info("Suggested strike: %d", pred.suggested_strike)
    for r in pred.reasons:
        logger.info("  • %s", r)


def _parse_date(value: str) -> "datetime.date":
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Invalid --date format. Use YYYY-MM-DD.") from exc


if __name__ == "__main__":
    main()
