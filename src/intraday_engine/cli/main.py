from __future__ import annotations

import argparse
import logging
import os
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
from intraday_engine.research.silent_accumulation_scanner import run_silent_accumulation_scan
from intraday_engine.research.fii_dii_trends import run_fii_dii_trends_scan
from intraday_engine.research.relative_strength_scanner import run_relative_strength_scan
from intraday_engine.research.minervini_trend_template_scanner import run_minervini_trend_template_scan
from intraday_engine.research.fundamentals_screener import run_fundamentals_scan
from intraday_engine.research.stock_news_scanner import run_news_scan
from intraday_engine.research.combined_signals_scanner import run_combined_scan
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
        help="Capture and store one option-chain snapshot (ATM ladder) to CSV.",
    )
    parser.add_argument(
        "--option-chain-scheduler",
        action="store_true",
        help="Option chain only, every 5 min. Prefer --session-scheduler for the full pipeline.",
    )
    parser.add_argument(
        "--session-scheduler",
        action="store_true",
        help=(
            "Full intraday pipeline every 5 min (Mon–Fri 09:15–15:30 IST): option chain → "
            "index analysis → options-trading scan. Dashboard reads stored output."
        ),
    )
    parser.add_argument(
        "--storage",
        choices=["write_to_files", "write_to_db"],
        default=None,
        help=(
            "Where session pipeline persists data. Default: write_to_files, or STORAGE_BACKEND env. "
            "Use write_to_db for InterServer MySQL."
        ),
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help=(
            "Create MySQL tables (signals, market_snapshots, option_chain_rows, json_artifacts). "
            "Use MYSQL_HOST=localhost on InterServer VPS, or run scripts/init_mysql_schema.sql in phpMyAdmin."
        ),
    )
    parser.add_argument(
        "--session-once",
        action="store_true",
        help="Run one full session cycle (option chain + index pipeline) and exit.",
    )
    parser.add_argument(
        "--index-pipeline-scheduler",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--index-pipeline-once",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--backfill-option-chain",
        action="store_true",
        help="Backfill 5-min option-chain bars from Kite historical OI (requires --date).",
    )
    parser.add_argument(
        "--backfill-full-day",
        action="store_true",
        help=(
            "Backfill entire session (09:15–15:25) for --underlying. "
            "Fills only missing bars unless --replace-existing."
        ),
    )
    parser.add_argument(
        "--bar-times",
        type=str,
        nargs="+",
        default=["11:10", "11:15"],
        help="Bar open times HH:MM (with --backfill-option-chain, not --backfill-full-day).",
    )
    parser.add_argument(
        "--strike-min",
        type=int,
        default=None,
        help="Min strike for backfill (default: auto from CSV or Index Analysis).",
    )
    parser.add_argument(
        "--strike-max",
        type=int,
        default=None,
        help="Max strike for backfill (default: auto from CSV or Index Analysis).",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="With backfill: drop existing captures in target bars before rebuilding.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="With --backfill-full-day: rewrite all session bars (same as --replace-existing).",
    )
    parser.add_argument(
        "--remove-chain-prefix",
        type=str,
        nargs="*",
        default=None,
        help="Drop snapshots whose timestamp starts with these prefixes before backfill.",
    )
    parser.add_argument(
        "--option-strikes",
        type=int,
        default=None,
        help="Legacy: symmetric CE/PE strike count override (default from OPTION_CHAIN_* env).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run FastAPI dashboard (uvicorn). Same as: uvicorn intraday_engine.dashboard:app",
    )
    parser.add_argument(
        "--no-option-chain-job",
        action="store_true",
        help="With --dashboard: do not schedule the built-in option-chain scraper job.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help=(
            "With --dashboard: stored data only — no background fetches, scans, or pipeline loops. "
            "Implies --no-option-chain-job and disables auto-trade/direction-engine loop."
        ),
    )
    parser.add_argument(
        "--no-auto-trade-loop",
        action="store_true",
        help="With --dashboard: do not run the 5-min direction-engine / auto-trade background loop.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Dashboard bind host (with --dashboard). Default 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Dashboard bind port (with --dashboard). Default 8000.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (with --dashboard).",
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
    parser.add_argument(
        "--silent-accumulation",
        action="store_true",
        help="Scan NIFTY 500 daily bars for silent (Wyckoff-style) institutional accumulation.",
    )
    parser.add_argument(
        "--silent-top",
        type=int,
        default=25,
        help="Top-N to keep in silent-accumulation output JSON. Default 25.",
    )
    parser.add_argument(
        "--silent-no-nse",
        action="store_true",
        help="Skip NSE delivery%% / bulk-deals download (OHLCV-only signals).",
    )
    parser.add_argument(
        "--fii-dii-snapshot",
        action="store_true",
        help="Refresh FII/DII + participant-OI 30-day trends JSON.",
    )
    parser.add_argument(
        "--relative-strength",
        action="store_true",
        help="Scan NIFTY 500 daily bars for stocks outperforming NIFTY 50 (RS line).",
    )
    parser.add_argument(
        "--minervini-template",
        action="store_true",
        help="Scan NIFTY 500 for Minervini Stage-2 Trend Template + VCP readiness.",
    )
    parser.add_argument(
        "--minervini-mode",
        choices=("setup", "extended"),
        default="setup",
        help="setup = pre-breakout VCP (default); extended = full Stage-2 template.",
    )
    parser.add_argument(
        "--minervini-top",
        type=int,
        default=50,
        help="Top-N Minervini candidates in output JSON. Default 50.",
    )
    parser.add_argument(
        "--minervini-rs-min",
        type=float,
        default=70.0,
        help="Minimum RS rank percentile for full pass (default 70).",
    )
    parser.add_argument(
        "--minervini-all-partial",
        action="store_true",
        help="Include partial template matches, not only full Trend Template + RS pass.",
    )
    parser.add_argument(
        "--rs-top",
        type=int,
        default=50,
        help="Top-N outperformers to keep in RS output JSON. Default 50.",
    )
    parser.add_argument(
        "--rs-include-all",
        action="store_true",
        help="Include underperformers in RS output (default keeps only outperformers).",
    )
    parser.add_argument(
        "--fundamentals",
        action="store_true",
        help="Scrape screener.in fundamentals for NIFTY 500 -> data/analysis/fundamentals/*.csv.",
    )
    parser.add_argument(
        "--fundamentals-workers",
        type=int,
        default=4,
        help="Parallel workers for fundamentals scrape (default 4; keep low to be polite).",
    )
    parser.add_argument(
        "--fundamentals-limit",
        type=int,
        default=None,
        help="Optional cap on symbols (for smoke tests).",
    )
    parser.add_argument(
        "--fundamentals-cache-hours",
        type=int,
        default=24,
        help="Reuse per-symbol screener.in cache younger than this (default 24h).",
    )
    parser.add_argument(
        "--fundamentals-force",
        action="store_true",
        help="Ignore per-symbol cache and re-scrape every stock.",
    )
    parser.add_argument(
        "--stock-news",
        action="store_true",
        help="Pull Google News RSS for NIFTY 500 + score sentiment -> data/analysis/news/*.csv.",
    )
    parser.add_argument(
        "--news-workers",
        type=int,
        default=8,
        help="Parallel workers for news scrape (default 8).",
    )
    parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=7,
        help="Only consider headlines published within the last N days (default 7).",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=None,
        help="Optional cap on symbols (for smoke tests).",
    )
    parser.add_argument(
        "--combined-signals",
        action="store_true",
        help="Join latest institutional volume + fundamentals + news into one ranked CSV.",
    )
    args = parser.parse_args()
    selected_date = _parse_date(args.date) if args.date else None
    underlying = args.underlying or None

    if args.storage:
        from intraday_engine.storage.backend import set_storage_backend

        set_storage_backend(args.storage)

    if args.init_db:
        _run_init_db()
        return

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

    if args.silent_accumulation:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        payload = run_silent_accumulation_scan(
            settings=settings,
            top_n=args.silent_top,
            use_nse_data=not args.silent_no_nse,
            trade_date=selected_date or date.today(),
        )
        rows = payload.get("rows", [])
        print(
            f"Silent accumulation: {len(rows)} top candidates of {payload.get('passed', 0)} passed "
            f"({payload.get('scanned', 0)} scanned). NSE data: {payload.get('nse_data_status')}"
        )
        for r in rows[:15]:
            print(
                f"  {r.get('stock'):12} score={r.get('score'):>5}  "
                f"OBV+{r.get('obv_slope_pct'):>5}%  CMF={r.get('cmf'):>6}  "
                f"U/D={r.get('up_down_vol_ratio')} deliv={r.get('avg_delivery_pct')}"
            )
        return

    if args.fii_dii_snapshot:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        payload = run_fii_dii_trends_scan(settings=settings, trade_date=selected_date or date.today())
        s = payload.get("summary", {})
        print(
            f"FII/DII: {len(payload.get('fii_dii') or [])} sessions, "
            f"OI rows: {len(payload.get('participant_oi') or [])}"
        )
        print(f"  FII net 5d/30d: {s.get('fii_net_5d')} / {s.get('fii_net_30d')}")
        print(f"  DII net 5d/30d: {s.get('dii_net_5d')} / {s.get('dii_net_30d')}")
        return

    if args.relative_strength:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        payload = run_relative_strength_scan(
            settings=settings,
            top_n=args.rs_top,
            only_outperformers=not args.rs_include_all,
            trade_date=selected_date or date.today(),
        )
        rows = payload.get("rows", [])
        np_ = payload.get("nifty_pct_change", {})
        print(
            f"Relative strength: {len(rows)} shown, {payload.get('passed', 0)} outperformers of "
            f"{payload.get('scanned', 0)} scanned. NIFTY 5d/20d/60d: "
            f"{np_.get('5d')}% / {np_.get('20d')}% / {np_.get('60d')}%"
        )
        for r in rows[:15]:
            print(
                f"  {r.get('stock'):12} strength={r.get('strength_score'):>6}  "
                f"RS20d={r.get('rs_slope_20d_pct')}%  vsN20d={r.get('excess_20d')}%"
            )
        return

    if args.minervini_template:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        payload = run_minervini_trend_template_scan(
            settings=settings,
            top_n=args.minervini_top,
            rs_min_percentile=args.minervini_rs_min,
            mode=args.minervini_mode,
            only_full_pass=not args.minervini_all_partial and args.minervini_mode == "extended",
            trade_date=selected_date or date.today(),
        )
        rows = payload.get("rows", [])
        print(
            f"Minervini {payload.get('scan_mode', 'setup')}: {payload.get('passed', 0)} pass, "
            f"{len(rows)} shown of {payload.get('evaluated', 0)} evaluated "
            f"({payload.get('scanned', 0)} symbols)"
        )
        for r in rows[:15]:
            score = r.get('setup_score') if r.get('scan_mode') == 'setup' else r.get('composite_score')
            print(
                f"  {r.get('stock'):12} score={score:>5}  "
                f"rules={r.get('criteria_passed')}/{r.get('criteria_total')}  "
                f"RS={r.get('rs_rank_percentile'):>5.0f}  VCP={r.get('vcp_score'):>4.0f}  "
                f"to pivot={r.get('dist_to_pivot_pct')}%"
            )
        return

    if args.fundamentals:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        sym_path = Path(args.nifty500_symbols_file) if args.nifty500_symbols_file else None
        payload = run_fundamentals_scan(
            settings=settings,
            symbols_file=sym_path,
            trade_date=selected_date or date.today(),
            cache_max_age_hours=args.fundamentals_cache_hours,
            max_workers=args.fundamentals_workers,
            stock_limit=args.fundamentals_limit,
            force_refresh=args.fundamentals_force,
        )
        print(
            f"Fundamentals: scanned={payload['scanned']} ok={payload['ok']} "
            f"failed={payload['failed']} -> {payload['output_csv']}"
        )
        return

    if args.stock_news:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        sym_path = Path(args.nifty500_symbols_file) if args.nifty500_symbols_file else None
        payload = run_news_scan(
            settings=settings,
            symbols_file=sym_path,
            trade_date=selected_date or date.today(),
            lookback_days=args.news_lookback_days,
            max_workers=args.news_workers,
            stock_limit=args.news_limit,
        )
        print(
            f"News: scanned={payload['scanned']} ok={payload['ok']} "
            f"failed={payload['failed']} -> {payload['output_csv']}"
        )
        return

    if args.combined_signals:
        settings = Settings.from_env(underlying="NIFTY")
        setup_logging(settings.log_level, settings.data_dir)
        payload = run_combined_scan(
            settings=settings,
            trade_date=selected_date or date.today(),
        )
        print(
            f"Combined: universe={payload['universe']} "
            f"(inst={payload['with_institutional']}, fund={payload['with_fundamentals']}, "
            f"news={payload['with_news']}) -> {payload['output_csv']}"
        )
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

    if args.backfill_full_day or args.backfill_option_chain:
        from intraday_engine.gamma.option_chain_fetcher import (
            backfill_option_chain_full_day,
            backfill_option_chain_historical,
        )

        td = selected_date or date.today()
        settings = Settings.from_env(underlying=underlying)
        setup_logging(settings.log_level, settings.data_dir)
        replace = args.replace_existing or args.no_skip_existing
        if args.backfill_full_day:
            result = backfill_option_chain_full_day(
                settings.data_dir,
                td,
                underlying,
                skip_existing=not replace,
                replace_existing=replace,
                strike_min=args.strike_min,
                strike_max=args.strike_max,
                remove_capture_prefixes=args.remove_chain_prefix,
            )
        else:
            strike_min = args.strike_min if args.strike_min is not None else 23600
            strike_max = args.strike_max if args.strike_max is not None else 24900
            result = backfill_option_chain_historical(
                settings.data_dir,
                td,
                underlying,
                args.bar_times,
                strike_min=strike_min,
                strike_max=strike_max,
                remove_capture_prefixes=args.remove_chain_prefix,
                replace_existing=True,
            )
        print(result)
        return

    if args.option_chain_scheduler:
        from intraday_engine.jobs.option_chain_scraper import run_option_chain_scheduler_loop

        settings = Settings.from_env(underlying=underlying)
        setup_logging(settings.log_level, settings.data_dir)
        run_option_chain_scheduler_loop(settings.data_dir)
        return

    if args.session_scheduler or args.index_pipeline_scheduler:
        if args.index_pipeline_scheduler:
            logging.getLogger(__name__).warning(
                "--index-pipeline-scheduler is deprecated; use --session-scheduler "
                "(includes option chain + index pipeline)."
            )
        _run_session_scheduler(underlying=underlying)
        return

    if args.session_once or args.index_pipeline_once:
        if args.index_pipeline_once:
            logging.getLogger(__name__).warning(
                "--index-pipeline-once is deprecated; use --session-once "
                "(includes option chain + index pipeline)."
            )
        _run_session_once(underlying=underlying, trade_date=selected_date)
        return

    if args.dashboard:
        _run_dashboard(
            host=args.host,
            port=args.port,
            reload=args.reload,
            no_option_chain_job=args.no_option_chain_job,
            read_only=args.read_only,
            no_auto_trade_loop=args.no_auto_trade_loop,
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


def _run_init_db() -> None:
    from intraday_engine.storage.db import ensure_schema, mysql_host_label, test_connection

    settings = Settings.from_env()
    setup_logging(settings.log_level, settings.data_dir)
    logger = logging.getLogger(__name__)
    logger.info("Connecting to MySQL at %s …", mysql_host_label())
    try:
        info = test_connection()
        logger.info("Connected. Existing tables before init: %s", ", ".join(info["tables"]) or "(none)")
        tables = ensure_schema(force=True)
        logger.info("Done. Tables: %s", ", ".join(tables))
    except Exception as exc:
        logger.error("MySQL init failed: %s", exc)
        raise SystemExit(1) from exc


def _ensure_mysql_schema_or_exit() -> None:
    from intraday_engine.storage.backend import get_storage_backend, write_to_db
    from intraday_engine.storage.db import ensure_schema, mysql_host_label

    if not write_to_db():
        return
    logger = logging.getLogger(__name__)
    try:
        tables = ensure_schema()
        logger.info(
            "Storage backend: %s → MySQL %s (%s)",
            get_storage_backend().value,
            mysql_host_label(),
            ", ".join(tables),
        )
    except ConnectionError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.error("MySQL schema init failed: %s", exc)
        raise SystemExit(1) from exc


def _run_session_scheduler(*, underlying: str | None) -> None:
    from intraday_engine.jobs.session_pipeline import run_session_scheduler_loop

    settings = Settings.from_env(underlying=underlying or "NIFTY")
    setup_logging(settings.log_level, settings.data_dir)
    _ensure_mysql_schema_or_exit()
    run_session_scheduler_loop(settings.data_dir)


def _run_session_once(*, underlying: str | None, trade_date: date | None) -> None:
    from intraday_engine.jobs.session_pipeline import run_session_cycle

    settings = Settings.from_env(underlying=underlying or "NIFTY")
    setup_logging(settings.log_level, settings.data_dir)
    _ensure_mysql_schema_or_exit()
    summary = run_session_cycle(settings.data_dir, trade_date=trade_date)
    logging.getLogger(__name__).info("Session pipeline once: %s", summary)


def _run_dashboard(
    *,
    host: str,
    port: int,
    reload: bool,
    no_option_chain_job: bool,
    read_only: bool,
    no_auto_trade_loop: bool,
) -> None:
    """Launch uvicorn dashboard; optionally skip background pipeline jobs."""
    if read_only:
        os.environ["INTRADAY_READ_ONLY_DASHBOARD"] = "1"
        os.environ["INTRADAY_NO_OPTION_CHAIN_JOB"] = "1"
    elif no_option_chain_job:
        os.environ["INTRADAY_NO_OPTION_CHAIN_JOB"] = "1"
    if read_only or no_auto_trade_loop:
        os.environ["INTRADAY_NO_AUTO_TRADE_LOOP"] = "1"

    settings = Settings.from_env()
    setup_logging(settings.log_level, settings.data_dir)
    logger = logging.getLogger(__name__)
    if read_only:
        logger.info("Dashboard starting in read-only mode (stored data only).")
    else:
        parts = []
        if no_option_chain_job or read_only:
            parts.append("option-chain job off")
        if no_auto_trade_loop or read_only:
            parts.append("auto-trade loop off")
        if parts:
            logger.info("Dashboard starting (%s).", ", ".join(parts))
        else:
            logger.info("Dashboard starting with all background jobs enabled.")

    import uvicorn

    uvicorn.run(
        "intraday_engine.dashboard:app",
        host=host,
        port=port,
        reload=reload,
    )


def _run_huge_move_or_capture(
    capture_only: bool,
    trade_date: date,
    underlying: str | None,
    num_strikes: int | None = None,
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
