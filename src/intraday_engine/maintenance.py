from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

from intraday_engine.analysis.eod_signal_analysis import analyze_signal_outcomes
from intraday_engine.core.underlyings import (
    default_fno_watchlist_csv_path,
    write_liquid_fno_watchlist_csv,
)
from intraday_engine.storage.legacy_migration import migrate_legacy_data, rewrite_partitioned_storage_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time maintenance utilities.")
    parser.add_argument(
        "--refresh-fno-watchlist",
        action="store_true",
        help="Fetch the latest liquid FnO symbols from Chartink and write data/reference/fno_watchlist.csv.",
    )
    parser.add_argument(
        "--analyze-signals-eod",
        action="store_true",
        help="Analyze the day's generated signals and write daily outcome CSV.",
    )
    parser.add_argument(
        "--migrate-legacy-data",
        action="store_true",
        help="Move legacy symbol-wise files into the new day-partitioned layout and delete obsolete migrated files.",
    )
    parser.add_argument(
        "--rewrite-partitioned-layout",
        action="store_true",
        help="Rewrite current partitioned files into FnO_Signals.csv, Index_Signals.csv, FnO_Analysis.csv, and Index_Analysis.csv.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Trade date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional data directory override. Defaults to DATA_DIR env or project data/.",
    )
    args = parser.parse_args()

    trade_date = _parse_date(args.date) if args.date else date.today()
    data_dir = _resolve_data_dir(args.data_dir)

    if args.refresh_fno_watchlist:
        path = write_liquid_fno_watchlist_csv(data_dir / "reference" / "fno_watchlist.csv")
        print(f"FnO watchlist written: {path}")

    if args.analyze_signals_eod:
        df, path = analyze_signal_outcomes(data_dir, trade_date)
        print(f"EOD analysis rows: {len(df)}")
        print(f"EOD analysis written: {path}")

    if args.migrate_legacy_data:
        report = migrate_legacy_data(data_dir)
        print(f"Migrated signal rows: {report.signal_rows}")
        print(f"Migrated snapshot rows: {report.snapshot_rows}")
        print(f"Migrated option-chain rows: {report.option_chain_rows}")
        print(f"Deleted files: {report.deleted_files}")
        print(f"Deleted dirs: {report.deleted_dirs}")

    if args.rewrite_partitioned_layout:
        report = rewrite_partitioned_storage_layout(data_dir)
        print(f"Rewritten signal rows: {report.signal_rows}")
        print(f"Rewritten analysis rows: {report.analysis_rows}")
        print(f"Deleted files: {report.deleted_files}")
        print(f"Deleted dirs: {report.deleted_dirs}")

    if not args.refresh_fno_watchlist and not args.analyze_signals_eod and not args.migrate_legacy_data and not args.rewrite_partitioned_layout:
        parser.print_help()


def _resolve_data_dir(cli_value: str | None) -> Path:
    data_dir = Path(cli_value or os.getenv("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = (_project_root() / data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Invalid --date format. Use YYYY-MM-DD.") from exc


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


if __name__ == "__main__":
    main()
