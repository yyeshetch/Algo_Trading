from __future__ import annotations

from datetime import date
from pathlib import Path

from intraday_engine.core.underlyings import list_index_underlyings


def normalize_underlying(underlying: str | None) -> str:
    value = (underlying or "NIFTY").strip().upper().replace(" ", "")
    return "BANKNIFTY" if value == "NIFTYBANK" else value


def asset_class_for_underlying(underlying: str | None) -> str:
    return "index" if normalize_underlying(underlying) in set(list_index_underlyings()) else "stock"


def signals_day_path(data_dir: Path, trade_date: date, asset_class: str) -> Path:
    filename = "Index_Signals.csv" if asset_class == "index" else "FnO_Signals.csv"
    return _partition_dir(data_dir / "signals", trade_date) / filename


def analysis_day_path(data_dir: Path, trade_date: date, asset_class: str) -> Path:
    filename = "Index_Analysis.csv" if asset_class == "index" else "FnO_Analysis.csv"
    return _partition_dir(data_dir / "analysis", trade_date) / filename


def option_chain_day_path(data_dir: Path, trade_date: date) -> Path:
    return _partition_dir(data_dir / "option_chain", trade_date) / "option_chain.csv"


def options_trading_signals_path(data_dir: Path, trade_date: date) -> Path:
    return _partition_dir(data_dir / "options_trading", trade_date) / "signals.json"


def signal_outcomes_day_path(data_dir: Path, trade_date: date) -> Path:
    return _partition_dir(data_dir / "analysis", trade_date) / "signal_outcomes.csv"


def watchlist_csv_path(data_dir: Path) -> Path:
    return data_dir / "reference" / "fno_watchlist.csv"


def accumulation_base_dir(data_dir: Path) -> Path:
    """Shared folder for NIFTY500 accumulation scanner (single-file OHLCV cache)."""
    p = data_dir / "analysis" / "accumulation"
    p.mkdir(parents=True, exist_ok=True)
    return p


def accumulation_ohlcv_master_path(data_dir: Path) -> Path:
    """Single incremental JSON store: symbol -> list of 1h OHLCV bars."""
    return accumulation_base_dir(data_dir) / "nifty500_1h_ohlcv_master.json"


def accumulation_partition_dir(data_dir: Path, trade_date: date) -> Path:
    """Date-partitioned outputs: analysis CSV, report text."""
    d = data_dir / "analysis" / f"date={trade_date.isoformat()}" / "accumulation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tomorrow_watchlist_json_path(data_dir: Path, trade_date: date) -> Path:
    """Stored NIFTY500 multi-TF watchlist for the stocks dashboard tab."""
    return data_dir / f"nifty500_tomorrow_watchlist_{trade_date.isoformat()}.json"


def nifty500_partition_dir(data_dir: Path, trade_date: date) -> Path:
    """Shared OHLCV export dir for incremental NIFTY500 cache."""
    _ = trade_date  # kept for backward-compatible call sites
    p = data_dir / "NIFTY500"
    p.mkdir(parents=True, exist_ok=True)
    return p


def nse_public_data_dir(data_dir: Path) -> Path:
    """Cache for NSE-published CSV/XLS artifacts (delivery, bulk deals, FII/DII, participant OI)."""
    p = data_dir / "reference" / "nse"
    p.mkdir(parents=True, exist_ok=True)
    return p


def nse_holidays_path(data_dir: Path) -> Path:
    return nse_public_data_dir(data_dir) / "nse_trading_holidays.json"


def silent_accumulation_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "silent_accumulation"
    p.mkdir(parents=True, exist_ok=True)
    return p


def silent_accumulation_path(data_dir: Path, trade_date: date) -> Path:
    return silent_accumulation_dir(data_dir) / f"silent_accumulation_{trade_date.isoformat()}.json"


def relative_strength_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "relative_strength"
    p.mkdir(parents=True, exist_ok=True)
    return p


def relative_strength_path(data_dir: Path, trade_date: date) -> Path:
    return relative_strength_dir(data_dir) / f"relative_strength_{trade_date.isoformat()}.json"


def fii_dii_trends_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "fii_dii"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fii_dii_trends_path(data_dir: Path, trade_date: date) -> Path:
    return fii_dii_trends_dir(data_dir) / f"fii_dii_trends_{trade_date.isoformat()}.json"


def nifty_index_daily_path(data_dir: Path) -> Path:
    """Cached NIFTY 50 daily OHLCV (used as RS benchmark)."""
    p = data_dir / "reference" / "nifty_index"
    p.mkdir(parents=True, exist_ok=True)
    return p / "NIFTY50_1D.csv"


def nifty_index_15min_path(data_dir: Path) -> Path:
    """Cached NIFTY 50 15-minute OHLCV (used as intraday RS benchmark)."""
    p = data_dir / "reference" / "nifty_index"
    p.mkdir(parents=True, exist_ok=True)
    return p / "NIFTY50_15Min.csv"


def intraday_relative_strength_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "intraday_relative_strength"
    p.mkdir(parents=True, exist_ok=True)
    return p


def intraday_relative_strength_path(data_dir: Path, trade_date: date) -> Path:
    return intraday_relative_strength_dir(data_dir) / f"intraday_rs_{trade_date.isoformat()}.json"


def sector_relative_strength_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "sector_relative_strength"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sector_relative_strength_path(data_dir: Path, trade_date: date) -> Path:
    return sector_relative_strength_dir(data_dir) / f"sector_rs_{trade_date.isoformat()}.json"


def institutional_volume_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "institutional_volume"
    p.mkdir(parents=True, exist_ok=True)
    return p


def institutional_volume_path(data_dir: Path, trade_date: date) -> Path:
    return institutional_volume_dir(data_dir) / f"institutional_volume_{trade_date.isoformat()}.json"


def fundamentals_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "fundamentals"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fundamentals_csv_path(data_dir: Path, trade_date: date) -> Path:
    return fundamentals_dir(data_dir) / f"fundamentals_{trade_date.isoformat()}.csv"


def fundamentals_cache_path(data_dir: Path, symbol: str) -> Path:
    """Per-symbol JSON cache so we don't re-scrape screener.in every run."""
    safe = "".join(c for c in symbol.upper() if c.isalnum() or c in ("-", "_", "&"))
    p = data_dir / "analysis" / "fundamentals" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.json"


def news_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "news"
    p.mkdir(parents=True, exist_ok=True)
    return p


def news_csv_path(data_dir: Path, trade_date: date) -> Path:
    return news_dir(data_dir) / f"news_{trade_date.isoformat()}.csv"


def market_overview_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "market_overview"
    p.mkdir(parents=True, exist_ok=True)
    return p


def market_overview_path(data_dir: Path, trade_date: date) -> Path:
    return market_overview_dir(data_dir) / f"market_overview_{trade_date.isoformat()}.json"


def combined_signals_dir(data_dir: Path) -> Path:
    p = data_dir / "analysis" / "combined"
    p.mkdir(parents=True, exist_ok=True)
    return p


def combined_signals_csv_path(data_dir: Path, trade_date: date) -> Path:
    return combined_signals_dir(data_dir) / f"combined_{trade_date.isoformat()}.csv"


def _partition_dir(root: Path, trade_date: date) -> Path:
    directory = root / f"date={trade_date.isoformat()}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
