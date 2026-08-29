"""Predefined underlying configs for NIFTY, NIFTY BANK, and F&O stocks."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnderlyingConfig:
    """Config for a single index/underlying."""

    name: str  # NFO instrument name (e.g. NIFTY, BANKNIFTY, RELIANCE)
    spot_symbol: str  # e.g. NSE:NIFTY 50, NSE:RELIANCE
    option_strike_step: int
    lot_size: int
    exchange: str = "NFO"  # NFO for NSE
    is_index: bool = True  # False for F&O stocks


# Predefined index underlyings (NFO)
INDEX_UNDERLYINGS: dict[str, UnderlyingConfig] = {
    "NIFTY": UnderlyingConfig(
        name="NIFTY",
        spot_symbol="NSE:NIFTY 50",
        option_strike_step=50,
        lot_size=50,
        is_index=True,
    ),
    "BANKNIFTY": UnderlyingConfig(
        name="BANKNIFTY",
        spot_symbol="NSE:NIFTY BANK",
        option_strike_step=100,
        lot_size=15,
        is_index=True,
    ),
    "FINNIFTY": UnderlyingConfig(
        name="FINNIFTY",
        spot_symbol="NSE:NIFTY FIN SERVICE",
        option_strike_step=50,
        lot_size=25,
        is_index=True,
    ),
    "MIDCPNIFTY": UnderlyingConfig(
        name="MIDCPNIFTY",
        spot_symbol="NSE:NIFTY MID SELECT",
        option_strike_step=25,
        lot_size=75,
        is_index=True,
    ),
    "NIFTYNXT50": UnderlyingConfig(
        name="NIFTYNXT50",
        spot_symbol="NSE:NIFTY NEXT 50",
        option_strike_step=5,
        lot_size=30,
        is_index=True,
    ),
}


FNO_LIQUID_SYMBOLS_CSV = "fno_liquid_symbols_sensibull.csv"

# Fallback snapshot when the Sensibull CSV is missing or unreadable.
LIQUID_FNO_STOCKS: tuple[str, ...] = (
    "HAL",
    "LT",
    "BSE",
    "WAAREEENER",
    "M&M",
    "MCX",
    "TCS",
    "MAZDOCK",
    "LUPIN",
    "HINDUNILVR",
    "ADANIENT",
    "BHARTIARTL",
    "SUNPHARMA",
    "CHOLAFIN",
    "HCLTECH",
    "ADANIPORTS",
    "COCHINSHIP",
    "INFY",
    "RELIANCE",
    "BDL",
    "AXISBANK",
    "VOLTAS",
    "ICICIBANK",
    "CDSL",
    "COFORGE",
    "CIPLA",
    "DRREDDY",
    "JSWSTEEL",
    "PAYTM",
    "ADANIENSOL",
    "TATACONSUM",
    "ADANIGREEN",
    "SBIN",
    "NAUKRI",
    "360ONE",
    "SHRIRAMFIN",
    "HINDALCO",
    "PREMIERENE",
    "MAXHEALTH",
    "BAJFINANCE",
    "INDUSINDBK",
    "HDFCBANK",
    "LODHA",
    "VEDL",
    "CGPOWER",
    "UPL",
    "INDHOTEL",
    "HDFCLIFE",
    "DLF",
    "HINDZINC",
    "JSWENERGY",
    "PGEL",
    "OIL",
    "COALINDIA",
    "KALYANKJIL",
    "BEL",
    "INDUSTOWER",
    "AMBUJACEM",
    "PFC",
    "JUBLFOOD",
    "VBL",
    "TATAPOWER",
    "NTPC",
    "KOTAKBANK",
    "RECLTD",
    "TMPV",
    "EXIDEIND",
    "ITC",
    "POWERGRID",
    "BPCL",
    "ANGELONE",
    "RVNL",
    "SWIGGY",
    "PETRONET",
    "CROMPTON",
    "ADANIPOWER",
    "VMM",
)

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def default_fno_watchlist_csv_path() -> Path:
    """Path to data/reference/fno_liquid_symbols_sensibull.csv (symbol column only)."""
    return _project_root() / "data" / "reference" / FNO_LIQUID_SYMBOLS_CSV


def get_underlying_config(underlying: str) -> UnderlyingConfig:
    """Return config for underlying. For stocks, creates dynamic config."""

    def _stock_config(name: str) -> UnderlyingConfig:
        return UnderlyingConfig(
            name=name,
            spot_symbol=f"NSE:{name}",
            option_strike_step=5,
            lot_size=1,  # Overridden from resolver when available
            is_index=False,
        )

    key = underlying.strip().upper().replace(" ", "")
    if key == "NIFTYBANK":
        key = "BANKNIFTY"
    if key in INDEX_UNDERLYINGS:
        return INDEX_UNDERLYINGS[key]
    return _stock_config(key)


def list_index_underlyings() -> list[str]:
    """Return list of index underlying keys."""
    return list(INDEX_UNDERLYINGS.keys())


def list_underlyings() -> list[str]:
    """Return list of index underlying keys. For stocks, use API /api/fno-stocks."""
    return list_index_underlyings()


def list_liquid_fno_stocks() -> list[str]:
    """Return the allowed FnO stock universe from the watchlist CSV."""
    return list(load_liquid_fno_stocks())


def filter_liquid_fno_stocks(symbols: Iterable[str]) -> list[str]:
    """Keep only allowed FnO stock symbols, preserving screener order."""
    live = {str(symbol).strip().upper() for symbol in symbols}
    return [symbol for symbol in load_liquid_fno_stocks() if symbol in live]


def load_liquid_fno_stocks(path: Path | None = None) -> tuple[str, ...]:
    """Load liquid FnO symbols from fno_liquid_symbols_sensibull.csv (symbol column)."""
    csv_path = path or default_fno_watchlist_csv_path()
    if not csv_path.exists():
        return LIQUID_FNO_STOCKS

    try:
        with csv_path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames or "symbol" not in reader.fieldnames:
                raise ValueError("CSV must have a 'symbol' column header.")
            symbols = tuple(
                str(row["symbol"]).strip().upper()
                for row in reader
                if str(row.get("symbol", "")).strip()
            )
        return symbols or LIQUID_FNO_STOCKS
    except Exception as exc:
        logger.warning("Using bundled FnO stock list because Sensibull CSV could not be read: %s", exc)
        return LIQUID_FNO_STOCKS

