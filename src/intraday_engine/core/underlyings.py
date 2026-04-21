"""Predefined underlying configs for NIFTY, NIFTY BANK, and F&O stocks."""

from __future__ import annotations

import csv
import html
import http.cookiejar
import json
import logging
import re
import urllib.parse
import urllib.request
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


CHARTINK_FNO_SCREENER_URL = "https://chartink.com/screener/fno-liquid-stocks-2"

# Fallback snapshot synced from the Chartink "FnO Liquid Stocks" screener.
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
    return _project_root() / "data" / "reference" / "fno_watchlist.csv"


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
    """Load the FnO watchlist from CSV, falling back to the bundled snapshot."""
    csv_path = path or default_fno_watchlist_csv_path()
    if not csv_path.exists():
        return LIQUID_FNO_STOCKS

    try:
        with csv_path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            symbols = tuple(
                str(row.get("symbol", "")).strip().upper()
                for row in reader
                if str(row.get("symbol", "")).strip()
            )
        return symbols or LIQUID_FNO_STOCKS
    except Exception as exc:
        logger.warning("Using bundled FnO stock list because watchlist CSV could not be read: %s", exc)
        return LIQUID_FNO_STOCKS


def fetch_liquid_fno_stocks_from_chartink() -> tuple[str, ...]:
    """Fetch the current liquid FnO stock universe from the Chartink screener."""

    try:
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        base_headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        }
        page = opener.open(
            urllib.request.Request(CHARTINK_FNO_SCREENER_URL, headers=base_headers),
            timeout=15,
        ).read().decode("utf-8", "ignore")
        csrf = re.search(r'csrf-token" content="([^"]+)"', page)
        scan_json = re.search(r':scan-json="([^"]+)"', page)
        if not csrf or not scan_json:
            raise RuntimeError("Chartink screener page did not expose the expected tokens.")

        payload = html.unescape(scan_json.group(1))
        scan_clause = json.loads(payload)["atlas_query"]
        post_headers = {
            **base_headers,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://chartink.com",
            "Referer": CHARTINK_FNO_SCREENER_URL,
            "X-CSRF-TOKEN": csrf.group(1),
            "X-Requested-With": "XMLHttpRequest",
        }
        response = opener.open(
            urllib.request.Request(
                "https://chartink.com/screener/process",
                data=urllib.parse.urlencode({"scan_clause": scan_clause}).encode(),
                headers=post_headers,
            ),
            timeout=15,
        ).read().decode("utf-8", "ignore")
        data = json.loads(response).get("data", [])
        symbols = tuple(
            str(row.get("nsecode", "")).strip().upper()
            for row in data
            if isinstance(row, dict) and row.get("nsecode")
        )
        if not symbols:
            raise RuntimeError("Chartink screener returned an empty symbol list.")
        return symbols
    except Exception as exc:
        logger.warning("Using fallback Chartink FnO stock list: %s", exc)
        return LIQUID_FNO_STOCKS


def write_liquid_fno_watchlist_csv(path: Path | None = None) -> Path:
    """Refresh the watchlist CSV from Chartink."""
    csv_path = path or default_fno_watchlist_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = fetch_liquid_fno_stocks_from_chartink()
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["symbol", "rank", "source"])
        writer.writeheader()
        for idx, symbol in enumerate(symbols, start=1):
            writer.writerow({"symbol": symbol, "rank": idx, "source": CHARTINK_FNO_SCREENER_URL})
    return csv_path

