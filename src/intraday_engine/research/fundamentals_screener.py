"""
Fundamentals scanner — scrapes screener.in for every NIFTY 500 stock and
emits a per-stock heuristic 0-100 fundamental score plus the raw ratios.

Why screener.in:
  - Best free India-equity coverage (consolidated + standalone, 10y history).
  - Pages are server-rendered HTML, so a plain HTTP GET works.

Per-symbol page layout we depend on:
  - https://www.screener.in/company/<SYM>/consolidated/  (preferred)
  - falls back to /company/<SYM>/  when consolidated is unavailable.

We extract:
  - Top ratio block (Market Cap, Current Price, Stock P/E, Book Value,
    Dividend Yield, ROCE, ROE, Face Value, High / Low, Debt-to-Equity, ...).
  - "Compounded Sales Growth" and "Compounded Profit Growth" CAGR cards
    (3yr / 5yr / 10yr / TTM).
  - Shareholding Pattern section (latest Promoters %, Pledged %).

Scoring (additive, capped at 100):
  ROE        > 20 → +20  | > 15 → +15  | > 10 → +10
  ROCE       > 20 → +20  | > 15 → +15  | > 10 → +10
  D/E        < 0.3 → +15 | < 0.5 → +10 | < 1.0 → +5  | > 2.0 → -10
  Sales3yCAGR > 20 → +15 | > 10 → +10  | > 5 → +5   | < 0 → -5
  Profit3yCAGR > 25 → +20 | > 15 → +15  | > 5 → +10  | < 0 → -10
  Promoter %  > 60 → +10 | > 50 → +5
  Pledged %   < 1  → +5  | > 30 → -15  | > 10 → -5
  PE          < 15 → +10 | < 25 → +5   | > 60 → -5

Caching: per-symbol JSON at data/analysis/fundamentals/cache/<SYM>.json.
A cache hit younger than ``cache_max_age_hours`` (default 24h) is reused —
this keeps a 500-stock refresh cheap (only stale rows hit the network).

Output CSV: data/analysis/fundamentals/fundamentals_<YYYY-MM-DD>.csv.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import (
    fundamentals_cache_path,
    fundamentals_csv_path,
)

logger = logging.getLogger(__name__)

SCREENER_BASE = "https://www.screener.in/company"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = 25
DEFAULT_CACHE_MAX_AGE_HOURS = 24
DEFAULT_THROTTLE_SEC = 0.55  # be polite — ~2 req/sec ceiling

_rate_lock = threading.Lock()
_rate_last = 0.0


def _throttle(min_interval: float = DEFAULT_THROTTLE_SEC) -> None:
    global _rate_last
    with _rate_lock:
        now = time.monotonic()
        wait = min_interval - (now - _rate_last)
        if wait > 0:
            time.sleep(wait)
        _rate_last = time.monotonic()


@dataclass
class FundamentalRow:
    stock: str
    source_url: str = ""
    market_cap_cr: float | None = None
    current_price: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    pe: float | None = None
    book_value: float | None = None
    pb: float | None = None
    dividend_yield: float | None = None
    roce: float | None = None
    roe: float | None = None
    face_value: float | None = None
    debt_to_equity: float | None = None
    sales_growth_3y: float | None = None
    sales_growth_5y: float | None = None
    sales_growth_ttm: float | None = None
    profit_growth_3y: float | None = None
    profit_growth_5y: float | None = None
    profit_growth_ttm: float | None = None
    promoter_holding: float | None = None
    pledged_pct: float | None = None
    fundamental_score: float = 0.0
    score_reasons: list[str] = field(default_factory=list)
    fetched_at: str = ""
    error: str = ""


def _http_get(url: str) -> str | None:
    """Return decoded HTML or None on failure. Throttled and retried once."""
    ctx = ssl.create_default_context()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-IN,en;q=0.8",
    }
    for attempt in range(2):
        _throttle()
        try:
            req = Request(url, headers=headers)
            with urlopen(req, context=ctx, timeout=HTTP_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            if e.code in (429, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            logger.debug("HTTP %s for %s", e.code, url)
            return None
        except (URLError, TimeoutError, OSError) as e:
            logger.debug("net err %s: %s", url, e)
            time.sleep(0.6 * (attempt + 1))
    return None


def _parse_number(text: str) -> float | None:
    """
    Parse screener.in cell text into a float.

    Handles: '1,234.56', '-12.3%', '₹ 1,234 Cr.', '12.3 %', 'NaN', '-'.
    For Cr. / Lakh suffixes we return the rupee value in *crores*.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s or s in {"-", "—", "NaN", "N.A.", "n/a"}:
        return None
    s = s.replace("\u20b9", "").replace("Rs.", "").replace(",", "").strip()
    s = s.replace("%", "").strip()
    mult = 1.0
    if s.lower().endswith("cr.") or s.lower().endswith("cr"):
        s = re.sub(r"(?i)cr\.?$", "", s).strip()
    elif s.lower().endswith("lakh") or s.lower().endswith("lakhs"):
        s = re.sub(r"(?i)lakhs?$", "", s).strip()
        mult = 0.01  # 1 lakh = 0.01 crore
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0)) * mult
    except ValueError:
        return None


def _last_numeric_cell(tr: Any) -> float | None:
    """Return the right-most parseable number from a <tr> (most recent FY column)."""
    cells = tr.find_all(["td", "th"])
    for cell in reversed(cells[1:]):  # skip the label column
        n = _parse_number(cell.get_text(" ", strip=True))
        if n is not None:
            return n
    return None


def _compute_debt_to_equity(soup: Any) -> float | None:
    """
    Fallback D/E parsing. Screener's free company view doesn't expose D/E as
    a top ratio; we derive it from the Balance Sheet section instead.
    """
    section = soup.select_one("#balance-sheet") or soup.select_one("#consolidated-balance-sheet")
    if section is None:
        return None
    borrowings = equity_cap = reserves = None
    for tr in section.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        label = tds[0].get_text(" ", strip=True).lower()
        if borrowings is None and "borrowing" in label:
            borrowings = _last_numeric_cell(tr)
        elif equity_cap is None and ("equity capital" in label or label.strip() == "equity"):
            equity_cap = _last_numeric_cell(tr)
        elif reserves is None and label.startswith("reserves"):
            reserves = _last_numeric_cell(tr)
    if borrowings is None:
        return None
    eq = (equity_cap or 0.0) + (reserves or 0.0)
    if eq <= 0:
        return None
    return round(borrowings / eq, 2)


def _parse_screener_page(html: str, symbol: str) -> dict[str, Any]:
    """Pull the ratios we care about out of a screener.in company page."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise RuntimeError(
            "beautifulsoup4 is required. Run `pip install -r requirements.txt`."
        ) from e

    soup = BeautifulSoup(html, "lxml")
    out: dict[str, Any] = {}

    # ---------- Top ratios (Market Cap, P/E, ROE, ROCE, ...)
    top = soup.select_one("#top-ratios") or soup.select_one("ul#top-ratios")
    if top:
        for li in top.find_all("li"):
            name_el = li.select_one(".name")
            val_el = li.select_one(".value") or li.select_one(".number")
            if not name_el or not val_el:
                continue
            key = name_el.get_text(strip=True).lower()
            val = val_el.get_text(" ", strip=True)
            out.setdefault("_top", {})[key] = val

    top_map = out.get("_top", {})

    def _top(*keys: str) -> float | None:
        for k in keys:
            for full, val in top_map.items():
                if k in full:
                    n = _parse_number(val)
                    if n is not None:
                        return n
        return None

    out["market_cap_cr"] = _top("market cap")
    out["current_price"] = _top("current price")
    out["pe"] = _top("stock p/e", "p/e")
    out["book_value"] = _top("book value")
    out["dividend_yield"] = _top("dividend yield")
    out["roce"] = _top("roce")
    out["roe"] = _top("roe")
    out["face_value"] = _top("face value")
    # 52w high/low are in a "High / Low" merged cell on screener
    hi_lo_text = next((v for k, v in top_map.items() if "high" in k and "low" in k), None)
    if hi_lo_text:
        nums = re.findall(r"-?\d+(?:\.\d+)?", hi_lo_text.replace(",", ""))
        if len(nums) >= 2:
            try:
                out["high_52w"] = float(nums[0])
                out["low_52w"] = float(nums[1])
            except ValueError:
                pass
    # D/E is rarely in the top-ratios block on screener.in's free view.
    # Compute it from the Balance Sheet section instead, using the most recent
    # column: D/E ≈ Borrowings / (Equity Capital + Reserves).
    out["debt_to_equity"] = _top("debt to equity")
    if out.get("debt_to_equity") is None:
        out["debt_to_equity"] = _compute_debt_to_equity(soup)

    if out.get("current_price") and out.get("book_value"):
        try:
            out["pb"] = round(out["current_price"] / out["book_value"], 2)
        except (TypeError, ZeroDivisionError):
            pass

    # ---------- Growth tables (Compounded Sales / Profit Growth)
    #
    # Screener renders each CAGR card as a <table class="ranges-table">. The
    # heading text (e.g. "Compounded Sales Growth") sits inside the *first*
    # <th> of the table itself — not in an h2/h3 sibling. Body rows look like:
    #     10 Years: | 18%
    #     5 Years:  | 12%
    #     3 Years:  | 22%
    #     TTM:      | 40%
    def _growth_section(heading_text: str) -> dict[str, float]:
        needle = heading_text.lower()
        target = None
        for tbl in soup.find_all("table", class_="ranges-table"):
            th = tbl.find("th")
            if th and needle in th.get_text(" ", strip=True).lower():
                target = tbl
                break
        if target is None:
            return {}
        rows: dict[str, float] = {}
        for tr in target.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue
            label = tds[0].get_text(" ", strip=True).lower()
            value = _parse_number(tds[-1].get_text(" ", strip=True))
            if value is None:
                continue
            if "10" in label:
                rows["10y"] = value
            elif "5" in label:
                rows["5y"] = value
            elif "3" in label:
                rows["3y"] = value
            elif "ttm" in label:
                rows["ttm"] = value
        return rows

    sales_g = _growth_section("Compounded Sales Growth")
    profit_g = _growth_section("Compounded Profit Growth")
    out["sales_growth_3y"] = sales_g.get("3y")
    out["sales_growth_5y"] = sales_g.get("5y")
    out["sales_growth_ttm"] = sales_g.get("ttm")
    out["profit_growth_3y"] = profit_g.get("3y")
    out["profit_growth_5y"] = profit_g.get("5y")
    out["profit_growth_ttm"] = profit_g.get("ttm")

    # ---------- Shareholding Pattern (Promoters %, Pledged %)
    shp_section = soup.select_one("#shareholding") or soup.find(
        lambda t: t.name in {"section", "div"}
        and "Shareholding Pattern" in t.get_text(strip=True)
    )
    if shp_section:
        promoters_pct: float | None = None
        pledged_pct: float | None = None
        for tr in shp_section.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue
            label = tds[0].get_text(" ", strip=True).lower()
            last_val = _parse_number(tds[-1].get_text(" ", strip=True))
            if last_val is None:
                continue
            if "promoter" in label and "pledged" not in label and promoters_pct is None:
                promoters_pct = last_val
            elif "pledged" in label and pledged_pct is None:
                pledged_pct = last_val
        out["promoter_holding"] = promoters_pct
        out["pledged_pct"] = pledged_pct

    # symbol echo for debug only
    out["_symbol"] = symbol
    return out


def _score_fundamentals(d: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    roe = d.get("roe")
    if roe is not None:
        if roe > 20:
            score += 20; reasons.append(f"ROE {roe:.1f}% (>20)")
        elif roe > 15:
            score += 15; reasons.append(f"ROE {roe:.1f}% (>15)")
        elif roe > 10:
            score += 10; reasons.append(f"ROE {roe:.1f}% (>10)")
        elif roe < 0:
            score -= 10; reasons.append(f"ROE {roe:.1f}% (negative)")

    roce = d.get("roce")
    if roce is not None:
        if roce > 20:
            score += 20; reasons.append(f"ROCE {roce:.1f}% (>20)")
        elif roce > 15:
            score += 15; reasons.append(f"ROCE {roce:.1f}% (>15)")
        elif roce > 10:
            score += 10; reasons.append(f"ROCE {roce:.1f}% (>10)")
        elif roce < 0:
            score -= 10; reasons.append(f"ROCE {roce:.1f}% (negative)")

    de = d.get("debt_to_equity")
    if de is not None:
        if de < 0.3:
            score += 15; reasons.append(f"D/E {de:.2f} (<0.3)")
        elif de < 0.5:
            score += 10; reasons.append(f"D/E {de:.2f} (<0.5)")
        elif de < 1.0:
            score += 5; reasons.append(f"D/E {de:.2f} (<1)")
        elif de > 2.0:
            score -= 10; reasons.append(f"D/E {de:.2f} (>2)")

    s3 = d.get("sales_growth_3y")
    if s3 is not None:
        if s3 > 20:
            score += 15; reasons.append(f"Sales3yCAGR {s3:.0f}% (>20)")
        elif s3 > 10:
            score += 10; reasons.append(f"Sales3yCAGR {s3:.0f}% (>10)")
        elif s3 > 5:
            score += 5; reasons.append(f"Sales3yCAGR {s3:.0f}% (>5)")
        elif s3 < 0:
            score -= 5; reasons.append(f"Sales3yCAGR {s3:.0f}% (negative)")

    p3 = d.get("profit_growth_3y")
    if p3 is not None:
        if p3 > 25:
            score += 20; reasons.append(f"Profit3yCAGR {p3:.0f}% (>25)")
        elif p3 > 15:
            score += 15; reasons.append(f"Profit3yCAGR {p3:.0f}% (>15)")
        elif p3 > 5:
            score += 10; reasons.append(f"Profit3yCAGR {p3:.0f}% (>5)")
        elif p3 < 0:
            score -= 10; reasons.append(f"Profit3yCAGR {p3:.0f}% (negative)")

    promo = d.get("promoter_holding")
    if promo is not None:
        if promo > 60:
            score += 10; reasons.append(f"Promoter {promo:.0f}% (>60)")
        elif promo > 50:
            score += 5; reasons.append(f"Promoter {promo:.0f}% (>50)")

    pledged = d.get("pledged_pct")
    if pledged is not None:
        if pledged < 1:
            score += 5; reasons.append(f"Pledged {pledged:.1f}% (<1)")
        elif pledged > 30:
            score -= 15; reasons.append(f"Pledged {pledged:.1f}% (>30)")
        elif pledged > 10:
            score -= 5; reasons.append(f"Pledged {pledged:.1f}% (>10)")

    pe = d.get("pe")
    if pe is not None and pe > 0:
        if pe < 15:
            score += 10; reasons.append(f"PE {pe:.1f} (<15)")
        elif pe < 25:
            score += 5; reasons.append(f"PE {pe:.1f} (<25)")
        elif pe > 60:
            score -= 5; reasons.append(f"PE {pe:.1f} (>60)")

    return round(max(-30.0, min(100.0, score)), 1), reasons


def _load_cache(path: Path, max_age_hours: int) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(data.get("fetched_at", ""))
        if datetime.now() - ts > timedelta(hours=max_age_hours):
            return None
        return data
    except Exception:
        return None


def _save_cache(path: Path, row: FundamentalRow) -> None:
    try:
        path.write_text(json.dumps(asdict(row), indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug("cache write failed for %s: %s", row.stock, e)


def fetch_one(
    symbol: str,
    data_dir: Path,
    cache_max_age_hours: int = DEFAULT_CACHE_MAX_AGE_HOURS,
    force_refresh: bool = False,
) -> FundamentalRow:
    """Return fundamentals + score for one symbol (cached when fresh)."""
    cache_path = fundamentals_cache_path(data_dir, symbol)
    if not force_refresh:
        cached = _load_cache(cache_path, cache_max_age_hours)
        if cached:
            try:
                cached.pop("_top", None)
                # FundamentalRow doesn't accept unknown keys.
                allowed = set(FundamentalRow.__dataclass_fields__.keys())
                cached = {k: v for k, v in cached.items() if k in allowed}
                return FundamentalRow(**cached)
            except Exception:
                pass

    row = FundamentalRow(stock=symbol, fetched_at=datetime.now().isoformat(timespec="seconds"))
    for path in (f"{SCREENER_BASE}/{symbol}/consolidated/", f"{SCREENER_BASE}/{symbol}/"):
        html = _http_get(path)
        if not html or "We couldn't find" in html or len(html) < 500:
            continue
        try:
            parsed = _parse_screener_page(html, symbol)
        except Exception as e:
            row.error = f"parse:{e}"[:160]
            continue
        if not parsed.get("current_price") and not parsed.get("market_cap_cr"):
            continue
        row.source_url = path
        row.market_cap_cr = parsed.get("market_cap_cr")
        row.current_price = parsed.get("current_price")
        row.high_52w = parsed.get("high_52w")
        row.low_52w = parsed.get("low_52w")
        row.pe = parsed.get("pe")
        row.book_value = parsed.get("book_value")
        row.pb = parsed.get("pb")
        row.dividend_yield = parsed.get("dividend_yield")
        row.roce = parsed.get("roce")
        row.roe = parsed.get("roe")
        row.face_value = parsed.get("face_value")
        row.debt_to_equity = parsed.get("debt_to_equity")
        row.sales_growth_3y = parsed.get("sales_growth_3y")
        row.sales_growth_5y = parsed.get("sales_growth_5y")
        row.sales_growth_ttm = parsed.get("sales_growth_ttm")
        row.profit_growth_3y = parsed.get("profit_growth_3y")
        row.profit_growth_5y = parsed.get("profit_growth_5y")
        row.profit_growth_ttm = parsed.get("profit_growth_ttm")
        row.promoter_holding = parsed.get("promoter_holding")
        row.pledged_pct = parsed.get("pledged_pct")
        score, reasons = _score_fundamentals(parsed)
        row.fundamental_score = score
        row.score_reasons = reasons
        _save_cache(cache_path, row)
        return row

    row.error = row.error or "not_found"
    return row


def run_fundamentals_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    trade_date: date | None = None,
    cache_max_age_hours: int = DEFAULT_CACHE_MAX_AGE_HOURS,
    max_workers: int = 4,
    stock_limit: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Run the fundamentals scan over NIFTY 500 and emit CSV."""
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()

    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)
    if stock_limit:
        symbols = symbols[: int(stock_limit)]

    rows: list[FundamentalRow] = []
    failed: list[str] = []

    def job(sym: str) -> FundamentalRow:
        try:
            return fetch_one(sym, settings.data_dir, cache_max_age_hours, force_refresh)
        except Exception as e:
            return FundamentalRow(stock=sym, error=str(e)[:200])

    # Low concurrency — screener.in is a small service, don't hammer it.
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 6))) as ex:
        futs = {ex.submit(job, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futs), start=1):
            row = fut.result()
            rows.append(row)
            if row.error:
                failed.append(row.stock)
            if i % 50 == 0:
                logger.info("fundamentals: %d/%d done (%d failed)", i, len(symbols), len(failed))

    rows.sort(key=lambda r: (-(r.fundamental_score or 0.0), r.stock))

    out_path = fundamentals_csv_path(settings.data_dir, td)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat: list[dict[str, Any]] = []
    for r in rows:
        d = asdict(r)
        d["score_reasons"] = " | ".join(d.get("score_reasons") or [])
        flat.append(d)
    pd.DataFrame(flat).to_csv(out_path, index=False)

    logger.info(
        "Fundamentals scan: %d symbols, %d failed. Saved %s",
        len(rows), len(failed), out_path,
    )
    return {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scanned": len(symbols),
        "ok": len(rows) - len(failed),
        "failed": len(failed),
        "output_csv": str(out_path),
    }


def load_latest_fundamentals_csv(data_dir: Path) -> pd.DataFrame:
    """Best-effort: return the most recent fundamentals CSV as a DataFrame."""
    d = data_dir / "analysis" / "fundamentals"
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.glob("fundamentals_*.csv"))
    if not files:
        return pd.DataFrame()
    try:
        return pd.read_csv(files[-1])
    except Exception:
        return pd.DataFrame()
