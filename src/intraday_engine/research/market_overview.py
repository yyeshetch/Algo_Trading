"""
Market Overview scanner.

Aggregates pre-market and global context into a single JSON for the dashboard:

  - GIFT Nifty            (pre-market NIFTY proxy via moneycontrol HTML scrape)
  - India VIX             (Zerodha Kite quote: NSE:INDIA VIX)
  - Global indices        (Yahoo Finance public chart API, no key required)
  - Commodities + macros  (Crude WTI/Brent, Gold, Silver, Dollar Index, US10Y)
  - FII / DII             (delegates to fii_dii_trends_scan; last 5/10/30d nets)
  - Global news headlines (Google News RSS for global markets / crude / commodities)
  - NIFTY plan for today  (latest BUY/SELL signal + key levels from stored snapshot)

Every section is wrapped in defensive try/except, so a single failing source
never breaks the whole scan. Output JSON shape is stable and consumed by
``/api/market-overview`` and the "Market Overview" tab on the index dashboard.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.research.fii_dii_trends import (
    load_stored_fii_dii_trends,
    run_fii_dii_trends_scan,
)
from intraday_engine.research.stock_news_scanner import (
    _BEAR_PATS,
    _BULL_PATS,
    _count_hits,
)
from intraday_engine.storage import DataStore
from intraday_engine.storage.layout import market_overview_path

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Generic HTTP helpers (urllib, no extra deps)
# ---------------------------------------------------------------------------


def _http_get(url: str, *, timeout: int = HTTP_TIMEOUT, accept: str = "*/*") -> bytes | None:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        ctx = ssl.create_default_context()
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except Exception as e:
        logger.debug("HTTP GET failed for %s: %s", url, e)
        return None


def _http_get_json(url: str, *, timeout: int = HTTP_TIMEOUT) -> Any | None:
    raw = _http_get(url, timeout=timeout, accept="application/json")
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("JSON decode failed for %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# GIFT Nifty (moneycontrol HTML scrape – the page is public, no auth)
# ---------------------------------------------------------------------------

_GIFT_NIFTY_URL = "https://www.moneycontrol.com/indian-indices/gift-nifty-50-126.html"
_GIFT_NIFTY_FALLBACK_URLS = [
    "https://www.moneycontrol.com/indian-indices/gift-nifty-50-126.html",
]


def _scrape_gift_nifty() -> dict[str, Any]:
    out: dict[str, Any] = {"source": "moneycontrol"}
    raw = _http_get(_GIFT_NIFTY_URL, accept="text/html")
    if raw is None:
        out["error"] = "fetch_failed"
        return out
    html = raw.decode("utf-8", errors="replace")

    def _first_number(*patterns: str) -> float | None:
        for pat in patterns:
            m = re.search(pat, html, re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            txt = m.group(1).replace(",", "").strip()
            try:
                return float(txt)
            except ValueError:
                continue
        return None

    last = _first_number(
        r'id="sp_val"[^>]*>\s*([\d,\.]+)',
        r'class="inprice1[^"]*"[^>]*>\s*([\d,\.]+)',
        r'"lastprice"\s*:\s*"?([\d,\.]+)',
    )
    change = _first_number(
        r'id="sp_ch"[^>]*>\s*([+-]?[\d,\.]+)',
        r'class="pricupdn[^"]*"[^>]*>\s*([+-]?[\d,\.]+)',
    )
    change_pct = _first_number(
        r'id="sp_perch"[^>]*>\s*\(?\s*([+-]?[\d,\.]+)\s*%?\s*\)?',
        r'class="pricupdn[^"]*"[^>]*>\s*[+-]?[\d,\.]+\s*<[^>]+>\s*\(?\s*([+-]?[\d,\.]+)\s*%',
    )
    out["last"] = last
    out["change"] = change
    out["change_pct"] = change_pct
    if last is None:
        out["error"] = "parse_failed"
    return out


# ---------------------------------------------------------------------------
# India VIX (Kite)
# ---------------------------------------------------------------------------


def _fetch_india_vix(client: ZerodhaClient) -> dict[str, Any]:
    out: dict[str, Any] = {"symbol": "NSE:INDIA VIX"}
    try:
        q = client.quote(["NSE:INDIA VIX"])
        node = (q or {}).get("NSE:INDIA VIX") or {}
        last = float(node.get("last_price") or 0.0)
        ohlc = node.get("ohlc") or {}
        prev = float(ohlc.get("close") or 0.0)
        change = last - prev if prev else None
        change_pct = (change / prev * 100.0) if (prev and change is not None) else None
        out.update(
            {
                "last": round(last, 4) if last else None,
                "prev_close": round(prev, 4) if prev else None,
                "change": round(change, 4) if change is not None else None,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
                "open": ohlc.get("open"),
                "high": ohlc.get("high"),
                "low": ohlc.get("low"),
            }
        )
    except Exception as e:
        logger.debug("INDIA VIX quote failed: %s", e)
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Global indices + commodities (Investing.com HTML scrape via curl_cffi)
# ---------------------------------------------------------------------------

INVESTING_INDICES_URL = "https://in.investing.com/indices/major-indices"
INVESTING_COMMODITIES_URL = "https://in.investing.com/commodities"


_CURL_IMPERSONATIONS = ("chrome124", "chrome120", "safari17_0")


def _curl_cffi_get_text(url: str, *, timeout: int = 25, retries: int = 2) -> str | None:
    """Fetch a Cloudflare-protected page using browser TLS fingerprint.

    Returns the HTML body on HTTP 200, or None on any failure (network, status,
    or missing dependency). We import curl_cffi lazily so the rest of the
    scanner still works even if the package isn't installed yet, and retry with
    different browser fingerprints on transient 403/503 challenges.
    """
    try:
        from curl_cffi import requests as cffi_requests  # type: ignore
    except Exception as e:
        logger.warning(
            "curl_cffi not available (%s). Install with: pip install curl_cffi", e
        )
        return None
    last_status: int | str = "no_response"
    attempts = max(1, retries + 1)
    for i in range(attempts):
        impersonate = _CURL_IMPERSONATIONS[i % len(_CURL_IMPERSONATIONS)]
        try:
            r = cffi_requests.get(url, impersonate=impersonate, timeout=timeout)
        except Exception as e:
            last_status = f"exc:{type(e).__name__}"
            logger.debug("curl_cffi GET failed for %s (try %d, %s): %s", url, i + 1, impersonate, e)
            continue
        if r.status_code == 200 and r.text:
            return r.text
        last_status = r.status_code
        logger.debug(
            "curl_cffi GET %s -> HTTP %s (try %d, impersonate=%s)",
            url, r.status_code, i + 1, impersonate,
        )
    logger.info("Investing.com fetch failed after %d tries: %s -> %s", attempts, url, last_status)
    return None


# Daily move grading (change %). Neutral band is explicit; moderate/extreme use ±1.5%.
_GRADE_NEUTRAL_LO = -0.5
_GRADE_NEUTRAL_HI = 0.5
_GRADE_MODERATE = 1.5


def grade_market_change(change_pct: float | None) -> dict[str, Any]:
    """Classify a daily % move into five market-sentiment buckets."""
    if change_pct is None:
        return {"grade": "UNKNOWN", "grade_label": "—", "grade_key": "unknown"}
    try:
        pct = float(change_pct)
    except (TypeError, ValueError):
        return {"grade": "UNKNOWN", "grade_label": "—", "grade_key": "unknown"}

    if pct < -_GRADE_MODERATE:
        return {
            "grade": "EXTREMELY_BEARISH",
            "grade_label": "Extremely Bearish",
            "grade_key": "extremely_bearish",
        }
    if pct < _GRADE_NEUTRAL_LO:
        return {"grade": "BEARISH", "grade_label": "Bearish", "grade_key": "bearish"}
    if pct <= _GRADE_NEUTRAL_HI:
        return {"grade": "NEUTRAL", "grade_label": "Neutral", "grade_key": "neutral"}
    if pct <= _GRADE_MODERATE:
        return {"grade": "BULLISH", "grade_label": "Bullish", "grade_key": "bullish"}
    return {
        "grade": "EXTREMELY_BULLISH",
        "grade_label": "Extremely Bullish",
        "grade_key": "extremely_bullish",
    }


def _summarize_grades(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "extremely_bearish",
        "bearish",
        "neutral",
        "bullish",
        "extremely_bullish",
        "unknown",
    )
    counts = {k: 0 for k in keys}
    for r in rows:
        key = str(r.get("grade_key") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _apply_row_grades(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in rows:
        r.update(grade_market_change(r.get("change_pct")))
    return rows


def _parse_number(text: str) -> float | None:
    if text is None:
        return None
    s = str(text).strip().replace(",", "").replace("\xa0", "").replace("%", "")
    if not s or s in {"-", "--", "N/A"}:
        return None
    s = s.lstrip("+")
    try:
        return float(s)
    except ValueError:
        return None


def _scrape_investing_tables(html: str) -> list[dict[str, Any]]:
    """Parse every price-style ``<table>`` on an Investing.com listing page.

    Auto-detects column layout from the ``<thead>`` of each table so this works
    for both the major-indices grid and the multiple per-category tables on the
    commodities page. Only tables that include a ``Last`` column are kept.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 missing; cannot parse investing.com HTML.")
        return []

    soup = BeautifulSoup(html, "lxml")
    rows_out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.select("thead th")]
        if "last" not in headers:
            continue
        # Build a header -> column-index map (some pages prepend a checkbox col).
        col_map: dict[str, int] = {}
        for i, h in enumerate(headers):
            if h:
                col_map[h] = i

        for tr in table.select("tr.dynamic-table-v2_row__ILVMx, tbody tr"):
            tds = tr.find_all("td", recursive=False)
            if not tds or len(tds) < len(headers) - 1:
                continue

            # Name (and URL) – prefer the anchor inside the "name" column.
            name = ""
            url = ""
            link = tr.find("a", href=True)
            if link:
                name = link.get_text(" ", strip=True)
                url = link.get("href", "")
            if not name:
                ckbox = tr.find("input", {"type": "checkbox"})
                if ckbox and ckbox.get("value"):
                    name = ckbox.get("value", "").strip()
            if not name:
                continue

            def cell_text(col: str) -> str:
                idx = col_map.get(col)
                if idx is None or idx >= len(tds):
                    return ""
                return tds[idx].get_text(" ", strip=True)

            row: dict[str, Any] = {
                "name": name,
                "symbol": (url.rstrip("/").rsplit("/", 1)[-1] if url else name),
                "url": url or None,
                "last": _parse_number(cell_text("last")),
                "prev_close": _parse_number(cell_text("prev.")) or _parse_number(cell_text("prev")),
                "high": _parse_number(cell_text("high")),
                "low": _parse_number(cell_text("low")),
                "change": _parse_number(cell_text("chg.")) or _parse_number(cell_text("change")),
                "change_pct": _parse_number(cell_text("chg. %")) or _parse_number(cell_text("change %")),
                "month": cell_text("month") or None,
            }

            # Time: prefer the <time datetime="..."> if present, else cell text.
            time_el = tr.find("time")
            if time_el:
                row["as_of"] = time_el.get("datetime") or time_el.get_text(strip=True)
            else:
                row["as_of"] = cell_text("time") or None

            # Detect direction from cell class names (handles weird signed text).
            chg_idx = col_map.get("chg.") or col_map.get("change")
            if chg_idx is not None and chg_idx < len(tds):
                klass = " ".join(tds[chg_idx].get("class", []))
                if "--down" in klass and row["change"] is not None and row["change"] > 0:
                    row["change"] = -row["change"]

            # Dedup by name (some sub-tables repeat headline items).
            key = row["name"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            rows_out.append(row)
    return rows_out


def _fetch_investing(url: str) -> list[dict[str, Any]]:
    html = _curl_cffi_get_text(url)
    if not html:
        return []
    rows = _scrape_investing_tables(html)
    for r in rows:
        r["source"] = "investing.com"
    return _apply_row_grades(rows)


def _build_investing_block(
    url: str, previous: dict[str, Any]
) -> dict[str, Any]:
    """Fetch latest from Investing.com but preserve last-good rows on failure.

    Returns a dict shaped like::

        {
          "source_url": "...",
          "rows": [...],          # latest if available, else last-good
          "error": None | "fetch_failed",
          "stale": True | False,  # True when we're showing previous data
          "previous_generated_at": "...",  # only when stale=True
        }
    """
    rows = _fetch_investing(url)
    if rows:
        return {
            "source_url": url,
            "rows": rows,
            "grade_summary": _summarize_grades(rows),
            "error": None,
            "stale": False,
        }
    fallback_rows = _apply_row_grades(list((previous or {}).get("rows") or []))
    if fallback_rows:
        return {
            "source_url": url,
            "rows": fallback_rows,
            "grade_summary": _summarize_grades(fallback_rows),
            "error": "fetch_failed",
            "stale": True,
            "previous_generated_at": previous.get("fetched_at")
            or previous.get("source_generated_at")
            or "unknown",
        }
    return {
        "source_url": url,
        "rows": [],
        "grade_summary": _summarize_grades([]),
        "error": "fetch_failed",
        "stale": False,
    }


# ---------------------------------------------------------------------------
# Global news (Google News RSS)
# ---------------------------------------------------------------------------

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
NEWS_TOPICS: list[tuple[str, str]] = [
    ("Stock Market", "global stock market today"),
    ("US Markets", "Wall Street S&P 500 Nasdaq"),
    ("Crude Oil", "crude oil price today"),
    ("Commodities", "gold copper commodities market"),
    ("Indian Markets", "NIFTY Sensex Indian stock market"),
]


def _classify_sentiment(bull: int, bear: int) -> tuple[str, float]:
    """Headline-level label + score in [-1, +1] from bullish/bearish keyword hits."""
    if bull == 0 and bear == 0:
        return "NEUTRAL", 0.0
    score = (bull - bear) / max(bull + bear, 1)
    score = max(-1.0, min(1.0, score))
    if score >= 0.2:
        return "POSITIVE", round(score, 3)
    if score <= -0.2:
        return "NEGATIVE", round(score, 3)
    return "NEUTRAL", round(score, 3)


def _score_headline(title: str) -> dict[str, Any]:
    text = title or ""
    bull = _count_hits(text, _BULL_PATS)
    bear = _count_hits(text, _BEAR_PATS)
    label, score = _classify_sentiment(bull, bear)
    return {
        "bullish_hits": bull,
        "bearish_hits": bear,
        "sentiment": label,
        "sentiment_score": score,
    }


def _aggregate_sentiment(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"label": "NEUTRAL", "score": 0.0, "positive": 0, "negative": 0, "neutral": 0, "articles": 0}
    pos = sum(1 for it in items if it.get("sentiment") == "POSITIVE")
    neg = sum(1 for it in items if it.get("sentiment") == "NEGATIVE")
    neu = len(items) - pos - neg
    bull_total = sum(int(it.get("bullish_hits") or 0) for it in items)
    bear_total = sum(int(it.get("bearish_hits") or 0) for it in items)
    label, score = _classify_sentiment(bull_total, bear_total)
    return {
        "label": label,
        "score": score,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "articles": len(items),
        "bullish_hits": bull_total,
        "bearish_hits": bear_total,
    }


def _parse_news_rss(xml_bytes: bytes, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return items
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        source_el = it.find("source")
        source = (source_el.text.strip() if source_el is not None and source_el.text else "")
        pub_raw = (it.findtext("pubDate") or "").strip()
        published_iso: str | None = None
        if pub_raw:
            try:
                published_iso = parsedate_to_datetime(pub_raw).astimezone(timezone.utc).isoformat()
            except Exception:
                published_iso = None
        if not title:
            continue
        item = {
            "title": title,
            "url": link,
            "source": source,
            "published": published_iso or pub_raw,
        }
        item.update(_score_headline(title))
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _fetch_news(lookback_hours: int = 24, per_topic_limit: int = 6) -> dict[str, Any]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)
    groups: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []
    for topic, query in NEWS_TOPICS:
        url = (
            f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}"
            "&hl=en-IN&gl=IN&ceid=IN:en"
        )
        raw = _http_get(url, accept="application/rss+xml,application/xml,text/xml")
        items: list[dict[str, Any]] = []
        if raw:
            items = _parse_news_rss(raw, limit=per_topic_limit * 2)
            kept: list[dict[str, Any]] = []
            for itm in items:
                pub = itm.get("published")
                try:
                    if pub and datetime.fromisoformat(pub.replace("Z", "+00:00")) < cutoff:
                        continue
                except Exception:
                    pass
                kept.append(itm)
                if len(kept) >= per_topic_limit:
                    break
            items = kept
        groups.append(
            {
                "topic": topic,
                "query": query,
                "items": items,
                "sentiment": _aggregate_sentiment(items),
            }
        )
        all_items.extend(items)
    return {"groups": groups, "overall": _aggregate_sentiment(all_items)}


# ---------------------------------------------------------------------------
# NIFTY plan for today (uses already-stored signals + snapshot)
# ---------------------------------------------------------------------------


def _json_safe_value(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
    return v


def _row_to_dict(row: pd.Series, keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        if k in row.index:
            out[k] = _json_safe_value(row[k])
    return out


def _build_nifty_plan(settings: Settings, trade_date: date) -> dict[str, Any]:
    plan: dict[str, Any] = {"trade_date": trade_date.isoformat(), "underlying": "NIFTY"}
    try:
        store = DataStore(settings.data_dir, underlying="NIFTY")
        sig_df = store.load_signals(trade_date=trade_date)
        snap_df = store.load_snapshots(trade_date=trade_date)
        latest_signal: dict[str, Any] | None = None
        if not sig_df.empty:
            if "timestamp" in sig_df.columns:
                sig_df = sig_df.sort_values(by="timestamp")
            actionable = sig_df[sig_df["signal"].isin(["BUY", "SELL"])] if "signal" in sig_df.columns else sig_df.iloc[0:0]
            if not actionable.empty:
                latest_signal = _row_to_dict(
                    actionable.iloc[-1],
                    [
                        "timestamp", "signal", "entry", "target", "stop_loss", "rr",
                        "confidence", "bias", "strike_price", "option_type",
                        "option_entry", "option_sl", "option_target", "reasons",
                    ],
                )
        plan["latest_signal"] = latest_signal

        levels: dict[str, Any] = {}
        if not snap_df.empty:
            last = snap_df.iloc[-1]
            levels = _row_to_dict(
                last,
                [
                    "timestamp", "spot", "open", "high", "low", "vwap",
                    "session_high", "session_low", "support", "resistance",
                    "spot_change_pct",
                ],
            )
            sess_high = None
            sess_low = None
            if "spot" in snap_df.columns:
                sess_high = float(snap_df["high"].max()) if "high" in snap_df.columns else float(snap_df["spot"].max())
                sess_low = float(snap_df["low"].min()) if "low" in snap_df.columns else float(snap_df["spot"].min())
            levels.setdefault("session_high", sess_high)
            levels.setdefault("session_low", sess_low)
        plan["levels"] = levels

        bias_counts: dict[str, int] = {}
        if not sig_df.empty and "bias" in sig_df.columns:
            for b in sig_df["bias"].astype(str).fillna("NEUTRAL"):
                bias_counts[b] = bias_counts.get(b, 0) + 1
        plan["bias_counts"] = bias_counts
        plan["dominant_bias"] = (
            sorted(bias_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
            if bias_counts else None
        )

        if not sig_df.empty and "signal" in sig_df.columns:
            plan["signal_counts"] = {
                "BUY": int((sig_df["signal"] == "BUY").sum()),
                "SELL": int((sig_df["signal"] == "SELL").sum()),
                "NO_TRADE": int(((~sig_df["signal"].isin(["BUY", "SELL"]))).sum()),
            }
        else:
            plan["signal_counts"] = {"BUY": 0, "SELL": 0, "NO_TRADE": 0}
    except Exception as e:
        logger.exception("NIFTY plan build failed: %s", e)
        plan["error"] = str(e)
    return plan


# ---------------------------------------------------------------------------
# FII / DII summary (reuse fii_dii_trends_scan output)
# ---------------------------------------------------------------------------


def _fii_dii_block(
    *,
    settings: Settings,
    trade_date: date,
    refresh: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    if refresh:
        try:
            payload = run_fii_dii_trends_scan(settings=settings, trade_date=trade_date)
        except Exception as e:
            logger.warning("FII/DII refresh failed: %s", e)
    if payload is None:
        payload = load_stored_fii_dii_trends(settings.data_dir, trade_date)
    if not payload:
        return {"error": "no_data"}
    rows = payload.get("fii_dii") or []
    last_row = rows[0] if rows else {}
    return {
        "as_of": payload.get("trade_date"),
        "generated_at": payload.get("generated_at"),
        "summary": payload.get("summary", {}),
        "latest": last_row,
        "recent_rows": rows[:10],
        "participant_oi_latest": (payload.get("summary") or {}).get("participants", {}),
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_market_overview_scan(
    *,
    settings: Settings | None = None,
    trade_date: date | None = None,
    refresh_fii_dii: bool = False,
    news_lookback_hours: int = 24,
    news_per_topic: int = 6,
) -> dict[str, Any]:
    """Compute market overview and persist to ``analysis/market_overview/...``."""
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()

    payload: dict[str, Any] = {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        payload["gift_nifty"] = _scrape_gift_nifty()
    except Exception as e:
        logger.warning("GIFT Nifty failed: %s", e)
        payload["gift_nifty"] = {"error": str(e)}

    try:
        client = ZerodhaClient(settings)
        payload["india_vix"] = _fetch_india_vix(client)
    except Exception as e:
        logger.warning("INDIA VIX failed: %s", e)
        payload["india_vix"] = {"error": str(e)}

    previous = load_stored_market_overview(settings.data_dir, td) or {}
    payload["global_indices"] = _build_investing_block(
        INVESTING_INDICES_URL, previous.get("global_indices") or {}
    )
    payload["commodities_macro"] = _build_investing_block(
        INVESTING_COMMODITIES_URL, previous.get("commodities_macro") or {}
    )

    payload["fii_dii"] = _fii_dii_block(
        settings=settings, trade_date=td, refresh=refresh_fii_dii
    )

    try:
        payload["news"] = _fetch_news(
            lookback_hours=news_lookback_hours, per_topic_limit=news_per_topic
        )
    except Exception as e:
        logger.warning("News fetch failed: %s", e)
        payload["news"] = {"groups": [], "overall": _aggregate_sentiment([])}

    payload["nifty_plan"] = _build_nifty_plan(settings, td)

    out_path = market_overview_path(settings.data_dir, td)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.warning("market_overview save failed: %s", e)
    logger.info("Market overview built (%s).", out_path)
    return payload


def load_stored_market_overview(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = market_overview_path(data_dir, trade_date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("market_overview_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
