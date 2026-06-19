"""
Stock News scanner — Google News RSS per NIFTY 500 symbol.

Google News exposes a free, key-less RSS endpoint:
    https://news.google.com/rss/search?q=<QUERY>&hl=en-IN&gl=IN&ceid=IN:en

We resolve each ticker to a company name (best-effort via the cached
``ind_nifty500list.csv`` reference) and build the query as
    "<COMPANY NAME>" stock OR shares
to cut noise from unrelated namesakes.

Headlines are passed through a small bullish/bearish keyword lexicon to
produce a per-stock sentiment score:

    sentiment = (bullish_hits - bearish_hits)
    normalized = sentiment / max(articles, 1)     # in [-1, +1]
    news_score = round(normalized * 50, 1)        # in [-50, +50]

Only the last ``lookback_days`` (default 7) of headlines are considered.

Output CSV: data/analysis/news/news_<YYYY-MM-DD>.csv with columns:
    stock, company, articles_count, recent_articles, bullish_hits,
    bearish_hits, sentiment_score, news_score, latest_headline,
    latest_url, latest_published, top_headlines, error
"""

from __future__ import annotations

import csv
import logging
import re
import ssl
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import news_csv_path

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = 20
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_TOP_HEADLINES = 5
DEFAULT_THROTTLE_SEC = 0.25  # Google RSS tolerates this

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


# ---------- Sentiment lexicons (lower-case, whole-ish word matched) ---------

BULLISH = {
    "surge", "surges", "surged", "jump", "jumps", "jumped", "rally", "rallies", "rallied",
    "soar", "soars", "soared", "gain", "gains", "gained", "rise", "rises", "rose", "climb",
    "climbs", "record high", "all-time high", "52-week high", "lifetime high",
    "beat", "beats", "beat estimates", "beats estimates", "tops", "topped estimates",
    "strong", "robust", "outperform", "outperforms", "upgrade", "upgrades", "upgraded",
    "buy rating", "raises target", "target raised", "price target raised", "raised target",
    "win", "wins", "won", "bags", "bagged", "secures", "secured order", "order win",
    "new order", "fresh order", "contract win", "contract awarded",
    "expansion", "expands", "expanded", "new plant", "capex", "capacity addition",
    "acquisition", "acquires", "acquired", "merger", "joint venture", "tie-up", "partnership",
    "dividend", "interim dividend", "special dividend", "bonus issue", "buyback", "stock split",
    "profit", "profits", "profit jumps", "profit rises", "net profit up", "net profit rises",
    "revenue jumps", "revenue rises", "revenue beats", "margin expansion",
    "earnings beat", "results beat", "guidance raised", "guidance upgrade",
    "approval", "approved", "regulatory approval", "fda approval", "clearance",
    "block deal buy", "promoter buying", "insider buying",
}

BEARISH = {
    "fall", "falls", "fell", "plunge", "plunges", "plunged", "drop", "drops", "dropped",
    "slump", "slumps", "slumped", "slide", "slides", "slid", "tumble", "tumbles", "tumbled",
    "crash", "crashes", "crashed", "record low", "52-week low", "lifetime low",
    "loss", "losses", "net loss", "swings to loss", "widens loss", "widening loss",
    "miss", "misses", "missed estimates", "below estimates", "guidance cut", "guidance lowered",
    "downgrade", "downgrades", "downgraded", "sell rating", "target cut", "price target cut",
    "probe", "investigation", "raid", "raids", "sebi probe", "income tax raid", "ed raid",
    "fraud", "scam", "default", "defaults", "defaulted", "bankruptcy", "insolvency", "nclt",
    "fine", "fines", "fined", "penalty", "penalised", "penalized",
    "ban", "banned", "suspended", "stake sale", "promoter selling", "block deal sell",
    "pledge", "pledged shares", "increased pledge",
    "layoff", "layoffs", "job cuts", "plant shutdown", "production halt", "strike",
    "regulatory action", "show cause", "rejection", "rejected", "warning letter",
    "downgrade rating", "rating downgrade", "moody's downgrade", "icra downgrade",
    "fire", "accident", "litigation", "lawsuit", "class action",
}

# Pre-compile word-boundary regexes so multi-word phrases match cleanly.
def _compile_lexicon(words: set[str]) -> list[re.Pattern[str]]:
    pats: list[re.Pattern[str]] = []
    for w in sorted(words, key=len, reverse=True):
        # \b doesn't play with multi-word phrases including hyphens etc; treat
        # them as plain substrings, but anchor single words with \b.
        if " " in w or "-" in w:
            pats.append(re.compile(re.escape(w), re.IGNORECASE))
        else:
            pats.append(re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE))
    return pats


_BULL_PATS = _compile_lexicon(BULLISH)
_BEAR_PATS = _compile_lexicon(BEARISH)


def _count_hits(text: str, patterns: list[re.Pattern[str]]) -> int:
    if not text:
        return 0
    return sum(1 for p in patterns if p.search(text))


@dataclass
class NewsArticle:
    title: str
    link: str
    published: str  # ISO date
    source: str
    bullish: int = 0
    bearish: int = 0


@dataclass
class NewsRow:
    stock: str
    company: str = ""
    articles_count: int = 0
    recent_articles: int = 0
    bullish_hits: int = 0
    bearish_hits: int = 0
    sentiment_score: float = 0.0  # raw normalized in [-1, +1]
    news_score: float = 0.0       # scaled in [-50, +50]
    latest_headline: str = ""
    latest_url: str = ""
    latest_published: str = ""
    top_headlines: list[str] = field(default_factory=list)
    fetched_at: str = ""
    error: str = ""


# ---------- Symbol -> company name resolution -------------------------------


def _build_symbol_company_map(data_dir: Path) -> dict[str, str]:
    """
    Use the cached ``ind_nifty500list.csv`` (already used by other scanners)
    to map ticker -> human company name. Falls back to ticker when missing.
    """
    csv_ref = data_dir / "reference" / "ind_nifty500list.csv"
    if not csv_ref.exists():
        return {}
    try:
        df = pd.read_csv(csv_ref)
    except Exception:
        return {}
    sym_col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), None)
    name_col = next(
        (c for c in df.columns if str(c).strip().lower() in {"company name", "company"}),
        None,
    )
    if not sym_col or not name_col:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        s = str(row.get(sym_col, "")).strip().upper()
        n = str(row.get(name_col, "")).strip()
        if s and n:
            out[s] = n
    return out


# ---------- RSS fetch + parse -----------------------------------------------


def _http_get(url: str) -> str | None:
    ctx = ssl.create_default_context()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml,*/*"}
    for attempt in range(2):
        _throttle()
        try:
            req = Request(url, headers=headers)
            with urlopen(req, context=ctx, timeout=HTTP_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug("news rss %s (try %d): %s", url, attempt + 1, e)
            time.sleep(0.5 * (attempt + 1))
    return None


def _parse_rss(xml_text: str) -> list[NewsArticle]:
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    chan = root.find("channel")
    if chan is None:
        return []
    items: list[NewsArticle] = []
    for it in chan.findall("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub_raw = (it.findtext("pubDate") or "").strip()
        source = ""
        src_el = it.find("source")
        if src_el is not None and src_el.text:
            source = src_el.text.strip()
        pub_iso = ""
        if pub_raw:
            try:
                dt = parsedate_to_datetime(pub_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                pub_iso = dt.astimezone(timezone.utc).isoformat(timespec="seconds")
            except Exception:
                pub_iso = ""
        if title:
            items.append(NewsArticle(title=title, link=link, published=pub_iso, source=source))
    return items


def fetch_news_for(
    symbol: str,
    company: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    top_n: int = DEFAULT_TOP_HEADLINES,
) -> NewsRow:
    row = NewsRow(
        stock=symbol,
        company=company,
        fetched_at=datetime.now().isoformat(timespec="seconds"),
    )
    query_name = company or symbol
    # Anchor on company name to suppress unrelated namesakes; add "stock OR shares"
    # to bias toward market coverage rather than corporate fluff.
    q = f'"{query_name}" (stock OR shares OR NSE OR BSE)'
    url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN:en"
    xml_text = _http_get(url)
    if not xml_text:
        row.error = "rss_fetch_failed"
        return row
    articles = _parse_rss(xml_text)
    row.articles_count = len(articles)

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    recent: list[NewsArticle] = []
    for a in articles:
        if not a.published:
            recent.append(a)
            continue
        try:
            if datetime.fromisoformat(a.published) >= cutoff:
                recent.append(a)
        except ValueError:
            recent.append(a)
    row.recent_articles = len(recent)
    if not recent:
        return row

    bullish_total = bearish_total = 0
    for a in recent:
        a.bullish = _count_hits(a.title, _BULL_PATS)
        a.bearish = _count_hits(a.title, _BEAR_PATS)
        bullish_total += a.bullish
        bearish_total += a.bearish
    row.bullish_hits = bullish_total
    row.bearish_hits = bearish_total

    net = bullish_total - bearish_total
    norm = net / max(len(recent), 1)
    row.sentiment_score = round(norm, 3)
    row.news_score = round(max(-50.0, min(50.0, norm * 50.0)), 1)

    recent.sort(key=lambda x: x.published, reverse=True)
    top = recent[:top_n]
    row.latest_headline = top[0].title
    row.latest_url = top[0].link
    row.latest_published = top[0].published
    row.top_headlines = [f"{a.published[:10]} | {a.title}" for a in top]
    return row


def run_news_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    trade_date: date | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    top_n: int = DEFAULT_TOP_HEADLINES,
    max_workers: int = 8,
    stock_limit: int | None = None,
) -> dict[str, Any]:
    """Scan Google News RSS for every NIFTY 500 stock and emit CSV."""
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()

    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)
    if stock_limit:
        symbols = symbols[: int(stock_limit)]

    sym_to_name = _build_symbol_company_map(settings.data_dir)

    rows: list[NewsRow] = []
    failed: list[str] = []

    def job(sym: str) -> NewsRow:
        try:
            return fetch_news_for(
                sym,
                sym_to_name.get(sym, sym),
                lookback_days=lookback_days,
                top_n=top_n,
            )
        except Exception as e:
            return NewsRow(stock=sym, error=str(e)[:200])

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 12))) as ex:
        futs = {ex.submit(job, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            rows.append(r)
            if r.error:
                failed.append(r.stock)
            if i % 100 == 0:
                logger.info("news: %d/%d done (%d failed)", i, len(symbols), len(failed))

    rows.sort(key=lambda r: (-(r.news_score or 0.0), -(r.recent_articles or 0), r.stock))

    out_path = news_csv_path(settings.data_dir, td)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stock", "company", "articles_count", "recent_articles",
            "bullish_hits", "bearish_hits", "sentiment_score", "news_score",
            "latest_headline", "latest_url", "latest_published",
            "top_headlines", "fetched_at", "error",
        ])
        for r in rows:
            d = asdict(r)
            w.writerow([
                d["stock"], d["company"], d["articles_count"], d["recent_articles"],
                d["bullish_hits"], d["bearish_hits"], d["sentiment_score"], d["news_score"],
                d["latest_headline"], d["latest_url"], d["latest_published"],
                " || ".join(d["top_headlines"] or []),
                d["fetched_at"], d["error"],
            ])

    logger.info(
        "News scan: %d symbols, %d failed. Saved %s",
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


def load_latest_news_csv(data_dir: Path) -> pd.DataFrame:
    d = data_dir / "analysis" / "news"
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.glob("news_*.csv"))
    if not files:
        return pd.DataFrame()
    try:
        return pd.read_csv(files[-1])
    except Exception:
        return pd.DataFrame()
