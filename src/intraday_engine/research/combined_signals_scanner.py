"""
Combined Signals scanner — join Institutional Volume + Fundamentals + News
into a single per-stock CSV with a composite score.

Inputs (cached by their own scanners):
  - data/analysis/institutional_volume/institutional_volume_<date>.json
  - data/analysis/fundamentals/fundamentals_<date>.csv
  - data/analysis/news/news_<date>.csv

Composite score (default weighting, all clipped before weighting):

    institutional_signal   in  [0, 100]   (mapped from biggest active label
                                           + recency + volume-multiple)
    fundamental_score      in  [-30, 100] (raw heuristic from fundamentals)
    news_score             in  [-50,  50] (raw normalized from news)

    combined = 0.50 * institutional + 0.35 * fundamental + 0.15 * news

The institutional sub-score rewards stocks that have *recent* (within the
last ~15 trading days) labelled prints — a "year" label today is worth more
than a "year" label 50 days ago. Volume multiple amplifies the effect.

Output CSV: data/analysis/combined/combined_<YYYY-MM-DD>.csv
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.research.fundamentals_screener import load_latest_fundamentals_csv
from intraday_engine.research.stock_news_scanner import load_latest_news_csv
from intraday_engine.storage.layout import (
    combined_signals_csv_path,
    institutional_volume_path,
)

logger = logging.getLogger(__name__)


# Bigger label => bigger conviction. Maps label -> base score (0-100 scale).
_LABEL_BASE = {
    "month": 30.0,
    "quarter": 50.0,
    "6_months": 70.0,
    "year": 85.0,
    "ever": 100.0,
}

# How many trailing trading days of the institutional candle to consider for
# the "recent" weighting. Older labels still count but with steep decay.
_RECENT_DAYS = 30


@dataclass
class CombinedRow:
    stock: str
    last_close: float | None = None
    last_date: str = ""
    institutional_signal: float = 0.0
    institutional_top_label: str = ""
    institutional_recent_date: str = ""
    institutional_volume_multiple: float | None = None
    institutional_pct_change: float | None = None
    fundamental_score: float | None = None
    market_cap_cr: float | None = None
    pe: float | None = None
    roe: float | None = None
    roce: float | None = None
    debt_to_equity: float | None = None
    sales_growth_3y: float | None = None
    profit_growth_3y: float | None = None
    promoter_holding: float | None = None
    pledged_pct: float | None = None
    news_score: float | None = None
    bullish_hits: int | None = None
    bearish_hits: int | None = None
    recent_articles: int | None = None
    latest_headline: str = ""
    latest_url: str = ""
    combined_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _load_institutional(data_dir: Path, td: date) -> dict[str, Any] | None:
    p = institutional_volume_path(data_dir, td)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("institutional_volume_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _label_rank(label: str) -> int:
    order = ["month", "quarter", "6_months", "year", "ever"]
    try:
        return order.index(label)
    except ValueError:
        return -1


def _institutional_subscore(row: dict[str, Any]) -> tuple[float, str, str, float | None, float | None]:
    """
    Return (score_0_100, top_label, top_date, vol_mult_at_top, pct_change_at_top).

    We look at the trailing ``_RECENT_DAYS`` candles for this stock and pick
    the one with the highest combined (label_rank, recency_weight,
    volume_multiple). Older candles are decayed linearly.
    """
    candles = row.get("candles") or []
    if not candles:
        return 0.0, "", "", None, None

    # Map "days ago" by walking from newest -> oldest.
    # candles are oldest->newest in the file, so index from the end.
    best: tuple[float, str, str, float | None, float | None] | None = None
    n = len(candles)
    for i, c in enumerate(candles):
        top_label = c.get("top_label") or ""
        if not top_label:
            continue
        days_ago = n - 1 - i  # 0 = most recent
        if days_ago > _RECENT_DAYS:
            continue
        base = _LABEL_BASE.get(top_label, 20.0)
        recency = max(0.0, 1.0 - (days_ago / float(_RECENT_DAYS + 1)))
        vol_mult = float(c.get("volume_multiple", 0) or 0)
        vol_bonus = min(15.0, max(0.0, (vol_mult - 1.0)) * 3.0)
        score = base * (0.55 + 0.45 * recency) + vol_bonus
        score = max(0.0, min(100.0, score))
        candidate = (score, top_label, c.get("date", ""), vol_mult, c.get("pct_change"))
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return 0.0, "", "", None, None
    return best


def _norm(v: Any) -> float | None:
    try:
        if v is None or v == "" or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def run_combined_scan(
    *,
    settings: Settings | None = None,
    trade_date: date | None = None,
    weights: tuple[float, float, float] = (0.50, 0.35, 0.15),
) -> dict[str, Any]:
    """
    Build combined CSV from the three latest cached scanner outputs.

    Parameters
    ----------
    weights : (institutional, fundamental, news), default (0.50, 0.35, 0.15).
        Must sum to ~1.0; institutional is the dominant driver.
    """
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()
    w_inst, w_fund, w_news = weights

    inst = _load_institutional(settings.data_dir, td) or {}
    fund_df = load_latest_fundamentals_csv(settings.data_dir)
    news_df = load_latest_news_csv(settings.data_dir)

    fund_map: dict[str, dict[str, Any]] = {}
    if not fund_df.empty and "stock" in fund_df.columns:
        for _, r in fund_df.iterrows():
            fund_map[str(r["stock"]).strip().upper()] = r.to_dict()

    news_map: dict[str, dict[str, Any]] = {}
    if not news_df.empty and "stock" in news_df.columns:
        for _, r in news_df.iterrows():
            news_map[str(r["stock"]).strip().upper()] = r.to_dict()

    # Universe = union of every symbol seen in any source — we still want
    # to surface a stock with strong fundamentals + great news even if it
    # doesn't have an institutional print, and vice versa.
    inst_rows: dict[str, dict[str, Any]] = {}
    for r in (inst.get("rows") or []):
        sym = str(r.get("stock", "")).strip().upper()
        if sym:
            inst_rows[sym] = r
    universe = set(inst_rows.keys()) | set(fund_map.keys()) | set(news_map.keys())

    combined: list[CombinedRow] = []
    for sym in sorted(universe):
        cr = CombinedRow(stock=sym)
        reasons: list[str] = []

        inst_row = inst_rows.get(sym)
        if inst_row is not None:
            cr.last_close = _norm(inst_row.get("last_close"))
            cr.last_date = str(inst_row.get("last_date", ""))
            score, label, idate, vm, pc = _institutional_subscore(inst_row)
            cr.institutional_signal = round(score, 1)
            cr.institutional_top_label = label
            cr.institutional_recent_date = idate
            cr.institutional_volume_multiple = vm
            cr.institutional_pct_change = pc
            if label:
                reasons.append(
                    f"Institutional {label} print on {idate} (vol×{vm:.1f}, {pc:+.1f}%)"
                    if vm is not None and pc is not None
                    else f"Institutional {label} print on {idate}"
                )

        fund = fund_map.get(sym)
        if fund:
            cr.fundamental_score = _norm(fund.get("fundamental_score"))
            cr.market_cap_cr = _norm(fund.get("market_cap_cr"))
            cr.pe = _norm(fund.get("pe"))
            cr.roe = _norm(fund.get("roe"))
            cr.roce = _norm(fund.get("roce"))
            cr.debt_to_equity = _norm(fund.get("debt_to_equity"))
            cr.sales_growth_3y = _norm(fund.get("sales_growth_3y"))
            cr.profit_growth_3y = _norm(fund.get("profit_growth_3y"))
            cr.promoter_holding = _norm(fund.get("promoter_holding"))
            cr.pledged_pct = _norm(fund.get("pledged_pct"))
            sr = str(fund.get("score_reasons") or "")
            if sr and sr != "nan":
                reasons.append(f"Fundamentals: {sr}")

        news = news_map.get(sym)
        if news:
            cr.news_score = _norm(news.get("news_score"))
            bh = _norm(news.get("bullish_hits"))
            beh = _norm(news.get("bearish_hits"))
            cr.bullish_hits = int(bh) if bh is not None else None
            cr.bearish_hits = int(beh) if beh is not None else None
            ra = _norm(news.get("recent_articles"))
            cr.recent_articles = int(ra) if ra is not None else None
            cr.latest_headline = str(news.get("latest_headline") or "")
            cr.latest_url = str(news.get("latest_url") or "")
            if cr.latest_headline and cr.latest_headline != "nan":
                reasons.append(f"News: {cr.latest_headline}")

        # weighted composite — missing sources contribute 0 (no penalty)
        i = cr.institutional_signal or 0.0
        f = cr.fundamental_score if cr.fundamental_score is not None else 0.0
        n = cr.news_score if cr.news_score is not None else 0.0
        cr.combined_score = round(w_inst * i + w_fund * f + w_news * n, 2)
        cr.reasons = reasons
        combined.append(cr)

    combined.sort(key=lambda r: r.combined_score, reverse=True)

    out_path = combined_signals_csv_path(settings.data_dir, td)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "stock", "last_close", "last_date",
        "combined_score",
        "institutional_signal", "institutional_top_label",
        "institutional_recent_date", "institutional_volume_multiple",
        "institutional_pct_change",
        "fundamental_score", "market_cap_cr", "pe", "roe", "roce",
        "debt_to_equity", "sales_growth_3y", "profit_growth_3y",
        "promoter_holding", "pledged_pct",
        "news_score", "bullish_hits", "bearish_hits", "recent_articles",
        "latest_headline", "latest_url",
        "reasons",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cr in combined:
            d = asdict(cr)
            d["reasons"] = " || ".join(d.get("reasons") or [])
            w.writerow([d.get(c, "") for c in cols])

    logger.info(
        "Combined scan: universe=%d (inst=%d, fund=%d, news=%d). Saved %s",
        len(combined), len(inst_rows), len(fund_map), len(news_map), out_path,
    )
    return {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe": len(combined),
        "with_institutional": len(inst_rows),
        "with_fundamentals": len(fund_map),
        "with_news": len(news_map),
        "weights": {"institutional": w_inst, "fundamental": w_fund, "news": w_news},
        "output_csv": str(out_path),
    }
