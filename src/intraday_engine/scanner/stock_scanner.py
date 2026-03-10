"""Orchestrate F&O stock scanner: fetch, metrics, score, rank."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.scanner.scanner_metrics import compute_stock_metrics
from intraday_engine.scanner.scanner_scoring import rank_stocks, score_stock
from intraday_engine.scanner.stock_scanner_fetcher import fetch_stock_15min_data

logger = logging.getLogger(__name__)


def run_stock_scan(
    trade_date: date | None = None,
    stock_limit: int | None = 50,
) -> list[dict[str, Any]]:
    """
    Run scanner on all F&O stocks. Returns ranked list of scored stocks.
    stock_limit: max stocks to scan (for performance; None = all)
    """
    settings = Settings.from_env(underlying="NIFTY")
    client = ZerodhaClient(settings)
    stock_names = client.fno_stock_names()
    if stock_limit:
        stock_names = stock_names[:stock_limit]

    all_metrics: list[dict[str, Any]] = []
    for i, name in enumerate(stock_names):
        try:
            data = fetch_stock_15min_data(client, name, trade_date)
            if not data:
                continue
            metrics = compute_stock_metrics(data)
            if not metrics:
                continue
            all_metrics.append(metrics)
        except Exception as e:
            logger.debug("Scan skip %s: %s", name, e)
            continue

    scored = [score_stock(m["stock"], m, all_metrics) for m in all_metrics]
    return rank_stocks(scored)
