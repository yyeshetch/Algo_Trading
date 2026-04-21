"""
Combined intraday FnO stock scan:

- Build BUY watchlist from liquid FnO stocks where 60m RSI > 55 and 15m RSI > 55
- Build SELL watchlist from liquid FnO stocks where 60m RSI < 45 and 15m RSI < 45
- Emit BUY/SELL intraday signals only when the symbol is already in the matching watchlist
- Watchlists update only on hourly slots anchored to market open: 9:15, 10:15, ... 15:15

Persists to: data/fno_intraday_signals_{date}.json
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.analysis.rsi import rsi_last
from intraday_engine.core.config import Settings
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.zerodha_client import ZerodhaClient

logger = logging.getLogger(__name__)


def _fno_i(key: str, default: int) -> int:
    from intraday_engine.core.tunables import get_int

    return get_int("fno_intraday_buy_scanner", key, default)


def _fno_f(key: str, default: float) -> float:
    from intraday_engine.core.tunables import get_float

    return get_float("fno_intraday_buy_scanner", key, default)


_rate_lock = threading.Lock()
_rate_last = 0.0


def _rate_limit() -> None:
    global _rate_last
    with _rate_lock:
        elapsed = time.monotonic() - _rate_last
        if elapsed < _fno_f("RATE_LIMIT_SEC", 0.25):
            time.sleep(_fno_f("RATE_LIMIT_SEC", 0.25) - elapsed)
        _rate_last = time.monotonic()


def _session_cutoff(trade_date: date) -> datetime:
    now = datetime.now()
    if now.date() != trade_date:
        return datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=25)
    return min(
        datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=30),
        now.replace(second=0, microsecond=0),
    )


def _drop_incomplete_candles(df: pd.DataFrame, trade_date: date, interval_minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    now = datetime.now()
    if trade_date != now.date():
        return df.reset_index(drop=True)
    cutoff = min(
        datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=30),
        now.replace(second=0, microsecond=0),
    )
    completed_mask = df["timestamp"] + pd.to_timedelta(interval_minutes, unit="m") <= cutoff
    return df.loc[completed_mask].reset_index(drop=True)


def _watchlist_periods(trade_date: date) -> list[tuple[datetime, datetime]]:
    """Hourly watchlist buckets anchored to market open; last bucket is 15:15-15:30."""
    start = datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=15)
    end = datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=30)
    periods: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(hours=1), end)
        periods.append((cur, nxt))
        cur = nxt
    return periods


def _active_watchlist_period(trade_date: date) -> tuple[datetime, datetime] | None:
    """Return the active watchlist bucket for the current/historical time."""
    periods = _watchlist_periods(trade_date)
    if not periods:
        return None
    now = datetime.now()
    if now.date() != trade_date:
        return periods[-1]
    for start, end in periods:
        if start <= now < end:
            return start, end
    if now < periods[0][0]:
        return periods[0]
    return periods[-1]


def _ts_to_date(ts: Any) -> date | None:
    if ts is None:
        return None
    if hasattr(ts, "date"):
        return ts.date()
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "")[:19]).date()
        except ValueError:
            return None
    return None


def _candles_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    raw = raw.rename(columns={"date": "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume", "oi"):
        if c in raw.columns:
            raw[c] = raw[c].astype(float)
    return raw


def _previous_day_levels(client: ZerodhaClient, token: int, trade_date: date) -> tuple[float | None, float | None]:
    from_dt = datetime.combine(trade_date, datetime.min.time()) - timedelta(days=90)
    to_dt = datetime.combine(trade_date, datetime.min.time())
    _rate_limit()
    try:
        rows = client.historical_data(token, from_dt, to_dt, interval="day")
    except Exception as e:
        logger.debug("Daily levels fetch failed: %s", e)
        return None, None
    if not rows:
        return None, None
    best: tuple[date, float, float] | None = None
    for r in rows:
        d = _ts_to_date(r.get("date"))
        if d is None or d >= trade_date:
            continue
        high = float(r.get("high", 0) or 0)
        low = float(r.get("low", 0) or 0)
        if high <= 0 or low <= 0:
            continue
        if best is None or d > best[0]:
            best = (d, high, low)
    if not best:
        return None, None
    return best[1], best[2]


def _session_vwap(df_5: pd.DataFrame) -> float | None:
    if df_5.empty or "volume" not in df_5.columns:
        return None
    tp = (df_5["high"] + df_5["low"] + df_5["close"]) / 3.0
    vol = df_5["volume"].astype(float)
    if vol.sum() <= 0:
        return None
    return float((tp * vol).sum() / vol.sum())


def _opening_range_5m(session_5: pd.DataFrame) -> tuple[float, float] | None:
    if session_5.empty:
        return None
    first = session_5.iloc[0]
    return float(first["high"]), float(first["low"])


def _volume_bias_from_candle(df_5: pd.DataFrame, candle_index: int) -> dict[str, Any]:
    if df_5.empty or candle_index < 0 or candle_index >= len(df_5) or candle_index < _fno_i("VOL_SMA_BARS", 20) - 1:
        return {
            "bias": "NEUTRAL",
            "volume": None,
            "volume_sma20": None,
            "volume_multiple": None,
            "liquidity_grab": None,
            "confirmation_timestamp": None,
            "liquidity_grab_status": "NONE",
            "liquidity_grab_hit_at": None,
        }

    candle = df_5.iloc[candle_index]
    volume_window = df_5["volume"].iloc[candle_index - _fno_i("VOL_SMA_BARS", 20) + 1 : candle_index + 1].astype(float)
    current_volume = float(candle["volume"])
    volume_sma20 = float(volume_window.mean()) if not volume_window.empty else 0.0
    volume_multiple = (current_volume / volume_sma20) if volume_sma20 > 0 else 0.0

    bias = "NEUTRAL"
    liquidity_grab: float | None = None
    candle_open = float(candle["open"])
    candle_close = float(candle["close"])
    if volume_sma20 > 0 and current_volume > _fno_f("VOLUME_BIAS_MULTIPLIER", 1.5) * volume_sma20:
        if candle_close > candle_open:
            bias = "BULLISH_BIAS"
            liquidity_grab = float(candle["low"])
        elif candle_close < candle_open:
            bias = "BEARISH_BIAS"
            liquidity_grab = float(candle["high"])

    return {
        "bias": bias,
        "volume": round(current_volume, 2),
        "volume_sma20": round(volume_sma20, 2) if volume_sma20 > 0 else None,
        "volume_multiple": round(volume_multiple, 2) if volume_sma20 > 0 else None,
        "liquidity_grab": round(liquidity_grab, 2) if liquidity_grab is not None else None,
        "confirmation_timestamp": (
            (pd.to_datetime(candle["timestamp"]) + pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
            if bias != "NEUTRAL"
            else None
        ),
        "liquidity_grab_status": "PENDING" if bias != "NEUTRAL" else "NONE",
        "liquidity_grab_hit_at": None,
    }


def _latest_volume_bias_state(session_5: pd.DataFrame) -> dict[str, Any]:
    latest = {
        "bias": "NEUTRAL",
        "volume": None,
        "volume_sma20": None,
        "volume_multiple": None,
        "liquidity_grab": None,
        "confirmation_timestamp": None,
        "liquidity_grab_status": "NONE",
        "liquidity_grab_hit_at": None,
    }
    if session_5.empty:
        return latest

    latest_idx = -1
    for idx in range(len(session_5)):
        candidate = _volume_bias_from_candle(session_5, idx)
        if candidate["bias"] != "NEUTRAL":
            latest = candidate
            latest_idx = idx

    if latest_idx < 0 or latest["liquidity_grab"] is None:
        return latest

    subsequent = session_5.iloc[latest_idx + 1 :].reset_index(drop=True)
    if subsequent.empty:
        return latest

    level = float(latest["liquidity_grab"])
    if latest["bias"] == "BULLISH_BIAS":
        hit = subsequent[subsequent["low"].astype(float) <= level]
    else:
        hit = subsequent[subsequent["high"].astype(float) >= level]

    if not hit.empty:
        latest["liquidity_grab_status"] = "GRABBED"
        latest["liquidity_grab_hit_at"] = (
            pd.to_datetime(hit.iloc[0]["timestamp"]) + pd.Timedelta(minutes=5)
        ).strftime("%Y-%m-%dT%H:%M:%S")
    return latest


def _rsi_at_or_before(df: pd.DataFrame, cutoff: datetime) -> float | None:
    """RSI from rows up to cutoff timestamp inclusive."""
    if df.empty or "timestamp" not in df.columns or "close" not in df.columns:
        return None
    subset = df[df["timestamp"] <= cutoff]
    if subset.empty:
        return None
    return rsi_last(subset["close"], _fno_i("RSI_PERIOD", 14))


def _liquid_fno_stock_names(client: ZerodhaClient) -> list[str]:
    """Return the allowed liquid FnO stocks that are live in Zerodha."""
    return client.fno_stock_names()


def _fetch_intraday_context(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date,
) -> dict[str, Any] | None:
    """Load all data needed to decide watchlist membership and intraday 5m signals."""
    settings = Settings.from_env(underlying=stock_name)
    spot_sym = settings.spot_symbol
    _rate_limit()
    try:
        quote = client.quote([spot_sym])
    except Exception as e:
        logger.debug("Quote %s: %s", stock_name, e)
        return None
    if not quote or spot_sym not in quote:
        return None
    spot_token = int(quote[spot_sym].get("instrument_token", 0) or 0)
    if not spot_token:
        return None

    session_cutoff = _session_cutoff(trade_date)
    session_start = datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=15)
    if session_cutoff < session_start:
        return None

    _rate_limit()
    try:
        m5 = _candles_to_df(
            client.historical_data(
                spot_token,
                session_start - timedelta(days=14),
                session_cutoff,
                interval="5minute",
            )
        )
    except Exception as e:
        logger.debug("5m %s: %s", stock_name, e)
        return None
    m5 = _drop_incomplete_candles(m5, trade_date, 5)
    if m5.empty or len(m5) < _fno_i("RSI_PERIOD", 14) + _fno_i("VOL_SMA_BARS", 20) + 2:
        return None

    _rate_limit()
    try:
        m15 = _candles_to_df(
            client.historical_data(
                spot_token,
                datetime.combine(trade_date, datetime.min.time()) - timedelta(days=40),
                session_cutoff,
                interval="15minute",
            )
        )
    except Exception as e:
        logger.debug("15m %s: %s", stock_name, e)
        return None
    m15 = _drop_incomplete_candles(m15, trade_date, 15)
    if m15.empty or len(m15) < _fno_i("RSI_PERIOD", 14) + 2:
        return None

    _rate_limit()
    try:
        h1 = _candles_to_df(
            client.historical_data(
                spot_token,
                datetime.combine(trade_date, datetime.min.time()) - timedelta(days=120),
                session_cutoff,
                interval="60minute",
            )
        )
    except Exception as e:
        logger.debug("60m %s: %s", stock_name, e)
        return None
    h1 = _drop_incomplete_candles(h1, trade_date, 60)
    if h1.empty or len(h1) < _fno_i("RSI_PERIOD", 14) + 2:
        return None

    pdh, pdl = _previous_day_levels(client, spot_token, trade_date)

    session_5 = m5[m5["timestamp"] >= session_start].reset_index(drop=True)
    if session_5.empty or len(session_5) < 3:
        return None
    or_levels = _opening_range_5m(session_5)
    if not or_levels:
        return None
    or_high, or_low = or_levels

    rsi_5 = rsi_last(m5["close"], _fno_i("RSI_PERIOD", 14))
    rsi_15 = rsi_last(m15["close"], _fno_i("RSI_PERIOD", 14))
    rsi_60 = rsi_last(h1["close"], _fno_i("RSI_PERIOD", 14))

    last_i = len(session_5) - 1
    last_close = float(session_5.iloc[last_i]["close"])
    last_vol = float(session_5.iloc[last_i]["volume"])
    vwap = _session_vwap(session_5.iloc[: last_i + 1])

    vol_sma: float | None = None
    volume_2x = False
    pos = len(m5) - 1
    if pos >= _fno_i("VOL_SMA_BARS", 20):
        prev_vols = m5["volume"].iloc[pos - _fno_i("VOL_SMA_BARS", 20) : pos].astype(float)
        vol_sma = float(prev_vols.mean())
        volume_2x = vol_sma > 0 and last_vol >= _fno_f("VOL_MULTIPLIER", 2.0) * vol_sma
    volume_bias = _latest_volume_bias_state(session_5)

    swing_high_level: float | None = None
    swing_high_breakout = False
    swing_low_level: float | None = None
    swing_low_breakdown = False
    if last_i >= _fno_i("SWING_LOOKBACK", 10):
        window = session_5.iloc[last_i - _fno_i("SWING_LOOKBACK", 10) : last_i]
        swing_high_level = float(window["high"].max())
        swing_low_level = float(window["low"].min())
        swing_high_breakout = last_close > swing_high_level
        swing_low_breakdown = last_close < swing_low_level

    fut_oi_last: float | None = None
    fut_oi_prev: float | None = None
    futures_oi_drop = False
    futures_oi_rise = False
    try:
        resolver = InstrumentResolver(client, settings)
        ref = float(session_5.iloc[0]["open"])
        symbols = resolver.resolve_for_date(ref, trade_date)
        fut_quote = client.quote([symbols.fut_symbol])
        if fut_quote and symbols.fut_symbol in fut_quote:
            fut_token = int(fut_quote[symbols.fut_symbol]["instrument_token"])
            _rate_limit()
            fut_df = _candles_to_df(
                client.historical_data(
                    fut_token,
                    session_start - timedelta(days=1),
                    session_cutoff,
                    interval="5minute",
                    oi=True,
                )
            )
            fut_df = _drop_incomplete_candles(fut_df, trade_date, 5)
            if not fut_df.empty and "oi" in fut_df.columns and len(fut_df) >= 2:
                oi_s = fut_df["oi"].astype(float)
                fut_oi_last = float(oi_s.iloc[-1])
                fut_oi_prev = float(oi_s.iloc[-2])
                futures_oi_drop = fut_oi_last < fut_oi_prev
                futures_oi_rise = fut_oi_last > fut_oi_prev
    except Exception as e:
        logger.debug("Fut OI %s: %s", stock_name, e)

    last_close_ts = pd.to_datetime(session_5.iloc[last_i]["timestamp"]) + pd.Timedelta(minutes=5)

    return {
        "stock": stock_name,
        "price": round(last_close, 2),
        "timestamp": last_close_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "_m15_df": m15,
        "_h1_df": h1,
        "rsi_60m": round(rsi_60, 2) if rsi_60 is not None else None,
        "rsi_15m": round(rsi_15, 2) if rsi_15 is not None else None,
        "rsi_5m": round(rsi_5, 2) if rsi_5 is not None else None,
        "pdh": round(pdh, 2) if pdh is not None else None,
        "pdl": round(pdl, 2) if pdl is not None else None,
        "or_high_5m": round(or_high, 2),
        "or_low_5m": round(or_low, 2),
        "vwap_5m": round(vwap, 2) if vwap is not None else None,
        "volume": int(last_vol),
        "vol_sma20": round(vol_sma, 2) if vol_sma is not None else None,
        "volume_bias": volume_bias["bias"],
        "volume_multiple": volume_bias["volume_multiple"],
        "liquidity_grab": volume_bias["liquidity_grab"],
        "liquidity_grab_status": volume_bias["liquidity_grab_status"],
        "liquidity_grab_hit_at": volume_bias["liquidity_grab_hit_at"],
        "volume_bias_confirmation_at": volume_bias["confirmation_timestamp"],
        "volume_bias_volume": volume_bias["volume"],
        "volume_bias_sma20": volume_bias["volume_sma20"],
        "fut_oi_last": fut_oi_last,
        "fut_oi_prev": fut_oi_prev,
        "swing_high_level": round(swing_high_level, 2) if swing_high_level is not None else None,
        "swing_low_level": round(swing_low_level, 2) if swing_low_level is not None else None,
        "buy_confirmations": {
            "orb_5m": last_close > or_high,
            "pdh_breakout": pdh is not None and last_close > pdh,
            "above_vwap_5m": vwap is not None and last_close > vwap,
            "rsi_5_above_55": rsi_5 is not None and rsi_5 > _fno_f("BUY_WATCHLIST_RSI", 55.0),
            "volume_2x_sma20": volume_2x,
            "futures_oi_drop": futures_oi_drop,
            "swing_high_breakout": swing_high_breakout,
        },
        "sell_confirmations": {
            "orb_5m_breakdown": last_close < or_low,
            "pdl_breakdown": pdl is not None and last_close < pdl,
            "below_vwap_5m": vwap is not None and last_close < vwap,
            "rsi_5_below_45": rsi_5 is not None and rsi_5 < _fno_f("SELL_WATCHLIST_RSI", 45.0),
            "volume_2x_sma20": volume_2x,
            "futures_oi_rise": futures_oi_rise,
            "swing_low_breakdown": swing_low_breakdown,
        },
    }


def _watchlist_side_from_rsi(rsi_60m: float | None, rsi_15m: float | None) -> str:
    if rsi_60m is not None and rsi_15m is not None and rsi_60m > _fno_f("BUY_WATCHLIST_RSI", 55.0) and rsi_15m > _fno_f("BUY_WATCHLIST_RSI", 55.0):
        return "BUY"
    if rsi_60m is not None and rsi_15m is not None and rsi_60m < _fno_f("SELL_WATCHLIST_RSI", 45.0) and rsi_15m < _fno_f("SELL_WATCHLIST_RSI", 45.0):
        return "SELL"
    return "NEUTRAL"


def _watchlist_entry(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "stock": ctx["stock"],
        "price": ctx["price"],
        "timestamp": ctx["timestamp"],
        "rsi_60m": ctx["rsi_60m"],
        "rsi_15m": ctx["rsi_15m"],
    }


def _public_ctx(ctx: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in ctx.items() if not k.startswith("_")}


def _finalize_row(ctx: dict[str, Any], watchlist_side: str) -> dict[str, Any]:
    buy_confirmations = {
        "watchlist_buy": watchlist_side == "BUY",
        **ctx["buy_confirmations"],
    }
    sell_confirmations = {
        "watchlist_sell": watchlist_side == "SELL",
        **ctx["sell_confirmations"],
    }
    buy_signal = all(buy_confirmations.values())
    sell_signal = all(sell_confirmations.values())
    signal = "BUY" if buy_signal else "SELL" if sell_signal else "NO_TRADE"
    return {
        **_public_ctx(ctx),
        "watchlist_side": watchlist_side,
        "buy_confirmations": buy_confirmations,
        "sell_confirmations": sell_confirmations,
        "signal": signal,
    }


def _store_path(data_dir: Path, trade_date: date) -> Path:
    return data_dir / f"fno_intraday_signals_{trade_date.isoformat()}.json"


def load_stored_intraday_scan(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    path = _store_path(data_dir, trade_date)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
        return None


def _save_intraday_scan(data_dir: Path, payload: dict[str, Any]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    td = payload.get("trade_date")
    if isinstance(td, str):
        path = _store_path(data_dir, datetime.strptime(td, "%Y-%m-%d").date())
    else:
        path = _store_path(data_dir, date.today())
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def run_fno_intraday_scan(
    trade_date: date | None = None,
    stock_limit: int | None = None,
) -> dict[str, Any]:
    """
    Single intraday refresh:
    1. Compute 60m/15m RSI for all liquid FnO stocks
    2. Build BUY/SELL watchlists
    3. Emit intraday BUY/SELL signals only for symbols already in the matching watchlist
    """
    settings = Settings.from_env(underlying="NIFTY")
    client = ZerodhaClient(settings)
    data_dir = settings.data_dir
    trade_date = trade_date or date.today()

    stock_names = _liquid_fno_stock_names(client)
    if stock_limit:
        stock_names = stock_names[:stock_limit]

    results_raw: list[dict[str, Any]] = []
    failed: list[str] = []

    def _scan_one(name: str) -> dict[str, Any] | None:
        return _fetch_intraday_context(client, name, trade_date)

    with ThreadPoolExecutor(max_workers=_fno_i("MAX_WORKERS", 5)) as ex:
        futs = {ex.submit(_scan_one, n): n for n in stock_names}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                row = fut.result()
                if row:
                    results_raw.append(row)
                else:
                    failed.append(name)
            except Exception as e:
                logger.debug("Intraday scan %s: %s", name, e)
                failed.append(name)

    results_raw.sort(key=lambda x: x.get("stock", ""))

    watchlist_timeline: list[dict[str, Any]] = []
    for slot_start, slot_end in _watchlist_periods(trade_date):
        buy_watchlist = []
        sell_watchlist = []
        for row in results_raw:
            rsi_60 = _rsi_at_or_before(row["_h1_df"], slot_start)
            rsi_15 = _rsi_at_or_before(row["_m15_df"], slot_start)
            side = _watchlist_side_from_rsi(rsi_60, rsi_15)
            if side == "BUY":
                buy_watchlist.append(
                    {
                        "stock": row["stock"],
                        "price": row["price"],
                        "timestamp": row["timestamp"],
                        "rsi_60m": round(rsi_60, 2) if rsi_60 is not None else None,
                        "rsi_15m": round(rsi_15, 2) if rsi_15 is not None else None,
                    }
                )
            elif side == "SELL":
                sell_watchlist.append(
                    {
                        "stock": row["stock"],
                        "price": row["price"],
                        "timestamp": row["timestamp"],
                        "rsi_60m": round(rsi_60, 2) if rsi_60 is not None else None,
                        "rsi_15m": round(rsi_15, 2) if rsi_15 is not None else None,
                    }
                )
        watchlist_timeline.append(
            {
                "slot_start": slot_start.isoformat(timespec="seconds"),
                "slot_end": slot_end.isoformat(timespec="seconds"),
                "buy_watchlist": buy_watchlist,
                "sell_watchlist": sell_watchlist,
            }
        )

    active_period = _active_watchlist_period(trade_date)
    active_slot_start = active_period[0].isoformat(timespec="seconds") if active_period else None
    active_slot_end = active_period[1].isoformat(timespec="seconds") if active_period else None
    active_timeline = next(
        (x for x in watchlist_timeline if x["slot_start"] == active_slot_start and x["slot_end"] == active_slot_end),
        None,
    )
    buy_watchlist = list(active_timeline.get("buy_watchlist", [])) if active_timeline else []
    sell_watchlist = list(active_timeline.get("sell_watchlist", [])) if active_timeline else []

    watchlist_map = {x["stock"]: "BUY" for x in buy_watchlist}
    watchlist_map.update({x["stock"]: "SELL" for x in sell_watchlist})

    results = [_finalize_row(r, watchlist_map.get(r["stock"], "NEUTRAL")) for r in results_raw]
    buy_signals = [r for r in results if r.get("signal") == "BUY"]
    sell_signals = [r for r in results if r.get("signal") == "SELL"]

    now = datetime.now()
    next_watchlist_update_at = None
    if active_period:
        periods = _watchlist_periods(trade_date)
        for idx, (start, _end) in enumerate(periods):
            if start == active_period[0]:
                if idx + 1 < len(periods):
                    next_watchlist_update_at = periods[idx + 1][0].isoformat(timespec="seconds")
                else:
                    next_watchlist_update_at = active_period[1].isoformat(timespec="seconds")
                break
    payload = {
        "trade_date": trade_date.isoformat(),
        "last_run_at": now.isoformat(timespec="seconds"),
        "next_refresh_at": (now + timedelta(minutes=5)).isoformat(timespec="seconds"),
        "watchlist_updated_at": now.isoformat(timespec="seconds"),
        "watchlist_slot_at": active_slot_start,
        "watchlist_slot_end_at": active_slot_end,
        "next_watchlist_update_at": next_watchlist_update_at,
        "watchlist_refreshed_this_run": True,
        "universe_name": "liquid_fno_stocks",
        "symbols_scanned": stock_names,
        "stock_limit": stock_limit,
        "watchlist_timeline": watchlist_timeline,
        "signals": results,
        "buy_watchlist": buy_watchlist,
        "sell_watchlist": sell_watchlist,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "failed_symbols": failed,
    }
    _save_intraday_scan(data_dir, payload)
    return payload
