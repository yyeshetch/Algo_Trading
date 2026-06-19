"""
Backtest: Institutional Expansion Pivot (15m entry, optional daily context).

Rules implemented:
- 20-period volume MA (VMA); RVOL = volume / VMA. Spike candle: RVOL >= vol_mult (default 2.0).
- Consolidation: lookback bars (10–15) BEFORE the spike with a tight horizontal range.
- Breakout: spike closes beyond consolidation high (long) or low (short).
- Optional spring/upthrust: a prior bar in the box pierces support/resistance on volume < VMA,
  then closes back inside (same direction as eventual breakout).
- Entry: next bar, 1 tick beyond spike high (long) or spike low (short); fill if price trades through.
- Stop: midpoint of spike candle (high+low)/2.
- Exit: trail — close vs 9 EMA; optional exhaustion — opposite-color bar with RVOL >= vol_mult.

Does not guarantee 1:8 RR; records planned 1:8 target distance for analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _tick_size(price: float, default: float) -> float:
    """Rough NSE cash tick: <100 → 0.05, else 0.10 (override with --tick)."""
    if default != 0.05 and default != 0.10:
        return float(default)
    return 0.05 if price < 100.0 else 0.10


@dataclass
class TradeResult:
    side: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    spike_idx: int
    spike_time: pd.Timestamp
    cons_high: float
    cons_low: float
    cons_bars: int
    rvol_spike: float
    planned_tp_8r: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    bars_held: int
    pnl_per_unit: float
    r_multiple: float
    had_spring: bool


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_daily_context(df: pd.DataFrame) -> pd.DataFrame:
    day = df["date"].dt.normalize()
    daily_close = df.groupby(day, sort=True)["close"].last().astype(float)
    d_ema20 = _ema(daily_close, 20)
    d_ema50 = _ema(daily_close, 50)
    out = df.copy()
    out["daily_ema20"] = day.map(d_ema20.shift(1))
    out["daily_ema50"] = day.map(d_ema50.shift(1))
    return out


def detect_consolidation(
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    i_spike: int,
    cons_bars: int,
    max_range_pct: float,
) -> tuple[float, float] | None:
    """Box from i_spike-cons_bars .. i_spike-1 inclusive."""
    a = i_spike - cons_bars
    b = i_spike - 1
    if a < 0:
        return None
    lo = float(np.min(lows[a : b + 1]))
    hi = float(np.max(highs[a : b + 1]))
    mid = (hi + lo) / 2.0
    if mid <= 0:
        return None
    if (hi - lo) / mid > max_range_pct:
        return None
    return hi, lo


def has_spring_long(
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    volumes: np.ndarray,
    vma: np.ndarray,
    cons_low: float,
    cons_high: float,
    a: int,
    b: int,
) -> bool:
    """Stop-run below box then close back inside (Wyckoff spring), low volume vs VMA."""
    for j in range(a, b + 1):
        if not np.isfinite(vma[j]) or vma[j] <= 0:
            continue
        if lows[j] < cons_low and closes[j] > cons_low and volumes[j] < vma[j]:
            return True
    return False


def has_spring_short(
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    volumes: np.ndarray,
    vma: np.ndarray,
    cons_low: float,
    cons_high: float,
    a: int,
    b: int,
) -> bool:
    """Upthrust above box then close back inside."""
    for j in range(a, b + 1):
        if not np.isfinite(vma[j]) or vma[j] <= 0:
            continue
        if highs[j] > cons_high and closes[j] < cons_high and volumes[j] < vma[j]:
            return True
    return False


def run_backtest(
    df: pd.DataFrame,
    *,
    vol_mult: float,
    cons_bars: int,
    max_range_pct: float,
    tick: float | None,
    use_daily_filter: bool,
    require_spring: bool,
    exhaustion_exit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    dates = df["date"]

    vma20 = df["volume"].rolling(window=20, min_periods=20).mean().to_numpy(dtype=float)
    rvol = np.where(vma20 > 0, vol / vma20, np.nan)
    ema9 = _ema(df["close"], 9).to_numpy(dtype=float)

    d_ok_long = np.ones(n, dtype=bool)
    d_ok_short = np.ones(n, dtype=bool)
    if use_daily_filter and "daily_ema20" in df.columns:
        de20 = df["daily_ema20"].to_numpy(dtype=float)
        d_ok_long = np.isfinite(de20) & (close > de20)
        d_ok_short = np.isfinite(de20) & (close < de20)

    trades: list[TradeResult] = []
    in_trade_until = -1

    for i in range(cons_bars + 20, n - 1):
        if i <= in_trade_until:
            continue

        box = detect_consolidation(lows, high, close, i, cons_bars, max_range_pct)
        if box is None:
            continue
        cons_high, cons_low = box

        if not np.isfinite(rvol[i]) or rvol[i] < vol_mult:
            continue

        spike_mid = (high[i] + lows[i]) / 2.0
        a_rng = i - cons_bars
        b_rng = i - 1
        spring_l = has_spring_long(lows, high, close, open_, vol, vma20, cons_low, cons_high, a_rng, b_rng)
        spring_s = has_spring_short(lows, high, close, open_, vol, vma20, cons_low, cons_high, a_rng, b_rng)

        # Long: bullish expansion, close above box
        long_setup = (
            close[i] > cons_high
            and close[i] > open_[i]
            and d_ok_long[i]
            and (not require_spring or spring_l)
        )
        # Short: bearish expansion, close below box
        short_setup = (
            close[i] < cons_low
            and close[i] < open_[i]
            and d_ok_short[i]
            and (not require_spring or spring_s)
        )

        if not long_setup and not short_setup:
            continue

        side = "LONG" if long_setup and not short_setup else ("SHORT" if short_setup and not long_setup else "")
        if side == "":
            # both fired — skip ambiguous bar
            continue

        tsize = _tick_size(float(close[i]), tick if tick is not None else 0.05)
        if side == "LONG":
            entry_stop = high[i] + tsize
            sl = spike_mid
            if sl >= entry_stop:
                continue
            risk = entry_stop - sl
        else:
            entry_stop = lows[i] - tsize
            sl = spike_mid
            if sl <= entry_stop:
                continue
            risk = sl - entry_stop

        if risk <= 0:
            continue

        planned_tp_8r = entry_stop + 8.0 * risk if side == "LONG" else entry_stop - 8.0 * risk

        # Entry bar: i+1
        j = i + 1
        o = open_[j]
        hi = high[j]
        lo = lows[j]

        if side == "LONG":
            if hi < entry_stop:
                continue
            if o > entry_stop:
                entry_price = float(o)
            else:
                entry_price = float(entry_stop)
        else:
            if lo > entry_stop:
                continue
            if o < entry_stop:
                entry_price = float(o)
            else:
                entry_price = float(entry_stop)

        # Recompute risk from actual entry vs same SL (mid spike)
        if side == "LONG":
            risk1 = entry_price - sl
        else:
            risk1 = sl - entry_price
        if risk1 <= 0:
            continue

        planned_tp_8r = entry_price + 8.0 * risk1 if side == "LONG" else entry_price - 8.0 * risk1

        exit_price = np.nan
        exit_reason = ""
        exit_idx = j
        bars_held = 0

        k = j
        while k < n:
            bars_held += 1
            ok = k
            c = close[ok]
            em = ema9[ok]
            rv = rvol[ok]
            o_ = open_[ok]
            hi_ = high[ok]
            lo_ = lows[ok]

            if side == "LONG":
                if lo_ <= sl:
                    exit_price = sl
                    exit_reason = "stop_mid_spike"
                    exit_idx = ok
                    break
                if np.isfinite(em) and c < em:
                    exit_price = float(c)
                    exit_reason = "ema9_trail"
                    exit_idx = ok
                    break
                if exhaustion_exit and np.isfinite(rv) and rv >= vol_mult and c < o_:
                    exit_price = float(c)
                    exit_reason = "exhaustion_vol_spike"
                    exit_idx = ok
                    break
            else:
                if hi_ >= sl:
                    exit_price = sl
                    exit_reason = "stop_mid_spike"
                    exit_idx = ok
                    break
                if np.isfinite(em) and c > em:
                    exit_price = float(c)
                    exit_reason = "ema9_trail"
                    exit_idx = ok
                    break
                if exhaustion_exit and np.isfinite(rv) and rv >= vol_mult and c > o_:
                    exit_price = float(c)
                    exit_reason = "exhaustion_vol_spike"
                    exit_idx = ok
                    break

            k += 1

        if np.isnan(exit_price):
            exit_price = float(close[n - 1])
            exit_reason = "data_end"
            exit_idx = n - 1

        pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        r_multiple = pnl / risk1

        trades.append(
            TradeResult(
                side=side,
                entry_time=dates.iloc[j],
                entry=float(entry_price),
                sl=float(sl),
                spike_idx=i,
                spike_time=dates.iloc[i],
                cons_high=cons_high,
                cons_low=cons_low,
                cons_bars=cons_bars,
                rvol_spike=float(rvol[i]),
                planned_tp_8r=float(planned_tp_8r),
                exit_time=dates.iloc[exit_idx],
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                bars_held=bars_held,
                pnl_per_unit=float(pnl),
                r_multiple=float(r_multiple),
                had_spring=bool(spring_l if side == "LONG" else spring_s),
            )
        )
        in_trade_until = exit_idx

    rows = [
        {
            "strategy": "institutional_expansion_pivot",
            "side": t.side,
            "entry_time": t.entry_time,
            "entry": t.entry,
            "sl_mid_spike": t.sl,
            "spike_time": t.spike_time,
            "cons_high": t.cons_high,
            "cons_low": t.cons_low,
            "cons_bars": t.cons_bars,
            "rvol_spike": t.rvol_spike,
            "planned_tp_8r": t.planned_tp_8r,
            "exit_time": t.exit_time,
            "exit": t.exit_price,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
            "pnl_per_unit": t.pnl_per_unit,
            "r_multiple": t.r_multiple,
            "had_spring": t.had_spring,
        }
        for t in trades
    ]
    trades_df = pd.DataFrame(rows)
    return trades_df, df


def main() -> None:
    p = argparse.ArgumentParser(description="Institutional Expansion Pivot 15m backtest")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("data/NIFTY500/AFCONS_15Min.csv"),
        help="15m OHLCV CSV with columns date,open,high,low,close,volume",
    )
    p.add_argument("--vol-mult", type=float, default=2.0, help="Min RVOL vs 20-VMA on spike candle")
    p.add_argument("--consolidation-bars", type=int, default=12, help="Bars in box before spike (10–15 typical)")
    p.add_argument(
        "--max-range-pct",
        type=float,
        default=0.012,
        help="Max (high-low)/mid of consolidation box as fraction (e.g. 0.012 = 1.2%%)",
    )
    p.add_argument("--tick", type=float, default=None, help="Tick size; default auto by price")
    p.add_argument("--daily-filter", action="store_true", help="Long above prior daily EMA20; short below")
    p.add_argument("--require-spring", action="store_true", help="Require spring/upthrust in box")
    p.add_argument(
        "--exhaustion-exit",
        action="store_true",
        help="Also exit on counter-trend bar with RVOL >= vol-mult",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output trades CSV (default: data/backtesting/iep_<stem>_<timestamp>.csv)",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = load_ohlcv(args.csv)
    df = add_daily_context(df)

    cons = args.consolidation_bars
    if cons < 10 or cons > 15:
        raise SystemExit("--consolidation-bars should be between 10 and 15")

    trades_df, _ = run_backtest(
        df,
        vol_mult=args.vol_mult,
        cons_bars=cons,
        max_range_pct=args.max_range_pct,
        tick=args.tick,
        use_daily_filter=args.daily_filter,
        require_spring=args.require_spring,
        exhaustion_exit=args.exhaustion_exit,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out
    if out is None:
        out = Path("data/backtesting") / f"iep_{args.csv.stem}_{ts}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out, index=False)

    n_tr = len(trades_df)
    win = (trades_df["pnl_per_unit"] > 0).sum() if n_tr else 0
    total = float(trades_df["pnl_per_unit"].sum()) if n_tr else 0.0
    print(f"Trades: {n_tr}  Win rate: {win / n_tr:.2%}" if n_tr else "Trades: 0")
    print(f"Sum pnl/unit: {total:.4f}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
