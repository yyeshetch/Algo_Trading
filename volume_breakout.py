import numpy as np
import pandas as pd

QUANTITY = 10
RSI_PERIOD = 14


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100.0 - (100.0 / (1.0 + rs))).astype(float)


df = pd.read_csv("data/NIFTY500/AFCONS_15Min.csv")

open_ = pd.to_numeric(df["open"], errors="coerce")
high = pd.to_numeric(df["high"], errors="coerce")
low = pd.to_numeric(df["low"], errors="coerce")
close = pd.to_numeric(df["close"], errors="coerce")
volume = pd.to_numeric(df["volume"], errors="coerce")

# Green: close > open | Red: close < open | Doji if equal
df["bar_color"] = pd.Series("", index=df.index, dtype="object")
df.loc[close > open_, "bar_color"] = "green"
df.loc[close < open_, "bar_color"] = "red"
df.loc[close == open_, "bar_color"] = "doji"

# 20 SMA on volume
df["volume_sma_20"] = volume.rolling(window=20, min_periods=20).mean()

# Volume multiplier = volume / 20 SMA volume
df["volume_multiplier"] = volume / df["volume_sma_20"]

df["Support"] = np.nan
df["Resistance"] = np.nan
_vol_spike = df["volume_multiplier"] > 3.5
_red = df["bar_color"] == "red"
_green = df["bar_color"] == "green"
df.loc[_vol_spike & _red, "Resistance"] = high.loc[_vol_spike & _red]
df.loc[_vol_spike & _green, "Support"] = low.loc[_vol_spike & _green]

# Trail stops using green candles with volume ≥ 2× 20-SMA volume (low as support)
df["Support_2x"] = np.nan
_vol_2x = df["volume_multiplier"] >= 2.0
df.loc[_vol_2x & _green, "Support_2x"] = low.loc[_vol_2x & _green]

# RSI 15m (Wilder, period 14)
df["rsi"] = wilder_rsi(close)

# Hourly: last 15m close per calendar-hour bucket → RSI + EMA 200 (mapped to each 15m row)
_hour_key = pd.to_datetime(df["date"], errors="coerce").dt.floor("h")
_hourly_close = df.groupby(_hour_key, sort=True)["close"].last().astype(float)
_rsi_h_series = wilder_rsi(_hourly_close)
df["rsi_hourly"] = _hour_key.map(_rsi_h_series)
_hourly_ema200 = _hourly_close.ewm(span=200, adjust=False).mean()
df["ema_200_hourly"] = _hour_key.map(_hourly_ema200)

# EMAs on close (15m)
df["ema_20"] = close.ewm(span=20, adjust=False).mean()
df["ema_50"] = close.ewm(span=50, adjust=False).mean()
df["ema_200"] = close.ewm(span=200, adjust=False).mean()

# Daily EMA on last close per calendar day; each 15m row uses prior completed day EMA (shift 1)
_day = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
# Session VWAP (resets each calendar day): cumsum(TP * V) / cumsum(V), TP = (H+L+C)/3
_tp_v = (high + low + close) / 3.0
_pv_v = _tp_v * volume
df["vwap"] = _pv_v.groupby(_day).cumsum() / volume.groupby(_day).cumsum().replace(0, np.nan)

_daily_close = df.groupby(_day, sort=True)["close"].last().astype(float)
_daily_ema20 = _daily_close.ewm(span=20, adjust=False).mean()
_daily_ema50 = _daily_close.ewm(span=50, adjust=False).mean()
df["daily_ema_20"] = _day.map(_daily_ema20.shift(1))
df["daily_ema_50"] = _day.map(_daily_ema50.shift(1))

_daily_rsi = wilder_rsi(_daily_close)
df["rsi_daily"] = _day.map(_daily_rsi.shift(1))

_daily_high = df.groupby(_day, sort=True)["high"].max().astype(float)
df["pdh"] = _day.map(_daily_high.shift(1))

# Last confirmed resistance before this bar (must break above this level)
df["recent_resistance_prior"] = df["Resistance"].ffill().shift(1)

long_trades: list[dict] = []
n = len(df)
dates = pd.to_datetime(df["date"], errors="coerce")
opens = df["open"].to_numpy(dtype=float)
lows = df["low"].to_numpy(dtype=float)
closes = df["close"].to_numpy(dtype=float)
support_trail = df["Support_2x"].to_numpy(dtype=float)
bar_colors = df["bar_color"].to_numpy()
rsis = df["rsi"].to_numpy(dtype=float)
rsis_h = df["rsi_hourly"].to_numpy(dtype=float)
rr_prior = df["recent_resistance_prior"].to_numpy(dtype=float)
ema20 = df["ema_20"].to_numpy(dtype=float)
ema50 = df["ema_50"].to_numpy(dtype=float)
d_ema20 = df["daily_ema_20"].to_numpy(dtype=float)
d_ema50 = df["daily_ema_50"].to_numpy(dtype=float)
rsis_d = df["rsi_daily"].to_numpy(dtype=float)
ema200 = df["ema_200"].to_numpy(dtype=float)
ema200_h = df["ema_200_hourly"].to_numpy(dtype=float)
vwap = df["vwap"].to_numpy(dtype=float)
pdh = df["pdh"].to_numpy(dtype=float)

i = 0
while i < n:
    if (
        bar_colors[i] == "green"
        and np.isfinite(rr_prior[i])
        and np.isfinite(closes[i])
        and closes[i] > rr_prior[i]
        and np.isfinite(rsis[i])
        and np.isfinite(rsis_h[i])
        and rsis[i] > 55.0
        and rsis[i] <= 72.0
        and rsis_h[i] > 55.0
        and np.isfinite(rsis_d[i])
        and rsis_d[i] > 55.0
        and np.isfinite(ema20[i])
        and np.isfinite(ema50[i])
        and closes[i] > ema20[i]
        and closes[i] > ema50[i]
        and np.isfinite(d_ema20[i])
        and np.isfinite(d_ema50[i])
        and closes[i] > d_ema20[i]
        and closes[i] > d_ema50[i]
        and np.isfinite(ema200[i])
        and np.isfinite(ema200_h[i])
        and closes[i] > ema200[i]
        and closes[i] > ema200_h[i]
        and np.isfinite(vwap[i])
        and closes[i] > vwap[i]
        and np.isfinite(pdh[i])
        and closes[i] > pdh[i]
    ):
        entry = float(closes[i])
        initial_sl = float(lows[i])
        risk = entry - initial_sl
        if risk > 0:
            # +3R reference only; exits trail on 2×-volume Support — no fixed take-profit at 3R
            target = entry + 3.0 * risk
            trail_sl = initial_sl
            entry_time = dates.iloc[i]

            exit_price = np.nan
            exit_time = None
            exit_reason = ""
            j = i + 1
            while j < n:
                # trail_sl here = stops in force at the open of bar j (2× vol supports through j-1)
                opn = opens[j]
                lo = lows[j]

                if np.isfinite(opn) and opn < trail_sl:
                    exit_price = float(opn)
                    exit_time = dates.iloc[j]
                    exit_reason = "gap_below_trail"
                    break

                if np.isfinite(lo) and lo <= trail_sl:
                    exit_price = trail_sl
                    exit_time = dates.iloc[j]
                    exit_reason = "trail_sl"
                    break

                # After bar j closes: new 2×-volume Support can tighten trail for bar j+1 onward only
                s = support_trail[j]
                if np.isfinite(s):
                    trail_sl = max(trail_sl, float(s))

                j += 1

            if exit_time is None:
                exit_price = float(closes[n - 1])
                exit_time = dates.iloc[n - 1]
                exit_reason = "data_end"

            pnl_per_unit = float(exit_price) - entry
            pnl = pnl_per_unit * QUANTITY

            long_trades.append(
                {
                    "side": "LONG",
                    "quantity": QUANTITY,
                    "entry_time": entry_time,
                    "entry": entry,
                    "resistance_broken": float(rr_prior[i]),
                    "initial_sl": initial_sl,
                    "target_1_3": target,
                    "exit_time": exit_time,
                    "exit": exit_price,
                    "exit_reason": exit_reason,
                    "trail_sl_at_exit": trail_sl,
                    "rsi_m15_at_entry": float(rsis[i]),
                    "rsi_hourly_at_entry": float(rsis_h[i]),
                    "rsi_daily_at_entry": float(rsis_d[i]),
                    "daily_ema_20_at_entry": float(d_ema20[i]),
                    "daily_ema_50_at_entry": float(d_ema50[i]),
                    "ema_200_m15_at_entry": float(ema200[i]),
                    "ema_200_hourly_at_entry": float(ema200_h[i]),
                    "vwap_at_entry": float(vwap[i]),
                    "pdh_at_entry": float(pdh[i]),
                    "pnl_per_unit": pnl_per_unit,
                    "pnl": pnl,
                }
            )
            i = j + 1
            continue
    i += 1

trades_df = pd.DataFrame(long_trades)

df.to_csv("AFCONS_15Min_with_indicators.csv", index=False)
trades_df.to_csv("AFCONS_long_trades.csv", index=False)