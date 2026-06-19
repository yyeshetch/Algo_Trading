# Removed Stocks Dashboard Tabs — Logic Reference

These three tabs were removed from the F&O Stocks dashboard (`/stocks`) on 2026-06-15. The logic below is preserved for reference. CLI commands and underlying modules may still exist where noted.

---

## 1. Top Picks

**Former module:** `src/intraday_engine/research/top_picks_aggregator.py`  
**Former API:** `GET /api/stocks/top-picks`  
**Storage:** None (read-only aggregator over cached scanner outputs)

### Purpose

Rank actionable long/short ideas by **confluence**: a stock appears only when **≥2 independent cached sources** agree on direction (default). Each pick includes a tight intraday trade plan from cached 15-min OHLCV.

### Input sources (each contributes at most once per side)

| Source key | Direction | Score weight | Logic |
|------------|-----------|--------------|-------|
| `daily_rs` | Long | 15–30 | From `relative_strength_<date>.json`: NIFTY 500 names outperforming NIFTY over ~20 days |
| `intraday_rs` | Long / Short | 15–35 | From `intraday_rs_<date>.json`: today’s open→last move vs NIFTY 50; \|excess\| > 0.3% |
| `intraday_scan_wl` | Long / Short | 15 | From `fno_intraday_signals_<date>.json` buy/sell **watchlists** (60m+15m RSI) |
| `intraday_scan_signal` | Long / Short | 25 | From same file: confirmed **5m BUY/SELL signals** |
| `mtf_watchlist` | Long / Short | 10–30 | From `nifty500_tomorrow_watchlist_<date>.json` by `next_day_bias` |
| `silent_acc` | Long only | 10–25 | From `silent_accumulation_<date>.json` (OBV/CMF/delivery score) |
| `sig_15m` | Long / Short | 25 | Latest per-stock BUY/SELL from today’s `FnO_Signals.csv` (15-min engine) |

### Conflict resolution

If the same symbol appears on both bull and bear buckets, keep the side with the **higher summed source score**.

### Trade plan (per surviving pick)

From `data/NIFTY500/{SYMBOL}_15Min.csv`:

- **Entry:** latest 15-min close  
- **SL (BUY):** max of prior two bars’ lows (fallback: entry × 0.996)  
- **SL (SELL):** min of prior two bars’ highs (fallback: entry × 1.004)  
- **Target:** entry ± 2 × risk (fixed **1:2 R:R**)  
- **Risk %:** \|entry − SL\| / entry × 100  

Optional filter: drop picks where risk % exceeds `max_risk_pct` (dashboard default 0.75%).

### Output

- `longs` / `shorts`: ranked by `(score desc, risk_pct asc)`  
- `sources_present`: which upstream caches existed  
- `macro`: FII/DII 5/10/30d nets (informational only)

---

## 2. 15-min Signals

**Former API:** `GET /api/stocks/signals`, `POST /api/stocks/refresh`  
**Runner:** `src/intraday_engine/engine/stock_cycle_runner.py` → `run_stocks_15min_cycle()`  
**Engine:** `src/intraday_engine/engine/stock_signal_engine.py`  
**CLI (still available):** `python -m intraday_engine.main --stocks-15min [--stocks-limit N]`  
**Storage:** `data/signals/date=YYYY-MM-DD/FnO_Signals.csv` + analysis snapshots per stock

### Data pipeline

1. For each F&O stock (optional limit): fetch **15-min** spot, nearest future, ATM CE/PE (with OI when available) from Kite.  
2. Merge into a rolling bar frame; persist snapshots via `DataStore.save_snapshots()`.  
3. For **each completed 15-min bar** in the session, run up to **four parallel strategy evaluators** and append all rows to signals CSV.

### Strategies (four rows per bar timestamp)

#### A. DIRECTIONAL (`_analyze_frame`)

Classic index-style scoring on the growing 15-min frame:

- Features: spot vs open/VWAP, futures premium, option CE/PE behaviour, OI changes  
- Day bias (`probable_day_bias`), momentum, support/resistance (rolling structure)  
- Sideways filter via `is_sideways_day` (range %, bias, momentum, breakout flags)  
- Weighted `score_signal()` → `build_trade_plan()` → BUY / SELL / NO_TRADE  
- RSI gate: BUY rejected if 15m RSI < 45; SELL rejected if 15m RSI > 55 (tunable)

#### B. FAILED_BIAS (`_analyze_failed_bias_frame`)

Trades **failed** opening bias (e.g. bearish open that reverses):

- Detects setup from prior bars + 15m history  
- Emits FAILED_BIAS strategy rows with `trigger_timestamp` / `fresh_until` for lifecycle

#### C. VOL_EMA_BULLISH (`_analyze_vol_ema_bullish_frame`)

Large green volume candle near 10/20/200 EMA on extended 15m closes; prior bar RSI > 55.

#### D. OVERBOUGHT_RSI (`_analyze_overbought_rsi_frame`)

Overbought RSI mean-reversion variant (SELL-biased).

### Dashboard display logic

- Load today’s `FnO_Signals.csv` for all stocks  
- Group by underlying; for each **strategy identity key**, show the **first** BUY/SELL trigger row of the day  
- Identity key: `strategy|strategy_label|signal|timestamp` (FAILED_BIAS uses trigger_timestamp)  
- Sort BUY/SELL tables by confidence; show RSI 60m/15m, fut OI %, RS vs NIFTY, ORB/pinbar flags, volume bias columns, reasons

### Scheduler

`--stocks-15min-scheduler` runs `run_every_15_minutes()` aligned to 15-min boundaries (default 50 stocks).

---

## 3. Intraday Signals

**Former module:** `src/intraday_engine/scanner/fno_intraday_buy_scanner.py`  
**Former API:** `GET /api/stocks/intraday-signals`, `POST /api/stocks/intraday-signals/refresh`  
**Storage:** `data/fno_intraday_signals_{YYYY-MM-DD}.json`  
**Tunables:** `config.json` → `fno_intraday_buy_scanner`

### Purpose

Separate **RSI watchlists** (60m + 15m) and **5m confirmation signals** for liquid F&O names. Signals fire only if the symbol is already on the matching watchlist for the active hourly slot.

### Watchlist rules

- **BUY watchlist:** 60m RSI > 55 **and** 15m RSI > 55 (at slot start)  
- **SELL watchlist:** 60m RSI < 45 **and** 15m RSI < 45  
- Slots anchored to market open: **9:15–10:15, 10:15–11:15, … 15:15–15:30**  
- Active slot = current hour bucket (or last slot for historical dates)

### Per-stock context (`_fetch_intraday_context`)

Parallel scan (default 5 workers) fetches for each symbol:

- 5m, 15m, 60m spot history (rate-limited)  
- Previous day high/low (PDH/PDL)  
- Opening range (first 5m bar high/low)  
- Session VWAP, volume vs 20-bar SMA, swing high/low over 10 bars  
- Nearest future OI change (5m)  
- Volume bias / liquidity grab state on 5m bars  

### BUY signal (all must be true)

Symbol on **BUY watchlist** plus:

| Gate | Condition |
|------|-----------|
| ORB 5m | Close > opening-range high |
| PDH | Close > previous day high |
| VWAP | Close > session VWAP |
| RSI 5m | RSI > 55 |
| Volume | Last 5m volume ≥ 2× 20-bar SMA |
| Fut OI | Latest OI < previous bar (drop) |
| Swing | Close > 10-bar swing high |

### SELL signal (all must be true)

Symbol on **SELL watchlist** plus symmetric breakdown gates (OR low, PDL, below VWAP, RSI 5m < 45, volume 2×, fut OI rise, swing low breakdown).

### Output JSON shape

- `buy_watchlist` / `sell_watchlist` for active slot  
- `buy_signals` / `sell_signals` filtered rows  
- `signals`: full universe with confirmation booleans  
- `watchlist_timeline`: per-slot watchlists for the day  
- Metadata: `last_run_at`, `next_refresh_at`, `failed_symbols`

### Shared refresh

Former **Refresh All Tabs** also ran `run_fno_intraday_scan` alongside the 15-min cycle and tomorrow watchlist.

---

## Files removed with the tabs

| File | Reason |
|------|--------|
| `src/intraday_engine/research/top_picks_aggregator.py` | Top Picks tab only |
| `src/intraday_engine/scanner/fno_intraday_buy_scanner.py` | Intraday Signals tab only |

## Files kept (CLI / other tabs)

| File | Still used by |
|------|----------------|
| `stock_cycle_runner.py` | CLI `--stocks-15min`, `--stocks-15min-scheduler` |
| `stock_signal_engine.py` | Above + writes OHLCV/signals used by other research scanners |

## API endpoints removed

- `GET /api/stocks/signals`  
- `POST /api/stocks/refresh`  
- `GET /api/stocks/intraday-signals`  
- `POST /api/stocks/intraday-signals/refresh`  
- `GET /api/stocks/top-picks`  

`POST /api/stocks/refresh-all` was simplified to run **Tomorrow watchlist** only.
