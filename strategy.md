# Signal Strategy Logic

This document describes all logic used to identify BUY/SELL signals for **indices** (NIFTY, BANKNIFTY) and **stocks** (F&O stocks).

---

## 1. Index Signals (NIFTY, BANKNIFTY)

**Engine:** `DirectionEngine`  
**Data:** 5-minute candles (spot, futures, ATM CE/PE with OI)  
**Cycle:** Runs every 5 minutes at candle boundaries

### 1.1 Data Requirements

- Minimum **3 completed bars** required. Otherwise: `NO_TRADE` with "Insufficient bars."
- Uses spot, futures, ATM call, ATM put (with OI when available).

### 1.2 Feature Computation

| Feature | Logic |
|---------|-------|
| `spot_above_open` / `spot_below_open` | Spot LTP vs session open |
| `spot_above_vwap` / `spot_below_vwap` | Spot LTP vs session VWAP |
| `fut_strength_pct` | % change of futures LTP vs spot LTP |
| `call_change_pct` / `put_change_pct` | ATM option premium change (prev bar → current) |
| `spot_change_pct` | Spot change (prev bar → current) |
| `fut_oi_bullish` | OI up + price up = longs adding; OI down + price up = short covering |
| `fut_oi_bearish` | OI up + price down = shorts adding; OI down + price down = long covering |
| `call_oi_change_pct` / `put_oi_change_pct` | ATM CE/PE OI change (first vs last candle) |

### 1.3 Support & Resistance

- **Lookback:** `LOOKBACK_BARS` (default 20) completed bars
- **Support:** Min of `spot_low` in lookback window
- **Resistance:** Max of `spot_high` in lookback window

### 1.4 Momentum

- **Window:** Last 4 bars
- **UP:** Move > +0.20%
- **DOWN:** Move < -0.20%
- **NEUTRAL:** Otherwise

### 1.5 Day Bias

| Bias | Condition |
|------|-----------|
| BULLISH_DAY | More bullish votes (spot above open+VWAP, fut strength > 3%, call expansion, fut OI bullish, CE OI up & PE OI down) |
| BEARISH_DAY | More bearish votes |
| NEUTRAL_DAY | Tie |

### 1.6 Structure Checks

- **Breakout:** `spot > resistance`
- **Breakdown:** `spot < support`
- **Follow-through:** `|spot_change_pct| > 0.06` (6%)
- **Mid-range:** Distance from range mid < 18% of range size
- **Stop too wide:** Min(long_stop_pct, short_stop_pct) > `MAX_STOP_PCT` (0.45% long, 0.65% short)

### 1.7 Sideways Day Filter

Returns `NO_TRADE` when:

1. **Narrow range:** `range_pct < MIN_DAY_RANGE_PCT` (0.40%)
2. **Chop + no break + neutral:** Chop around open/VWAP, no breakout/breakdown, neutral bias & momentum
3. **Chop + no follow-through:** Chop with no structure break and no follow-through

### 1.8 Scoring Weights

| Factor | Weight | Bullish | Bearish |
|--------|--------|---------|---------|
| Spot vs open/VWAP | 0.18 | Above both | Below both |
| Futures strength | 0.12 | fut_strength > 3% | fut_strength < -3% |
| Futures OI | 0.10 | Longs adding / short covering | Shorts adding / long covering |
| Options expansion | 0.15 | Call up, put down | Put up, call down |
| Options OI | 0.10 | CE OI up >2%, PE OI down <-2% | PE OI up >2%, CE OI down <-2% |
| Breakout follow-through | 0.18 | Breakout + follow-through | Breakdown + follow-through |
| Momentum | 0.12 | UP | DOWN |
| Structure quality | 0.05 | BULLISH_DAY bias | BEARISH_DAY bias |

**Penalties:** Chop around open/VWAP (0.10), conflicting options (0.08), no follow-through (0.08), neutral momentum (0.05), mid-range (0.05), stop too wide (0.15).

### 1.9 Trade Plan

- **BUY:** `final_score > 0` and `confidence >= min_confidence` (0.70, or 0.85 on NEUTRAL_DAY)
- **SELL:** `final_score < 0` and `confidence >= min_confidence`
- **Entry:** Spot LTP
- **Stop:** Support (long) / Resistance (short)
- **Target:** Resistance (long) / Support (short), or entry ± risk × `MIN_RR` (1.8)
- **Option:** ATM CE for BUY, ATM PE for SELL

**Risk filters:** RR ≥ `MIN_RR`, stop % ≤ `MAX_STOP_PCT` / `MAX_STOP_PCT_SHORT`.

---

## 2. Stock Signals (15-min Cycle)

**Engine:** `stock_signal_engine.run_stock_cycle`  
**Data:** 15-minute candles (spot, futures, optional ATM CE/PE with OI)  
**Cycle:** Runs every 15 minutes for all F&O stocks

Uses the **same analysis logic** as indices:

- Same features, support/resistance, momentum, bias, sideways filter, scoring, trade plan
- Uses `lookback_bars`, `min_confidence`, `max_stop_pct`, etc. from config
- Fallback: if options unavailable, uses spot + futures only (features adapt)

---

## 3. Stock Scanner (F&O Scanner Dashboard)

**Purpose:** Rank F&O stocks by directional conviction for screening, not execution.

### 3.1 Data

- 15-min candles: spot, futures, ATM CE/PE (with OI)
- Session open → last completed candle

### 3.2 Metrics

| Metric | Formula |
|--------|---------|
| `spot_change_pct` | (close - open) / open × 100 |
| `call_oi_change_pct` / `put_oi_change_pct` | (last - first) / first × 100 |
| `call_premium_change_pct` / `put_premium_change_pct` | (close - open) / open × 100 |
| `spot_volume` / `futures_volume` | Sum of volumes |

### 3.3 Scoring (Scanner)

| Signal | Condition | Score |
|--------|-----------|-------|
| Spot | spot_chg > 0.3% | +0.25 bullish |
| Spot | spot_chg < -0.3% | +0.25 bearish |
| CE OI | call_oi_chg > 2% | +0.15 bullish |
| CE OI | call_oi_chg < -2% | +0.15 bearish |
| PE OI | put_oi_chg > 2% | +0.15 bearish |
| PE OI | put_oi_chg < -2% | +0.15 bullish |
| CE premium | call_prem_chg > 2% | +0.15 bullish |
| CE premium | call_prem_chg < -2% | +0.15 bearish |
| PE premium | put_prem_chg > 2% | +0.15 bearish |
| PE premium | put_prem_chg < -2% | +0.15 bullish |
| Volume | Normalized spot + fut volume | +0.15 × vol_score (both directions) |

**Final:** `bullish - bearish`; `direction` = BULLISH | BEARISH | NEUTRAL. Ranked by `|final_score|`.

---

## 4. ORB (Opening Range Breakout) – Stocks Only

**Engine:** `orb_scanner.run_orb_scan`  
**Data:** 15-min candles; first candle = Opening Range (OR)

### 4.1 Opening Range

- **OR High:** High of first 15-min candle (9:15–9:30)
- **OR Low:** Low of first 15-min candle
- Cached in `data/orb_ranges_YYYY-MM-DD.json`

### 4.2 Variation

- **Variation:** ±0.2%
- **Upper break level:** `or_high × (1 - 0.002)` = OR high − 0.2%
- **Lower break level:** `or_low × (1 + 0.002)` = OR low + 0.2%

### 4.3 Signal Logic

- **BUY:** Latest 15-min close ≥ upper break level
- **SELL:** Latest 15-min close ≤ lower break level
- **NO_TRADE:** Close between lower and upper break levels

---

## 5. Pinbar Scanner – Stocks Only

**Engine:** `orb_scanner.run_pinbar_scan`  
**Data:** Last completed 15-min candle (OHLC)

### 5.1 Bullish Pinbar (BUY)

- Long lower wick: `lower_wick ≥ 2 × body`
- Small body: `body ≤ 0.35 × range`
- Upper wick small: `upper_wick ≤ 0.5 × lower_wick`
- Interpretation: Rejection at low, potential reversal up

### 5.2 Bearish Pinbar (SELL)

- Long upper wick: `upper_wick ≥ 2 × body`
- Small body: `body ≤ 0.35 × range`
- Lower wick small: `lower_wick ≤ 0.5 × upper_wick`
- Interpretation: Rejection at high, potential reversal down

### 5.3 Definitions

- `body` = |close − open|
- `range` = high − low
- `lower_wick` = min(open, close) − low
- `upper_wick` = high − max(open, close)

---

## 6. Config Parameters (Env)

| Variable | Default | Description |
|----------|---------|--------------|
| `LOOKBACK_BARS` | 20 | Bars for support/resistance |
| `MIN_RR` | 1.8 | Minimum reward:risk |
| `MIN_CONFIDENCE` | 0.70 | Min confidence for BUY/SELL |
| `MIN_CONFIDENCE_NEUTRAL_DAY` | 0.85 | Min confidence on NEUTRAL_DAY |
| `MAX_STOP_PCT` | 0.45 | Max stop % (long) |
| `MAX_STOP_PCT_SHORT` | 0.65 | Max stop % (short) |
| `OPTION_STOP_RATIO` | 0.35 | Option SL as % of premium |
| `MIN_DAY_RANGE_PCT` | 0.40 | Min day range % (sideways filter) |

---

## 7. Summary by Asset Type

| Asset | Data | Signal Types | Key Logic |
|-------|------|--------------|-----------|
| NIFTY, BANKNIFTY | 5-min spot, fut, CE, PE | BUY, SELL, NO_TRADE | Features + scoring + sideways filter + trade plan |
| F&O Stocks (15-min) | 15-min spot, fut, CE, PE | BUY, SELL, NO_TRADE | Same as indices |
| F&O Stocks (Scanner) | 15-min session | BULLISH, BEARISH, NEUTRAL | OI + premium + volume scoring |
| F&O Stocks (ORB) | 15-min OR + latest | BUY, SELL, NO_TRADE | Close vs OR ± 0.2% |
| F&O Stocks (Pinbar) | Last 15-min candle | BUY, SELL | Wick/body ratio patterns |
