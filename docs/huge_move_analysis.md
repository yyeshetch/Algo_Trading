# Huge Move Analysis & Market Condition Detection

## 1. Today's NIFTY 264-Point Fall (Post 13:40) – What Constituted It?

Based on market reports (March 19, 2025), the sharp fall was attributed to:

| Factor | Impact |
|--------|--------|
| **Crude oil spike** | Brent crude +8.5% to ~$116/barrel after Israel-Iran tensions |
| **Geopolitical** | Iran-Israel conflict escalation ("energy war") |
| **Fed hawkish stance** | Rate-cut expectations pushed back |
| **Rupee depreciation** | Currency weakness added to FII selling pressure |
| **FII selling** | Institutional exit often concentrated post-lunch (13:30–14:30) |

**Intraday pattern:** Large moves often occur 13:40–14:30 when:
- FII activity picks up post US pre-market
- Low liquidity (lunch hour) amplifies moves
- Stop-loss cascades trigger once key levels break

---

## 2. Market Condition Detection (Normal vs Pressure)

### 2.1 Regime Classification

| Regime | Indicators | Interpretation |
|--------|------------|-----------------|
| **NORMAL** | Range < 0.8%, PCR 0.7–1.3, low volume vs avg | Choppy, avoid aggressive trades |
| **PRESSURE (Bearish)** | PCR > 1.3, PE OI building, put premium expanding | Downside risk elevated |
| **PRESSURE (Bullish)** | PCR < 0.7, CE OI building, call premium expanding | Upside potential |
| **VOLATILITY SPIKE** | Range > 1.5%, volume 2× avg, premium expansion | Large move in progress or imminent |

### 2.2 Indicators to Add

1. **Session range vs average** – `range_pct` vs 5-day avg range
2. **Volume ratio** – Spot + futures volume vs 5-day avg
3. **PCR (OI-based)** – PE_OI / CE_OI across 5–10 strikes
4. **PCR (Volume-based)** – PE_vol / CE_vol (faster signal)
5. **Premium expansion** – ATM CE/PE % change vs session open
6. **OI shift** – CE OI vs PE OI change (build-up direction)

---

## 3. Option Chain Indicators for Huge Moves (100+ Points)

### 3.1 Pre-Move Signals

| Signal | Bullish Huge Up | Bearish Huge Down |
|--------|-----------------|-------------------|
| **PCR (OI)** | < 0.6 (extreme call buying) | > 1.5 (extreme put buying) |
| **PCR (Volume)** | PE_vol/CE_vol < 0.5 | PE_vol/CE_vol > 2.0 |
| **OI build-up** | CE OI +5%+, PE OI -5%+ | PE OI +5%+, CE OI -5%+ |
| **Premium** | Call +15%+, Put -15%+ | Put +15%+, Call -15%+ |
| **Max Pain deviation** | Spot > Max Pain + 50 pts | Spot < Max Pain - 50 pts |
| **Strike concentration** | High CE OI at OTM strikes | High PE OI at OTM strikes |

### 3.2 Max Pain

- Strike where total option writer payout is minimized
- Price often gravitates toward Max Pain near expiry
- **Deviation > 50 pts** suggests potential snap-back or continuation

### 3.3 Gamma Squeeze (Expiry Day)

- High OI at ATM → dealers hedge aggressively
- Break of ATM can trigger cascading delta hedging → acceleration

---

## 4. Implementation Plan

### 4.1 Option Chain Capture (5–10 Strikes)

- Fetch ATM ± 5 to ± 10 strikes for **nearest weekly expiry** (not just expiry day)
- Store: timestamp, spot, strike, CE/PE OI, volume, LTP, OHLC
- Persist to `data/option_chain_{underlying}_{date}.jsonl` (append each poll)

### 4.2 Market Condition Detector

- Compute: range_pct, volume_ratio, PCR_oi, PCR_vol, premium_expansion
- Classify: NORMAL | PRESSURE_BULLISH | PRESSURE_BEARISH | VOLATILITY_SPIKE
- Output: regime + confidence

### 4.3 Huge Move Predictor

- Use stored option chain history
- Compute: PCR, OI change, premium change, max pain
- Score: P(huge_up), P(huge_down), P(no_move)
- Threshold: 100+ points spot move or 100+ points option premium move

### 4.4 Scheduler Integration

- Set `CAPTURE_OPTION_CHAIN=1` in `.env` to capture option chain every 5 min with the direction engine
- Set `OPTION_STRIKES=5` (default) or `10` for number of strikes each side of ATM
- Run huge move prediction: `python -m intraday_engine.cli.main --huge-move`
- Capture only (no prediction): `python -m intraday_engine.cli.main --capture-option-chain`
