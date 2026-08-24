# Intraday Direction Engine (Zerodha Kite)

Production-oriented, explainable, rule-based intraday signal engine that runs every 5 minutes and emits:
- `data/signals.csv` (NIFTY) or `data/{underlying}/signals.csv` (BANKNIFTY)
- `data/signals.jsonl` or `data/{underlying}/signals.jsonl`
- terminal output

**Supported underlyings:** NIFTY, NIFTY BANK (BANKNIFTY), and all F&O stocks (RELIANCE, TCS, INFY, etc.).

## What it fetches every 5 minutes
- Full-day 5-minute candles from market open (09:15) to previous completed 5-minute candle
- For each timestamp: index spot, nearest futures, ATM call, ATM put

## Output fields
- Spot: entry, target, stop_loss, rr
- Option (CE for BUY, PE for SELL): strike_price, option_type, option_entry, option_sl, option_target

## Core logic
- Weighted scoring for bullish, bearish, and no-trade penalties
- Day bias from spot/open/VWAP + futures + options behavior
- Intraday support/resistance from rolling structure
- Momentum from recent multi-bar move
- Trade plan with structure-based stop and minimum reward:risk checks
- Rolling analysis per 5-minute candle; each new completed candle gets its own signal row

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with valid Kite credentials and symbol settings.

## Run
One cycle (default: NIFTY):
```bash
PYTHONPATH=src python -m intraday_engine.main --once
```

Specific underlying (NIFTY, BANKNIFTY, or F&O stock):
```bash
PYTHONPATH=src python -m intraday_engine.main --once --underlying BANKNIFTY
```

Specific date backfill (weekly ATM option + monthly future resolution on that date):
```bash
PYTHONPATH=src python -m intraday_engine.main --date 2026-03-09
PYTHONPATH=src python -m intraday_engine.main --date 2026-03-09 --underlying BANKNIFTY
```

Continuous scheduler:
```bash
PYTHONPATH=src python -m intraday_engine.main
```

## Dashboard
Web UI to view signals, refresh data, and execute orders:
```bash
PYTHONPATH=src uvicorn intraday_engine.dashboard:app --reload --host 0.0.0.0 --port 8000
```
Open http://localhost:8000
- **Underlying selector**: Switch between NIFTY, NIFTY BANK, or any F&O stock
- **Refresh**: Fetches latest data and generates signals for the selected index
- **Execute**: Places market order on Zerodha for the latest BUY/SELL signal (lots editable, default 2)

## Folder structure
```
src/intraday_engine/
  core/          # config.py, models.py
  utils/         # logging_setup.py, output.py
  storage/       # data_store.py
  engine/        # direction_engine.py, scheduler.py
  api/           # dashboard.py, templates/dashboard.html
  cli/           # main.py
  analysis/      # scoring, momentum, bias, S/R, trade plan, sideways
  fetch/         # Zerodha client, instrument resolver, market_data
  features/      # feature engineering from merged bars
  main.py        # CLI entry (delegates to cli.main)
  dashboard.py   # API entry (delegates to api.dashboard)
```




# Scan today (only runs on expiry day – Tuesday for Nifty)
PYTHONPATH=src python -m intraday_engine.main --gamma-blast

# Scan a specific date
PYTHONPATH=src python -m intraday_engine.main --gamma-blast --date 2026-03-11

## F&O Stocks (15-min)
Fetch 15-min data for all F&O stocks and generate signals (spot + futures; options when available):
```bash
PYTHONPATH=src python -m intraday_engine.main --stocks-15min
PYTHONPATH=src python -m intraday_engine.main --stocks-15min --stocks-limit 100
```
Continuous 15-min scheduler:
```bash
PYTHONPATH=src python -m intraday_engine.main --stocks-15min-scheduler
```
Stocks dashboard at http://localhost:8000/stocks shows stored signals. Use Refresh to fetch and generate.

## 15-min ORB (Opening Range Breakout)
Uses latest 15-min candle close (not LTP). BUY if close ≥ OR high − 0.2%, SELL if close ≤ OR low + 0.2%.
```bash
PYTHONPATH=src python -m intraday_engine.main --orb --orb-limit 200
```
Dashboard: http://localhost:8000/stocks → ORB tab. Parallel fetch (5 workers), one call per stock.

## Pinbars (15-min)
Bullish/bearish pinbar on last 15-min candle:
```bash
PYTHONPATH=src python -m intraday_engine.main --pinbar --pinbar-limit 200
```
Dashboard: http://localhost:8000/stocks → Pinbars tab.



# Predict huge move (captures option chain, runs prediction)
python -m intraday_engine.cli.main --huge-move

# Capture and store option chain only
python -m intraday_engine.cli.main --capture-option-chain

# Use 10 strikes each side
python -m intraday_engine.cli.main --huge-move --option-strikes 10


# Position Monitoring
python -m intraday_engine --btst
python -m intraday_engine --trail

python -m intraday_engine.position_monitor btst
python -m intraday_engine.position_monitor trail

python -m intraday_engine --btst --underlying NIFTY
python -m intraday_engine --trail --underlying NIFTY


# Maintenance
PYTHONPATH=src .venv/bin/python -m intraday_engine.maintenance --refresh-fno-watchlist

PYTHONPATH=src .venv/bin/python -m intraday_engine.maintenance --migrate-legacy-data

PYTHONPATH=src .venv/bin/python -m intraday_engine.maintenance --rewrite-partitioned-layout

# EOD Signal Analysis
PYTHONPATH=src .venv/bin/python -m intraday_engine.maintenance --analyze-signals-eod --date YYYY-MM-DD

# Accumulation
PYTHONPATH=src python3 -m intraday_engine.main --nifty500-accumulation

PYTHONPATH=src python3 -m intraday_engine.main \
  --nifty500-accumulation \
  --nifty500-top 15 \
  --nifty500-workers 4 \
  --nifty500-out data/reference/nifty500_accumulation_top.json


PYTHONPATH=src python3 -m intraday_engine.main \
  --nifty500-accumulation \
  --nifty500-symbols-file /path/to/symbols.txt

# Next-Day Watchlist
python3 -m intraday_engine.main --tomorrow-watchlist
# optional:
python3 -m intraday_engine.main --tomorrow-watchlist --tw-top 20 --tw-workers 4 --tw-limit 30



PYTHONPATH=src python -m intraday_engine.backtest.uhv --data-dir data/NIFTY500 --output-dir data/backtesting/uhv_backtest



python institutional_expansion_pivot.py --csv data/NIFTY500/AFCONS_15Min.csv

python backtest_strategies.py --symbol AFCONS --exit rr

python backtest_strategies.py --data-dir data/NIFTY500 --exit hybrid --output data/backtesting/orb_trades.csv

python volume_breakout.py

PYTHONPATH=src python3 -m intraday_engine.cli --uhv-bt


# Running dashboard
PYTHONPATH=src uvicorn intraday_engine.dashboard:app --reload --host 0.0.0.0 --port 8000

# Read-only dashboard (stored data only — no background fetches/scans)
PYTHONPATH=src python -m intraday_engine.cli.main \
  --dashboard --read-only --reload --host 0.0.0.0 --port 8000

# Full session pipeline (option chain + index analysis + options-trading scan, every 5 min)
# Also builds Market Overview once daily at ~09:05 IST (GIFT Nifty, VIX, news, NIFTY plan).
# On macOS, use caffeinate so the Mac does not sleep through the 09:15 open:
#   PYTHONPATH=src caffeinate -dims python -m intraday_engine.cli.main --session-scheduler
PYTHONPATH=src python -m intraday_engine.cli.main --session-scheduler

# One full session cycle (manual)
PYTHONPATH=src python -m intraday_engine.cli.main --session-once

# Typical setup (2 terminals):
#   1) --session-scheduler
#   2) --dashboard --read-only

# Low-level (usually not needed — use --session-scheduler instead):
# PYTHONPATH=src python -m intraday_engine.cli.main --option-chain-scheduler
# PYTHONPATH=src python -m intraday_engine.cli.main --capture-option-chain


# Running dashboard only
PYTHONPATH=src python -m intraday_engine.cli.main \ 
  --dashboard --read-only --reload --host 0.0.0.0 --port 8000

# Running scraper in non sleep mode
PYTHONPATH=src caffeinate -dims python -m intraday_engine.cli.main --session-scheduler

# Option chain backfill
PYTHONPATH=src python -m intraday_engine.cli.main \
  --backfill-option-chain \
  --date 2026-07-30 \
  --underlying NIFTY \
  --bar-times 11:10 11:15 \
  --strike-min 23600 \
  --strike-max 24900 \
  --remove-chain-prefix 2026-07-30T11:17

PYTHONPATH=src python -m intraday_engine.cli.main \
  --backfill-option-chain --date 2026-07-30 --underlying NIFTY \
  --bar-times 11:10 11:15 --strike-min 23600 --strike-max 24900

PYTHONPATH=src python -m intraday_engine.cli.main \
  --backfill-option-chain --date 2026-07-30 --underlying NIFTYBANK \
  --bar-times 11:10 11:15 --strike-min 56200 --strike-max 57600


# 
        action="store_true",
        help="Backfill missing 5-min option-chain bars from Kite historical OI (requires --date).",
        help="Backfill 5-min option-chain bars from Kite historical OI (requires --date).",
    )
    if args.backfill_option_chain:
        from intraday_engine.gamma.option_chain_fetcher import backfill_option_chain_historical
    if args.backfill_full_day or args.backfill_option_chain:
                remove_capture_prefixes=args.remove_chain_prefix,
                replace_existing=replace or True,
                replace_existing=True,
            )
        newline="",
    )
FileNotFoundError: [Errno 2] No such file or directory: 'data/option_chain/date=2026-07-30/option_chain.csv'
session bars 75 09:15 15:25
present bars 0 sample [] ... []
  --remove-chain-prefix [REMOVE_CHAIN_PREFIX ...]
--
                        prefixes before backfill.
  --option-strikes OPTION_STRIKES
                        Legacy: symmetric CE/PE strike count override (default
Full-day option chain backfill is implemented. Here’s how to use it.

New CLI flag: --backfill-full-day
Backfills all 75 session bars (09:15 → 15:25, every 5 min) from Kite historical OI.

# Fill only missing bars
PYTHONPATH=src python -m intraday_engine.cli.main \
  --backfill-full-day \
  --date 2026-07-30 \
  --underlying NIFTY

# Rebuild the entire day (drop + refill all session bars):
PYTHONPATH=src python -m intraday_engine.cli.main \
  --backfill-full-day \
  --date 2026-07-30 \
  --underlying NIFTY \
  --replace-existing

PYTHONPATH=src python -m intraday_engine.cli.main \
  --backfill-full-day \
  --date 2026-07-30 \
  --underlying BANKNIFTY