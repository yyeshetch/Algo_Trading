# Intraday Direction Engine (Zerodha Kite)

Production-oriented, explainable, rule-based intraday signal engine that runs every 5 minutes and emits:
- `data/signals.csv` (NIFTY) or `data/{underlying}/signals.csv` (BANKNIFTY)
- `data/signals.jsonl` or `data/{underlying}/signals.jsonl`
- terminal output

**Supported underlyings:** NIFTY, NIFTY BANK (BANKNIFTY).

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

Specific underlying (NIFTY, BANKNIFTY):
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
- **Index selector**: Switch between NIFTY, NIFTY BANK
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