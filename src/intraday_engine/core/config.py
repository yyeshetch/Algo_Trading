from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """Project root (directory containing src/). Stable regardless of cwd or folder rename."""
    return Path(__file__).resolve().parent.parent.parent.parent


# Load .env from project root
load_dotenv(_project_root() / ".env")


@dataclass(frozen=True)
class Settings:
    kite_api_key: str
    kite_access_token: str
    underlying: str
    spot_symbol: str
    option_strike_step: int
    lot_size: int
    poll_interval_minutes: int
    lookback_bars: int
    min_rr: float
    min_confidence: float
    min_confidence_neutral_day: float
    max_stop_pct: float
    max_stop_pct_short: float
    option_stop_ratio: float
    min_day_range_pct: float
    data_dir: Path
    log_level: str
    daily_sl_rupees: float
    default_sl_points: float

    @staticmethod
    def from_env() -> "Settings":
        data_dir_val = os.getenv("DATA_DIR", "data")
        data_dir = Path(data_dir_val)
        if not data_dir.is_absolute():
            data_dir = (_project_root() / data_dir).resolve()
        return Settings(
            kite_api_key=_required("KITE_API_KEY"),
            kite_access_token=_required("KITE_ACCESS_TOKEN"),
            underlying=os.getenv("UNDERLYING", "NIFTY").strip().upper(),
            spot_symbol=os.getenv("SPOT_SYMBOL", "NSE:NIFTY 50").strip(),
            option_strike_step=int(os.getenv("OPTION_STRIKE_STEP", "50")),
            lot_size=int(os.getenv("LOT_SIZE", "50")),
            poll_interval_minutes=int(os.getenv("POLL_INTERVAL_MINUTES", "5")),
            lookback_bars=int(os.getenv("LOOKBACK_BARS", "20")),
            min_rr=float(os.getenv("MIN_RR", "1.8")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.70")),
            min_confidence_neutral_day=float(os.getenv("MIN_CONFIDENCE_NEUTRAL_DAY", "0.85")),
            max_stop_pct=float(os.getenv("MAX_STOP_PCT", "0.45")),
            max_stop_pct_short=float(os.getenv("MAX_STOP_PCT_SHORT", "0.65")),
            option_stop_ratio=float(os.getenv("OPTION_STOP_RATIO", "0.35")),
            min_day_range_pct=float(os.getenv("MIN_DAY_RANGE_PCT", "0.40")),
            data_dir=data_dir,
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            daily_sl_rupees=float(os.getenv("DAILY_SL_RUPEES", "4000")),
            default_sl_points=float(os.getenv("DEFAULT_SL_POINTS", "10")),
        )


def _required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value.strip()
