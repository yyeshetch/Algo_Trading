from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from intraday_engine.core.underlyings import get_underlying_config


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
    def from_env(underlying: str | None = None) -> "Settings":
        data_dir_val = os.getenv("DATA_DIR", "data")
        data_dir = Path(data_dir_val)
        if not data_dir.is_absolute():
            data_dir = (_project_root() / data_dir).resolve()
        underlying_val = (underlying or os.getenv("UNDERLYING", "NIFTY")).strip().upper()
        underlying_val = underlying_val.replace(" ", "").replace("NIFTYBANK", "BANKNIFTY") or "NIFTY"
        try:
            uc = get_underlying_config(underlying_val)
            env_underlying = os.getenv("UNDERLYING", "NIFTY").strip().upper().replace(" ", "").replace("NIFTYBANK", "BANKNIFTY")
            use_env = env_underlying and env_underlying == underlying_val
            spot_symbol = (os.getenv("SPOT_SYMBOL") or uc.spot_symbol) if use_env else uc.spot_symbol
            option_strike_step = int((os.getenv("OPTION_STRIKE_STEP") or uc.option_strike_step) if use_env else uc.option_strike_step)
            lot_size = int((os.getenv("LOT_SIZE") or uc.lot_size) if use_env else uc.lot_size)
        except KeyError:
            spot_symbol = os.getenv("SPOT_SYMBOL", "NSE:NIFTY 50").strip()
            option_strike_step = int(os.getenv("OPTION_STRIKE_STEP", "50"))
            lot_size = int(os.getenv("LOT_SIZE", "50"))

        from intraday_engine.core.tunables import get_float, get_int, get_str

        def _ei(env_key: str, tk: str, default: int) -> int:
            v = os.getenv(env_key)
            if v is not None and str(v).strip() != "":
                return int(v)
            return get_int("settings", tk, default)

        def _ef(env_key: str, tk: str, default: float) -> float:
            v = os.getenv(env_key)
            if v is not None and str(v).strip() != "":
                return float(v)
            return get_float("settings", tk, default)

        def _es_log(env_key: str, tk: str, default: str) -> str:
            v = os.getenv(env_key)
            if v is not None and str(v).strip() != "":
                return v.strip().upper()
            return get_str("settings", tk, default).strip().upper()

        return Settings(
            kite_api_key=_required("KITE_API_KEY"),
            kite_access_token=_required("KITE_ACCESS_TOKEN"),
            underlying=underlying_val,
            spot_symbol=spot_symbol.strip(),
            option_strike_step=option_strike_step,
            lot_size=lot_size,
            poll_interval_minutes=_ei("POLL_INTERVAL_MINUTES", "poll_interval_minutes", 5),
            lookback_bars=_ei("LOOKBACK_BARS", "lookback_bars", 20),
            min_rr=_ef("MIN_RR", "min_rr", 1.8),
            min_confidence=_ef("MIN_CONFIDENCE", "min_confidence", 0.70),
            min_confidence_neutral_day=_ef("MIN_CONFIDENCE_NEUTRAL_DAY", "min_confidence_neutral_day", 0.85),
            max_stop_pct=_ef("MAX_STOP_PCT", "max_stop_pct", 0.45),
            max_stop_pct_short=_ef("MAX_STOP_PCT_SHORT", "max_stop_pct_short", 0.65),
            option_stop_ratio=_ef("OPTION_STOP_RATIO", "option_stop_ratio", 0.35),
            min_day_range_pct=_ef("MIN_DAY_RANGE_PCT", "min_day_range_pct", 0.40),
            data_dir=data_dir,
            log_level=_es_log("LOG_LEVEL", "log_level", "INFO"),
            daily_sl_rupees=_ef("DAILY_SL_RUPEES", "daily_sl_rupees", 4000.0),
            default_sl_points=_ef("DEFAULT_SL_POINTS", "default_sl_points", 10.0),
        )


def _required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value.strip()
