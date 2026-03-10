"""Predefined underlying configs for NIFTY, NIFTY BANK."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnderlyingConfig:
    """Config for a single index/underlying."""

    name: str  # NFO instrument name (e.g. NIFTY, BANKNIFTY)
    spot_symbol: str  # e.g. NSE:NIFTY 50, NSE:NIFTY BANK
    option_strike_step: int
    lot_size: int
    exchange: str = "NFO"  # NFO for NSE


# Predefined underlyings (NFO)
UNDERLYINGS: dict[str, UnderlyingConfig] = {
    "NIFTY": UnderlyingConfig(
        name="NIFTY",
        spot_symbol="NSE:NIFTY 50",
        option_strike_step=50,
        lot_size=50,
    ),
    "BANKNIFTY": UnderlyingConfig(
        name="BANKNIFTY",
        spot_symbol="NSE:NIFTY BANK",
        option_strike_step=100,
        lot_size=15,
    ),
}


def get_underlying_config(underlying: str) -> UnderlyingConfig:
    """Return config for underlying. Raises KeyError if unknown."""
    key = underlying.strip().upper().replace(" ", "")
    if key == "NIFTYBANK":
        key = "BANKNIFTY"
    return UNDERLYINGS[key]


def list_underlyings() -> list[str]:
    """Return list of supported underlying keys."""
    return list(UNDERLYINGS.keys())
