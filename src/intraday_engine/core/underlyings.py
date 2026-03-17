"""Predefined underlying configs for NIFTY, NIFTY BANK, and F&O stocks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnderlyingConfig:
    """Config for a single index/underlying."""

    name: str  # NFO instrument name (e.g. NIFTY, BANKNIFTY, RELIANCE)
    spot_symbol: str  # e.g. NSE:NIFTY 50, NSE:RELIANCE
    option_strike_step: int
    lot_size: int
    exchange: str = "NFO"  # NFO for NSE
    is_index: bool = True  # False for F&O stocks


# Predefined index underlyings (NFO)
INDEX_UNDERLYINGS: dict[str, UnderlyingConfig] = {
    "NIFTY": UnderlyingConfig(
        name="NIFTY",
        spot_symbol="NSE:NIFTY 50",
        option_strike_step=50,
        lot_size=50,
        is_index=True,
    ),
    "BANKNIFTY": UnderlyingConfig(
        name="BANKNIFTY",
        spot_symbol="NSE:NIFTY BANK",
        option_strike_step=100,
        lot_size=15,
        is_index=True,
    ),
    "FINNIFTY": UnderlyingConfig(
        name="FINNIFTY",
        spot_symbol="NSE:NIFTY FIN SERVICE",
        option_strike_step=50,
        lot_size=25,
        is_index=True,
    ),
    "MIDCPNIFTY": UnderlyingConfig(
        name="MIDCPNIFTY",
        spot_symbol="NSE:NIFTY MID SELECT",
        option_strike_step=25,
        lot_size=75,
        is_index=True,
    ),
    "NIFTYNXT50": UnderlyingConfig(
        name="NIFTYNXT50",
        spot_symbol="NSE:NIFTY NEXT 50",
        option_strike_step=5,
        lot_size=30,
        is_index=True,
    ),
}


def get_underlying_config(underlying: str) -> UnderlyingConfig:
    """Return config for underlying. For stocks, creates dynamic config."""

    def _stock_config(name: str) -> UnderlyingConfig:
        return UnderlyingConfig(
            name=name,
            spot_symbol=f"NSE:{name}",
            option_strike_step=5,
            lot_size=1,  # Overridden from resolver when available
            is_index=False,
        )

    key = underlying.strip().upper().replace(" ", "")
    if key == "NIFTYBANK":
        key = "BANKNIFTY"
    if key in INDEX_UNDERLYINGS:
        return INDEX_UNDERLYINGS[key]
    return _stock_config(key)


def list_index_underlyings() -> list[str]:
    """Return list of index underlying keys."""
    return list(INDEX_UNDERLYINGS.keys())


def list_underlyings() -> list[str]:
    """Return list of index underlying keys. For stocks, use API /api/fno-stocks."""
    return list_index_underlyings()

