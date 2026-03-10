from __future__ import annotations

from typing import Tuple

import pandas as pd


def calculate_support_resistance(df: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    # Use completed bars only for structure so current price can actually break levels.
    source = df.iloc[:-1] if len(df) > 1 else df
    window = source.tail(max(lookback, 3))
    support = float(window["spot_low"].min())
    resistance = float(window["spot_high"].max())
    return support, resistance

