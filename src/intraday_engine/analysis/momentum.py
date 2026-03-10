from __future__ import annotations

import pandas as pd


def momentum_direction(df: pd.DataFrame) -> str:
    if len(df) < 4:
        return "NEUTRAL"

    window = df.tail(4)
    start = float(window.iloc[0]["spot_ltp"])
    end = float(window.iloc[-1]["spot_ltp"])
    if start == 0:
        return "NEUTRAL"

    move = ((end - start) / start) * 100
    if move > 0.20:
        return "UP"
    if move < -0.20:
        return "DOWN"
    return "NEUTRAL"

