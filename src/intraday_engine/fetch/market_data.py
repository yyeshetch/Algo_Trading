from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.zerodha_client import ZerodhaClient


class MarketDataFetcher:
    def __init__(self, client: ZerodhaClient, resolver: InstrumentResolver, settings: Settings) -> None:
        self.client = client
        self.resolver = resolver
        self.settings = settings

    def fetch_intraday_frame(self) -> pd.DataFrame:
        return self.fetch_intraday_frame_for_date(None)

    def fetch_intraday_frame_for_date(self, trade_date: date | None) -> pd.DataFrame:
        from_dt, to_dt = _market_window(trade_date)
        if to_dt < from_dt:
            raise RuntimeError("No completed 5-minute candle available for the requested session.")
        effective_date = from_dt.date()

        spot_quote_raw = self.client.quote([self.settings.spot_symbol])
        spot_quote = spot_quote_raw[self.settings.spot_symbol]
        spot_token = int(spot_quote["instrument_token"])
        spot_rows = self.client.historical_data(spot_token, from_dt, to_dt)
        spot_df = _drop_incomplete_candles(_to_candle_df(spot_rows, "spot"), trade_date, 5)
        if spot_df.empty:
            raise RuntimeError("No spot candles returned for today's session window.")

        reference_spot = float(spot_df.iloc[0]["spot_open_raw"])
        symbols = self.resolver.resolve_for_date(reference_spot, effective_date)
        deriv_quotes = self.client.quote([symbols.fut_symbol, symbols.ce_symbol, symbols.pe_symbol])

        fut_token = int(deriv_quotes[symbols.fut_symbol]["instrument_token"])
        ce_token = int(deriv_quotes[symbols.ce_symbol]["instrument_token"])
        pe_token = int(deriv_quotes[symbols.pe_symbol]["instrument_token"])

        fut_df = _drop_incomplete_candles(
            _to_candle_df(self.client.historical_data(fut_token, from_dt, to_dt, oi=True), "future", include_oi=True),
            trade_date,
            5,
        )
        ce_df = _drop_incomplete_candles(
            _to_candle_df(self.client.historical_data(ce_token, from_dt, to_dt, oi=True), "call", include_oi=True),
            trade_date,
            5,
        )
        pe_df = _drop_incomplete_candles(
            _to_candle_df(self.client.historical_data(pe_token, from_dt, to_dt, oi=True), "put", include_oi=True),
            trade_date,
            5,
        )

        merged = spot_df.merge(fut_df, on="timestamp", how="inner")
        ce_cols = ["timestamp", "call_close", "call_volume"]
        if "call_oi" in ce_df.columns:
            ce_cols.append("call_oi")
        merged = merged.merge(ce_df[ce_cols], on="timestamp", how="inner")
        pe_cols = ["timestamp", "put_close", "put_volume"]
        if "put_oi" in pe_df.columns:
            pe_cols.append("put_oi")
        merged = merged.merge(pe_df[pe_cols], on="timestamp", how="inner")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        spot_day_open = float(merged.iloc[0]["spot_open_raw"])
        fut_day_open = float(merged.iloc[0]["future_open_raw"])

        merged["spot_symbol"] = self.settings.spot_symbol
        merged["future_symbol"] = symbols.fut_symbol
        merged["call_symbol"] = symbols.ce_symbol
        merged["put_symbol"] = symbols.pe_symbol
        merged["ce_symbol"] = symbols.ce_symbol
        merged["pe_symbol"] = symbols.pe_symbol
        merged["fut_symbol"] = symbols.fut_symbol
        merged["atm_strike"] = symbols.atm_strike

        merged["spot_ltp"] = merged["spot_close"]
        merged["future_ltp"] = merged["future_close"]
        merged["call_ltp"] = merged["call_close"]
        merged["put_ltp"] = merged["put_close"]

        merged["spot_open"] = spot_day_open
        merged["future_open"] = fut_day_open
        merged["spot_high"] = merged["spot_high_raw"]
        merged["spot_low"] = merged["spot_low_raw"]
        merged["spot_close"] = merged["spot_close"]
        merged["future_high"] = merged["future_high_raw"]
        merged["future_low"] = merged["future_low_raw"]
        merged["future_close"] = merged["future_close"]

        merged["spot_vwap"] = _session_vwap(merged["spot_close"], merged["spot_volume"])
        merged["future_vwap"] = _session_vwap(merged["future_close"], merged["future_volume"])
        merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

        out_cols = [
            "timestamp",
            "spot_symbol",
            "spot_ltp",
            "spot_open",
            "spot_high",
            "spot_low",
            "spot_close",
            "spot_vwap",
            "future_symbol",
            "future_ltp",
            "future_open",
            "future_high",
            "future_low",
            "future_close",
            "future_vwap",
            "call_symbol",
            "call_ltp",
            "put_symbol",
            "put_ltp",
            "atm_strike",
            "ce_symbol",
            "pe_symbol",
            "fut_symbol",
        ]
        if "future_oi" in merged.columns:
            out_cols.append("future_oi")
        if "call_oi" in merged.columns:
            out_cols.append("call_oi")
        if "put_oi" in merged.columns:
            out_cols.append("put_oi")
        return merged[[c for c in out_cols if c in merged.columns]]


def _market_window(trade_date: date | None) -> tuple[datetime, datetime]:
    now = datetime.now()
    target_date = trade_date or now.date()
    session_start = datetime.combine(target_date, datetime.min.time()).replace(hour=9, minute=15)
    # 15:30 includes the final 5-minute candle (15:25–15:30).
    session_end = datetime.combine(target_date, datetime.min.time()).replace(hour=15, minute=30)
    if target_date < now.date():
        return session_start, session_end
    if target_date > now.date():
        return session_start, session_start - timedelta(minutes=5)
    return session_start, min(session_end, now.replace(second=0, microsecond=0))


def _to_candle_df(rows: list[dict[str, object]], prefix: str, include_oi: bool = False) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp"])

    raw = pd.DataFrame(rows)
    raw = raw.rename(columns={"date": "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    cols = {
        "timestamp": raw["timestamp"],
        f"{prefix}_open_raw": raw["open"].astype(float),
        f"{prefix}_high_raw": raw["high"].astype(float),
        f"{prefix}_low_raw": raw["low"].astype(float),
        f"{prefix}_close": raw["close"].astype(float),
        f"{prefix}_volume": raw["volume"].astype(float),
    }
    if include_oi and "oi" in raw.columns:
        cols[f"{prefix}_oi"] = raw["oi"].astype(float)
    return pd.DataFrame(cols)


def _drop_incomplete_candles(
    df: pd.DataFrame,
    trade_date: date | None,
    interval_minutes: int,
) -> pd.DataFrame:
    if df.empty:
        return df
    now = datetime.now()
    target_date = trade_date or now.date()
    if target_date != now.date():
        return df
    cutoff = min(
        datetime.combine(target_date, datetime.min.time()).replace(hour=15, minute=30),
        now.replace(second=0, microsecond=0),
    )
    completed_mask = df["timestamp"] + pd.to_timedelta(interval_minutes, unit="m") <= cutoff
    return df.loc[completed_mask].reset_index(drop=True)


def _session_vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
    if float(volume.fillna(0.0).sum()) <= 0.0:
        # Indices can have zero/empty volume in historical candles; use session mean as VWAP proxy.
        return price.expanding().mean()
    numerator = (price * volume).cumsum()
    denominator = volume.cumsum().replace(0, 1.0)
    return numerator / denominator

