from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from kiteconnect import KiteConnect
from tenacity import retry, stop_after_attempt, wait_exponential

from intraday_engine.core.config import Settings

logger = logging.getLogger(__name__)


class ZerodhaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.kite = KiteConnect(api_key=settings.kite_api_key)
        self.kite.set_access_token(settings.kite_access_token)
        self._nfo_cache: List[Dict[str, object]] = []
        self._cache_refreshed_at: datetime | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
    def quote(self, symbols: List[str]) -> Dict[str, Dict[str, object]]:
        logger.debug("Fetching quotes for symbols: %s", symbols)
        return self.kite.quote(symbols)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
    def historical_data(
        self,
        instrument_token: int,
        from_dt: datetime,
        to_dt: datetime,
        interval: str = "5minute",
        oi: bool = False,
    ) -> List[Dict[str, object]]:
        """Fetch historical OHLCV. Set oi=True for options/futures to include open interest."""
        logger.debug(
            "Fetching historical data token=%s from=%s to=%s interval=%s oi=%s",
            instrument_token,
            from_dt,
            to_dt,
            interval,
            oi,
        )
        return self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval,
            continuous=False,
            oi=oi,
        )

    def get_day_pnl(self) -> float | None:
        """Return today's total P&L from positions. None if fetch fails (e.g. market closed)."""
        try:
            data = self.kite.positions()
            if not data or not isinstance(data, dict):
                return 0.0
            total = 0.0
            for p in data.get("net", []):
                if isinstance(p, dict):
                    total += float(p.get("pnl", 0) or 0)
            return total
        except Exception as e:
            logger.warning("Could not fetch day P&L: %s", e)
            return None

    def get_trade_summary(self) -> Dict[str, object] | None:
        """Return day P&L, positions, and orders. None if fetch fails."""
        try:
            day_pnl = self.get_day_pnl()
            positions = []
            orders = []
            try:
                pos_data = self.kite.positions()
                if isinstance(pos_data, dict):
                    positions = pos_data.get("net", []) or []
            except Exception as e:
                logger.debug("Could not fetch positions: %s", e)
            try:
                orders = self.kite.orders() or []
            except Exception as e:
                logger.debug("Could not fetch orders: %s", e)
            return {
                "day_pnl": day_pnl if day_pnl is not None else 0.0,
                "positions": positions,
                "orders": orders,
            }
        except Exception as e:
            logger.warning("Could not fetch trade summary: %s", e)
            return None

    def get_instrument_token(self, tradingsymbol: str, exchange: str = "NFO") -> int | None:
        """Get instrument_token for tradingsymbol from NFO instruments."""
        sym = str(tradingsymbol).replace("NFO:", "").strip()
        for inst in self.nfo_instruments():
            if isinstance(inst, dict) and inst.get("tradingsymbol") == sym:
                return int(inst.get("instrument_token", 0))
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str = "MIS",
        order_type: str = "MARKET",
        trigger_price: float | None = None,
        price: float | None = None,
    ) -> str:
        """Place a market order (or SL-M). Returns order_id."""
        if tradingsymbol.startswith("NFO:"):
            tradingsymbol = tradingsymbol[4:]
        if exchange.upper() != "NFO":
            exchange = "NFO"
        ot = order_type.upper() if isinstance(order_type, str) else order_type
        if ot == "SL-M":
            ot = self.kite.ORDER_TYPE_SLM
        elif ot == "MARKET":
            ot = self.kite.ORDER_TYPE_MARKET
        elif ot == "SL":
            ot = self.kite.ORDER_TYPE_SL
        params: Dict[str, object] = {
            "variety": self.kite.VARIETY_REGULAR,
            "tradingsymbol": tradingsymbol,
            "exchange": exchange,
            "transaction_type": transaction_type.upper(),
            "quantity": quantity,
            "product": self.kite.PRODUCT_MIS if product == "MIS" else product,
            "order_type": ot,
        }
        if trigger_price is not None:
            params["trigger_price"] = round(trigger_price, 2)
        if price is not None:
            params["price"] = round(price, 2)
        order_id = self.kite.place_order(**params)
        logger.info("Order placed: %s %s %s qty=%s order_id=%s", transaction_type, tradingsymbol, exchange, quantity, order_id)
        return order_id

    def place_sl_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        trigger_price: float,
        product: str = "MIS",
    ) -> str:
        """Place SL-M order. For NFO options, use SL-L if SL-M fails (limit = trigger ± buffer)."""
        if tradingsymbol.startswith("NFO:"):
            tradingsymbol = tradingsymbol[4:]
        exchange = "NFO" if exchange.upper() != "NFO" else exchange
        quantity = abs(quantity)
        try:
            return self.place_order(
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type="SL-M",
                trigger_price=trigger_price,
            )
        except Exception as e:
            if "SL-M" in str(e) or "SLM" in str(e):
                buffer = max(1.0, trigger_price * 0.02)
                price = trigger_price - buffer if transaction_type == "SELL" else trigger_price + buffer
                return self.place_order(
                    tradingsymbol=tradingsymbol,
                    exchange=exchange,
                    transaction_type=transaction_type,
                    quantity=quantity,
                    product=product,
                    order_type="SL",
                    trigger_price=trigger_price,
                    price=round(price, 2),
                )
            raise

    def modify_sl_order(self, order_id: str, trigger_price: float) -> str:
        """Modify SL order trigger price."""
        return self.kite.modify_order(
            variety=self.kite.VARIETY_REGULAR,
            order_id=str(order_id),
            trigger_price=round(trigger_price, 2),
        )

    def cancel_order(self, order_id: str) -> str:
        """Cancel an order."""
        return self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=str(order_id))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
    def nfo_instruments(self) -> List[Dict[str, object]]:
        now = datetime.now()
        if self._cache_refreshed_at and (now - self._cache_refreshed_at) < timedelta(hours=6):
            return self._nfo_cache

        logger.info("Refreshing NFO instruments cache.")
        records = self.kite.instruments("NFO")
        self._nfo_cache = records
        self._cache_refreshed_at = now
        return records

