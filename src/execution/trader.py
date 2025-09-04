# src/execution/trader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config.settings import get_settings
from src.utils.helpers import get_logger, async_retry

log = get_logger(__name__)


@dataclass
class Order:
    id: str
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: float
    price: Optional[float]  # None for market
    status: str = "NEW"


class Trader:
    """
    Placeholder execution layer.
    - Wraps REST calls later (place_order, cancel_order, get_balance).
    """
    def __init__(self):
        self.s = get_settings()

    @async_retry()
    async def place_order(self, symbol: str, side: str, qty: float, price: float | None = None) -> Order:
        # No real API calls in Phase 1. Return a stub.
        oid = f"stub-{symbol}-{side}-qty{qty}"
        log.info(f"[STUB] place_order {symbol} {side} qty={qty} price={price}")
        return Order(id=oid, symbol=symbol, side=side, qty=qty, price=price, status="FILLED")

    @async_retry()
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        log.info(f"[STUB] cancel_order {order_id}")
        return {"order_id": order_id, "status": "CANCELED"}

    @async_retry()
    async def get_balance(self) -> Dict[str, float]:
        log.info("[STUB] get_balance")
        return {"USDT": 10_000.0}
