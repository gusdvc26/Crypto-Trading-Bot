# src/ingestion/ws_coinbase.py
from __future__ import annotations
import asyncio, json
from typing import AsyncGenerator, Iterable, Dict, Any
from datetime import datetime, timezone
import websockets

from config.settings import get_settings
from src.utils.helpers import get_logger

log = get_logger(__name__)
S = get_settings()

def _to_ms(ts: str) -> int:
    if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
    return int(datetime.fromisoformat(ts).timestamp() * 1000)

async def coinbase_ticker_stream(product_ids: Iterable[str]) -> AsyncGenerator[Dict[str, Any], None]:
    uri = getattr(S.api, "ws_url", "wss://ws-feed.exchange.coinbase.com")
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
                sub = {"type": "subscribe", "product_ids": list(product_ids), "channels": ["ticker"]}
                await ws.send(json.dumps(sub))
                log.info(f"WS connected: {uri} ({len(list(product_ids))} symbols)")
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if data.get("type") != "ticker": continue
                        pid, price, time = data.get("product_id"), data.get("price"), data.get("time")
                        if not (pid and price and time): continue
                        size = data.get("last_size") or "0"
                        yield {"product_id": pid, "t": _to_ms(time), "price": float(price), "size": float(size)}
                    except Exception as e:
                        log.debug(f"WS parse error: {e}")
        except Exception as e:
            log.warning(f"WS connection error: {e}; reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff); backoff = min(backoff * 2, 30.0)
        else:
            backoff = 1.0
