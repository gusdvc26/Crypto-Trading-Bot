# src/app/scanner.py
from __future__ import annotations

import asyncio
import logging
import os
from typing import List

from config.settings import get_settings
from src.utils.helpers import setup_logging, get_logger
from src.ingestion.fetch_data import fetch_ohlcv, shutdown_session
from src.utils.persist import persist_ohlcv_row, persist_decision
from src.signals.strategies import SmaCross

log = get_logger(__name__)

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))  # 60s default


async def scan_once(symbols: List[str], interval: str, limit: int, exchange: str) -> None:
    strat = SmaCross(fast=5, slow=20)

    async def handle_symbol(sym: str):
        try:
            candles = await fetch_ohlcv(sym, interval, limit=limit, use_cache=True)
            if not candles:
                log.warning(f"{sym}: no candles returned")
                return

            # Tier-0: persist all candles
            for r in candles:
                await persist_ohlcv_row(exchange, sym, r)

            # Tier-1: decision
            sig = strat.generate(candles)
            last_ts = candles[-1]["t"]
            await persist_decision(
                exchange=exchange,
                symbol=sym,
                strategy=strat.name,
                strategy_version="v1",
                action=sig.action.value,
                confidence=sig.confidence,
                t=last_ts,
            )
            log.info(f"{sym} -> {sig.action.value} (conf={sig.confidence:.3f})")
        except Exception as e:
            log.exception(f"Error scanning {sym}: {e}")

    await asyncio.gather(*(handle_symbol(s) for s in symbols))


async def main() -> int:
    s = get_settings()
    setup_logging(s)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("Scanner bootstrapped")

    symbols = list(s.default_symbols)
    interval = s.default_interval
    limit = 200

    try:
        while True:
            log.info(f"Scan tick: {symbols} interval={interval} limit={limit}")
            await scan_once(symbols, interval, limit, s.api.exchange)
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)
    finally:
        try:
            await shutdown_session()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
