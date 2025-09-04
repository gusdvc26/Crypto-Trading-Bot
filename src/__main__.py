# src/__main__.py
from __future__ import annotations

import os
import asyncio, logging
from config.settings import get_settings
from src.utils.helpers import setup_logging, get_logger
from src.ingestion.fetch_data import fetch_ohlcv
from src.signals.strategies import SmaCross
from src.utils.persist import persist_ohlcv_row, persist_decision

RUN_TAG = os.getenv("RUN_TAG","").strip()
SV_SUFFIX = f"|{RUN_TAG}" if RUN_TAG else ""


log = get_logger(__name__)
SCAN_EVERY_S = int(os.getenv("CORE_SCAN_EVERY_S", "60"))

async def process_symbol(sym: str, s, strat: SmaCross):
    candles = await fetch_ohlcv(sym, s.default_interval, limit=200, use_cache=False)
    if not candles:
        log.warning(f"No candles for {sym}")
        return
    log.info(f"Fetched OHLCV candles for {sym}: {len(candles)} rows")

    # persist the latest bar only (keeps I/O light)
    await persist_ohlcv_row(s.api.exchange, sym, candles[-1])

    sig = strat.generate(candles)
    f = sig.meta.get("fast_ma")
    sl = sig.meta.get("slow_ma")
    if isinstance(f, (int, float)) and isinstance(sl, (int, float)):
        log.info(f"[EVAL] {sym} class={sig.action.value} conf={sig.confidence:.3f} fast={f:.6f} slow={sl:.6f}")
    else:
        log.info(f"[EVAL] {sym} class={sig.action.value} conf={sig.confidence:.3f}")

    await persist_decision(
        exchange=s.api.exchange,
        symbol=sym,
        strategy="SmaCross",
        strategy_version="v1|core|1m",
        action=sig.action.value,
        confidence=float(sig.confidence),
        t=candles[-1]["t"],
        features=None,
    )

async def main():
    s = get_settings()
    setup_logging(s)
    logging.getLogger("httpx").setLevel(logging.INFO)
    log.info("Bootstrapped app")

    strat = SmaCross(fast=10, slow=20, conf_scale=0.001)
    symbols = s.default_symbols

    while True:
        await asyncio.gather(*(process_symbol(sym, s, strat) for sym in symbols))
        await asyncio.sleep(SCAN_EVERY_S)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
