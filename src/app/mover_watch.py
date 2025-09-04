# src/app/mover_watch.py
from __future__ import annotations

import os, asyncio, logging
from typing import List, Tuple, Dict, Any

import httpx

from config.settings import get_settings
from src.utils.helpers import setup_logging, get_logger, load_config
from src.ingestion.fetch_data import fetch_ohlcv
from src.universe.universe import select_universe, UniversePolicy
from src.utils.persist import persist_ohlcv_row, persist_decision
from src.signals.movers import MoversV1

# ---- run tag ----
RUN_TAG = os.getenv("RUN_TAG", "").strip()
SV_SUFFIX = f"|{RUN_TAG}" if RUN_TAG else ""

log = get_logger(__name__)

# ---- Tunables (env overrides) ----
SCAN_EVERY_S   = int(os.getenv("MOVER_SCAN_EVERY_S", "60"))
TOP_PRINT      = int(os.getenv("MOVER_TOP_PRINT", "5"))
LIMIT_BARS     = int(os.getenv("MOVER_LIMIT_BARS", "30"))
CONC_LIMIT     = int(os.getenv("MOVER_CONCURRENCY", "8"))

# Gating + quality
GATE_TOP_K               = int(os.getenv("GATE_TOP_K", "3"))
FILTER_BREAKOUT_MIN_BPS  = float(os.getenv("FILTER_BREAKOUT_MIN_BPS", "0"))
FILTER_VOL_SPIKE_MIN     = float(os.getenv("FILTER_VOL_SPIKE_MIN", "0"))
COOLDOWN_S               = int(os.getenv("COOLDOWN_S", "0"))
MOVER_429_SLEEP_S        = float(os.getenv("MOVER_429_SLEEP_S", "2.5"))

_last_buy: Dict[str, int] = {}  # per-symbol last BUY timestamp (ms)


async def _fetch_one(sym: str, interval: str) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        bars = await fetch_ohlcv(sym, interval, limit=LIMIT_BARS, use_cache=True)
        return sym, bars or []
    except Exception:
        log.exception(f"fetch failed: {sym}")
        return sym, []


async def scan_once(symbols: List[str], interval: str, exchange: str, strat: MoversV1) -> List[Tuple[str, float]]:
    sem = asyncio.Semaphore(CONC_LIMIT)

    async def guarded(sym: str):
        async with sem:
            return await _fetch_one(sym, interval)

    pairs = await asyncio.gather(*(guarded(s) for s in symbols))
    results: List[Dict[str, Any]] = []

    # compute scores & persist latest bar
    for sym, candles in pairs:
        if not candles:
            continue
        last_bar = candles[-1]
        await persist_ohlcv_row(exchange, sym, last_bar)
        res = strat.generate(candles)
        results.append({"sym": sym, "t": int(last_bar["t"]), "score": float(res.score), "meta": res.meta or {}})

    if not results:
        return []

    # Top-K
    results.sort(key=lambda r: r["score"], reverse=True)
    top_syms = {r["sym"] for r in results[:max(0, GATE_TOP_K)]}

    # Decide action
    out_for_print: List[Tuple[str, float]] = []
    for r in results:
        sym, t, score, meta = r["sym"], r["t"], r["score"], r["meta"]

        allow = (sym in top_syms) and (score >= strat.thresh)
        if allow and FILTER_BREAKOUT_MIN_BPS > 0:
            allow = allow and (float(meta.get("breakout_bps", 0.0)) >= FILTER_BREAKOUT_MIN_BPS)
        if allow and FILTER_VOL_SPIKE_MIN > 0:
            allow = allow and (float(meta.get("vol_spike", 1.0)) >= FILTER_VOL_SPIKE_MIN)
        if allow and COOLDOWN_S > 0:
            last = _last_buy.get(sym, 0)
            if (t - last) < COOLDOWN_S * 1000:
                allow = False
            else:
                _last_buy[sym] = t

        action = "BUY" if allow else "HOLD"

        await persist_decision(
            exchange=exchange,
            symbol=sym,
            strategy=strat.name,
            strategy_version=f"v1|movers|1m{SV_SUFFIX}",
            action=action,
            confidence=score,
            t=t,
            features=None,
        )
        out_for_print.append((sym, score))

    out_for_print.sort(key=lambda x: x[1], reverse=True)
    return out_for_print


async def main() -> int:
    s = get_settings()
    setup_logging(s)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("Mover watcher booted")

    cfg_assets = {}
    try:
        cfg_assets = load_config('assets')
    except Exception:
        cfg_assets = {}
    pol = UniversePolicy(
        max_symbols=int(os.getenv("UNIVERSE_MAX", str(cfg_assets.get('universe_max', 25)))),
        movers_window_m=int(os.getenv("UNIVERSE_MOVERS_WINDOW", "5")),
        refresh_ttl_s=int(os.getenv("UNIVERSE_REFRESH_TTL", str(cfg_assets.get('refresh_ttl_s', 300)))),
        min_24h_notional=float(os.getenv("UNIVERSE_MIN_NOTIONAL", str(cfg_assets.get('min_24h_notional', 100000)))),
        # exclude=("BTC-USD","ETH-USD","SOL-USD"),
    )
    cfg_risk = {}
    try:
        cfg_risk = load_config('risk')
    except Exception:
        cfg_risk = {}
    strat = MoversV1(
        thresh=float(os.getenv("MOVER_THRESH", str(cfg_risk.get('mover_thresh', 0.010)))),
        require_trend=bool(int(os.getenv("MOVER_REQUIRE_TREND", str(int(bool(cfg_risk.get('require_trend', 0))))))),
        min_atr_bps=float(os.getenv("MOVER_MIN_ATR_BPS", str(cfg_risk.get('mover_min_atr_bps', 5)))),
    )

    interval, exchange = s.default_interval, s.api.exchange

    try:
        while True:
            # 429-safe universe refresh
            try:
                symbols = await select_universe(pol)
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 429:
                    ra = e.response.headers.get("Retry-After")
                    wait_s = float(ra) if ra else MOVER_429_SLEEP_S
                    log.warning(f"Universe refresh hit 429; sleeping {wait_s:.1f}s and continuing.")
                    await asyncio.sleep(wait_s)
                    continue
                log.warning(f"Universe refresh HTTP error: {e}; backing off 3s.")
                await asyncio.sleep(3)
                continue
            except Exception as e:
                log.warning(f"Universe refresh error: {e}; backing off 3s.")
                await asyncio.sleep(3)
                continue

            log.info(f"Universe={len(symbols)} interval={interval}")

            # Scan (also 429-safe)
            try:
                top = await scan_once(symbols, interval, exchange, strat)
                if top:
                    head = ", ".join(f"{sym}:{sc:.4f}" for sym, sc in top[:int(os.getenv('MOVER_TOP_PRINT','5'))])
                    log.info(f"Top movers (1m): {head}")
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 429:
                    ra = e.response.headers.get("Retry-After")
                    wait_s = float(ra) if ra else MOVER_429_SLEEP_S
                    log.warning(f"Scan hit 429; sleeping {wait_s:.1f}s and continuing.")
                    await asyncio.sleep(wait_s)
                else:
                    log.warning(f"Scan HTTP error: {e}; backing off 3s.")
                    await asyncio.sleep(3)
            except Exception as e:
                log.warning(f"Scan error: {e}; backing off 3s.")
                await asyncio.sleep(3)

            await asyncio.sleep(SCAN_EVERY_S)

    except (asyncio.CancelledError, KeyboardInterrupt):
        log.info("Mover watcher stopped by user.")
        return 0
    except Exception as e:
        log.exception(f"Mover watcher crashed: {e}")
        return 1


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
