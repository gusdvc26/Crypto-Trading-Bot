# src/app/sniper_paper.py
from __future__ import annotations

import asyncio, os, logging
from typing import Dict

from config.settings import get_settings
from src.utils.helpers import setup_logging, get_logger
from src.universe.universe import select_universe, UniversePolicy
from src.ingestion.ws_coinbase import coinbase_ticker_stream
from src.processing.microbars import MicroBarAgg, RollingBars
from src.signals.movers import MoversV1
from src.utils.persist import persist_ohlcv_row, persist_decision

# ---- run tag (helps isolate runs in reports) ----
RUN_TAG = os.getenv("RUN_TAG", "").strip()
SV_SUFFIX = f"|{RUN_TAG}" if RUN_TAG else ""

log = get_logger(__name__)

# ---- Tunables (env overrides) ----
WINDOW_S      = int(os.getenv("SNIPER_BAR_SECONDS", "10"))   # 5–15 typical
HISTORY_BARS  = int(os.getenv("SNIPER_HISTORY_BARS", "60"))
TOP_PRINT     = int(os.getenv("SNIPER_TOP_PRINT", "8"))

# Gating + filters
GATE_TOP_K              = int(os.getenv("GATE_TOP_K", "3"))
FILTER_BREAKOUT_MIN_BPS = float(os.getenv("FILTER_BREAKOUT_MIN_BPS", "0"))
FILTER_VOL_SPIKE_MIN    = float(os.getenv("FILTER_VOL_SPIKE_MIN", "0"))
COOLDOWN_S              = int(os.getenv("COOLDOWN_S", "0"))

_last_buy: Dict[str, int] = {}  # per-symbol last BUY timestamp (ms)
scores: Dict[str, float]  = {}  # live scores per symbol


async def main() -> int:
    s = get_settings()
    setup_logging(s)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info(f"Sniper (log-only) on {WINDOW_S}s bars booted")

    # Universe policy (stagger TTL from Movers to reduce REST bursts)
    pol = UniversePolicy(
        max_symbols=int(os.getenv("UNIVERSE_MAX", "30")),
        movers_window_m=int(os.getenv("UNIVERSE_MOVERS_WINDOW", "5")),
        refresh_ttl_s=int(os.getenv("UNIVERSE_REFRESH_TTL", "540")),
        min_24h_notional=float(os.getenv("UNIVERSE_MIN_NOTIONAL", "300000")),
        # exclude=("BTC-USD","ETH-USD","SOL-USD"),
    )
    symbols = await select_universe(pol)
    log.info(f"Universe size: {len(symbols)}")

    # Strategy tuned for microbars
    strat = MoversV1(
        thresh=float(os.getenv("MOVER_THRESH", "0.035")),
        min_atr_bps=float(os.getenv("MOVER_MIN_ATR_BPS", "8")),
        require_trend=bool(int(os.getenv("MOVER_REQUIRE_TREND", "0"))),
    )

    agg  = MicroBarAgg(window_ms=WINDOW_S * 1000)
    hist = RollingBars(maxlen=HISTORY_BARS)

    try:
        async for tick in coinbase_ticker_stream(symbols):
            flushed = agg.update(tick)           # may flush 0..N bars
            pid = tick["product_id"]

            for bar in flushed:
                # Tier-0 persistence: bar
                await persist_ohlcv_row(s.api.exchange, pid, bar)

                bars = hist.append(pid, bar)
                if len(bars) < 30:
                    continue

                res = strat.generate(bars)
                scores[pid] = float(res.score)

                # ---- Top-K gating across current scores ----
                top_syms = {
                    k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max(0, GATE_TOP_K)]
                }

                # Allow only if Top-K and above threshold
                allow = (pid in top_syms) and (res.score >= strat.thresh)

                # Hard filters
                if allow and FILTER_BREAKOUT_MIN_BPS > 0:
                    allow = allow and (float(res.meta.get("breakout_bps", 0.0)) >= FILTER_BREAKOUT_MIN_BPS)
                if allow and FILTER_VOL_SPIKE_MIN > 0:
                    allow = allow and (float(res.meta.get("vol_spike", 1.0)) >= FILTER_VOL_SPIKE_MIN)

                # Cooldown
                if allow and COOLDOWN_S > 0:
                    last = _last_buy.get(pid, 0)
                    if (bar["t"] - last) < COOLDOWN_S * 1000:
                        allow = False
                    else:
                        _last_buy[pid] = bar["t"]

                action = "BUY" if allow else "HOLD"

                # Tier-1 persistence: decision
                await persist_decision(
                    exchange=s.api.exchange,
                    symbol=pid,
                    strategy=strat.name + f"_{WINDOW_S}s",
                    strategy_version=f"v1|sniper|{WINDOW_S}s{SV_SUFFIX}",
                    action=action,
                    confidence=float(res.score),
                    t=bar["t"],
                    features=None,
                )

            if flushed and scores:
                head = ", ".join(
                    f"{k}:{scores[k]:.4f}"
                    for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:TOP_PRINT]
                )
                log.info(f"Top movers ({WINDOW_S}s): {head}")

    except (asyncio.CancelledError, KeyboardInterrupt):
        log.info("Sniper stopped by user.")
        return 0
    except Exception as e:
        log.exception(f"Sniper crashed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
