# src/universe/universe.py
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config.settings import get_settings
from src.utils.helpers import get_logger, TTLCache

log = get_logger(__name__)
S = get_settings()

# ---------------- SSL/HTTP client (mirrors fetch_data) ----------------
_client_lock = asyncio.Lock()
_client: Optional[httpx.AsyncClient] = None

def _resolve_ssl_verify_and_trust_env() -> tuple[Optional[bool | str], bool]:
    mode: str = getattr(S.api, "ssl_verify", "certifi")
    path: str = getattr(S.api, "ssl_verify_path", "")
    trust_env: bool = bool(getattr(S.api, "ssl_trust_env", False))
    verify: Optional[bool | str]
    if mode == "false":
        verify = False
    elif mode == "path" and path:
        verify = path
    elif mode == "certifi":
        try:
            import certifi  # type: ignore
            verify = certifi.where()
        except Exception:
            verify = True
    else:
        verify = True
    return verify, trust_env

async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client and not _client.is_closed:
        return _client
    async with _client_lock:
        if _client and not _client.is_closed:
            return _client
        limits = httpx.Limits(
            max_connections=S.api.http_conn_limit,
            max_keepalive_connections=S.api.http_conn_limit,
        )
        verify, trust_env = _resolve_ssl_verify_and_trust_env()
        _client = httpx.AsyncClient(
            base_url=S.api.base_url.rstrip("/"),
            timeout=S.api.http_timeout,
            limits=limits,
            verify=verify,
            trust_env=trust_env,
        )
        log.debug(f"[universe] httpx client for {S.api.exchange} @ {S.api.base_url}")
        return _client

async def _get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    cli = await _get_client()
    r = await cli.get(path.lstrip("/"), params=params)
    r.raise_for_status()
    return r.json()

# ---------------- Simple rate limiter to avoid 429s ----------------
# Coinbase public REST is ~10 req/s per IP. We'll default to 8 rps.
_RPS_LIMIT = max(1, int(os.getenv("UNIVERSE_RPS", "8")))
_CONC_LIMIT = max(1, int(os.getenv("UNIVERSE_CONCURRENCY", "10")))
_rate_lock = asyncio.Lock()
_last_req_time = 0.0

async def _respect_rps() -> None:
    """Ensure we don't exceed the configured RPS across concurrent tasks."""
    global _last_req_time
    min_interval = 1.0 / float(_RPS_LIMIT)
    async with _rate_lock:
        now = asyncio.get_running_loop().time()
        wait = max(0.0, _last_req_time + min_interval - now)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_req_time = asyncio.get_running_loop().time()

# ---------------- Config & policy ----------------
@dataclass
class UniversePolicy:
    max_symbols: int = 20                      # how many to keep
    quotes: Tuple[str, ...] = ("USD", "USDT")  # allowed quote currencies
    include: Tuple[str, ...] = ()              # always include
    exclude: Tuple[str, ...] = ()              # never include
    min_price: float = 0.005                   # filter dust
    min_24h_notional: float = 50_000.0         # $ filter for liquidity
    refresh_ttl_s: int = 300                   # refresh universe every 5 min
    movers_window_m: int = 0                   # 0 = off; set to 5/15 to boost movers

# Global cache (ttl will be aligned to policy at runtime)
_universe_cache = TTLCache(ttl_seconds=300, max_items=8)

# ---------------- Exchange adapters ----------------
async def _coinbase_list_products() -> List[Dict[str, Any]]:
    prods = await _get_json("/products")
    out = []
    for p in prods:
        quote = (p.get("quote_currency") or p.get("quote") or "").upper()
        base = (p.get("base_currency") or p.get("base") or "").upper()
        pid  = (p.get("id") or f"{base}-{quote}")
        status = p.get("status") or ("online" if not p.get("trading_disabled") else "offline")
        out.append({"id": pid, "base": base, "quote": quote, "status": status})
    return out

async def _coinbase_ticker(product_id: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        await _respect_rps()
        t = await _get_json(f"/products/{product_id}/ticker")
        price = float(t.get("price")) if t.get("price") is not None else None
        vol = float(t.get("volume")) if t.get("volume") is not None else None  # 24h base vol
        return price, vol
    except Exception as e:
        log.debug(f"[universe] ticker fail {product_id}: {e}")
        return None, None

async def _kraken_list_pairs() -> List[Dict[str, Any]]:
    data = await _get_json("/0/public/AssetPairs")
    res = data.get("result", {})
    out = []
    for _k, v in res.items():
        base = (v.get("base", "")).upper().replace("XBT", "BTC")
        quote = (v.get("quote", "")).upper().replace("XBT", "BTC")
        sym = f"{base}-{quote}"
        out.append({"id": sym, "base": base, "quote": quote, "status": "online"})
    return out

async def _kraken_ticker(pair_norm: str) -> Tuple[Optional[float], Optional[float]]:
    base, quote = pair_norm.split("-")
    if base == "BTC":
        base = "XBT"
    pair = f"{base}{quote}"
    try:
        await _respect_rps()
        data = await _get_json("/0/public/Ticker", {"pair": pair})
        obj = next(iter(data.get("result", {}).values()), None)
        if not obj:
            return None, None
        price = float(obj["c"][0])
        vol24 = float(obj["v"][1])
        return price, vol24
    except Exception as e:
        log.debug(f"[universe] kraken ticker fail {pair}: {e}")
        return None, None

# ---------------- Ranking helpers ----------------
async def _rank_candidates(products: List[Dict[str, Any]], policy: UniversePolicy) -> List[Tuple[str, float]]:
    """
    Return list of (symbol, 24h_notional_usd) sorted desc.
    """
    x = S.api.exchange.lower()
    sem = asyncio.Semaphore(_CONC_LIMIT)
    results: List[Tuple[str, float]] = []

    async def one(p: Dict[str, Any]):
        if p["quote"] not in policy.quotes or p["status"] != "online":
            return
        sym = p["id"]
        async with sem:
            # small retry loop for transient 429s/none
            for attempt in range(4):
                if x == "coinbase":
                    price, vol = await _coinbase_ticker(sym)
                else:
                    price, vol = await _kraken_ticker(sym)
                if price is not None and vol is not None:
                    break
                await asyncio.sleep(0.25 * (2 ** attempt))
        if price is None or vol is None:
            return
        if price < policy.min_price:
            return
        notional = price * vol
        if notional < policy.min_24h_notional:
            return
        results.append((sym, notional))

    await asyncio.gather(*(one(p) for p in products))
    results.sort(key=lambda t: t[1], reverse=True)
    return results

# Optional movers boost using recent % change (kept light)
async def _apply_movers_boost(
    ranked: List[Tuple[str, float]],
    window_m: int,
    boost_weight: float = 0.2,
) -> List[Tuple[str, float]]:
    if window_m <= 0 or not ranked:
        return ranked
    topK = ranked[: min(50, len(ranked))]
    from src.ingestion.fetch_data import fetch_ohlcv
    sem = asyncio.Semaphore(min(30, _CONC_LIMIT))
    boosted: List[Tuple[str, float]] = []

    async def one(sym: str, base_score: float):
        async with sem:
            candles = await fetch_ohlcv(sym, interval="1m", limit=window_m + 1, use_cache=True)
        if len(candles) >= 2:
            c0, c1 = candles[0]["c"], candles[-1]["c"]
            pct = (c1 - c0) / (abs(c0) + 1e-12)
        else:
            pct = 0.0
        boosted.append((sym, base_score * (1.0 + boost_weight * abs(pct))))

    await asyncio.gather(*(one(sym, sc) for sym, sc in topK))
    boosted.sort(key=lambda t: t[1], reverse=True)
    rest = [r for r in ranked if r[0] not in {s for s, _ in boosted}]
    return boosted + rest

# ---------------- Public API ----------------
async def select_universe(policy: UniversePolicy | None = None) -> List[str]:
    """
    Builds a dynamic symbol universe according to policy and current exchange.
    Result is cached for policy.refresh_ttl_s seconds.
    """
    global _universe_cache
    policy = policy or UniversePolicy()

    # Align cache TTL to policy (no ttl_override arg)
    if getattr(_universe_cache, "ttl_seconds", None) != policy.refresh_ttl_s:
        _universe_cache = TTLCache(ttl_seconds=policy.refresh_ttl_s, max_items=8)

    cache_key = (
        "universe", S.api.exchange, policy.max_symbols, policy.quotes,
        policy.min_24h_notional, policy.movers_window_m
    )
    cached = await _universe_cache.get(cache_key)
    if cached:
        return cached

    x = S.api.exchange.lower()
    if x == "coinbase":
        products = await _coinbase_list_products()
    elif x == "kraken":
        products = await _kraken_list_pairs()
    else:
        raise ValueError(f"Unsupported exchange: {S.api.exchange}")

    ranked = await _rank_candidates(products, policy)
    if policy.movers_window_m > 0:
        ranked = await _apply_movers_boost(ranked, policy.movers_window_m)

    included = list(policy.include)
    excluded = set(policy.exclude)
    for sym, _score in ranked:
        if sym in excluded or sym in included:
            continue
        included.append(sym)
        if len(included) >= policy.max_symbols:
            break

    await _universe_cache.set(cache_key, included)  # default TTL applies
    return included

# Convenience CLI
async def _cli():
    pol = UniversePolicy(
        max_symbols=int(os.getenv("UNIVERSE_MAX", "20")),
        movers_window_m=int(os.getenv("UNIVERSE_MOVERS_WINDOW", "0")),
        refresh_ttl_s=int(os.getenv("UNIVERSE_REFRESH_TTL", "300")),
    )
    syms = await select_universe(pol)
    print(f"{S.api.exchange} universe ({len(syms)}): {', '.join(syms)}")

if __name__ == "__main__":
    asyncio.run(_cli())
