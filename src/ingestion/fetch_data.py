# src/ingestion/fetch_data.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx

from config.settings import get_settings
from src.utils.helpers import get_logger, async_retry, TTLCache

log = get_logger(__name__)
S = get_settings()

# ---------------------------------
# Shared HTTP client + small TTL cache
# ---------------------------------
_client_lock = asyncio.Lock()
_client: Optional[httpx.AsyncClient] = None

_cache = TTLCache(
    ttl_seconds=S.cache.ttl_seconds,
    max_items=S.cache.max_items,
)

# ---------------------------------
# SSL verify configuration (robust to bad env)
# ---------------------------------
def _resolve_ssl_verify_and_trust_env() -> tuple[Optional[bool | str], bool]:
    """
    Decide how httpx should verify SSL and whether to respect shell env (SSL_CERT_FILE, etc.).
    Falls back to safe defaults if new config fields are missing.
    """
    # Defaults if fields are not present in your settings yet
    ssl_verify_mode: str = getattr(getattr(S, "api"), "ssl_verify", "certifi")
    ssl_verify_path: str = getattr(getattr(S, "api"), "ssl_verify_path", "")
    ssl_trust_env: bool = bool(getattr(getattr(S, "api"), "ssl_trust_env", False))

    verify: Optional[bool | str]
    if ssl_verify_mode == "false":
        verify = False
    elif ssl_verify_mode == "path" and ssl_verify_path:
        verify = ssl_verify_path
    elif ssl_verify_mode == "certifi":
        try:
            import certifi  # type: ignore
            verify = certifi.where()
        except Exception:
            # Certifi not installed; use system default
            verify = True
    else:
        # "system" or unknown -> system default
        verify = True

    return verify, ssl_trust_env


# ---------------------------------
# HTTP client helpers
# ---------------------------------
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
        headers = {"X-MBX-APIKEY": S.api.api_key} if S.api.api_key else None
        verify, trust_env = _resolve_ssl_verify_and_trust_env()

        _client = httpx.AsyncClient(
            base_url=S.api.base_url.rstrip("/"),
            timeout=S.api.http_timeout,
            limits=limits,
            headers=headers,
            verify=verify,
            trust_env=trust_env,  # if False, ignores bad SSL_CERT_FILE/REQUESTS_CA_BUNDLE
        )
        log.debug(f"Created httpx AsyncClient for {S.api.exchange} @ {S.api.base_url}")
        return _client


async def shutdown_session() -> None:
    """Gracefully close the shared client (call at app shutdown)."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        log.debug("Closed httpx AsyncClient")


@async_retry(attempts=S.api.http_retries, base_delay=S.api.http_backoff_base)
async def _get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    client = await _get_client()
    resp = await client.get(path.lstrip("/"), params=params)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------
# Exchange-specific normalization
# ---------------------------------
def _norm_interval_coinbase(interval: str) -> int:
    """
    Coinbase Exchange granularity in seconds: 60, 300, 900, 3600, 21600, 86400
    Maps common strings to supported granularities.
    """
    m = (interval or "").lower().strip()
    return {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "6h": 21600,
        "1d": 86400,
    }.get(m, 60)


def _norm_symbol_coinbase(symbol: str) -> str:
    """
    Normalize to Coinbase product_id format: BASE-QUOTE (e.g., BTC-USD / ETH-USDT)
    Accepts BTCUSD/BTCUSDT/BTC-USD, etc.
    """
    s = symbol.upper().replace(" ", "").replace("_", "").replace("/", "")
    if "-" in s:
        return s
    if s.endswith("USDT"):
        return f"{s[:-4]}-USDT"
    if s.endswith("USD"):
        return f"{s[:-3]}-USD"
    return f"{s}-USD"


def _norm_interval_kraken(interval: str) -> int:
    """
    Kraken interval is in minutes: 1, 5, 15, 60, 240, 1440.
    """
    m = (interval or "").lower().strip()
    return {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }.get(m, 1)


def _norm_symbol_kraken(symbol: str) -> str:
    """
    Normalize to Kraken REST OHLC pairs (e.g., XBTUSD, ETHUSD).
    Kraken uses XBT for BTC.
    """
    s = symbol.upper().replace("-", "").replace("/", "")
    for quote in ("USDT", "USD", "EUR", "GBP"):
        if s.endswith(quote):
            base, q = s[: -len(quote)], quote
            if base == "BTC":
                base = "XBT"
            return f"{base}{q}"
    base = "XBT" if s == "BTC" else s
    return f"{base}USD"


# ---------------------------------
# Parsers
# ---------------------------------
def _parse_coinbase_candles(raw: List[List[Any]]) -> List[Dict[str, Any]]:
    """
    Coinbase candles schema: [time, low, high, open, close, volume]
    time is seconds (convert to ms).
    """
    out: List[Dict[str, Any]] = []
    for r in raw:
        t, lo, hi, op, cl, vol = r
        out.append(
            {
                "t": int(t) * 1000,
                "o": float(op),
                "h": float(hi),
                "l": float(lo),
                "c": float(cl),
                "v": float(vol),
            }
        )
    out.sort(key=lambda x: x["t"])
    return out


def _parse_kraken_candles(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Kraken OHLC schema:
    {"result": {"PAIR": [[time, open, high, low, close, vwap, volume, count], ...]}}
    """
    result = raw.get("result", {})
    series = next(iter(result.values())) if result else []
    out: List[Dict[str, Any]] = []
    for t, op, hi, lo, cl, vwap, vol, cnt in series:
        out.append(
            {
                "t": int(t) * 1000,
                "o": float(op),
                "h": float(hi),
                "l": float(lo),
                "c": float(cl),
                "v": float(vol),
            }
        )
    out.sort(key=lambda x: x["t"])
    return out


# ---------------------------------
# Public ingestion API
# ---------------------------------
async def fetch_ohlcv(
    symbol: str,
    interval: str | None = None,
    limit: int = 100,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV candles as a list of dicts: {t,o,h,l,c,v}, ascending by time.
    Supported exchanges: coinbase, kraken (set via Settings/api.exchange).
    """
    x = S.api.exchange.lower()
    interval = interval or S.default_interval

    if x == "coinbase":
        product_id = _norm_symbol_coinbase(symbol)
        gran = _norm_interval_coinbase(interval)
        cache_key = ("ohlcv", "coinbase", product_id, gran, limit)
        if use_cache and S.cache.enabled:
            cached = await _cache.get(cache_key)
            if cached is not None:
                return cached

        # GET /products/{product_id}/candles?granularity=seconds
        raw = await _get_json(f"/products/{product_id}/candles", {"granularity": gran})
        parsed = _parse_coinbase_candles(raw)[-limit:]

        if use_cache and S.cache.enabled:
            await _cache.set(cache_key, parsed)
        return parsed

    elif x == "kraken":
        pair = _norm_symbol_kraken(symbol)
        kint = _norm_interval_kraken(interval)
        cache_key = ("ohlcv", "kraken", pair, kint, limit)
        if use_cache and S.cache.enabled:
            cached = await _cache.get(cache_key)
            if cached is not None:
                return cached

        # GET /0/public/OHLC?pair=PAIR&interval=minutes
        raw = await _get_json("/0/public/OHLC", {"pair": pair, "interval": kint})
        parsed = _parse_kraken_candles(raw)[-limit:]

        if use_cache and S.cache.enabled:
            await _cache.set(cache_key, parsed)
        return parsed

    else:
        raise ValueError(f"Unsupported exchange in settings: {S.api.exchange}")


async def fetch_ohlcv_many(
    symbols: List[str],
    interval: str | None = None,
    limit: int = 100,
    use_cache: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Concurrently fetch candles for many symbols (with a concurrency cap).
    """
    sem = asyncio.Semaphore(max(1, int(S.perf.max_concurrency)))

    async def _one(sym: str):
        async with sem:
            return sym, await fetch_ohlcv(sym, interval=interval, limit=limit, use_cache=use_cache)

    results = await asyncio.gather(*[_one(s) for s in symbols])
    return {k: v for k, v in results}


async def fetch_orderbook_top(symbol: str) -> Dict[str, Optional[float]]:
    """
    Return best bid/ask snapshot for the configured exchange.
    """
    x = S.api.exchange.lower()

    if x == "coinbase":
        product_id = _norm_symbol_coinbase(symbol)
        # GET /products/{product_id}/book?level=1
        raw = await _get_json(f"/products/{product_id}/book", {"level": 1})
        bid = float(raw["bids"][0][0]) if raw.get("bids") else None
        ask = float(raw["asks"][0][0]) if raw.get("asks") else None
        return {"bid": bid, "ask": ask}

    elif x == "kraken":
        pair = _norm_symbol_kraken(symbol)
        # GET /0/public/Depth?pair=PAIR&count=5
        raw = await _get_json("/0/public/Depth", {"pair": pair, "count": 5})
        book = next(iter(raw.get("result", {}).values()), {})
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        bid = float(bids[0][0]) if bids else None
        ask = float(asks[0][0]) if asks else None
        return {"bid": bid, "ask": ask}

    else:
        raise ValueError(f"Unsupported exchange in settings: {S.api.exchange}")
