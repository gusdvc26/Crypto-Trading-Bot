# src/utils/helpers.py
from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import asdict
from datetime import datetime, timezone
import time
import math
from typing import Optional, Dict
import os
import ast
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Coroutine, Optional, Type

from config.settings import get_settings, Settings


# -----------------------
# Time utilities
# -----------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_sec() -> int:
    return int(utc_now().timestamp())


def ts_ms() -> int:
    return int(utc_now().timestamp() * 1000)


# ---- New tiny helpers for UTC ms timestamps ----
def now_utc_ms() -> int:
    """Return current UTC epoch milliseconds as int.

    Small helper used when recording decision timestamps.
    """
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def to_utc_ms(x) -> int:
    """Convert an ISO string or numeric ms to UTC epoch ms (int).

    If ``x`` is already an int/float, it is returned as int(ms).
    Otherwise, parse an ISO 8601 string (tolerates trailing 'Z') and return ms.
    """
    from datetime import datetime, timezone as _tz
    if isinstance(x, (int, float)):
        return int(x)
    return int(datetime.fromisoformat(str(x).replace('Z', '+00:00')).replace(tzinfo=_tz.utc).timestamp() * 1000)


# -----------------------
# Rate limiting and backoff
# -----------------------
class TokenBucketLimiter:
    """
    Lightweight per-key token bucket limiter for async code.

    - Each key has its own bucket with `capacity` tokens and `rate_per_sec` refill speed.
    - `acquire(key)` waits until a token is available and consumes one.
    """

    def __init__(self, rate_per_sec: float = 5.0, capacity: int = 5):
        self._rate = float(rate_per_sec)
        self._cap = int(capacity)
        self._state: Dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    def _refill(self, tokens: float, last_ts: float) -> tuple[float, float]:
        now = time.monotonic()
        dt = max(0.0, now - last_ts)
        tokens = min(self._cap, tokens + dt * self._rate)
        return tokens, now

    async def acquire(self, key: str):
        """
        Await until a token is available for `key`. Returns an async context manager
        which is a no-op (kept for symmetry with other limiters).
        """
        async with self._lock:
            tokens, last = self._state.get(key, (float(self._cap), time.monotonic()))
            tokens, last = self._refill(tokens, last)
            if tokens < 1.0:
                # compute time to 1 token
                need = 1.0 - tokens
                wait_s = need / max(self._rate, 1e-9)
                await asyncio.sleep(wait_s)
                tokens, last = self._refill(tokens, last)
            # consume
            tokens -= 1.0
            self._state[key] = (tokens, last)

        class _Noop:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return _Noop()


class GlobalLimiter:
    """
    Simple global in-flight limiter built on asyncio.Semaphore.
    Use to cap concurrent network requests process-wide.
    """

    def __init__(self, max_in_flight: int = 100):
        self._sem = asyncio.Semaphore(max(1, int(max_in_flight)))

    async def acquire(self):
        await self._sem.acquire()

        class _Releaser:
            def __init__(self, sem: asyncio.Semaphore):
                self._sem = sem

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self._sem.release()
                return False

        return _Releaser(self._sem)


def backoff_delay(attempt: int, base: float = 0.25, factor: float = 2.0, jitter: float = 0.1, max_delay: float = 5.0) -> float:
    """
    Jittered exponential backoff delay in seconds for the given attempt (1-based).

    delay = min(max_delay, base * factor**(attempt-1)) + random_jitter
    where random_jitter in [0, jitter).
    """
    import random

    deterministic = base * (factor ** max(0, attempt - 1))
    delay = min(max_delay, deterministic)
    return float(delay + (random.random() * max(0.0, jitter)))


# -----------------------
# Config loader (YAML with env overrides)
# -----------------------
def _simple_yaml_parse(text: str) -> dict:
    """Parse a minimal subset of YAML for flat key: value pairs.

    Supports comments (#), strings, ints/floats/bools, and Python-literal lists/dicts
    when provided inline (parsed via ast.literal_eval).
    """
    out: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            continue
        key, val = line.split(':', 1)
        key = key.strip()
        val = val.strip()
        # Remove optional quotes
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            out[key] = val[1:-1]
            continue
        # Try python literal (for [a,b] etc.)
        try:
            out[key] = ast.literal_eval(val)
            continue
        except Exception:
            pass
        # Try bool/int/float
        low = val.lower()
        if low in ("true", "false"):
            out[key] = (low == "true")
            continue
        try:
            if '.' in val:
                out[key] = float(val)
            else:
                out[key] = int(val)
            continue
        except Exception:
            pass
        out[key] = val
    return out


def _coerce_env_value(raw: str):
    """Coerce env string to bool/int/float/list if possible; else raw string."""
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def load_config(name: str) -> dict:
    """Load YAML config from `configs/{name}.yml` and apply env overrides.

    Env overrides apply to flat top-level keys using the pattern
    `{NAME}_{KEY}` uppercased. For example, `risk.yml` key `tp_bps` can be
    overridden with env `RISK_TP_BPS=30`.

    Raises FileNotFoundError with a helpful message if the config file is missing.
    """
    from config.settings import get_settings as _get_settings

    s = _get_settings()
    path = (s.root_dir / 'configs' / f'{name}.yml')
    if not path.exists():
        raise FileNotFoundError(
            f"Config '{name}' not found at {path}. Create it or choose one of: assets, risk, exec."
        )
    text = path.read_text(encoding='utf-8')

    data: dict
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        data = dict(loaded or {})
    except Exception:
        # Fall back to minimal parser without extra dependencies
        data = _simple_yaml_parse(text)

    # Apply env overrides for flat keys
    prefix = name.upper() + '_'
    for k in list(data.keys()):
        env_key = prefix + k.upper()
        if env_key in os.environ:
            data[k] = _coerce_env_value(os.environ[env_key])

    return data


# -----------------------
# Logging setup
# -----------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).replace(tzinfo=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(settings: Optional[Settings] = None) -> None:
    s = settings or get_settings()

    root = logging.getLogger()
    root.setLevel(getattr(logging, s.logging.level.upper(), logging.INFO))
    # drop existing handlers to avoid duplicates in notebooks / reloads
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(root.level)
    ch.setFormatter(JsonFormatter() if s.logging.json else logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    root.addHandler(ch)

    # Rotating file handler (optional)
    if s.logging.to_file:
        file_path = s.logs_dir / s.logging.filename
        fh = RotatingFileHandler(file_path, maxBytes=10_000_000, backupCount=5)
        fh.setLevel(root.level)
        fh.setFormatter(JsonFormatter() if s.logging.json else logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    # Ensure logging is configured at first call
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)


# -----------------------
# Async retry with jitter (exponential backoff)
# -----------------------
def async_retry(
    exceptions: tuple[Type[BaseException], ...] = (Exception,),
    attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 5.0,
    jitter: float = 0.1,
) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, Any]]]:
    """
    Decorator for async functions. Retries on `exceptions` with exponential backoff + jitter.
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= attempts:
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += random.random() * jitter
                    await asyncio.sleep(delay)
        return wrapper
    return decorator


# -----------------------
# Tiny in-memory TTL cache (async-safe)
# -----------------------
class TTLCache:
    """
    Minimal async-safe TTL cache for hot paths (e.g., last candles).
    Not a replacement for Redis; just enough for local speedups.
    """
    __slots__ = ("_data", "_ttl", "_max", "_lock")

    def __init__(self, ttl_seconds: int = 3, max_items: int = 4096):
        self._data: dict[Any, tuple[int, Any]] = {}
        self._ttl = ttl_seconds
        self._max = max_items
        self._lock = asyncio.Lock()

    async def get(self, key: Any) -> Optional[Any]:
        now = ts_sec()
        async with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires, value = item
            if now >= expires:
                self._data.pop(key, None)
                return None
            return value

    async def set(self, key: Any, value: Any) -> None:
        now = ts_sec()
        async with self._lock:
            if len(self._data) >= self._max:
                # simple drop: remove one arbitrary expired or first key
                self._data.pop(next(iter(self._data)), None)
            self._data[key] = (now + self._ttl, value)

    async def purge_expired(self) -> None:
        now = ts_sec()
        async with self._lock:
            expired = [k for k, (exp, _) in self._data.items() if now >= exp]
            for k in expired:
                self._data.pop(k, None)
