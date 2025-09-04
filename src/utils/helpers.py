# src/utils/helpers.py
from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import asdict
from datetime import datetime, timezone
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
