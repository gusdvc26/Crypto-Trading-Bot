# config/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path


class Env(str, Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


# Resolve project root from this file’s location
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../crypto-signal-bot
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")
    json: bool = bool(int(os.getenv("LOG_JSON", "0")))          # 1 -> JSON logs
    to_file: bool = bool(int(os.getenv("LOG_TO_FILE", "1")))    # 1 -> write file
    dir: Path = LOGS_DIR
    filename: str = os.getenv("LOG_FILE_NAME", "app.log")


@dataclass(frozen=True)
class APISettings:
    # Default to Coinbase Exchange (US-friendly)
    exchange: str = os.getenv("EXCHANGE", "coinbase")
    base_url: str = os.getenv("BASE_URL", "https://api.exchange.coinbase.com")
    ws_url: str = os.getenv("WS_URL", "wss://ws-feed.exchange.coinbase.com")
    api_key: str = os.getenv("API_KEY", "")
    api_secret: str = os.getenv("API_SECRET", "")

    # Networking / retries
    http_timeout: float = float(os.getenv("HTTP_TIMEOUT", "10"))    # seconds
    http_conn_limit: int = int(os.getenv("HTTP_CONN_LIMIT", "100"))
    http_retries: int = int(os.getenv("HTTP_RETRIES", "3"))
    http_backoff_base: float = float(os.getenv("HTTP_BACKOFF_BASE", "0.25"))  # seconds


@dataclass(frozen=True)
class CacheSettings:
    enabled: bool = bool(int(os.getenv("CACHE_ENABLED", "1")))
    ttl_seconds: int = int(os.getenv("CACHE_TTL", "3"))
    max_items: int = int(os.getenv("CACHE_MAX_ITEMS", "4096"))


@dataclass(frozen=True)
class PerfSettings:
    use_uvloop: bool = bool(int(os.getenv("USE_UVLOOP", "0")))   # Linux/macOS optional
    timezone: str = os.getenv("TIMEZONE", "UTC")
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "200"))

# --- add near the other dataclasses ---
@dataclass(frozen=True)
class PersistSettings:
    enable_raw: bool = bool(int(os.getenv("PERSIST_RAW", "1")))
    enable_decisions: bool = bool(int(os.getenv("PERSIST_DECISIONS", "1")))
    # Keep features inline? (off = just hash now; turn on later if needed)
    inline_features: bool = bool(int(os.getenv("PERSIST_INLINE_FEATURES", "0")))
    fmt: str = os.getenv("PERSIST_FMT", "jsonl.gz")  # jsonl.gz | jsonl
    flush_every: int = int(os.getenv("PERSIST_FLUSH_EVERY", "100"))  # buffer before flush
    dir_raw: Path = (DATA_DIR / "raw")
    dir_ml: Path = (DATA_DIR / "processed")  # decisions/labels live here

# In Settings dataclass:
persist: PersistSettings = field(default_factory=PersistSettings)


@dataclass(frozen=True)
class Settings:
    env: Env = Env(os.getenv("ENV", "dev"))
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    logs_dir: Path = LOGS_DIR

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APISettings = field(default_factory=APISettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    perf: PerfSettings = field(default_factory=PerfSettings)

    # Use Coinbase-style symbols by default
    default_symbols: tuple[str, ...] = ("BTC-USD", "ETH-USD")
    default_interval: str = "1m"
    persist: PersistSettings = field(default_factory=PersistSettings)

    ssl_trust_env: bool = bool(int(os.getenv("SSL_TRUST_ENV", "0")))   # ignore shell SSL vars by default
    ssl_verify: str = os.getenv("SSL_VERIFY", "certifi")               # certifi | system | false | path
    ssl_verify_path: str = os.getenv("SSL_VERIFY_PATH", "")            # used if SSL_VERIFY=path




@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache settings exactly once per process.
    Also ensures expected directories exist.
    """
    s = Settings()
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
