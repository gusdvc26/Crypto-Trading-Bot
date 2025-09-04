# src/utils/persist.py
from __future__ import annotations

import asyncio
import gzip
import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from config.settings import get_settings
from src.utils.helpers import get_logger, now_utc_ms

log = get_logger(__name__)
S = get_settings()

def _ts_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

class JsonlSink:
    """
    Minimal buffered JSONL writer with optional gzip.
    Not thread-safe; intended for single-process async usage.
    """
    def __init__(self, file_path: Path, flush_every: int = 100, compressed: bool = True):
        self.file_path = file_path
        self.flush_every = flush_every
        self.compressed = compressed
        self._buf: list[str] = []
        self._lock = asyncio.Lock()
        _ensure_parent(file_path)

    async def write(self, row: Dict[str, Any]) -> None:
        if is_dataclass(row):
            row = asdict(row)
        line = json.dumps(row, ensure_ascii=False)
        async with self._lock:
            self._buf.append(line)
            if len(self._buf) >= self.flush_every:
                await self.flush()

    async def flush(self) -> None:
        if not self._buf:
            return
        data = ("\n".join(self._buf) + "\n").encode("utf-8")
        if self.compressed:
            # append mode gzip without reopening whole file each time
            with gzip.open(self.file_path, "ab") as f:
                f.write(data)
        else:
            with open(self.file_path, "ab") as f:
                f.write(data)
        self._buf.clear()

    async def close(self) -> None:
        await self.flush()

def _daily_path(base: Path, exchange: str, symbol: str, date_str: str, name: str) -> Path:
    # Example: data/raw/coinbase/BTC-USD/2025-08-22/ohlcv.jsonl.gz
    suffix = ".jsonl.gz" if S.persist.fmt.endswith(".gz") else ".jsonl"
    return base / exchange / symbol / date_str / f"{name}{suffix}"

# --- Public convenience writers ---

async def persist_ohlcv_row(exchange: str, symbol: str, row: Dict[str, Any]) -> None:
    if not S.persist.enable_raw:
        return
    date_str = datetime.utcfromtimestamp(row["t"] / 1000).strftime("%Y-%m-%d")
    path = _daily_path(S.persist.dir_raw, exchange, symbol, date_str, "ohlcv")
    sink = JsonlSink(path, flush_every=S.persist.flush_every, compressed=S.persist.fmt.endswith(".gz"))
    await sink.write({
        "ts": _ts_iso(),
        "exchange": exchange,
        "symbol": symbol,
        **row,  # includes t,o,h,l,c,v
    })
    await sink.close()  # simple/explicit for Phase 1; can be optimized later

async def persist_decision(
    exchange: str,
    symbol: str,
    strategy: str,
    strategy_version: str,
    action: str,
    confidence: float,
    t: int,
    features: Optional[Dict[str, Any]] = None,
    params_hash: Optional[str] = None,
    features_hash: Optional[str] = None,
) -> str:
    if not S.persist.enable_decisions:
        return ""
    decision_id = str(uuid.uuid4())
    date_str = datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d")
    path = _daily_path(S.persist.dir_ml, exchange, symbol, date_str, "decisions")
    # Decision row: ensure `ts_ms` (UTC ms) and `symbol` present.
    ts_ms_value = int(t) if isinstance(t, (int, float)) else now_utc_ms()
    payload: Dict[str, Any] = {
        "decision_id": decision_id,
        "ts": _ts_iso(),
        "ts_ms": ts_ms_value,
        "t": t,
        "exchange": exchange,
        "symbol": symbol,
        "strategy": strategy,
        "strategy_version": strategy_version,
        "action": action,
        "confidence": confidence,
        "params_hash": params_hash,
        "features_hash": features_hash,
        "dataset_version": "v1",  

    }
    if S.persist.inline_features and features:
        payload["features"] = features

    sink = JsonlSink(path, flush_every=S.persist.flush_every, compressed=S.persist.fmt.endswith(".gz"))
    await sink.write(payload)
    await sink.close()
    return decision_id
