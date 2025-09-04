# src/processing/microbars.py
from __future__ import annotations
from collections import deque
from typing import Dict, Any, Deque, List

class MicroBarAgg:
    def __init__(self, window_ms: int = 5000):
        self.window_ms = int(window_ms)
        self.state: Dict[str, Dict[str, Any]] = {}

    def _bucket(self, t_ms: int) -> int:
        return (t_ms // self.window_ms) * self.window_ms

    def update(self, tick: Dict[str, Any]) -> List[Dict[str, Any]]:
        pid = tick["product_id"]; t = int(tick["t"]); p = float(tick["price"]); sz = float(tick.get("size", 0.0))
        bucket = self._bucket(t)
        st = self.state.get(pid); out: List[Dict[str, Any]] = []

        if st is None:
            self.state[pid] = {"t0": bucket, "o": p, "h": p, "l": p, "c": p, "v": sz}
            return out

        if bucket > st["t0"]:
            out.append({"t": st["t0"], "o": st["o"], "h": st["h"], "l": st["l"], "c": st["c"], "v": st["v"]})
            self.state[pid] = {"t0": bucket, "o": p, "h": p, "l": p, "c": p, "v": sz}
            return out

        st["c"] = p
        if p > st["h"]: st["h"] = p
        if p < st["l"]: st["l"] = p
        st["v"] += sz
        return out

class RollingBars:
    def __init__(self, maxlen: int = 60):
        self.buf: Dict[str, Deque[Dict[str, Any]]] = {}
        self.maxlen = maxlen
    def append(self, symbol: str, bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        dq = self.buf.setdefault(symbol, deque(maxlen=self.maxlen))
        dq.append(bar)
        return list(dq)
