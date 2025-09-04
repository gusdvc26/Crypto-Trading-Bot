# src/signals/movers.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"

@dataclass(frozen=True)
class MoverResult:
    action: Action
    score: float
    meta: Dict[str, Any]

def _sma(vals: List[float], n: int) -> float | None:
    if n <= 0 or len(vals) < n: return None
    return sum(vals[-n:]) / n

def _atr_like(ohlcv: List[Dict[str, Any]], n: int = 14) -> float | None:
    if len(ohlcv) < n: return None
    trs = [(r["h"] - r["l"]) for r in ohlcv[-n:]]
    return sum(trs) / n

class MoversV1:
    """Simple movers detector for OHLCV bars."""
    def __init__(
        self,
        w1: float = 1.0, w2: float = 0.7, w3: float = 0.8, w4: float = 0.5,
        thresh: float = 0.01, lookback_breakout: int = 20,
        vol_spike_w1: int = 5, vol_spike_w2: int = 20,
        require_trend: bool = False, min_atr_bps: float = 5.0,
    ):
        self.name = "MoversV1"
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4
        self.thresh = thresh
        self.lb = lookback_breakout
        self.v1, self.v2 = vol_spike_w1, vol_spike_w2
        self.require_trend = require_trend
        self.min_atr_bps = min_atr_bps

    def generate(self, ohlcv: List[Dict[str, Any]]) -> MoverResult:
        if len(ohlcv) < max(6, self.lb + 1, self.v2 + 1):
            return MoverResult(Action.HOLD, 0.0, {"reason": "insufficient_data"})

        closes = [r["c"] for r in ohlcv]
        highs  = [r["h"] for r in ohlcv]
        vols   = [r["v"] for r in ohlcv]

        c0 = closes[-2]; c1 = closes[-1]
        ret_1m = (c1 - c0) / (abs(c0) + 1e-12)
        ret_5m = (c1 - closes[-6]) / (abs(closes[-6]) + 1e-12)

        prev_high = max(highs[-(self.lb+1):-1])
        breakout_bps = max(0.0, (c1 - prev_high) / (abs(prev_high) + 1e-12))

        vol_spike = (sum(vols[-self.v1:]) / max(1e-12, (sum(vols[-self.v2:]) / self.v2)))

        trend_ok = True
        if self.require_trend:
            sma5 = _sma(closes, 5); sma20 = _sma(closes, 20)
            trend_ok = (sma5 is not None and sma20 is not None and sma5 > sma20)

        atr = _atr_like(ohlcv, 14) or 0.0
        atr_bps = atr / (abs(c1) + 1e-12)

        raw = (self.w1*ret_1m) + (self.w2*ret_5m) + (self.w3*breakout_bps) + (self.w4*max(0.0, vol_spike-1.0))
        score = max(0.0, raw)

        action = Action.BUY if (score >= self.thresh and trend_ok and atr_bps >= (self.min_atr_bps/10_000.0)) else Action.HOLD
        return MoverResult(action=action, score=score, meta={
            "ret_1m": ret_1m, "ret_5m": ret_5m, "breakout_bps": breakout_bps,
            "vol_spike": vol_spike, "trend_ok": trend_ok, "atr_bps": atr_bps,
        })
