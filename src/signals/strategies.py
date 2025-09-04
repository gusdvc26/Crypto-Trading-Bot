# src/signals/strategies.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass(frozen=True)
class SignalResult:
    action: SignalAction
    confidence: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

def _sma(closes: List[float], n: int) -> float | None:
    if len(closes) < n or n <= 0:
        return None
    return sum(closes[-n:]) / float(n)

class SmaCross:
    def __init__(self, fast: int = 10, slow: int = 20, conf_scale: float = 0.001):
        if fast <= 0 or slow <= 0:
            raise ValueError("fast/slow must be > 0")
        self.fast = fast
        self.slow = slow
        self.conf_scale = conf_scale  # ~10 bps gap → conf ~1.0 if 0.001

    def generate(self, candles: List[Dict[str, Any]]) -> SignalResult:
        if not candles:
            return SignalResult(SignalAction.HOLD, 0.0, {"reason": "no_data"})
        closes = [r["c"] for r in candles if "c" in r]
        f = _sma(closes, self.fast)
        s = _sma(closes, self.slow)
        if f is None or s is None:
            return SignalResult(SignalAction.HOLD, 0.0, {"reason": "insufficient_history"})

        gap = (f - s) / (abs(s) + 1e-12)
        # confidence from relative SMA gap
        confidence = max(0.0, min(1.0, abs(gap) / self.conf_scale))

        if gap > 0:
            action = SignalAction.BUY
        elif gap < 0:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        return SignalResult(
            action=action,
            confidence=confidence,
            meta={"fast_ma": f, "slow_ma": s, "gap": gap}
        )
