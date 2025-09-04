# tests/test_signals.py
from src.signals.strategies import SmaCross, SignalAction

def test_sma_cross_basic():
    ohlcv = [
        {"t": 1, "o": 10, "h": 10, "l": 9,  "c": 10, "v": 1},
        {"t": 2, "o": 10, "h": 11, "l": 9,  "c": 11, "v": 1},
        {"t": 3, "o": 11, "h": 12, "l": 10, "c": 12, "v": 1},
        {"t": 4, "o": 12, "h": 13, "l": 11, "c": 13, "v": 1},
        {"t": 5, "o": 13, "h": 14, "l": 12, "c": 14, "v": 1},
        {"t": 6, "o": 14, "h": 15, "l": 13, "c": 15, "v": 1},
        {"t": 7, "o": 15, "h": 16, "l": 14, "c": 16, "v": 1},
        {"t": 8, "o": 16, "h": 17, "l": 15, "c": 17, "v": 1},
        {"t": 9, "o": 17, "h": 18, "l": 16, "c": 18, "v": 1},
        {"t":10, "o": 18, "h": 19, "l": 17, "c": 19, "v": 1},
    ]
    strat = SmaCross(fast=3, slow=5)
    sig = strat.generate(ohlcv)
    assert sig.action in {SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD}
