# src/backtesting/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.signals.strategies import Strategy, SignalAction, SignalResult


@dataclass(frozen=True)
class BacktestConfig:
    starting_cash: float = 10_000.0
    fee_rate: float = 0.0005  # 5 bps per side (placeholder)
    slippage_bps: float = 1.0  # placeholder


@dataclass
class BacktestReport:
    total_return_pct: float
    trades: int
    win_rate_pct: float
    notes: Dict[str, Any]


class Backtester:
    """
    Ultra-minimal placeholder:
    - Walks OHLCV sequentially
    - Applies the strategy on each step (using closes)
    - Executes naive market BUY/SELL with size=all-in or flat
    """
    def __init__(self, cfg: BacktestConfig | None = None):
        self.cfg = cfg or BacktestConfig()

    def run(self, strategy: Strategy, ohlcv: List[Dict[str, Any]]) -> BacktestReport:
        if not ohlcv:
            return BacktestReport(0.0, 0, 0.0, {"reason": "no_data"})

        cash = self.cfg.starting_cash
        qty = 0.0
        trades = 0
        wins = 0
        last_entry_price = None

        for i in range(1, len(ohlcv) + 1):
            window = ohlcv[:i]
            signal: SignalResult = strategy.generate(window)
            close = window[-1]["c"]

            if signal.action == SignalAction.BUY and cash > 0:
                # all-in buy (placeholder)
                price = close * (1 + self.cfg.slippage_bps / 10_000)
                qty = (cash * (1 - self.cfg.fee_rate)) / price
                cash = 0.0
                trades += 1
                last_entry_price = price

            elif signal.action == SignalAction.SELL and qty > 0:
                # full close (placeholder)
                price = close * (1 - self.cfg.slippage_bps / 10_000)
                proceeds = qty * price * (1 - self.cfg.fee_rate)
                cash += proceeds
                # win if exit above entry
                if last_entry_price is not None and price > last_entry_price:
                    wins += 1
                qty = 0.0
                trades += 1
                last_entry_price = None

        # Mark-to-market if still holding
        if qty > 0:
            cash += qty * ohlcv[-1]["c"]
            qty = 0.0

        total_return_pct = (cash / self.cfg.starting_cash - 1.0) * 100
        win_rate_pct = (wins / trades * 100) if trades else 0.0

        return BacktestReport(
            total_return_pct=total_return_pct,
            trades=trades,
            win_rate_pct=win_rate_pct,
            notes={"strategy": getattr(strategy, "name", type(strategy).__name__)},
        )
