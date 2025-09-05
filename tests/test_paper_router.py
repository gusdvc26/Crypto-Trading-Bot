import pandas as pd

from src.exec.paper_router import simulate_replay, route


def test_slippage_calc_buy_sell():
    df = pd.DataFrame([
        {"symbol": "BTC-USD", "action": "BUY", "ts_ms": 1000, "price": 100.0},
        {"symbol": "ETH-USD", "action": "SELL", "ts_ms": 2000, "price": 200.0},
    ])
    cfg = {"slippage_bps": 50, "cooldown_s": 0, "max_notional_usd": 1000}
    orders, fills, _ = simulate_replay(df, cfg)
    assert len(orders) == 2 and len(fills) == 2
    # BUY: +0.50% -> 100 * 1.005 = 100.5
    fb = fills[fills.symbol == "BTC-USD"].iloc[0]
    assert abs(fb.fill_price - 100.0 * 1.005) < 1e-9
    # SELL: -0.50% -> 200 * 0.995 = 199.0
    fs = fills[fills.symbol == "ETH-USD"].iloc[0]
    assert abs(fs.fill_price - 200.0 * 0.995) < 1e-9


def test_cooldown_enforced():
    df = pd.DataFrame([
        {"symbol": "BTC-USD", "action": "BUY", "ts_ms": 1000, "price": 100.0},
        {"symbol": "BTC-USD", "action": "BUY", "ts_ms": 1500, "price": 101.0},  # within 1s cooldown
        {"symbol": "BTC-USD", "action": "BUY", "ts_ms": 2500, "price": 102.0},  # after cooldown
    ])
    cfg = {"slippage_bps": 0, "cooldown_s": 1, "max_notional_usd": 1000}
    orders, fills, _ = simulate_replay(df, cfg)
    # Expect only first and third to be accepted
    assert len(orders) == 2
    ts_list = list(orders.ts_ms)
    assert ts_list == [1000, 2500]


def test_route_orders_and_pnl_qty1():
    # Signals: buy then sell, same symbol. Zero fees/slippage and zero cooldown.
    sig = pd.DataFrame([
        {"ts_ms": 1_000, "symbol": "BTC-USD", "side": "buy", "confidence": 0.9},
        {"ts_ms": 2_000, "symbol": "BTC-USD", "side": "sell", "confidence": 0.8},
    ])
    # Mid: deterministic, equal price so qty=1 and PnL≈sell - buy = 0.0
    mid = pd.DataFrame([
        {"ts_ms": 1_000, "symbol": "BTC-USD", "mid": 100.0},
        {"ts_ms": 2_000, "symbol": "BTC-USD", "mid": 100.0},
    ])
    cfg = {
        "taker_fee_bps": 0.0,
        "default_slippage_bps": 0.0,
        "max_notional_per_order": 100.0,  # qty = 100/100 = 1
        "per_symbol_cooldown_s": 0,
    }
    orders, fills, summary = route(sig, mid, cfg)
    assert len(orders) == 2
    assert len(fills) == 2
    # Expected PnL equals sell_px - buy_px for qty=1 in this setup
    expected = 100.0 - 100.0
    assert abs(summary["pnl"] - expected) < 1e-9


def test_route_cooldown_skips_second_signal():
    sig = pd.DataFrame([
        {"ts_ms": 1_000, "symbol": "ETH-USD", "side": "buy", "confidence": 0.7},
        {"ts_ms": 1_050, "symbol": "ETH-USD", "side": "sell", "confidence": 0.6},  # within cooldown
    ])
    mid = pd.DataFrame([
        {"ts_ms": 1_000, "symbol": "ETH-USD", "mid": 50.0},
        {"ts_ms": 1_050, "symbol": "ETH-USD", "mid": 50.5},
    ])
    cfg = {
        "taker_fee_bps": 0.0,
        "default_slippage_bps": 0.0,
        "max_notional_per_order": 50.0,
        "per_symbol_cooldown_s": 1,  # 1s cooldown
    }
    orders, fills, summary = route(sig, mid, cfg)
    assert len(orders) == 1
    assert summary["orders"] == 1
