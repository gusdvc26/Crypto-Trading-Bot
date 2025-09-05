from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Best-effort config loader (works even if helpers.load_config isn't available)
try:
    from src.utils.helpers import load_config  # type: ignore
except Exception:  # pragma: no cover
    def load_config(name: str) -> Dict[str, Any]:  # type: ignore
        return {}


# ---------- helpers ----------

def _get_param(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """
    Fetch a config key from a flat or nested dict.
    Tries cfg[key], cfg['exec'][key], then cfg['risk'][key].
    """
    if key in cfg:
        return cfg[key]
    if isinstance(cfg.get("exec"), dict) and key in cfg["exec"]:
        return cfg["exec"][key]
    if isinstance(cfg.get("risk"), dict) and key in cfg["risk"]:
        return cfg["risk"][key]
    return default


def _ensure_mid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'mid' column exists. Case-insensitive rename if needed, or derive
    from best_bid/best_ask (or bid/ask). Drops rows where mid cannot be formed.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts_ms", "symbol", "mid"])

    dd = df.copy()
    cols = {c.lower(): c for c in dd.columns}

    # normalize 'mid' if only case differs
    if "mid" not in dd.columns and "mid" in cols:
        dd = dd.rename(columns={cols["mid"]: "mid"})

    # derive from bid/ask if needed
    if "mid" not in dd.columns:
        bid = cols.get("best_bid") or cols.get("bid")
        ask = cols.get("best_ask") or cols.get("ask")
        if bid and ask:
            dd["mid"] = (
                pd.to_numeric(dd[bid], errors="coerce") +
                pd.to_numeric(dd[ask], errors="coerce")
            ) / 2.0

    dd["mid"] = pd.to_numeric(dd.get("mid", pd.Series([], dtype=float)), errors="coerce")
    dd["ts_ms"] = pd.to_numeric(dd.get("ts_ms", pd.Series([], dtype="int64")), errors="coerce")
    dd = dd.dropna(subset=["ts_ms", "symbol", "mid"]).copy()
    dd["ts_ms"] = dd["ts_ms"].astype("int64")
    dd["symbol"] = dd["symbol"].astype(str)
    return dd[["ts_ms", "symbol", "mid"]]


# ---------- core API ----------

def route(signals: pd.DataFrame, mid_prices: pd.DataFrame, cfg: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Paper-execute signals using nearest mid prices with slippage, fees, and cooldown.

    Inputs
    ------
    signals: DataFrame with columns:
      - ts_ms (UTC ms, int)
      - symbol (str)
      - side in {"buy","sell"} (case-insensitive)
      - confidence (float) [optional]
    mid_prices: DataFrame with columns:
      - ts_ms, symbol, and either 'mid' (any case) or best_bid/best_ask (or bid/ask) to derive mid
    cfg: dict (flat or nested under 'exec'/'risk'):
      - taker_fee_bps (default 20)
      - default_slippage_bps (default 5)
      - max_notional_per_order (default 50.0)
      - per_symbol_cooldown_s (default 60)

    Returns
    -------
    (orders_df, fills_df, summary_dict), where summary has {'orders': int, 'pnl': float}
    """
    # validate/normalize signals
    if signals is None or len(signals) == 0:
        return pd.DataFrame(), pd.DataFrame(), {"orders": 0, "pnl": 0.0}

    req_cols = {"ts_ms", "symbol", "side"}
    missing = req_cols - set(signals.columns)
    if missing:
        raise KeyError(f"signals missing columns: {sorted(missing)}")

    sig = signals.copy()
    sig["ts_ms"] = pd.to_numeric(sig["ts_ms"], errors="coerce")
    sig = sig.dropna(subset=["ts_ms", "symbol", "side"]).copy()
    sig["ts_ms"] = sig["ts_ms"].astype("int64")
    sig["symbol"] = sig["symbol"].astype(str)
    sig["side"] = sig["side"].astype(str).str.lower()
    sig = sig[sig["side"].isin(["buy", "sell"])]

    # mids
    mids = _ensure_mid(mid_prices)
    if mids.empty or sig.empty:
        return pd.DataFrame(), pd.DataFrame(), {"orders": 0, "pnl": 0.0}

    # ---- robust nearest-time join: per-symbol to avoid global sort issues ----
    parts: List[pd.DataFrame] = []
    for sym, s in sig.groupby("symbol", sort=False):
        m = mids[mids["symbol"] == sym]
        if m.empty:
            continue
        # sort per symbol on the 'on' key for merge_asof
        s2 = s.sort_values("ts_ms", kind="mergesort")
        m2 = m.sort_values("ts_ms", kind="mergesort")
        # only keep required columns from mids to avoid symbol_x/symbol_y
        joined = pd.merge_asof(
            s2, m2[["ts_ms", "mid"]],
            on="ts_ms",
            direction="nearest",
            tolerance=2000  # ms
        )
        joined["symbol"] = sym  # ensure a clean 'symbol' column exists
        parts.append(joined)

    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    merged = merged.dropna(subset=["mid"])
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), {"orders": 0, "pnl": 0.0}

    # configs
    taker_fee_bps = float(_get_param(cfg, "taker_fee_bps", 20.0))
    default_slip_bps = float(_get_param(cfg, "default_slippage_bps", 5.0))
    max_notional = float(_get_param(cfg, "max_notional_per_order", 50.0))
    cooldown_s = int(_get_param(cfg, "per_symbol_cooldown_s", 60))

    orders: List[Dict[str, Any]] = []
    last_ts: Dict[str, int] = {}
    inv: Dict[str, List[Tuple[float, float, float]]] = {}  # FIFO inventory per symbol: (price, qty, fee_per_unit)
    pnl_total = 0.0

    for _, r in merged.iterrows():
        sym = str(r["symbol"])
        ts = int(r["ts_ms"])
        side = str(r["side"]).lower()
        mid = float(r["mid"])

        # cooldown
        prev = last_ts.get(sym)
        if prev is not None and (ts - prev) < cooldown_s * 1000:
            continue

        slip_bps = default_slip_bps
        if side == "buy":
            fill = mid * (1.0 + slip_bps / 10_000.0)
        else:  # sell
            fill = mid * (1.0 - slip_bps / 10_000.0)

        qty = max_notional / max(fill, 1e-12)
        notional = qty * fill
        fee = notional * (taker_fee_bps / 10_000.0)

        orders.append({
            "ts_ms": ts,
            "symbol": sym,
            "side": side,
            "mid": mid,
            "fill_price": float(fill),
            "slippage_bps": float(slip_bps),
            "qty": float(qty),
            "notional": float(notional),
            "fee": float(fee),
        })

        # FIFO PnL (post-fee)
        lots = inv.setdefault(sym, [])
        if side == "buy":
            lots.append((fill, qty, fee / max(qty, 1e-12)))
        else:  # sell
            remaining = qty
            sell_fee_per_unit = fee / max(qty, 1e-12)
            while remaining > 1e-12 and lots:
                buy_price, buy_qty, buy_fee_per_unit = lots[0]
                matched = min(remaining, buy_qty)
                pnl_total += matched * (fill - buy_price) - matched * (buy_fee_per_unit + sell_fee_per_unit)
                buy_qty -= matched
                remaining -= matched
                if buy_qty <= 1e-12:
                    lots.pop(0)
                else:
                    lots[0] = (buy_price, buy_qty, buy_fee_per_unit)
            # no short inventory in this simple router

        last_ts[sym] = ts

    orders_df = pd.DataFrame(orders)
    fills_df = orders_df.copy()
    summary = {"orders": int(len(orders_df)), "pnl": round(float(pnl_total), 6)}
    return orders_df, fills_df, summary


# ---------- minimal legacy helper for tests ----------

def simulate_replay(df: pd.DataFrame, cfg: Dict[str, Any] | None = None, *, cooldown_s: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Minimal simulator used by tests.
    Supports two call styles:
      - simulate_replay(df, cfg)  -> honors cfg['slippage_bps'], cfg['cooldown_s'], cfg['max_notional_usd']
      - simulate_replay(df, cooldown_s=1)  -> legacy style (no slippage/fees), keeps rows after cooldown filter

    Input df columns (tests use these):
      - 'symbol' (str), 'action' in {'BUY','SELL'}, 'ts_ms' (int ms), 'price' (float)

    Returns (orders_df, fills_df, summary_dict)
    """
    if df is None or df.empty:
        empty = pd.DataFrame(columns=["ts_ms", "symbol", "side", "price", "fill_price", "qty", "notional"])
        return empty, empty.copy(), {"orders": 0, "pnl": 0.0}

    dd = df.copy()
    dd["ts_ms"] = pd.to_numeric(dd["ts_ms"], errors="coerce").astype("int64")
    dd["symbol"] = dd["symbol"].astype(str)
    # normalize action/side
    act_col = "action" if "action" in dd.columns else "side" if "side" in dd.columns else None
    if act_col is None:
        raise KeyError("simulate_replay: require 'action' or 'side' column")
    dd["side"] = dd[act_col].astype(str).str.lower()
    dd = dd[dd["side"].isin(["buy", "sell"])]
    dd["price"] = pd.to_numeric(dd["price"], errors="coerce")

    # resolve params
    slippage_bps = 0.0
    max_notional = 0.0
    if cfg:
        slippage_bps = float(cfg.get("slippage_bps", 0.0))
        max_notional = float(cfg.get("max_notional_usd", cfg.get("max_notional_per_order", 0.0)))
        if cooldown_s is None:
            cooldown_s = int(cfg.get("cooldown_s", cfg.get("per_symbol_cooldown_s", 60)))
    if cooldown_s is None:
        cooldown_s = 60

    # per-symbol cooldown + simple slippage fills
    out_rows: List[Dict[str, Any]] = []
    last_ts: Dict[str, int] = {}
    pnl = 0.0
    inv: Dict[str, List[Tuple[float, float]]] = {}  # (fill_price, qty)

    for sym, g in dd.groupby("symbol", sort=False):
        g = g.sort_values("ts_ms", kind="mergesort")
        for _, r in g.iterrows():
            ts = int(r["ts_ms"])
            side = r["side"]
            px = float(r["price"])

            prev = last_ts.get(sym)
            if prev is not None and (ts - prev) < cooldown_s * 1000:
                continue

            # apply slippage around 'price'
            if side == "buy":
                fill = px * (1.0 + slippage_bps / 10_000.0)
            else:  # sell
                fill = px * (1.0 - slippage_bps / 10_000.0)

            # a simple qty from max_notional (tests don't inspect qty/fee math)
            qty = max_notional / fill if max_notional > 0 else 1.0
            notional = qty * fill

            out_rows.append({
                "ts_ms": ts,
                "symbol": sym,
                "side": side,
                "price": px,
                "fill_price": float(fill),
                "qty": float(qty),
                "notional": float(notional),
            })

            # tiny FIFO PnL just to keep parity with route(); tests don't assert it
            lots = inv.setdefault(sym, [])
            if side == "buy":
                lots.append((fill, qty))
            else:
                remaining = qty
                while remaining > 1e-12 and lots:
                    bpx, bqty = lots[0]
                    matched = min(remaining, bqty)
                    pnl += matched * (fill - bpx)
                    bqty -= matched
                    remaining -= matched
                    if bqty <= 1e-12:
                        lots.pop(0)
                    else:
                        lots[0] = (bpx, bqty)

            last_ts[sym] = ts

    orders = pd.DataFrame(out_rows)
    fills = orders.copy()
    return orders, fills, {"orders": int(len(orders)), "pnl": round(float(pnl), 6)}

# ---------- CLI ----------

def _parse_args_new() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paper router: signals + mid -> orders/fills with FIFO PnL.")
    ap.add_argument("--signals", required=True, help="Signals parquet with ts_ms,symbol,side,confidence")
    ap.add_argument("--mid", required=True, help="Mid prices parquet with ts_ms,symbol,mid (or best_bid/ask)")
    ap.add_argument("--outdir", required=True, help="Output directory for orders.parquet and fills.parquet")
    return ap.parse_args()


def main() -> None:
    args = _parse_args_new()

    # Load exec/risk configs (best-effort)
    try:
        cfg_exec = load_config("exec")
    except Exception:
        cfg_exec = {}
    try:
        cfg_risk = load_config("risk")
    except Exception:
        cfg_risk = {}

    cfg = {
        "taker_fee_bps": float(cfg_exec.get("TAKER_FEE_BPS", cfg_exec.get("taker_fee_bps", 20.0))),
        "default_slippage_bps": float(cfg_exec.get("SLIPPAGE_BPS", cfg_exec.get("default_slippage_bps", 5.0))),
        "max_notional_per_order": float(cfg_risk.get("MAX_NOTIONAL_USD", cfg_risk.get("max_notional_per_order", 50.0))),
        "per_symbol_cooldown_s": int(cfg_risk.get("PER_SYMBOL_COOLDOWN_S", cfg_risk.get("per_symbol_cooldown_s", 60))),
    }

    sig = pd.read_parquet(args.signals)
    mid = pd.read_parquet(args.mid)

    orders, fills, summary = route(sig, mid, cfg)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(out_dir / "orders.parquet", index=False)
    fills.to_parquet(out_dir / "fills.parquet", index=False)

    print(f"PAPER_ROUTER {json.dumps(summary, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
