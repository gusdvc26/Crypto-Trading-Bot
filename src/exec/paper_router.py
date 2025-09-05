from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

from src.utils.helpers import load_config


def _to_utc_date(ms: int) -> str:
    return datetime.utcfromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d")


def _load_decisions_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if p.is_dir():
        # Recursively read known decision files
        files = list(p.rglob("*.parquet")) + list(p.rglob("*.jsonl")) + list(p.rglob("*.jsonl.gz"))
        frames = [
            _read_file(f) for f in files if ("decisions" in f.name or f.suffix in (".parquet",))
        ]
        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = _read_file(p)
        return df if df is not None else pd.DataFrame()


def _read_file(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        if p.suffixes[-2:] == [".jsonl", ".gz"] or p.suffix == ".jsonl":
            import gzip
            import json as _json
            rows: List[Dict[str, Any]] = []
            if p.suffix.endswith(".gz") or p.name.endswith(".gz"):
                opener = gzip.open
                mode = "rt"
            else:
                opener = open
                mode = "r"
            with opener(p, mode, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(_json.loads(line))
            return pd.DataFrame(rows)
    except Exception:
        return None
    return None


def _slip(side: str, price: float, bps: float) -> Tuple[float, float]:
    """Return (mid, fill) given side and slippage in bps."""
    mid = float(price)
    sgn = 1.0 if side.upper() == "BUY" else -1.0 if side.upper() == "SELL" else 0.0
    fill = mid * (1.0 + sgn * (bps / 10_000.0)) if sgn != 0.0 else mid
    return mid, float(fill)

def _get_param(cfg: dict, key: str, default: Any) -> Any:
    """
    Fetch a config key from flat or nested dict.
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
    Ensure a 'mid' column exists, deriving from best_bid/best_ask if present.
    Drops rows with non-finite mids.
    """
    dd = df.copy()
    cols = {c.lower(): c for c in dd.columns}
    if "mid" not in dd.columns and "mid" in cols:
        dd.rename(columns={cols["mid"]: "mid"}, inplace=True)
    if "mid" not in dd.columns:
        bid = cols.get("best_bid") or cols.get("bid")
        ask = cols.get("best_ask") or cols.get("ask")
        if bid and ask:
            dd["mid"] = (pd.to_numeric(dd[bid], errors="coerce") + pd.to_numeric(dd[ask], errors="coerce")) / 2.0
    dd["mid"] = pd.to_numeric(dd.get("mid", pd.Series([], dtype=float)), errors="coerce")
    dd = dd[pd.to_numeric(dd.get("ts_ms"), errors="coerce").notna() & dd["mid"].notna()]
    dd["ts_ms"] = pd.to_numeric(dd["ts_ms"], errors="coerce").astype("int64")
    return dd

def route(signals: pd.DataFrame, mid_prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Paper-execute signals using nearest mid prices with slippage, fees, and cooldown.

    - signals: columns ts_ms (UTC ms), symbol, side in {'buy','sell'}, confidence (float optional).
    - mid_prices: columns ts_ms, symbol, mid (or best_bid/best_ask to derive mid).
    - cfg keys (flat or nested under 'exec'/'risk'):
      taker_fee_bps (default 20), default_slippage_bps (default 5),
      max_notional_per_order (default 50.0), per_symbol_cooldown_s (default 60).

    Returns (orders_df, fills_df, summary_dict), where summary_dict includes {'orders', 'pnl'}.
    """
    if signals is None or len(signals) == 0:
        return pd.DataFrame(), pd.DataFrame(), {"orders": 0, "pnl": 0.0}

    # Normalize signals
    sig = signals.copy()
    req_cols = {"ts_ms", "symbol", "side"}
    missing = req_cols - set(sig.columns)
    if missing:
        raise KeyError(f"signals missing columns: {sorted(missing)}")
    sig["ts_ms"] = pd.to_numeric(sig["ts_ms"], errors="coerce")
    sig = sig.dropna(subset=["ts_ms", "symbol", "side"]).copy()
    sig["ts_ms"] = sig["ts_ms"].astype("int64")
    sig["symbol"] = sig["symbol"].astype(str)
    sig["side"] = sig["side"].astype(str).str.lower()
    sig = sig[sig["side"].isin(["buy", "sell"])]

    # Prepare mids and merge_asof by symbol with 2s tolerance
    mids = _ensure_mid(mid_prices)
    if mids.empty:
        return pd.DataFrame(), pd.DataFrame(), {"orders": 0, "pnl": 0.0}

    # Use datetime keys for tolerance in Timedelta
    sig["ts_dt"] = pd.to_datetime(sig["ts_ms"], unit="ms", utc=True)
    mids["ts_dt"] = pd.to_datetime(mids["ts_ms"], unit="ms", utc=True)
    sig = sig.sort_values(["symbol", "ts_dt"]).reset_index(drop=True)
    mids = mids.sort_values(["symbol", "ts_dt"]).reset_index(drop=True)
    merged = pd.merge_asof(
        sig, mids, on="ts_dt", by="symbol",
        direction="nearest", tolerance=pd.Timedelta("2s")
    )
    merged = merged.dropna(subset=["mid"]).reset_index(drop=True)

    # Configs
    taker_fee_bps = float(_get_param(cfg, "taker_fee_bps", 20.0))
    default_slip_bps = float(_get_param(cfg, "default_slippage_bps", 5.0))
    max_notional = float(_get_param(cfg, "max_notional_per_order", 50.0))
    cooldown_s = int(_get_param(cfg, "per_symbol_cooldown_s", 60))

    orders: List[Dict[str, Any]] = []
    last_ts: Dict[str, int] = {}
    # FIFO inventory per symbol: list of lots (price, qty, fee_per_unit)
    inv: Dict[str, List[Tuple[float, float, float]]] = {}
    pnl_total = 0.0

    for _, r in merged.iterrows():
        sym = r["symbol"]
        ts = int(r.get("ts_ms", r.get("ts_ms_x")))
        side = str(r["side"]).lower()
        mid = float(r["mid"])

        prev = last_ts.get(sym)
        if prev is not None and (ts - prev) < cooldown_s * 1000:
            continue  # cooldown skip

        slip_bps = default_slip_bps
        if side == "buy":
            fill = mid * (1.0 + slip_bps / 10_000.0)
        else:
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

        # FIFO PnL on matched qty
        lots = inv.setdefault(sym, [])
        if side == "buy":
            lots.append((fill, qty, fee / max(qty, 1e-12)))
        else:
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
            # No shorting: ignore any unmatched remainder
        last_ts[sym] = ts

    orders_df = pd.DataFrame(orders)
    fills_df = orders_df.copy()
    summary = {"orders": int(len(orders_df)), "pnl": round(float(pnl_total), 6)}
    return orders_df, fills_df, summary


def simulate_replay(decisions: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate a paper router over a decisions DataFrame.

    Required columns in decisions: symbol, action, ts_ms or t. Optional: price.
    Config keys used (from exec.yml or env-overrides):
      - cooldown_s: int seconds between orders per symbol
      - slippage_bps: float bps applied against mid for fill
      - max_notional_usd: float cap per order used to size qty (qty = max_notional / price)
    """
    if decisions is None or decisions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Defaults
    cooldown_s = int(cfg.get("cooldown_s", 0))
    slippage_bps = float(cfg.get("slippage_bps", 1.0))
    max_notional = float(cfg.get("max_notional_usd", 100.0))

    # Normalize timestamps
    dec = decisions.copy()
    if "ts_ms" not in dec.columns:
        if "t" in dec.columns:
            dec["ts_ms"] = pd.to_numeric(dec["t"], errors="coerce").astype("Int64")
        else:
            dec["ts_ms"] = pd.NA
    dec = dec.dropna(subset=["symbol", "action", "ts_ms"]).copy()
    dec["ts_ms"] = pd.to_numeric(dec["ts_ms"], errors="coerce")
    dec = dec.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)

    # Routing
    last_ts: Dict[str, int] = {}
    orders: List[Dict[str, Any]] = []
    fills: List[Dict[str, Any]] = []

    for _, r in dec.iterrows():
        sym = str(r.get("symbol"))
        side = str(r.get("action", "")).upper()
        if side not in ("BUY", "SELL"):
            continue
        ts_ms = int(r.get("ts_ms"))
        prev = last_ts.get(sym)
        if prev is not None and (ts_ms - prev) < cooldown_s * 1000:
            # Cooldown skip
            continue

        # Price handling: use explicit 'price' if present; else skip sizing but still record order with 0 qty
        price = r.get("price")
        if price is None or pd.isna(price):
            mid, fill_px = (float("nan"), float("nan"))
            qty = 0.0
            notional = 0.0
        else:
            price = float(price)
            mid, fill_px = _slip(side, price, slippage_bps)
            qty = max_notional / max(fill_px, 1e-12)
            notional = qty * fill_px

        oid = str(uuid.uuid4())
        order = {
            "order_id": oid,
            "ts_ms": ts_ms,
            "symbol": sym,
            "side": side,
            "price": float(price) if price is not None and not pd.isna(price) else None,
            "qty": float(qty),
            "notional_usd": float(notional),
        }
        fill = {
            "order_id": oid,
            "ts_ms": ts_ms,
            "symbol": sym,
            "side": side,
            "mid_price": float(mid) if not pd.isna(mid) else None,
            "slippage_bps": float(slippage_bps),
            "fill_price": float(fill_px) if not pd.isna(fill_px) else None,
            "qty": float(qty),
            "notional_usd": float(notional),
        }
        orders.append(order)
        fills.append(fill)
        last_ts[sym] = ts_ms

    orders_df = pd.DataFrame(orders)
    fills_df = pd.DataFrame(fills)

    if orders_df.empty:
        summary = pd.DataFrame(columns=["symbol", "orders", "gross_notional_usd"])
    else:
        summary = (
            orders_df.groupby("symbol")["notional_usd"].agg(["count", "sum"]).reset_index()
            .rename(columns={"count": "orders", "sum": "gross_notional_usd"})
        )
    return orders_df, fills_df, summary


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paper router replay over a decisions stream.")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD UTC start date (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD UTC end date (inclusive)")
    ap.add_argument("--in", dest="inp", required=True, help="Input decisions (parquet/jsonl or folder)")
    ap.add_argument("--out", required=True, help="Output directory for orders/fills/summary")
    return ap.parse_args()


def main_cli() -> None:
    args = _parse_args()
    cfg_exec = {}
    try:
        cfg_exec = load_config("exec")
    except Exception:
        cfg_exec = {}

    df = _load_decisions_any(args.inp)
    if df.empty:
        print("No decisions found.")
        return

    # Filter date range using ts_ms/t
    if "ts_ms" in df.columns:
        df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce")
    elif "t" in df.columns:
        df["ts_ms"] = pd.to_numeric(df["t"], errors="coerce")
    else:
        raise KeyError("Decisions must contain 'ts_ms' or 't' for time filtering")

    start_dt = pd.to_datetime(args.start, utc=True)
    end_dt = pd.to_datetime(args.end, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    mask = (pd.to_datetime(df["ts_ms"], unit="ms", utc=True) >= start_dt) & (
        pd.to_datetime(df["ts_ms"], unit="ms", utc=True) <= end_dt
    )
    df = df[mask].copy().reset_index(drop=True)

    orders, fills, summary = simulate_replay(df, cfg_exec)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(out_dir / "orders.parquet", index=False)
    fills.to_parquet(out_dir / "fills.parquet", index=False)
    (out_dir / "pnl_summary.json").write_text(summary.to_json(orient="records"), encoding="utf-8")
    print(f"Wrote orders/fills to {out_dir}")


if __name__ == "__main__":
    import sys as _sys
    argv = " ".join(_sys.argv[1:]).lower()
    if "--signals" in argv or "--mid" in argv or "--outdir" in argv:
        def _parse_args_new() -> argparse.Namespace:
            ap = argparse.ArgumentParser(description="Paper paper-router: signals + mid -> orders/fills with FIFO PnL.")
            ap.add_argument("--signals", required=True, help="Signals parquet with ts_ms,symbol,side,confidence")
            ap.add_argument("--mid", required=True, help="Mid prices parquet with ts_ms,symbol,mid (or best_bid/ask)")
            ap.add_argument("--outdir", required=True, help="Output directory for orders.parquet and fills.parquet")
            return ap.parse_args()

        def main() -> None:
            """
            New CLI: --signals/--mid/--outdir. Loads exec/risk configs, routes, writes outputs, prints summary.
            """
            args = _parse_args_new()
            try:
                cfg_exec = load_config("exec")
            except Exception:
                cfg_exec = {}
            try:
                cfg_risk = load_config("risk")
            except Exception:
                cfg_risk = {}
            cfg = {
                "taker_fee_bps": float(cfg_exec.get("taker_fee_bps", 20.0)),
                "default_slippage_bps": float(cfg_exec.get("default_slippage_bps", 5.0)),
                "max_notional_per_order": float(cfg_risk.get("max_notional_per_order", 50.0)),
                "per_symbol_cooldown_s": int(cfg_risk.get("per_symbol_cooldown_s", 60)),
            }
            sig = pd.read_parquet(args.signals)
            mid = pd.read_parquet(args.mid)
            orders, fills, summary = route(sig, mid, cfg)
            out_dir = Path(args.outdir)
            out_dir.mkdir(parents=True, exist_ok=True)
            orders.to_parquet(out_dir / "orders.parquet", index=False)
            fills.to_parquet(out_dir / "fills.parquet", index=False)
            print(f"PAPER_ROUTER " + json.dumps(summary, separators=(",", ":")))

        main()
    else:
        main_cli()
