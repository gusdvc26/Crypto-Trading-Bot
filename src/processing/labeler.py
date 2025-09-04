# src/processing/labeler.py
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import get_settings
from src.utils.helpers import get_logger

log = get_logger(__name__)
S = get_settings()


# ---------- IO helpers ----------

def _raw_ohlcv_path(exchange: str, symbol: str, date_str: str) -> Path:
    return S.persist.dir_raw / exchange / symbol / date_str / "ohlcv.jsonl.gz"

def _decisions_path(exchange: str, symbol: str, date_str: str) -> Path:
    return S.persist.dir_ml / exchange / symbol / date_str / "decisions.jsonl.gz"

def _labels_path(exchange: str, symbol: str, date_str: str) -> Path:
    out_dir = S.persist.dir_ml / exchange / symbol / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "labels.jsonl.gz"


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl_gz(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with gzip.open(path, "ab") as f:
        for r in rows:
            f.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))


# ---------- Labeling utilities ----------

def _infer_step_ms(ohlcv: List[Dict[str, Any]]) -> Optional[int]:
    if len(ohlcv) < 2:
        return None
    return int(ohlcv[1]["t"] - ohlcv[0]["t"])

@dataclass
class OhlcvIndex:
    ts: List[int]
    by_ts: Dict[int, Dict[str, float]]

def _build_index(ohlcv: List[Dict[str, Any]]) -> OhlcvIndex:
    ts = [int(r["t"]) for r in ohlcv]
    by_ts = {int(r["t"]): {"o": float(r["o"]), "h": float(r["h"]), "l": float(r["l"]), "c": float(r["c"])} for r in ohlcv}
    return OhlcvIndex(ts=ts, by_ts=by_ts)

def _window_indices(ts_sorted: List[int], t0: int, t1: int) -> Tuple[int, int]:
    """Return slice [i_start, i_end] where t in (t0, t1] (open on left, closed on right)."""
    # binary search (manual to avoid deps)
    lo, hi = 0, len(ts_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_sorted[mid] <= t0:
            lo = mid + 1
        else:
            hi = mid
    i_start = lo
    lo, hi = i_start, len(ts_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_sorted[mid] <= t1:
            lo = mid + 1
        else:
            hi = mid
    i_end = lo - 1
    return i_start, i_end  # inclusive end; if i_start > i_end -> empty

def _first_touch_times(
    ts_sorted: List[int],
    by_ts: Dict[int, Dict[str, float]],
    t0: int,
    t1: int,
    c0: float,
    action: str,
    tp: float,
    sl: float,
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Scan forward and detect first hit of TP/SL inside (t0, t1].
    For BUY: TP at price >= c0*(1+tp), SL at price <= c0*(1-sl)
    For SELL: TP at price <= c0*(1-tp), SL at price >= c0*(1+sl)
    Returns (t_tp, t_sl, first_touch)
    """
    i_start, i_end = _window_indices(ts_sorted, t0, t1)
    if i_start > i_end:
        return None, None, "None"

    tp_price_buy = c0 * (1 + tp)
    sl_price_buy = c0 * (1 - sl)
    tp_price_sell = c0 * (1 - tp)
    sl_price_sell = c0 * (1 + sl)

    first_touch = "None"
    t_tp: Optional[int] = None
    t_sl: Optional[int] = None

    for i in range(i_start, i_end + 1):
        t = ts_sorted[i]
        row = by_ts[t]
        hi = row["h"]
        lo = row["l"]
        if action == "SELL":
            if t_tp is None and lo <= tp_price_sell:  # profit for short
                t_tp = t
                if first_touch == "None":
                    first_touch = "TP"
            if t_sl is None and hi >= sl_price_sell:
                t_sl = t
                if first_touch == "None":
                    first_touch = "SL"
        else:  # BUY or HOLD treated same for labeling
            if t_tp is None and hi >= tp_price_buy:
                t_tp = t
                if first_touch == "None":
                    first_touch = "TP"
            if t_sl is None and lo <= sl_price_buy:
                t_sl = t
                if first_touch == "None":
                    first_touch = "SL"

        if first_touch != "None":
            # We keep scanning to record both times, but you could break here if you only want first
            pass

    return t_tp, t_sl, first_touch


def _label_one(
    d: Dict[str, Any],
    idx: OhlcvIndex,
    horizons_min: List[int],
    tp_bps: int,
    sl_bps: int,
) -> Dict[str, Any]:
    """Compute labels for a single decision row."""
    # Decision timestamp (UTC ms) copied through for auditing/joins
    # Prefer explicit `ts_ms`, fallback to numeric `t`, then ISO `ts`.
    try:
        decision_ts_ms = int(d.get("ts_ms", d.get("t")))  # type: ignore[arg-type]
    except Exception:
        try:
            iso = d.get("ts")
            decision_ts_ms = int(datetime.fromisoformat(str(iso).replace('Z','+00:00')).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except Exception:
            decision_ts_ms = int(d.get("t", 0))
    out: Dict[str, Any] = {
        "decision_id": d.get("decision_id"),
        "ts_label": datetime.now(timezone.utc).isoformat(),
        "exchange": d.get("exchange"),
        "symbol": d.get("symbol"),
        "t": int(d["t"]),
        "decision_ts_ms": int(decision_ts_ms),
        "strategy": d.get("strategy"),
        "strategy_version": d.get("strategy_version"),
        "action": d.get("action"),
        "dataset_version": d.get("dataset_version", "v1"),  # <-- add this

    }

    t0 = int(d["t"])
    row0 = idx.by_ts.get(t0)
    if row0 is None:
        return out  # missing alignment; keep shell so pipeline doesn't break

    c0 = row0["c"]
    tp = tp_bps / 10_000.0
    sl = sl_bps / 10_000.0

    for H in horizons_min:
        future_t = t0 + H * 60_000
        # Close-to-close
        cH = idx.by_ts.get(future_t, {}).get("c")
        ret_close = None if cH is None else (cH - c0) / (c0 + 1e-12)

        # Path-aware extremes within (t0, future_t]
        i_start, i_end = _window_indices(idx.ts, t0, future_t)
        if i_start <= i_end:
            highs = [idx.by_ts[t]["h"] for t in idx.ts[i_start:i_end + 1]]
            lows  = [idx.by_ts[t]["l"] for t in idx.ts[i_start:i_end + 1]]
            runup = (max(highs) - c0) / (c0 + 1e-12)
            drawdown = (min(lows) - c0) / (c0 + 1e-12)  # ≤ 0
        else:
            runup = 0.0
            drawdown = 0.0

        # Directional normalization (SELL flips sign so "good" is positive)
        is_sell = (d.get("action") == "SELL")
        d_ret_close = None if ret_close is None else (-ret_close if is_sell else ret_close)
        favorable = (-drawdown if not is_sell else runup)
        adverse   = (runup if not is_sell else -drawdown)

        # TP/SL first-touch times and outcome
        t_tp, t_sl, first_touch = _first_touch_times(idx.ts, idx.by_ts, t0, future_t, c0, d.get("action", "HOLD"), tp, sl)
        time_to_tp = None if t_tp is None else (t_tp - t0) // 1000
        time_to_sl = None if t_sl is None else (t_sl - t0) // 1000
        tp_hit = t_tp is not None
        sl_hit = t_sl is not None

        # Write fields
        key = f"{H}m"
        # Auditing: window end timestamp for this horizon (UTC ms)
        out[f"window_end_ts_{key}_ms"] = int(future_t)
        out[f"ret_{key}"] = ret_close
        out[f"d_ret_{key}"] = d_ret_close
        out[f"runup_{key}"] = runup
        out[f"drawdown_{key}"] = drawdown
        out[f"favorable_{key}"] = favorable
        out[f"adverse_{key}"] = adverse
        out[f"tp_hit_{key}"] = tp_hit
        out[f"sl_hit_{key}"] = sl_hit
        out[f"time_to_tp_{key}_s"] = time_to_tp
        out[f"time_to_sl_{key}_s"] = time_to_sl
        out[f"first_touch_{key}"] = first_touch

    return out


def _label_decisions_for_day(
    exchange: str,
    symbol: str,
    date_str: str,
    horizons_min: List[int],
    tp_bps: int,
    sl_bps: int,
) -> Tuple[int, int]:
    ohlcv = _read_jsonl_gz(_raw_ohlcv_path(exchange, symbol, date_str))
    decisions = _read_jsonl_gz(_decisions_path(exchange, symbol, date_str))

    if not decisions:
        log.info(f"No decisions for {exchange}/{symbol} on {date_str}")
        return 0, 0
    if not ohlcv:
        log.warning(f"No OHLCV for {exchange}/{symbol} on {date_str} — skipping labels")
        return len(decisions), 0

    idx = _build_index(ohlcv)

    out_rows: List[Dict[str, Any]] = []
    for d in decisions:
        out_rows.append(_label_one(d, idx, horizons_min, tp_bps, sl_bps))

    _append_jsonl_gz(_labels_path(exchange, symbol, date_str), out_rows)
    labeled = sum(1 for r in out_rows)  # we always emit a row; some fields may be None
    return len(decisions), labeled


# ---------- CLI ----------

def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def main_cli():
    ap = argparse.ArgumentParser(description="Compute rich labels (returns, excursions, TP/SL touches) for decisions.")
    ap.add_argument("--exchange", default=S.api.exchange, help="Exchange (e.g., coinbase)")
    ap.add_argument("--symbol", required=True, help="Symbol (e.g., BTC-USD)")
    ap.add_argument("--date", default=_today_utc_str(), help="Partition date YYYY-MM-DD (UTC)")
    ap.add_argument("--horizons", default="1,5,15", help="Comma list of horizons in minutes")
    ap.add_argument("--tp-bps", type=int, default=20, help="Take-profit threshold in bps (1bp=0.01%)")
    ap.add_argument("--sl-bps", type=int, default=20, help="Stop-loss threshold in bps")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    total, labeled = _label_decisions_for_day(
        args.exchange, args.symbol, args.date, horizons_min=horizons, tp_bps=args.tp_bps, sl_bps=args.sl_bps
    )
    print(f"Labeled {labeled}/{total} decisions for {args.exchange}/{args.symbol} on {args.date} "
          f"at horizons={horizons}, tp_bps={args.tp_bps}, sl_bps={args.sl_bps}")

if __name__ == "__main__":
    main_cli()
