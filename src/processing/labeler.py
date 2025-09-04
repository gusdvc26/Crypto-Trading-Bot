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
            try:
                rows.append(json.loads(line))
            except Exception:
                # ignore malformed
                continue
    return rows

def _append_jsonl_gz(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with gzip.open(path, "ab") as f:
        for r in rows:
            f.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))

# ---------- Labeling utilities ----------

def _to_ms_safe(x: Any) -> int:
    """
    Convert epoch seconds/ms or ISO8601 → integer milliseconds UTC.
    """
    # numeric?
    try:
        xn = float(x)
        # seconds < 1e12; ms >= 1e12
        return int(round(xn * 1000)) if xn < 1e12 else int(round(xn))
    except Exception:
        pass
    # ISO8601 fallback
    ts = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)

def _infer_step_ms(ohlcv: List[Dict[str, Any]]) -> Optional[int]:
    if len(ohlcv) < 2:
        return None
    return int(ohlcv[1]["t"] - ohlcv[0]["t"])

@dataclass
class OhlcvIndex:
    ts: List[int]  # ascending epoch ms
    by_ts: Dict[int, Dict[str, float]]  # t -> {o,h,l,c}

def _build_index(ohlcv: List[Dict[str, Any]]) -> OhlcvIndex:
    ts = [int(r["t"]) for r in ohlcv]
    by_ts = {int(r["t"]): {"o": float(r["o"]), "h": float(r["h"]), "l": float(r["l"]), "c": float(r["c"])} for r in ohlcv}
    return OhlcvIndex(ts=ts, by_ts=by_ts)

def _rightmost_le(ts_sorted: List[int], t0: int) -> Optional[int]:
    """
    Index of the rightmost timestamp <= t0, or None if all > t0.
    """
    lo, hi = 0, len(ts_sorted) - 1
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if ts_sorted[mid] <= t0:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans

def _window_indices(ts_sorted: List[int], t0: int, t1: int) -> Tuple[int, int]:
    """Return slice [i_start, i_end] where t in (t0, t1] (open left, closed right)."""
    # first index with ts > t0
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
    return i_start, i_end  # inclusive end; empty if i_start > i_end

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
    Detect first hit of TP/SL inside (t0, t1].
    BUY: TP if high >= c0*(1+tp), SL if low <= c0*(1-sl)
    SELL: TP if low <= c0*(1-tp), SL if high >= c0*(1+sl)
    """
    i_start, i_end = _window_indices(ts_sorted, t0, t1)
    if i_start > i_end:
        return None, None, "None"

    tp_b = c0 * (1 + tp)
    sl_b = c0 * (1 - sl)
    tp_s = c0 * (1 - tp)
    sl_s = c0 * (1 + sl)

    first = "None"
    t_tp: Optional[int] = None
    t_sl: Optional[int] = None

    for i in range(i_start, i_end + 1):
        t = ts_sorted[i]
        row = by_ts[t]
        hi = row["h"]
        lo = row["l"]
        if action == "SELL":
            if t_tp is None and lo <= tp_s:
                t_tp = t
                if first == "None":
                    first = "TP"
            if t_sl is None and hi >= sl_s:
                t_sl = t
                if first == "None":
                    first = "SL"
        else:  # BUY / HOLD
            if t_tp is None and hi >= tp_b:
                t_tp = t
                if first == "None":
                    first = "TP"
            if t_sl is None and lo <= sl_b:
                t_sl = t
                if first == "None":
                    first = "SL"

    return t_tp, t_sl, first

def _label_one(
    d: Dict[str, Any],
    idx: OhlcvIndex,
    horizons_min: List[int],
    tp_bps: int,
    sl_bps: int,
) -> Dict[str, Any]:
    """
    Compute labels for a single decision. Compatible with dataset_builder single-file merge:
    emits decision_ts_ms, mask_{H}m, tp_first_{H}m, sl_first_{H}m, dir_ret_{H}m.
    """
    # Robust decision time (ms)
    if "ts_ms" in d:
        decision_ts_ms = _to_ms_safe(d["ts_ms"])
    elif "t" in d:
        decision_ts_ms = _to_ms_safe(d["t"])
    elif "ts" in d:
        decision_ts_ms = _to_ms_safe(d["ts"])
    else:
        # cannot label without a time; return shell row
        decision_ts_ms = 0

    t0 = decision_ts_ms
    # Get price at or before t0 (align 10s decisions to 1m bars)
    i0 = _rightmost_le(idx.ts, t0)
    if i0 is None:
        # all bars after decision; cannot label
        shell = {
            "symbol": d.get("symbol"),
            "decision_ts_ms": int(decision_ts_ms),
            "strategy_version": d.get("strategy_version"),
        }
        for H in horizons_min:
            shell[f"mask_{H}m"] = False
            shell[f"tp_first_{H}m"] = False
            shell[f"sl_first_{H}m"] = False
            shell[f"dir_ret_{H}m"] = None
        return shell

    c0 = idx.by_ts[idx.ts[i0]]["c"]
    tp = tp_bps / 10_000.0
    sl = sl_bps / 10_000.0
    action = str(d.get("action", "HOLD")).upper()

    out: Dict[str, Any] = {
        "decision_id": d.get("decision_id"),
        "ts_label": datetime.now(timezone.utc).isoformat(),
        "exchange": d.get("exchange"),
        "symbol": d.get("symbol"),
        "t": int(decision_ts_ms),              # keep canonical decision time as 't' too
        "decision_ts_ms": int(decision_ts_ms), # explicit join key (ms)
        "strategy": d.get("strategy"),
        "strategy_version": d.get("strategy_version"),
        "action": action,
        "dataset_version": d.get("dataset_version", "v1"),
    }

    for H in horizons_min:
        horizon_ms = H * 60_000
        t1 = t0 + horizon_ms

        # horizon window indices (open-left, closed-right)
        i_start, i_end = _window_indices(idx.ts, t0, t1)
        has_window = i_start <= i_end
        out[f"mask_{H}m"] = bool(has_window)

        # close-to-close return (if we have exact bar at t1, else None)
        cH = idx.by_ts.get(t1, {}).get("c")
        if cH is None:
            dir_ret = None
        else:
            raw_ret = (cH - c0) / (c0 + 1e-12)
            dir_ret = -raw_ret if action == "SELL" else raw_ret
        out[f"dir_ret_{H}m"] = dir_ret

        # first-touch logic only if window exists
        if has_window:
            t_tp, t_sl, first = _first_touch_times(idx.ts, idx.by_ts, t0, t1, c0, action, tp, sl)
            out[f"tp_first_{H}m"] = (first == "TP")
            out[f"sl_first_{H}m"] = (first == "SL")
        else:
            out[f"tp_first_{H}m"] = False
            out[f"sl_first_{H}m"] = False

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
        # ensure symbol carried
        d.setdefault("symbol", symbol)
        out_rows.append(_label_one(d, idx, horizons_min, tp_bps, sl_bps))

    # QA print (concise and robust)
    try:
        # compute from emitted rows
        n = len(out_rows)
        null_pct = sum(1 for r in out_rows if not r.get("decision_ts_ms")) / max(1, n)
        tms = [r["decision_ts_ms"] for r in out_rows if r.get("decision_ts_ms")]
        tmin = datetime.fromtimestamp(min(tms)/1000, tz=timezone.utc).isoformat() if tms else None
        tmax = datetime.fromtimestamp(max(tms)/1000, tz=timezone.utc).isoformat() if tms else None
        m5 = [r.get("mask_5m", False) for r in out_rows]
        m5_pct = (sum(bool(x) for x in m5) / max(1, len(m5))) if m5 else 0.0
        print(f"LABEL_QA {symbol} {date_str} n={n} null_decision_ts_ms={null_pct:.2%} "
              f"tmin={tmin} tmax={tmax} mask_5m_true_pct={m5_pct:.2%}")
    except Exception as e:
        log.warning(f"LABEL_QA compute failed for {symbol}/{date_str}: {e}")

    _append_jsonl_gz(_labels_path(exchange, symbol, date_str), out_rows)
    labeled = len(out_rows)  # we emit one row per decision
    return len(decisions), labeled

# ---------- CLI ----------

def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def main_cli():
    ap = argparse.ArgumentParser(description="Compute labels (TP/SL first-touch + returns) for decisions.")
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
