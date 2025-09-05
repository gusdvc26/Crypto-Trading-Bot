from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd

PRICE_SYNONYMS = [
    "mid","best_bid","best_ask","bid","ask",
    "last","price","trade_price","px",
    "close","close_price","mark","mark_price","vwap","open","avg_price","average_price"
]

def _find_cols(df: pd.DataFrame, names: Iterable[str]) -> dict:
    """Case-insensitive mapping of canonical->actual if present in df."""
    lower = {c.lower(): c for c in df.columns}
    out = {}
    for n in names:
        if n in lower:
            out[n] = lower[n]
    return out

def _list_numeric_candidates(df: pd.DataFrame) -> list[str]:
    bad_prefixes = ("mask_","dir_ret_","tp_first_")
    bad_names = set(["confidence"])
    num_cols = []
    for c in df.columns:
        lc = c.lower()
        if lc.startswith(bad_prefixes) or lc in bad_names:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.notna().sum()
        if valid > 0:
            num_cols.append((c, valid))
    num_cols.sort(key=lambda x: -x[1])
    return [c for c,_ in num_cols]

def derive_mid(df: pd.DataFrame, *, price_col: Optional[str]=None, bid_col: Optional[str]=None, ask_col: Optional[str]=None) -> pd.Series:
    """
    Create a numeric 'mid' series using overrides or best-available columns:
      1) If bid_col & ask_col provided → mid=(bid+ask)/2
      2) Else if price_col provided → use that column
      3) Else if 'mid' exists → use it
      4) Else if best_bid & best_ask exist → (bid+ask)/2
      5) Else try one of: last, price, trade_price, px, close, close_price, mark, mark_price, vwap, open, avg_price
    """
    # Overrides (case-insensitive lookup)
    cols = {c.lower(): c for c in df.columns}
    if bid_col and ask_col:
        b = cols.get(bid_col.lower()); a = cols.get(ask_col.lower())
        if not b or not a:
            raise SystemExit(f"ERROR: override columns not found (bid={bid_col}, ask={ask_col})")
        return (pd.to_numeric(df[b], errors="coerce") + pd.to_numeric(df[a], errors="coerce")) / 2.0
    if price_col:
        p = cols.get(price_col.lower())
        if not p:
            raise SystemExit(f"ERROR: override price column not found: {price_col}")
        return pd.to_numeric(df[p], errors="coerce")

    # Heuristics
    present = _find_cols(df, PRICE_SYNONYMS)
    # direct mid
    if "mid" in present:
        return pd.to_numeric(df[present["mid"]], errors="coerce")
    # bid/ask pair
    b = present.get("best_bid") or present.get("bid")
    a = present.get("best_ask") or present.get("ask")
    if b and a:
        return (pd.to_numeric(df[b], errors="coerce") + pd.to_numeric(df[a], errors="coerce")) / 2.0
    # price-like fallbacks
    for k in ("last","price","trade_price","px","close","close_price","mark","mark_price","vwap","open","avg_price","average_price"):
        if k in present:
            return pd.to_numeric(df[present[k]], errors="coerce")

    # Nothing found
    cand = _list_numeric_candidates(df)[:20]
    raise SystemExit(
        "ERROR: cannot derive 'mid' (no mid, bid/ask, or price-like columns).\n"
        "Hint: re-run with overrides, e.g.:\n"
        "  python -m src.tools.build_mids --dataset <path> --out <path> --price-col <col>\n"
        "  python -m src.tools.build_mids --dataset <path> --out <path> --bid-col <bid> --ask-col <ask>\n"
        f"Numeric candidates in this dataset: {cand}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="input dataset parquet")
    ap.add_argument("--out", required=True, help="output mids parquet path")
    ap.add_argument("--price-col", help="force a specific price column to use as mid")
    ap.add_argument("--bid-col", help="force bid column name for mid=(bid+ask)/2")
    ap.add_argument("--ask-col", help="force ask column name for mid=(bid+ask)/2")
    ap.add_argument("--list", action="store_true", help="list numeric candidate columns and exit")
    ap.add_argument("--constant", type=float, help="force a constant mid value for all rows (smoke/testing)")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise SystemExit(f"ERROR: dataset parquet not found: {ds_path}")

    df = pd.read_parquet(ds_path)
    if not {"ts_ms", "symbol"}.issubset(df.columns):
        raise SystemExit("ERROR: dataset must contain ts_ms and symbol")

    if args.list:
        print("NUMERIC_CANDIDATES", _list_numeric_candidates(df)[:50])
        return

    # Resolve overrides from CLI and env (CLI takes precedence)
    try:
        import os
        price_col = args.price_col or os.environ.get("MID_PRICE_COL")
        bid_col = args.bid_col or os.environ.get("MID_BID_COL")
        ask_col = args.ask_col or os.environ.get("MID_ASK_COL")
        const_env = None
        if os.environ.get("MID_CONSTANT"):
            try:
                const_env = float(os.environ["MID_CONSTANT"])
            except Exception:
                const_env = None
    except Exception:
        price_col = args.price_col
        bid_col = args.bid_col
        ask_col = args.ask_col
        const_env = None

    # Selection order: bid/ask -> price -> constant -> heuristics
    if bid_col and ask_col:
        mid = derive_mid(df, bid_col=bid_col, ask_col=ask_col)
    elif price_col:
        mid = derive_mid(df, price_col=price_col)
    elif args.constant is not None:
        mid = pd.Series(float(args.constant), index=df.index)
    elif const_env is not None:
        mid = pd.Series(float(const_env), index=df.index)
    else:
        mid = derive_mid(df)
    mids = df[["ts_ms", "symbol"]].copy()
    mids["mid"] = mid

    mids = mids.dropna(subset=["ts_ms", "symbol", "mid"]).copy()
    mids["ts_ms"] = pd.to_numeric(mids["ts_ms"], errors="coerce").astype("int64")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mids.to_parquet(out, index=False)
    print(f"Wrote mids -> {out}")

if __name__ == "__main__":
    main()
