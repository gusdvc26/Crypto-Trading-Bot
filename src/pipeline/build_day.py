# src/pipeline/build_day.py
from __future__ import annotations
import argparse, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone

from config.settings import get_settings
from src.utils.helpers import get_logger

log = get_logger(__name__)
S = get_settings()

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def discover_symbols(exchange: str, date_str: str) -> list[str]:
    base = S.persist.dir_ml / exchange
    if not base.exists():
        return []
    syms: set[str] = set()
    for p in base.glob(f"*/{date_str}/decisions.jsonl.gz"):
        # .../exchange/<SYMBOL>/<YYYY-MM-DD>/decisions.jsonl.gz
        try:
            syms.add(p.parts[-3])  # symbol folder
        except Exception:
            pass
    return sorted(syms)

def run_module(mod: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", mod] + args
    log.info("RUN: " + " ".join(cmd))
    res = subprocess.run(cmd)
    return res.returncode

def main():
    ap = argparse.ArgumentParser(description="Label and build dataset for a given day.")
    ap.add_argument("--exchange", default=S.api.exchange)
    ap.add_argument("--start", default=_today_utc())
    ap.add_argument("--end", default=None)
    ap.add_argument("--symbols", default="", help="Comma list; empty = auto-discover from decisions")
    args = ap.parse_args()

    start = args.start
    end = args.end or args.start
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = discover_symbols(args.exchange, start)

    if not symbols:
        print(f"No symbols found for {args.exchange} {start}")
        return 0

    # 1) Label each symbol for the day
    for sym in symbols:
        rc = run_module("src.processing.labeler", ["--symbol", sym, "--date", start])
        if rc != 0:
            log.warning(f"Labeler failed for {sym} {start} (rc={rc})")

    # 2) Build unified dataset for the date range
    rc = run_module(
        "src.processing.dataset_builder",
        ["--symbols", ",".join(symbols), "--start", start, "--end", end, "--out", "data/processed/train/dataset_v1"]
    )
    if rc != 0:
        log.warning(f"Dataset builder rc={rc}")

    print("Pipeline complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
