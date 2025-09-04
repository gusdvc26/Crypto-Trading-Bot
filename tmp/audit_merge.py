import os, json, gzip, glob
import pandas as pd
from datetime import datetime, timezone

P = r"data/processed/train/dataset_dbg.parquet"
df = pd.read_parquet(P)

assert any(c.startswith("mask_") for c in df.columns), "No mask_* columns in dataset"

def to_date_utc(ts):
    # try to robustly parse decision timestamp to date (UTC)
    if pd.api.types.is_datetime64_any_dtype(df["t"]):
        return pd.to_datetime(ts, utc=True).date()
    # if epoch ms
    try:
        return datetime.fromtimestamp(ts/1000, tz=timezone.utc).date()
    except:
        return pd.to_datetime(ts, utc=True).date()

sample = df.sample(min(30, len(df)), random_state=7)
misses = []

for _, r in sample.iterrows():
    sym = r.get("symbol") or r.get("asset") or r.get("ticker")
    if sym is None: 
        continue
    t = r["t"]
    d = to_date_utc(t).isoformat()
    label_glob = fr"data/processed/coinbase/{sym}/{d}/labels_5m.jsonl.gz"
    files = glob.glob(label_glob)
    if not files:
        misses.append((sym, d, "NO_LABEL_FILE"))
        continue
    # peek at label file to confirm rows & ts range
    path = files[0]
    n = 0
    tmin = None; tmax = None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            n += 1
            try:
                obj = json.loads(line)
                tt = obj.get("t") or obj.get("ts") or obj.get("timestamp")
                if tt is not None:
                    tt = pd.to_datetime(tt, utc=True, errors="coerce")
                    if tmin is None or tt < tmin: tmin = tt
                    if tmax is None or tt > tmax: tmax = tt
            except:
                pass
            if n >= 3 and tmin is not None and tmax is not None:
                break
    if n == 0:
        misses.append((sym, d, "EMPTY_LABEL_FILE"))
        continue
    # compute delta vs decision time
    tt_dec = pd.to_datetime(t, utc=True, errors="coerce")
    misses.append((sym, d, f"LABEL_EXIST n>=1; label_ts˜[{tmin},{tmax}] ; decision_t={tt_dec}"))

print("Audit (symbol, date, note):")
for m in misses[:25]:
    print(m)

# Summaries
print("\nColumns present that matter:")
print([c for c in df.columns if c.startswith(("mask_","tp_first_","sl_first_","dir_ret_"))])

for h in (1,5,15):
    c = f"mask_{h}m"
    if c in df.columns:
        s = df[c]
        print(c, "true:", int((s==True).sum()), " / rows:", len(s))
