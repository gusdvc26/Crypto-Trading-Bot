import json
from pathlib import Path

import pandas as pd

from src.pipeline.materialize_gold import materialize_from_dataset


def test_materialize_gold_manifest(tmp_path: Path):
    # Build a tiny dataset parquet with required columns
    df = pd.DataFrame(
        {
            "symbol": ["BTC-USD", "ETH-USD"],
            "ts_ms": [1_700_000_000_000, 1_700_000_000_500],  # same day
            "t": [1_700_000_000_000, 1_700_000_000_500],
            "strategy_version": ["v1|movers|1m", "v1|sniper|10s"],
            "mask_5m": [True, False],
        }
    )
    ds_path = tmp_path / "dataset.parquet"
    df.to_parquet(ds_path, index=False)

    out_root = tmp_path / "gold" / "merged"
    n = materialize_from_dataset(str(ds_path), out_root=str(out_root))
    assert n == 2

    # Find manifest
    # Partition uses yyyy/mm/dd from ts_ms; precompute the parts for test timestamp
    day = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.strftime("%Y/%m/%d").iloc[0]
    part_dir = out_root / day
    assert part_dir.exists()
    mf = part_dir / "manifest.json"
    assert mf.exists()
    payload = json.loads(mf.read_text(encoding="utf-8"))
    # Validate schema and row count
    assert set(["row_count", "strategy_versions", "mask_5m_true_pct", "bad_rows_pct", "build_ts_utc"]) <= set(payload.keys())
    assert int(payload["row_count"]) > 0

