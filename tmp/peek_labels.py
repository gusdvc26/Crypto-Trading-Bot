import pandas as pd

p = r"data/processed/train/dataset_cs6.parquet"
df = pd.read_parquet(p)

cols = [c for c in df.columns if c.startswith(("mask_","dir_ret_","tp_first_","sl_first_"))]
print("Label columns:", cols)

print("\nmask True counts:")
for h in (1,5,15):
    c = f"mask_{h}m"
    if c in df.columns:
        s = df[c]
        print(c, int((s==True).sum()), "true of", len(s))

print("\nstrategy_version sample:")
print(df["strategy_version"].value_counts().head(8))
