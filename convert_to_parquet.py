"""Optional speed-up: converts the CSV data files to Parquet (the dashboard loads Parquet 3-5x faster).
Run once:  python convert_to_parquet.py      (needs pandas + pyarrow, already in requirements.txt)"""
import pandas as pd
from pathlib import Path
d = Path(__file__).parent / "data" / "processed"
for f in sorted(list(d.glob("*.csv.gz")) + list(d.glob("*.csv"))):
    out = d / (f.name.replace(".csv.gz", "").replace(".csv", "") + ".parquet")
    df = pd.read_csv(f, low_memory=False)
    for c in ("match_id", "id", "key_pass_id"):
        if c in df: df[c] = df[c].astype(str)
    df.to_parquet(out, index=False); print(f.name, "->", out.name, f"({out.stat().st_size / 1e6:.1f} MB)")
print("Done. You can delete the .csv.gz files if you like.")
