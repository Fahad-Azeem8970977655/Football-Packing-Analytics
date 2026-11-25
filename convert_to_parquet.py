import pandas as pd
from pathlib import Path

d = Path(__file__).parent / "data" / "processed"
STR_COLS = ["match_id", "id", "key_pass_id", "season", "competition", "event_uuid"]
CAT_COLS = ["team", "player", "position", "action_type", "play_pattern", "pass_height",
            "pass_body_part", "pass_type", "recipient", "competition", "season",
            "opp_team", "possession_team", "match_date", "match_id"]
DROP = {"actions": ["progress_m", "pack_gk", "lines_broken", "opp_block_width"]}

for f in sorted(list(d.glob("*.csv.gz")) + list(d.glob("*.csv"))):
    name = f.name.replace(".csv.gz", "").replace(".csv", "")
    head = pd.read_csv(f, nrows=0).columns
    df = pd.read_csv(f, low_memory=False, dtype={c: str for c in STR_COLS if c in head})
    df = df.drop(columns=DROP.get(name, []), errors="ignore")
    if name == "actions":                      # only the big table gets categories
        for c in CAT_COLS:
            if c in df: df[c] = df[c].astype("category")
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    df.to_parquet(d / f"{name}.parquet", index=False, compression="zstd")
    print(f"{name:26s} {len(df):>8,} rows  {(d / f'{name}.parquet').stat().st_size / 1e6:6.1f} MB")