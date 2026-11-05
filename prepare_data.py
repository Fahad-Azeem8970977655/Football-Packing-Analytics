"""STEP 1 - turn your 4 big parquet files into small, dashboard-ready files.

    python prepare_data.py --src "C:/Users/DELL/Desktop/New"        (folder with the 4 parquet files)

Output (in ./data/processed):
    actions.parquet        one row per Pass / Carry with packing + line-breaking metrics
    matches.parquet        cleaned match table
    minutes.parquet        minutes played per player per match (from lineups)
    frames_by_match/       freeze frames of every Pass/Carry, one folder per match (for the viewer)
    summary.json           coverage numbers
Runs once (a few minutes, low memory: frames are processed one row-group at a time).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from packing import packing_chunk

EVENT_COLS = ["match_id", "id", "index", "period", "minute", "second", "type", "possession",
              "play_pattern", "team", "player", "position", "location", "pass.end_location",
              "pass.length", "pass.angle", "pass.height", "pass.body_part", "pass.outcome",
              "pass.recipient", "pass.type", "carry.end_location", "under_pressure", "duration"]
FRAME_COLS = ["match_id", "event_uuid", "player_index", "x", "y", "teammate", "actor", "keeper"]


def xy(series):
    arr = np.full((len(series), 2), np.nan)
    for i, v in enumerate(series.to_numpy()):
        if v is not None and len(v) >= 2:
            arr[i, 0], arr[i, 1] = v[0], v[1]
    return arr


def get(d, key):
    return d.get(key) if isinstance(d, dict) else None


def clean_matches(src):
    m = pd.read_parquet(src / "matches.parquet",
                        columns=["match_id", "match_date", "competition", "season", "home_team",
                                 "away_team", "home_score", "away_score", "match_status_360"])
    out = pd.DataFrame({
        "match_id": m["match_id"].astype(str),
        "match_date": m["match_date"].astype(str),
        "competition": m["competition"].map(lambda d: get(d, "competition_name")),
        "season": m["season"].map(lambda d: get(d, "season_name")),
        "home_team": m["home_team"].map(lambda d: get(d, "home_team_name")),
        "away_team": m["away_team"].map(lambda d: get(d, "away_team_name")),
        "home_score": m["home_score"], "away_score": m["away_score"],
        "status_360": m["match_status_360"],
    })
    out["label"] = (out["match_date"] + " | " + out["home_team"].astype(str) + " " +
                    out["home_score"].astype(str) + "-" + out["away_score"].astype(str) + " " +
                    out["away_team"].astype(str) + " | " + out["competition"].astype(str) +
                    " " + out["season"].astype(str))
    return out


def load_actions(src, matches):
    ev = pd.read_parquet(src / "events.parquet", columns=EVENT_COLS)
    ev = ev[ev["type"].isin(["Pass", "Carry"])].reset_index(drop=True)
    is_pass = (ev["type"] == "Pass").to_numpy()
    st_ = xy(ev["location"])
    pe, ce = xy(ev["pass.end_location"]), xy(ev["carry.end_location"])
    en = np.where(is_pass[:, None], pe, ce)
    a = pd.DataFrame({
        "id": ev["id"], "match_id": ev["match_id"].astype(str), "period": ev["period"],
        "minute": ev["minute"], "second": ev["second"], "team": ev["team"],
        "player": ev["player"], "position": ev["position"], "action_type": ev["type"],
        "play_pattern": ev["play_pattern"],
        "sx": st_[:, 0].astype("float32"), "sy": st_[:, 1].astype("float32"),
        "ex": en[:, 0].astype("float32"), "ey": en[:, 1].astype("float32"),
        "completed": np.where(is_pass, ev["pass.outcome"].isna().to_numpy(), True),
        "under_pressure": ev["under_pressure"].fillna(False).astype(bool),
        "pass_height": ev["pass.height"], "pass_body_part": ev["pass.body_part"],
        "pass_type": ev["pass.type"], "recipient": ev["pass.recipient"],
    })
    a = a.dropna(subset=["sx", "sy", "ex", "ey"]).reset_index(drop=True)
    a["length"] = np.hypot(a.ex - a.sx, a.ey - a.sy).astype("float32")
    a["forward_m"] = (a.ex - a.sx).astype("float32")
    goal = lambda x, y: np.hypot(120 - x, 40 - y)
    a["progress_m"] = (goal(a.sx, a.sy) - goal(a.ex, a.ey)).astype("float32")
    a["progressive"] = (a["progress_m"] >= 10) & (a["ex"] >= a["sx"])
    a["final_third_entry"] = (a.sx < 80) & (a.ex >= 80)
    a["box_entry"] = ~((a.sx >= 102) & (a.sy.between(18, 62))) & (a.ex >= 102) & (a.ey.between(18, 62))
    a = a.merge(matches[["match_id", "competition", "season", "match_date"]], on="match_id", how="left")
    return a


def process_frames(src, actions, out_dir, write_frames):
    pf = pq.ParquetFile(src / "frames.parquet")
    ids = pd.Index(actions["id"])
    sx, ex = actions["sx"].to_numpy(float), actions["ex"].to_numpy(float)
    root = out_dir / "frames_by_match"
    if write_frames:
        shutil.rmtree(root, ignore_errors=True)
    results, carry = [], None

    def handle(fr):
        if fr is None or fr.empty:
            return
        pos = ids.get_indexer(fr["event_uuid"].to_numpy())
        res = packing_chunk(fr, pos, sx, ex)
        if not res.empty:
            results.append(res)
        if write_frames and (pos >= 0).any():
            k = fr.loc[pos >= 0, FRAME_COLS].copy()
            for c in ("teammate", "actor", "keeper"):
                k[c] = k[c].fillna(False).astype(bool)
            k["x"], k["y"] = k["x"].astype("float32"), k["y"].astype("float32")
            k["player_index"] = k["player_index"].astype("int16")
            pq.write_to_dataset(pa.Table.from_pandas(k, preserve_index=False),
                                root_path=str(root), partition_cols=["match_id"])

    for g in range(pf.num_row_groups):
        print(f"   frames row-group {g + 1}/{pf.num_row_groups}", end="\r")
        fr = pf.read_row_group(g, columns=FRAME_COLS).to_pandas()
        if carry is not None:
            fr = pd.concat([carry, fr], ignore_index=True)
        last = fr["event_uuid"].iloc[-1]          # last event may continue in next row-group
        is_last = (fr["event_uuid"].to_numpy() == last)
        carry, fr = fr[is_last], fr[~is_last]
        handle(fr)
    handle(carry)
    print()
    return pd.concat(results) if results else pd.DataFrame()


def minutes_played(src, matches):
    ev = pd.read_parquet(src / "events.parquet", columns=["match_id", "minute"])
    end = ev.groupby("match_id")["minute"].max().add(1).to_dict()
    lu = pd.read_parquet(src / "lineups.parquet", columns=["match_id", "team", "player_name", "positions"])

    def tmin(s):
        if s is None or not isinstance(s, str):
            return None
        p = s.split(":")
        return int(p[0]) + int(p[1]) / 60

    rows = []
    for mid, team, name, poss in lu.itertuples(index=False, name=None):
        end_min, tot = end.get(str(mid), 95), 0.0
        for p in (poss if poss is not None else []):
            f, t = tmin(p.get("from")), tmin(p.get("to"))
            f = 0.0 if f is None else f
            t = end_min if t is None else t
            tot += max(min(t, end_min) - f, 0)
        rows.append((str(mid), team, name, tot))
    m = pd.DataFrame(rows, columns=["match_id", "team", "player", "minutes"])
    return m[m["minutes"] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".", help="folder holding the 4 parquet files")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--no-frames", action="store_true", help="skip freeze-frame viewer data")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1/5 matches");  matches = clean_matches(src)
    print("2/5 events (Pass + Carry)");  actions = load_actions(src, matches)
    print(f"    {len(actions):,} passes/carries")
    print("3/5 freeze frames -> packing (slowest step)")
    metrics = process_frames(src, actions, out, not args.no_frames)
    actions = actions.join(metrics)
    actions["has_frame"] = actions.index.isin(metrics.index)
    actions["pack_cmp"] = actions["pack"].where(actions["completed"], 0).where(actions["has_frame"])
    for c in ("team", "player", "position", "action_type", "play_pattern", "pass_height",
              "pass_body_part", "pass_type", "recipient", "competition", "season"):
        actions[c] = actions[c].astype("category")
    print("4/5 minutes played");  mins = minutes_played(src, matches)
    print("5/5 saving")
    actions.to_parquet(out / "actions.parquet", index=False)
    matches.to_parquet(out / "matches.parquet", index=False)
    mins.to_parquet(out / "minutes.parquet", index=False)
    f = actions[actions.has_frame]
    summary = {"matches": int(matches.shape[0]), "actions": int(len(actions)),
               "actions_with_frame": int(len(f)),
               "frame_coverage_pct": round(100 * len(f) / max(len(actions), 1), 1),
               "avg_packing_completed_pass": round(float(f[f.action_type == "Pass"]["pack_cmp"].mean()), 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Done. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
