"""Shared helpers: data loading, sidebar filters, pitch drawing."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data" / "processed"


CATS = ["team", "player", "position", "action_type", "play_pattern", "pass_height", "pass_body_part",
        "pass_type", "recipient", "competition", "season", "opp_team"]


def _table(name, **kw):
    """Reads data/processed/<name>.parquet, or .csv.gz / .csv if no parquet exists."""
    for p, reader in ((DATA / f"{name}.parquet", pd.read_parquet), (DATA / f"{name}.csv.gz", pd.read_csv),
                      (DATA / f"{name}.csv", pd.read_csv)):
        if p.exists():
            return reader(p) if reader is pd.read_parquet else reader(p, **kw)
    st.error(f"data/processed/{name}.(parquet|csv.gz|csv) not found. Copy your processed data files into data/processed.")
    st.stop()


@st.cache_resource(show_spinner="Loading actions...")
def load_actions() -> pd.DataFrame:       # cache_resource: shared object, never mutate it
    a = _table("actions", dtype={**{c: str for c in CATS}, "match_id": str, "id": str}, low_memory=False)
    a["match_id"] = a["match_id"].astype(str)
    for c in ("completed", "under_pressure", "progressive", "final_third_entry", "box_entry", "has_frame"):
        a[c] = a[c].astype(bool)
    for c in CATS:
        a[c] = a[c].astype("category")
    for c in a.select_dtypes("float64").columns:
        a[c] = a[c].astype("float32")
    return a


@st.cache_resource
def load_matches() -> pd.DataFrame:
    m = _table("matches", dtype={"match_id": str, "season": str, "competition": str})
    m["match_id"] = m["match_id"].astype(str)
    return m


@st.cache_resource
def load_minutes() -> pd.DataFrame:
    m = _table("minutes", dtype={"match_id": str})
    m["match_id"] = m["match_id"].astype(str)
    return m


@st.cache_resource
def load_possessions() -> pd.DataFrame:
    p = _table("possessions", dtype={"match_id": str, "possession_team": str, "play_pattern": str,
                                     "competition": str, "season": str})
    p["match_id"] = p["match_id"].astype(str)
    return p


@st.cache_resource
def load_shots() -> pd.DataFrame:
    sh = _table("shots", dtype={"match_id": str, "key_pass_id": str, "id": str})
    sh["match_id"] = sh["match_id"].astype(str)
    return sh


@st.cache_resource
def load_game_states() -> pd.DataFrame:
    g = _table("game_states", dtype={"match_id": str})
    g["match_id"] = g["match_id"].astype(str)
    return g


@st.cache_resource
def load_team_matches() -> pd.DataFrame:
    t = _table("team_match_metrics", dtype={"match_id": str})
    t["match_id"] = t["match_id"].astype(str)
    t["comp"] = t["competition"].astype(str) + " " + t["season"].astype(str) + " [" + t["source"] + "]"
    return t


@st.cache_resource
def load_league_shots() -> pd.DataFrame:
    w = _table("league_shots", dtype={"match_id": str})
    w["comp"] = w["competition"].astype(str) + " " + w["season"].astype(str) + " [" + w["source"] + "]"
    return w


@st.cache_resource
def load_player_metrics() -> pd.DataFrame:
    p = _table("player_metrics")
    p["comp"] = p["competition"].astype(str) + " " + p["season"].astype(str) + " [" + p["source"] + "]"
    return p


@st.cache_resource
def load_def_events() -> pd.DataFrame:
    return _table("defensive_events", dtype={"match_id": str})


@st.cache_resource
def load_counterpress() -> pd.DataFrame:
    return _table("counterpress_losses", dtype={"match_id": str})


@st.cache_resource
def load_regains() -> pd.DataFrame:
    return _table("regains", dtype={"match_id": str})


@st.cache_resource
def load_pressing_tm() -> pd.DataFrame:
    return _table("pressing_team_match", dtype={"match_id": str})


@st.cache_resource
def load_shots_detail() -> pd.DataFrame:
    d = _table("shots_detail", dtype={"match_id": str, "key_pass_id": str})
    d["cs"] = d["competition"].astype(str) + " " + d["season"].astype(str)
    return d


@st.cache_resource
def load_league_xpoints() -> pd.DataFrame:
    return _table("league_xpoints", dtype={"match_id": str})


@st.cache_resource
def load_stints() -> pd.DataFrame:
    return _table("lineup_stints", dtype={"match_id": str})


@st.cache_resource
def load_cards() -> pd.DataFrame:
    return _table("cards", dtype={"match_id": str})


@st.cache_resource
def load_tactical_shifts() -> pd.DataFrame:
    return _table("tactical_shifts", dtype={"match_id": str})


def comp_filter(key="cf_sel"):
    """Sidebar competition-season filter for pages that do not need freeze frames. Returns the matching rows of the matches table (with a 'cs' column)."""
    m = load_matches().copy()
    m["cs"] = m["competition"].astype(str) + " " + m["season"].astype(str)
    st.sidebar.header("Filters")
    opts = sorted(m["cs"].unique())
    sel = st.sidebar.multiselect("Competition-seasons", opts, default=opts, key=key)
    out = m[m["cs"].isin(sel)]
    if out.empty:
        st.warning("Select at least one competition-season.")
        st.stop()
    return out


@st.cache_resource
def load_frames_top() -> pd.DataFrame:
    """Freeze frames of each match's top-packing actions (used when frames_by_match/ does not exist)."""
    return _table("frames_top", dtype={"match_id": str})


def get_data():
    """Sidebar filters. Returns (base, df, opt).
    base = competition/season/type filtered (all actions)
    df   = base restricted to actions that have a freeze frame (if ticked) + column pack_used"""
    a = load_actions()
    st.sidebar.header("Filters")
    comps = sorted(a["competition"].dropna().unique().tolist())
    seasons = sorted(a["season"].dropna().unique().tolist())
    c = st.sidebar.multiselect("Competition", comps, default=comps, key="f_comp")
    s = st.sidebar.multiselect("Season", seasons, default=seasons, key="f_season")
    t = st.sidebar.multiselect("Action type", ["Pass", "Carry"], default=["Pass", "Carry"], key="f_type")
    completed_only = st.sidebar.checkbox("Count packing of completed actions only", True, key="f_cmp",
                                         help="Impect-style: packing only counts if the pass is completed (carries always are).")
    frames_only = st.sidebar.checkbox("Only actions with a 360 freeze frame", True, key="f_frames",
                                      help="Packing can only be measured where a freeze frame exists.")
    mask = a["competition"].isin(c) & a["season"].isin(s) & a["action_type"].isin(t)
    base = a.loc[mask]
    df = base[base["has_frame"]] if frames_only else base
    df = df.copy()
    df["pack_used"] = df["pack"].where(df["completed"] | (not completed_only), 0)
    if df.empty:
        st.warning("No data for these filters.")
        st.stop()
    return base, df, {"completed_only": completed_only, "frames_only": frames_only}


POS_COLS = ["position"]


@st.cache_resource
def load_xt_grid():
    return pd.read_csv(DATA / "xt_grid.csv").to_numpy()


def agg_players(d):
    """Common per-group aggregation columns."""
    d = d.assign(is_pass=(d.action_type == "Pass").astype(int), is_carry=(d.action_type == "Carry").astype(int),
                 lb=(d.pack_used >= 1).astype(int), prog=d.progressive.astype(int),
                 pack_pass=d.pack_used.where(d.action_type == "Pass", 0),
                 pack_carry=d.pack_used.where(d.action_type == "Carry", 0),
                 f3=d.final_third_entry.astype(int), box=d.box_entry.astype(int),
                 db=d.breaks_def.fillna(0), mb=d.breaks_mid.fillna(0))
    return d
