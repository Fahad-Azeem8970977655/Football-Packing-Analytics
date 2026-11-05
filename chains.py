"""Links every shot to the passes and carries of its possession (the 'chain') - shared by the Chain Value page."""
import numpy as np
import pandas as pd
import streamlit as st

from utils import load_actions, load_shots


@st.cache_resource(show_spinner="Linking every shot to its build-up chain...")
def get_chains():
    A = load_actions()
    S = load_shots()
    S = S[(S["type"] == "Shot") & (S["shot_type"] != "Penalty") & (S["period"] < 5)].copy()
    S["t"] = S["minute"] * 60 + S["second"]
    S["team"] = S["team"].astype(str)
    a = A[["id", "match_id", "possession", "team", "player", "position", "recipient", "action_type", "minute", "second", "sx", "sy", "ex", "ey", "pack_cmp", "xt", "completed"]].copy()
    a["row"] = np.arange(len(a))                                    # file order = event order
    a["team"], a["player"] = a["team"].astype(str), a["player"].astype(str)
    a["recipient"], a["position"] = a["recipient"].astype(object), a["position"].astype(object)
    a["t"] = a["minute"] * 60 + a["second"]
    sh = S[["id", "match_id", "possession", "team", "t", "player", "xg", "is_goal", "key_pass_id"]].rename(columns={"id": "shot_id", "t": "shot_t", "player": "shooter"})
    sh["shooter"] = sh["shooter"].astype(str)
    m = sh.merge(a, on=["match_id", "possession", "team"])
    m = m[m["t"] <= m["shot_t"]].copy()
    m["is_key"] = m["id"] == m["key_pass_id"]
    return sh, m
