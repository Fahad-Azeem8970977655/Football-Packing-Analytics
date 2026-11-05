import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models import ON_TARGET, get_psxg
from pitch import GOAL_ZONES_Y, GOAL_ZONES_Z, new_goal, new_pitch
from ui import setup
from utils import load_shots_detail

setup("Shooting & Finishing", "🥅", "Where shots are taken, where they go in the goal, who finishes above expectation - and what the placement is worth (post-shot xG).")
SD = load_shots_detail()
st.sidebar.header("Filters")
opts = sorted(SD["cs"].unique())
sel = st.sidebar.multiselect("Competition-seasons", opts, default=opts, key="sf_sel")
no_pen = st.sidebar.checkbox("Exclude penalties", True, key="sf_pen")
S = SD[SD["cs"].isin(sel)].copy()
if no_pen:
    S = S[~S["penalty"]]
if len(S) < 100:
    st.warning("Too few shots - widen the filters."); st.stop()
S["team"], S["player"] = S["team"].astype(str), S["player"].astype(str)
S["dist"] = np.hypot(120 - S["sx"], 40 - S["sy"])
S["on_target"] = S["outcome"].isin(ON_TARGET)
k = st.columns(5)
k[0].metric("Shots", f"{len(S):,}")
k[1].metric("Goals", f"{int(S['is_goal'].sum()):,}")
k[2].metric("xG", f"{S['xg'].sum():,.0f}")
k[3].metric("On target", f"{S['on_target'].mean():.0%}")
k[4].metric("Average distance", f"{S['dist'].mean():.1f} yd")
t1, t2, t3, t4 = st.tabs(["Shot map", "Goal-mouth placement", "Finishing & post-shot xG", "Chances"])

OUT_STYLE = {"Goal": ("#ff1744", "star"), "Saved": ("#ffd166", "circle"), "Saved to Post": ("#ffd166", "circle"), "Post": ("#b0bec5", "diamond"),
             "Blocked": ("#9aa5b1", "x"), "Off T": ("#4cc9f0", "circle"), "Wayward": ("#4cc9f0", "circle"), "Saved Off Target": ("#ffd166", "circle")}
with t1:
    c = st.columns(2)
    team = c[0].selectbox("Team", ["All teams"] + sorted(S["team"].unique()), key="sf_team")
    X = S if team == "All teams" else S[S["team"] == team]
    player = c[1].selectbox("Player", ["All players"] + sorted(X["player"].unique()), key="sf_player")
    X = X if player == "All players" else X[X["player"] == player]
    X = X.dropna(subset=["sx", "sy"])
    fig = new_pitch(f"{player if player != 'All players' else team}: {len(X):,} shots, {int(X['is_goal'].sum())} goals, {X['xg'].sum():.1f} xG", 640, half=True)
    for out, (col, sym) in OUT_STYLE.items():
        s_ = X[X["outcome"] == out]
        if len(s_):
            fig.add_trace(go.Scatter(x=s_["sx"], y=s_["sy"], mode="markers", name=f"{out} ({len(s_):,})", hoverinfo="text", text=s_["player"] + " | " + s_["minute"].astype(str) + "' | xG " + s_["xg"].round(2).astype(str),
                                     marker=dict(size=7 + 38 * s_["xg"].fillna(0.05).clip(upper=0.8), color=col, symbol=sym, opacity=0.8, line=dict(color="white", width=0.6))))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Marker size = xG. Stars are goals; yellow circles saved shots; grey crosses blocked; blue circles off target; white diamonds hit the post. Attacking left to right.")

with t2:
    o = st.multiselect("Outcomes", ["Goal", "Saved", "Saved to Post", "Post", "Off T", "Wayward"], default=["Goal", "Saved", "Saved to Post", "Post"], key="sf_out")
    c = st.columns(3)
    team = c[0].selectbox("Team ", ["All teams"] + sorted(S["team"].unique()), key="sf_team2")
    body = c[1].selectbox("Body part", ["All"] + sorted(S["body_part"].dropna().astype(str).unique()), key="sf_body")
    st_ = c[2].selectbox("Shot type", ["All"] + sorted(S["shot_type"].dropna().astype(str).unique()), key="sf_stype")
    G = S[S["outcome"].isin(o) & S["end_y"].notna() & S["end_z"].notna() & S["end_y"].between(30, 50) & S["end_z"].between(0, 3.5)]
    G = G if team == "All teams" else G[G["team"] == team]
    G = G if body == "All" else G[G["body_part"].astype(str) == body]
    G = G if st_ == "All" else G[G["shot_type"].astype(str) == st_]
    fig = new_goal(f"Where shots end up in the goal ({len(G):,} shots)", 460)
    fig.add_trace(go.Histogram2d(x=G["end_y"], y=G["end_z"], xbins=dict(start=32, end=48, size=0.5), ybins=dict(start=0, end=3.5, size=0.35), colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,230,80,0.55)"]],
                                 showscale=False, hoverinfo="skip", name="density"))
    for out in o:
        s_ = G[G["outcome"] == out]
        col, sym = OUT_STYLE.get(out, ("#4cc9f0", "circle"))
        if len(s_):
            fig.add_trace(go.Scatter(x=s_["end_y"], y=s_["end_z"], mode="markers", name=f"{out} ({len(s_):,})", hoverinfo="text", text=s_["player"] + " | xG " + s_["xg"].round(2).astype(str),
                                     marker=dict(size=7, color=col, symbol=sym, opacity=0.65, line=dict(color="white", width=0.4))))
    st.plotly_chart(fig, use_container_width=True)
    T_ = S[S["outcome"].isin(ON_TARGET) & S["end_y"].notna() & S["end_z"].notna()].copy()
    T_ = T_ if team == "All teams" else T_[T_["team"] == team]
    T_["col"] = pd.cut(T_["end_y"], list(GOAL_ZONES_Y[:1]) + [38.667, 41.333] + list(GOAL_ZONES_Y[-1:]), labels=["left", "centre", "right"], include_lowest=True)
    T_["row"] = pd.cut(T_["end_z"], [0, 0.889, 1.778, 2.667], labels=["low", "middle", "high"], include_lowest=True)
    zz = T_.dropna(subset=["col", "row"]).groupby(["row", "col"], observed=True).agg(shots=("is_goal", "size"), goals=("is_goal", "sum")).reset_index()
    zz["conversion"] = 100 * zz["goals"] / zz["shots"]
    piv = zz.pivot(index="row", columns="col", values="conversion").reindex(index=["high", "middle", "low"])
    cnt = zz.pivot(index="row", columns="col", values="shots").reindex(index=["high", "middle", "low"])
    st.subheader("Goal conversion by target zone (shots on target)")
    st.plotly_chart(px.imshow(piv, text_auto=".0f", color_continuous_scale="YlOrRd", aspect="auto", labels=dict(color="% scored", x="", y="")).update_layout(height=340), use_container_width=True)
    st.caption("Zones as the shooter sees the goal. Counts: " + "; ".join(f"{r} {c_}: {int(cnt.loc[r, c_])}" for r in cnt.index for c_ in cnt.columns if pd.notna(cnt.loc[r, c_])))

with t3:
    psxg, stats = get_psxg()
    k = st.columns(4)
    k[0].metric("On-target shots modelled", f"{stats['n']:,}")
    k[1].metric("AUC: xG only", f"{stats['auc_xg_only']:.3f}")
    k[2].metric("AUC: xG + placement", f"{stats['auc']:.3f}", help="Held-out matches, 5-fold. The gain is what placement adds.")
    k[3].metric("Brier: xG vs PSxG", f"{stats['brier_xg']:.3f} / {stats['brier']:.3f}")
    S["psxg"] = psxg.reindex(S.index)
    mn = st.slider("Minimum shots", 10, 100, 25, 5, key="sf_min")
    g = S.groupby("player").agg(team=("team", lambda s: s.mode().iat[0]), shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"), on_target=("on_target", "sum")).reset_index()
    ot = S[S["psxg"].notna()].groupby("player").agg(psxg=("psxg", "sum"), xg_on_target=("xg", "sum")).reset_index()
    g = g.merge(ot, on="player", how="left").fillna({"psxg": 0, "xg_on_target": 0})
    g = g[g["shots"] >= mn].copy()
    g["goals_minus_xg"] = g["goals"] - g["xg"]
    g["placement_value"] = g["psxg"] - g["xg_on_target"]
    g["on_target_%"] = 100 * g["on_target"] / g["shots"]
    g["xg_per_shot"] = g["xg"] / g["shots"]
    if len(g) >= 5:
        fig = px.scatter(g, x="xg", y="goals", size="shots", color="placement_value", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, hover_name="player", hover_data=["team", "shots"],
                         labels={"xg": "expected goals (xG)", "goals": "goals", "placement_value": "placement value"})
        mx = float(max(g["xg"].max(), g["goals"].max()))
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dot", color="gray"), name="goals = xG"))
        st.plotly_chart(fig.update_layout(height=560), use_container_width=True)
        st.caption("Above the dotted line = scoring more than the chances deserved. Colour = placement value: post-shot xG minus pre-shot xG on his on-target shots (positive = he places shots where keepers struggle).")
        cols = ["player", "team", "shots", "goals", "xg", "goals_minus_xg", "placement_value", "on_target_%", "xg_per_shot"]
        a_, b_ = st.columns(2)
        a_.markdown("**Most goals above xG**")
        a_.dataframe(g.sort_values("goals_minus_xg", ascending=False)[cols].head(15).round(2), hide_index=True, use_container_width=True)
        b_.markdown("**Best shot placement** (post-shot xG minus xG)")
        b_.dataframe(g.sort_values("placement_value", ascending=False)[cols].head(15).round(2), hide_index=True, use_container_width=True)
        tt = S.groupby("team").agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum")).reset_index()
        tt = tt[tt["shots"] >= 5 * mn]
        tt["goals_minus_xg"] = tt["goals"] - tt["xg"]
        st.plotly_chart(px.bar(tt.sort_values("goals_minus_xg").tail(25), x="goals_minus_xg", y="team", orientation="h", color="goals_minus_xg", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                               labels={"team": ""}).update_layout(height=620, coloraxis_showscale=False, title="Team finishing: goals minus xG"), use_container_width=True)
    else:
        st.info("No player reaches that many shots - lower the slider.")

with t4:
    S["assisted"] = np.where(S["key_pass_id"].notna(), "assisted (key pass)", "unassisted")
    a = S.groupby("assisted").agg(shots=("xg", "size"), xg=("xg", "mean"), conversion=("is_goal", "mean")).reset_index()
    a["conversion"] = 100 * a["conversion"]
    l, r = st.columns(2)
    l.plotly_chart(px.bar(a, x="assisted", y=["xg"], labels={"value": "average xG per shot", "assisted": ""}, color_discrete_sequence=["#00d4a6"]).update_layout(height=340, title="Assisted vs unassisted: shot quality"), use_container_width=True)
    r.plotly_chart(px.bar(a, x="assisted", y="conversion", text=a["shots"].map("{:,}".format), labels={"conversion": "% scored", "assisted": ""}, color_discrete_sequence=["#ff006e"]).update_layout(height=340, title="Assisted vs unassisted: conversion"), use_container_width=True)
    orig = S[S["first_time"].notna()]
    if len(orig):
        f = orig.assign(first_time=np.where(orig["first_time"].astype(bool), "first-time", "controlled")).groupby("first_time").agg(shots=("xg", "size"), xg=("xg", "mean"), conversion=("is_goal", "mean")).reset_index()
        f["conversion"] = 100 * f["conversion"]
        st.markdown("**First-time vs controlled shots** (freeze-frame competitions only)")
        st.dataframe(f.round(3), hide_index=True, use_container_width=True)
    big = S[(S["xg"] >= 0.3)]
    st.subheader("Big chances (xG of 0.3 or more)")
    c = st.columns(3)
    c[0].metric("Big chances", f"{len(big):,}")
    c[1].metric("Conversion", f"{big['is_goal'].mean():.0%}")
    c[2].metric("Average xG", f"{big['xg'].mean():.2f}")
    st.markdown("**Biggest chances missed**")
    miss = big[~big["is_goal"]].sort_values("xg", ascending=False).head(25)
    st.dataframe(miss[["cs", "team", "player", "minute", "xg", "outcome", "shot_type", "body_part"]].round(2), hide_index=True, use_container_width=True)
    tq = S.dropna(subset=["technique"])
    if len(tq):
        tt = tq.groupby("technique").agg(shots=("xg", "size"), xg=("xg", "mean"), conversion=("is_goal", "mean")).reset_index()
        tt["conversion"] = 100 * tt["conversion"]
        st.plotly_chart(px.bar(tt.sort_values("shots", ascending=False), x="technique", y="conversion", text=tt.sort_values("shots", ascending=False)["shots"], labels={"conversion": "% scored"},
                               color_discrete_sequence=["#ffb703"]).update_layout(height=340, title="Shot technique (freeze-frame competitions)"), use_container_width=True)
