import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, add_caption, lerp_color, new_pitch, role_group, ROLE_COLORS
from ui import setup
from utils import get_data, load_actions, load_matches, load_shots

setup("Possession Sequences", "🔗", "The anatomy of a chance: every pass and carry that led to a shot, numbered on the pitch, with the packing and threat added at each step.")
base, df, opt = get_data()
matches = load_matches()
mids = tuple(sorted(base["match_id"].unique()))
S = load_shots()
S = S[(S["type"] == "Shot") & (S["shot_type"] != "Penalty") & S["match_id"].isin(mids)].copy()
S["t"] = S["minute"] * 60 + S["second"]


@st.cache_data(show_spinner="Linking shots to their build-up sequences...")
def build(match_ids):
    A = load_actions()
    a = A.loc[A["match_id"].isin(match_ids), ["match_id", "possession", "team", "minute", "second", "action_type", "player", "position", "recipient", "sx", "sy", "ex", "ey",
                                             "completed", "pack_cmp", "xt", "under_pressure"]].copy()
    a["row"] = np.arange(len(a))                              # file order = event order
    a["team"] = a["team"].astype(str)
    a["t"] = a["minute"] * 60 + a["second"]
    return a


a = build(mids)
sh = S.assign(team=S["team"].astype(str))[["id", "match_id", "possession", "team", "t", "player", "xg", "is_goal", "outcome", "sx", "sy", "minute", "body_part"]].rename(columns={"id": "shot_id", "t": "shot_t", "player": "shooter", "sx": "shot_x", "sy": "shot_y"})
m = sh.merge(a, on=["match_id", "possession", "team"])
m = m[m["t"] <= m["shot_t"]]
agg = m.groupby("shot_id").agg(steps=("row", "size"), passes=("action_type", lambda s: int((s == "Pass").sum())), packing=("pack_cmp", "sum"), xt=("xt", "sum"),
                              start_t=("t", "min"), players=("player", "nunique")).reset_index()
agg = sh.merge(agg, on="shot_id", how="left")
agg["duration_s"] = (agg["shot_t"] - agg["start_t"]).clip(lower=0)
lab = matches.set_index("match_id")["label"]
agg["match"] = agg["match_id"].map(lab)

t1, t2 = st.tabs(["Anatomy of a chance", "Goal gallery & statistics"])
with t1:
    ml = matches[matches["match_id"].isin(mids)].sort_values("match_date")
    label = st.selectbox("Match", ml["label"].tolist(), key="sq_match")
    mid = ml.loc[ml["label"] == label, "match_id"].iloc[0]
    c = st.columns(3)
    goals_only = c[0].checkbox("Goals only", False, key="sq_goals")
    order = c[1].selectbox("Order shots by", ["Time", "xG (highest first)", "Packing in the sequence"], key="sq_order")
    ms = agg[agg["match_id"] == mid].copy()
    if goals_only:
        ms = ms[ms["is_goal"]]
    ms = ms.sort_values({"Time": "shot_t", "xG (highest first)": "xg", "Packing in the sequence": "packing"}[order], ascending=order == "Time")
    if ms.empty:
        st.info("No shots in this selection.")
    else:
        names = {r.shot_id: f"{r.minute}' {r.shooter} ({r.team}) - xG {r.xg:.2f} - {'GOAL' if r.is_goal else r.outcome}" for r in ms.itertuples()}
        sid = c[2].selectbox("Shot", list(names), format_func=names.get, key="sq_shot")
        shot = ms[ms["shot_id"] == sid].iloc[0]
        seq = m[m["shot_id"] == sid].sort_values("row").tail(25).reset_index(drop=True)
        k = st.columns(5)
        k[0].metric("Actions in the sequence", len(seq))
        k[1].metric("Players involved", int(shot["players"]) if pd.notna(shot["players"]) else 0)
        k[2].metric("Opponents bypassed", int(shot["packing"]) if pd.notna(shot["packing"]) else 0)
        k[3].metric("Threat added (xT)", f"{shot['xt']:.2f}" if pd.notna(shot["xt"]) else "n/a")
        k[4].metric("Shot xG", f"{shot['xg']:.2f}")
        fig = new_pitch(f"{shot['team']} - {shot['minute']}' {shot['shooter']}" + (" - GOAL" if shot["is_goal"] else ""), 680)
        n = len(seq)
        for i, r in seq.iterrows():
            col = lerp_color("#ffe066", "#ff006e", i / max(n - 1, 1))
            fig.add_annotation(x=r.ex, y=r.ey, ax=r.sx, ay=r.sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.2,
                               arrowwidth=3.2 if r.action_type == "Pass" else 2, arrowcolor=col, opacity=0.95)
            pos = r.position
            fig.add_trace(go.Scatter(x=[r.sx], y=[r.sy], mode="markers+text", text=[str(i + 1)], textfont=dict(size=11, color="#111", family="Arial Black"), showlegend=False,
                                     hoverinfo="text", hovertext=f"{i + 1}. {r.player} ({pos})<br>{r.action_type}" + ("" if r.completed else " (failed)") + f"<br>packing {0 if pd.isna(r.pack_cmp) else int(r.pack_cmp)} | xT {r.xt:+.3f}",
                                     marker=dict(size=24, symbol="circle" if r.action_type == "Pass" else "square", color=ROLE_COLORS[role_group(pos)], line=dict(color="white", width=2))))
        if len(seq) and pd.notna(shot["shot_x"]):
            last = seq.iloc[-1]
            fig.add_trace(go.Scatter(x=[last.ex, shot["shot_x"]], y=[last.ey, shot["shot_y"]], mode="lines", showlegend=False, hoverinfo="skip", line=dict(color="white", width=2, dash="dot")))
        if pd.notna(shot["shot_x"]):
            fig.add_trace(go.Scatter(x=[shot["shot_x"]], y=[shot["shot_y"]], mode="markers", name="Goal" if shot["is_goal"] else "Shot", hoverinfo="text",
                                     hovertext=f"{shot['shooter']} - xG {shot['xg']:.2f} - {shot['outcome']}",
                                     marker=dict(size=34, symbol="star", color="#ff1744" if shot["is_goal"] else "#ffd166", line=dict(color="white", width=2))))
            add_caption(fig, shot["shot_x"] - 2, shot["shot_y"] - 6, f"{shot['shooter']}<br>xG {shot['xg']:.2f}", size=11)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Numbers show the order of the actions (circle = pass, square = carry); arrow colour runs from yellow (start) to pink (the last action before the shot). Circle colour = role of the player "
                   "(GK gold, DF blue, MD green, FW red). The star is the shot. Sequences are limited to the last 25 actions.")
        l, r = st.columns(2)
        seq["step"] = np.arange(1, n + 1)
        seq["pack"] = seq["pack_cmp"].fillna(0)
        seq["cum_xt"] = seq["xt"].cumsum()
        l.plotly_chart(px.bar(seq, x="step", y="pack", color="action_type", labels={"pack": "opponents bypassed", "step": "step"}).update_layout(height=320, title="Packing at each step"), use_container_width=True)
        r.plotly_chart(px.line(seq, x="step", y="cum_xt", markers=True, labels={"cum_xt": "cumulative xT", "step": "step"}).update_layout(height=320, title="Threat built up"), use_container_width=True)
        st.dataframe(seq[["step", "player", "position", "action_type", "recipient", "pack", "xt", "completed", "under_pressure"]].round(3), hide_index=True, use_container_width=True)

with t2:
    G = agg.dropna(subset=["steps"]).copy()
    goals = G[G["is_goal"]]
    c = st.columns(4)
    c[0].metric("Shots analysed", f"{len(G):,}")
    c[1].metric("Goals", f"{len(goals):,}")
    c[2].metric("Median packing: goals", f"{goals['packing'].median():.0f}" if len(goals) else "n/a")
    c[3].metric("Median packing: other shots", f"{G[~G['is_goal']]['packing'].median():.0f}")
    top = goals.sort_values("packing", ascending=False).head(15)
    if len(top):
        st.subheader("Goals created by the most line-breaking build-up")
        fig = px.bar(top.iloc[::-1], x="packing", y=top.iloc[::-1]["shooter"] + " (" + top.iloc[::-1]["minute"].astype(str) + "')", orientation="h", color="xt", color_continuous_scale="Magma",
                     hover_data=["match", "steps", "players", "duration_s"], labels={"packing": "opponents bypassed in the build-up", "y": ""})
        st.plotly_chart(fig.update_layout(height=520), use_container_width=True)
        st.dataframe(top[["match", "shooter", "minute", "steps", "players", "packing", "xt", "duration_s", "xg"]].round(2), hide_index=True, use_container_width=True)
    st.subheader("Sequence length vs packing")
    st.plotly_chart(px.scatter(G.assign(result=np.where(G["is_goal"], "goal", "no goal")), x="steps", y="packing", color="result", size="xg", opacity=0.6, hover_name="shooter",
                               color_discrete_map={"goal": "#ff006e", "no goal": "#4cc9f0"}, labels={"steps": "actions before the shot", "packing": "opponents bypassed"})
                    .update_layout(height=480), use_container_width=True)
    st.caption("Bubble size = shot xG. Long, patient build-ups and short, direct line-breaking attacks both produce goals - the chart shows which dominate in the current filters.")
