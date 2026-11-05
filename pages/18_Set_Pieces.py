import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import new_pitch
from ui import setup
from utils import get_data, load_actions, load_possessions, load_shots

setup("Set Pieces", "🚩", "Corners, free kicks, throw-ins and goal kicks: where deliveries land, what they create, and who takes them best.")
base, df, opt = get_data()
mids = set(base["match_id"].unique())
A = load_actions()
Pq = load_possessions()
Pq = Pq[Pq["match_id"].isin(mids)]
t1, t2, t3 = st.tabs(["Corners", "Free kicks", "Throw-ins & goal kicks"])

with t1:
    C = A[(A["action_type"] == "Pass") & (A["pass_type"] == "Corner") & A["match_id"].isin(mids)].copy()
    corner_poss = Pq[Pq["play_pattern"] == "From Corner"][["match_id", "possession", "shots", "xg", "goals", "has_shot"]]
    C = C.merge(corner_poss, on=["match_id", "possession"], how="left").fillna({"shots": 0, "xg": 0, "goals": 0, "has_shot": 0})
    C["team"], C["player"] = C["team"].astype(str), C["player"].astype(str)
    if len(C) < 50:
        st.warning("Too few corners in these filters.")
    else:
        k = st.columns(5)
        k[0].metric("Corners", f"{len(C):,}")
        k[1].metric("Found a teammate", f"{C['completed'].mean():.0%}")
        k[2].metric("Led to a shot", f"{C['has_shot'].mean():.0%}")
        k[3].metric("Goals", int(C["goals"].sum()))
        k[4].metric("xG per corner", f"{C['xg'].mean():.3f}")
        side = st.radio("Corner side", ["Both", "Upper (y near 0)", "Lower (y near 80)"], horizontal=True, key="sp_side")
        D = C if side == "Both" else C[C["sy"] < 40] if side.startswith("Upper") else C[C["sy"] >= 40]
        fig = new_pitch(f"Corner deliveries ({len(D):,}) - colour = what happened next", 640, half=True)
        S_ = D.sample(min(260, len(D)), random_state=0)
        for r in S_.itertuples():
            col = "#ff1744" if r.goals > 0 else "#ffb703" if r.has_shot > 0 else "rgba(255,255,255,0.45)"
            fig.add_annotation(x=r.ex, y=r.ey, ax=r.sx, ay=r.sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=0.9,
                               arrowwidth=2.2 if r.has_shot else 1.2, arrowcolor=col, opacity=0.9)
        for nm, col in (("Goal", "#ff1744"), ("Shot", "#ffb703"), ("No shot", "#9aa5b1")):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=4), name=nm))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Which delivery zones create the most xG?")
        E = D[D["completed"] | (D["ex"] > 0)].dropna(subset=["ex", "ey"])
        nx_, ny_ = 6, 6
        xi = np.clip(((E["ex"] - 90) // 5).astype(int), 0, nx_ - 1)
        yi = np.clip(((E["ey"] - 15) // 8.33).astype(int), 0, ny_ - 1)
        gz = E.assign(xi=xi, yi=yi).groupby(["yi", "xi"]).agg(xg=("xg", "mean"), n=("id", "size")).reset_index()
        z, n = np.full((ny_, nx_), np.nan), np.zeros((ny_, nx_))
        for r in gz.itertuples():
            n[r.yi, r.xi] = r.n
            if r.n >= 12:
                z[r.yi, r.xi] = r.xg
        fig = new_pitch("Average xG of the possession by where the corner lands (zones with 12+ corners)", 560, half=True)
        fig.add_trace(go.Heatmap(x=90 + 5 * (np.arange(nx_) + 0.5), y=15 + 8.33 * (np.arange(ny_) + 0.5), z=z, customdata=n, opacity=0.8, colorscale="YlOrRd",
                                 colorbar=dict(title="xG / corner"), hovertemplate="xG %{z:.3f}<br>%{customdata:.0f} corners<extra></extra>"))
        st.plotly_chart(fig, use_container_width=True)
        l, r = st.columns(2)
        h = C.groupby(C["pass_height"].astype(str)).agg(corners=("id", "size"), xg=("xg", "mean"), shot_rate=("has_shot", "mean")).reset_index()
        l.plotly_chart(px.bar(h, x="pass_height", y="xg", text=h["corners"], labels={"pass_height": "delivery height", "xg": "xG per corner"}, color_discrete_sequence=["#00d4a6"])
                       .update_layout(height=340, title="By delivery height"), use_container_width=True)
        f = C.assign(side=np.where(C["sy"] < 40, "upper corner", "lower corner")).groupby(["side", C["pass_body_part"].astype(str)]).agg(corners=("id", "size"), xg=("xg", "mean")).reset_index()
        r.plotly_chart(px.bar(f, x="side", y="xg", color="pass_body_part", barmode="group", labels={"xg": "xG per corner"}).update_layout(height=340, title="By corner side and taking foot (proxy for in/out-swingers)"),
                       use_container_width=True)
        mn = st.slider("Minimum corners (teams / takers)", 10, 100, 25, 5, key="sp_min")
        a_, b_ = st.columns(2)
        tt = C.groupby("team").agg(corners=("id", "size"), found_teammate=("completed", "mean"), shot_rate=("has_shot", "mean"), goals=("goals", "sum"), xg_per_corner=("xg", "mean")).reset_index()
        a_.markdown("**Teams**")
        a_.dataframe(tt[tt["corners"] >= mn].sort_values("xg_per_corner", ascending=False).round(3), hide_index=True, use_container_width=True)
        pp = C.groupby("player").agg(corners=("id", "size"), found_teammate=("completed", "mean"), xg_per_corner=("xg", "mean"), avg_length=("length", "mean")).reset_index()
        b_.markdown("**Takers**")
        b_.dataframe(pp[pp["corners"] >= mn].sort_values("xg_per_corner", ascending=False).round(3), hide_index=True, use_container_width=True)

with t2:
    sh = load_shots()
    F = sh[(sh["type"] == "Shot") & (sh["shot_type"] == "Free Kick") & sh["match_id"].isin(mids) & sh["sx"].notna()].copy()
    if len(F) < 20:
        st.warning("Too few direct free kicks in these filters.")
    else:
        F["dist"] = np.hypot(120 - F["sx"], 40 - F["sy"])
        k = st.columns(4)
        k[0].metric("Direct free-kick shots", f"{len(F):,}")
        k[1].metric("Goals", int(F["is_goal"].sum()))
        k[2].metric("Conversion", f"{F['is_goal'].mean():.1%}")
        k[3].metric("Average xG", f"{F['xg'].mean():.3f}")
        fig = new_pitch("Direct free kicks - shot origin and path to goal", 600, half=True)
        for r in F.sample(min(220, len(F)), random_state=0).itertuples():
            fig.add_annotation(x=120, y=40 + (r.sy - 40) * 0.12, ax=r.sx, ay=r.sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=0.8,
                               arrowwidth=2.2 if r.is_goal else 1, arrowcolor="#ff1744" if r.is_goal else "rgba(255,255,255,0.45)", opacity=0.9)
        for nm, col in (("Goal", "#ff1744"), ("Other", "#9aa5b1")):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=4), name=nm))
        st.plotly_chart(fig, use_container_width=True)
        F["dist_bin"] = pd.cut(F["dist"], [0, 20, 25, 30, 40, 80], labels=["<20", "20-25", "25-30", "30-40", "40+"])
        b = F.groupby("dist_bin", observed=True).agg(shots=("id", "size"), goal_rate=("is_goal", "mean"), xg=("xg", "mean")).reset_index()
        b["goal_rate"] = 100 * b["goal_rate"]
        st.plotly_chart(px.bar(b, x="dist_bin", y=["goal_rate", "xg"], barmode="group", labels={"dist_bin": "distance to goal (yards)", "value": "goal rate % / xG"})
                        .update_layout(height=360), use_container_width=True)
        tk = F.groupby("player").agg(shots=("id", "size"), goals=("is_goal", "sum"), xg=("xg", "sum")).reset_index()
        tk["goals_minus_xg"] = tk["goals"] - tk["xg"]
        st.markdown("**Free-kick takers** (3+ shots)")
        st.dataframe(tk[tk["shots"] >= 3].sort_values("goals_minus_xg", ascending=False).round(2).head(20), hide_index=True, use_container_width=True)

with t3:
    T = A[(A["action_type"] == "Pass") & A["pass_type"].isin(["Throw-in", "Goal Kick"]) & A["match_id"].isin(mids)].copy()
    if T.empty:
        st.warning("No throw-ins or goal kicks in these filters.")
    else:
        T["kind"] = T["pass_type"].astype(str)
        s = T.groupby("kind").agg(count=("id", "size"), completion=("completed", "mean"), avg_length=("length", "mean"), forward_m=("forward_m", "mean")).reset_index()
        s["completion"] = 100 * s["completion"]
        st.dataframe(s.round(2), hide_index=True, use_container_width=True)
        G = T[T["kind"] == "Goal Kick"].merge(Pq[Pq["play_pattern"] == "From Goal Kick"][["match_id", "possession", "has_shot", "xg"]], on=["match_id", "possession"], how="left")
        G["style"] = np.where(G["length"] >= 35, "long (35+ yd)", "short (<35 yd)")
        gs = G.groupby("style").agg(goal_kicks=("id", "size"), completion=("completed", "mean"), shot_rate=("has_shot", "mean"), xg=("xg", "mean")).reset_index()
        for c_ in ("completion", "shot_rate"):
            gs[c_] = 100 * gs[c_]
        l, r = st.columns(2)
        l.plotly_chart(px.bar(gs.melt(id_vars="style", value_vars=["completion", "shot_rate"]), x="style", y="value", color="variable", barmode="group",
                              labels={"value": "%", "style": ""}).update_layout(height=360, title="Goal kicks: short vs long"), use_container_width=True)
        r.markdown("**Goal-kick style**")
        r.dataframe(gs.round(3), hide_index=True, use_container_width=True)
        TI = Pq[Pq["play_pattern"] == "From Throw In"].copy()
        TI["zone"] = pd.cut(TI["start_x"], [-1, 40, 80, 121], labels=["own third", "middle third", "attacking third"])
        tz = TI.groupby("zone", observed=True).agg(throw_ins=("has_shot", "size"), shot_rate=("has_shot", "mean"), xg=("xg", "mean")).reset_index()
        tz["shot_rate"] = 100 * tz["shot_rate"]
        st.plotly_chart(px.bar(tz, x="zone", y="shot_rate", text=tz["throw_ins"], labels={"shot_rate": "% of possessions ending in a shot", "zone": "throw-in taken in"}, color_discrete_sequence=["#ffb703"])
                        .update_layout(height=340, title="Throw-ins: how often do they lead to a shot?"), use_container_width=True)
