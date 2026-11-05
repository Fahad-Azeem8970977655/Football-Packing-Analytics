import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, add_arrows, add_player, new_pitch, role_group, ROLE_COLORS
from ui import setup
from utils import get_data, load_minutes

setup("Receivers & Targets", "🧲", "Who receives the ball between the opponent's lines or in behind them? Receiving maps, line-breaking duos and the players defences struggle to pick up.")
base, df, opt = get_data()
mins = load_minutes()
P = df[(df["action_type"] == "Pass") & df["completed"] & df["recipient"].notna()].copy()
P["recipient"], P["player"], P["team"] = P["recipient"].astype(str), P["player"].astype(str), P["team"].astype(str)
has = P["def_line_x"].notna() & P["mid_line_x"].notna()
P["between"] = has & (P["ex"] >= P["mid_line_x"]) & (P["ex"] <= P["def_line_x"])
P["behind"] = P["def_line_x"].notna() & (P["ex"] > P["def_line_x"])
pos_of = df.groupby(df["player"].astype(str), observed=True)["position"].agg(lambda s: s.dropna().astype(str).mode().iat[0] if s.notna().any() else None)

g = P.groupby(["recipient", "team"]).agg(receptions=("id", "size"), between_lines=("between", "sum"), behind_line=("behind", "sum"), packing_received=("pack_used", "sum"),
                                         xt_received=("xt", "sum"), avg_x=("ex", "mean"), avg_y=("ey", "mean")).reset_index()
m = mins[mins["match_id"].isin(df["match_id"].unique())].groupby(["player", "team"])["minutes"].sum().reset_index().rename(columns={"player": "recipient"})
m["team"] = m["team"].astype(str)
g = g.merge(m, on=["recipient", "team"], how="left")
mm = st.slider("Minimum minutes", 200, 3000, 600, 100, key="rt_min")
g = g[g["minutes"] >= mm].copy()
if len(g) < 5:
    st.warning("Too few players - lower the minutes slider or widen the sidebar filters."); st.stop()
for c, n in (("receptions", "receptions_p90"), ("between_lines", "between_lines_p90"), ("behind_line", "behind_line_p90"), ("packing_received", "packing_received_p90"), ("xt_received", "xt_received_p90")):
    g[n] = g[c] / g["minutes"] * 90
g["position"] = g["recipient"].map(pos_of).astype(object)
g["role"] = g["position"].map(role_group)
k = st.columns(4)
k[0].metric("Completed passes analysed", f"{len(P):,}")
k[1].metric("Landed between the lines", f"{P['between'].sum() / len(P):.1%}", help="Destination between the opponent's midfield line and defensive line.")
k[2].metric("Landed in behind", f"{P['behind'].sum() / len(P):.1%}")
k[3].metric("Receivers", len(g))

t1, t2, t3 = st.tabs(["Leaderboard", "Receiving map", "Line-breaking duos"])
with t1:
    METR = {"between_lines_p90": "Receptions between the lines / 90", "behind_line_p90": "Receptions in behind / 90", "packing_received_p90": "Packing received / 90",
            "xt_received_p90": "Threat (xT) received / 90", "receptions_p90": "Receptions / 90"}
    l, r = st.columns([1, 2])
    mt = l.selectbox("Rank by", list(METR), format_func=METR.get, key="rt_metric")
    n = l.slider("Players shown", 5, 40, 20, key="rt_n")
    roles = l.multiselect("Role", ["GK", "DF", "MD", "FW"], default=["DF", "MD", "FW"], key="rt_role")
    G = g[g["role"].isin(roles)]
    top = G.sort_values(mt, ascending=False).head(n)
    fig = px.bar(top.iloc[::-1], x=mt, y=top.iloc[::-1]["recipient"] + " (" + top.iloc[::-1]["position"].astype(str).map(abbr) + ")", color="role", color_discrete_map=ROLE_COLORS, orientation="h",
                 hover_data=["team", "minutes"], labels={mt: METR[mt], "y": ""})
    r.plotly_chart(fig.update_layout(height=max(380, 24 * n)), use_container_width=True)
    st.plotly_chart(px.scatter(G, x="between_lines_p90", y="behind_line_p90", size="receptions", color="role", color_discrete_map=ROLE_COLORS, hover_name="recipient", hover_data=["team", "position"],
                               labels={"between_lines_p90": "receptions between the lines / 90", "behind_line_p90": "receptions in behind / 90"}).update_layout(height=520), use_container_width=True)
    st.dataframe(G.sort_values(mt, ascending=False)[["recipient", "team", "position", "minutes", "receptions", "between_lines_p90", "behind_line_p90", "packing_received_p90", "xt_received_p90"]].round(2).head(100),
                 hide_index=True, use_container_width=True)

with t2:
    who = st.selectbox("Player", sorted((g["recipient"] + " | " + g["team"]).tolist()), key="rt_player")
    name, team = who.split(" | ", 1)
    R = P[(P["recipient"] == name) & (P["team"] == team)]
    pos = pos_of.get(name)
    fig = new_pitch(f"Where {name} ({abbr(pos)}) receives the ball", 640)
    fig.add_trace(go.Histogram2d(x=R["ex"], y=R["ey"], xbins=dict(start=0, end=120, size=6), ybins=dict(start=0, end=80, size=6), colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,230,80,0.75)"]],
                                 showscale=False, hoverinfo="skip", name="receptions"))
    S_ = R.sort_values("pack_used", ascending=False).head(120)
    add_arrows(fig, S_, max_n=120)
    r_ = g[(g["recipient"] == name) & (g["team"] == team)].iloc[0]
    add_player(fig, r_.avg_x, r_.avg_y, abbr(pos), ROLE_COLORS[role_group(pos)], symbol="star", size=44, name="Average reception spot", showlegend=True, hover=f"{name} ({pos})")
    st.plotly_chart(fig, use_container_width=True)
    c = st.columns(4)
    c[0].metric("Receptions", int(r_.receptions))
    c[1].metric("Between the lines", int(r_.between_lines))
    c[2].metric("In behind", int(r_.behind_line))
    c[3].metric("Packing received", int(r_.packing_received))
    st.caption("Yellow glow = where the passes land (denser = more receptions). Arrows = the 120 passes he received that bypassed the most opponents, coloured by packing. The star is his average receiving spot.")
    top_p = R.groupby("player").agg(passes=("id", "size"), packing=("pack_used", "sum")).sort_values("passes", ascending=False).head(8).reset_index()
    st.markdown("**Who finds him most**")
    st.dataframe(top_p.rename(columns={"player": "passer"}), hide_index=True, use_container_width=True)

with t3:
    D = P.groupby(["player", "recipient", "team"]).agg(passes=("id", "size"), packing=("pack_used", "sum"), between=("between", "sum"), xt=("xt", "sum")).reset_index()
    mp = st.slider("Minimum passes between the pair", 5, 80, 20, key="rt_duo")
    D = D[D["passes"] >= mp].copy()
    D["packing_per_pass"] = D["packing"] / D["passes"]
    D["duo"] = D["player"] + " -> " + D["recipient"]
    if D.empty:
        st.info("No pair reaches that many passes - lower the slider.")
    else:
        top = D.sort_values("packing", ascending=False).head(20)
        st.plotly_chart(px.bar(top.iloc[::-1], x="packing", y="duo", color="packing_per_pass", color_continuous_scale="Magma", orientation="h", hover_data=["team", "passes", "between", "xt"],
                               labels={"packing": "opponents bypassed together", "duo": ""}).update_layout(height=620), use_container_width=True)
        st.dataframe(D.sort_values("packing_per_pass", ascending=False).head(20).round(2), hide_index=True, use_container_width=True)
        st.caption("Top: duos that together bypass the most opponents. Table: duos whose passes bypass the most opponents per pass.")
