import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, add_player, lerp_color, new_pitch, role_group, short_name, ROLE_COLORS
from ui import setup
from utils import get_data, load_matches

setup("Team Shape & Passing Network", "🕸️", "Where each player really plays (average position, labelled with name and position) and who passes to whom - arrows show direction, thickness the number of passes, colour the packing created.")
base, df, opt = get_data()
matches = load_matches()

team = st.selectbox("Team", sorted(df["team"].astype(str).unique()))
td = df[df["team"].astype(str) == team].copy()
td["player"] = td["player"].astype(str)
ml = matches[matches["match_id"].isin(td["match_id"].unique())].sort_values("match_date")
scope = st.selectbox("Scope", ["All filtered matches"] + ml["label"].tolist())
if scope != "All filtered matches":
    td = td[td["match_id"] == ml.loc[ml["label"] == scope, "match_id"].iloc[0]]
single = scope != "All filtered matches"
c = st.columns(3)
n_players = c[0].slider("Players shown (most involved)", 8, 16, 11 if single else 13)
min_pass = c[1].slider("Minimum passes per link", 1, 40, 3 if single else 12)
heat = c[2].checkbox("Show touch heatmap", True)

touch = td.groupby("player").agg(touches=("id", "size"), x=("sx", "mean"), y=("sy", "mean"), pack=("pack_used", "sum"), xt=("xt", "sum"),
                                 position=("position", lambda s: s.dropna().astype(str).mode().iat[0] if s.notna().any() else None)).reset_index()
touch = touch.sort_values("touches", ascending=False).head(n_players)
if touch.empty:
    st.warning("No data."); st.stop()
touch["abbr"] = touch["position"].astype(str).map(abbr)
touch["role"] = touch["position"].map(role_group)
pas = td[(td["action_type"] == "Pass") & td["recipient"].notna() & td["completed"]].copy()
pas["recipient"] = pas["recipient"].astype(str)
pas = pas[pas["player"].isin(touch["player"]) & pas["recipient"].isin(touch["player"])]
e = pas.groupby(["player", "recipient"]).agg(n=("id", "size"), pack=("pack_used", "sum")).reset_index()
e = e[(e["n"] >= min_pass) & (e["player"] != e["recipient"])]
P = touch.set_index("player")

fig = new_pitch(f"{team} - average positions & passing links", 700)
if heat:
    fig.add_trace(go.Histogram2d(x=td["sx"], y=td["sy"], xbins=dict(start=0, end=120, size=6), ybins=dict(start=0, end=80, size=6), colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,255,255,0.45)"]],
                                 showscale=False, hoverinfo="skip", name="touches"))
mx = max(e["n"].max(), 1) if len(e) else 1
mp = max(e["pack"].max(), 1) if len(e) else 1
for r in e.itertuples():
    p0, p1 = P.loc[r.player], P.loc[r.recipient]
    v = np.array([p1.x - p0.x, p1.y - p0.y]); L = float(np.hypot(*v))
    if L < 1:
        continue
    u = v / L; off = np.array([-u[1], u[0]]) * 0.9                # parallel offset so A->B and B->A do not overlap
    a0 = np.array([p0.x, p0.y]) + u * 3.6 + off; a1 = np.array([p1.x, p1.y]) - u * 3.6 + off
    fig.add_annotation(x=a1[0], y=a1[1], ax=a0[0], ay=a0[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.0,
                       arrowwidth=1 + 6 * r.n / mx, arrowcolor=lerp_color("#9aa5b1", "#ff006e", r.pack / mp), opacity=0.9)
    fig.add_trace(go.Scatter(x=[(a0[0] + a1[0]) / 2], y=[(a0[1] + a1[1]) / 2], mode="markers", showlegend=False, hoverinfo="text",
                             hovertext=f"{r.player} -> {r.recipient}<br>{r.n} passes, packing {int(r.pack)}", marker=dict(size=10, color="rgba(0,0,0,0)")))
for role, name in (("GK", "Goalkeeper"), ("DF", "Defenders"), ("MD", "Midfielders"), ("FW", "Forwards")):
    s = touch[touch["role"] == role]
    if s.empty:
        continue
    fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="markers+text", name=name, text=s["abbr"], textposition="middle center",
                             textfont=dict(size=10, color="#111", family="Arial Black"), hoverinfo="text",
                             hovertext=s["player"] + " (" + s["position"].astype(str) + ")<br>touches " + s["touches"].astype(str) + "<br>packing " + s["pack"].astype(int).astype(str) + "<br>xT " + s["xt"].round(2).astype(str),
                             marker=dict(color=ROLE_COLORS[role], size=14 + 30 * s["touches"] / touch["touches"].max(), line=dict(color="white", width=2))))
fig.add_trace(go.Scatter(x=touch["x"], y=touch["y"] + 6.2, mode="text", text=touch["player"].map(short_name), textfont=dict(color="white", size=11), showlegend=False, hoverinfo="skip"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Each circle sits at the player's average action location and shows his position (RB, CM, CF ...). Size = involvement. Arrow thickness = passes between two players, "
           "arrow colour from grey to pink = packing those passes created. The team attacks left to right.")

G = nx.DiGraph()
for r in e.itertuples():
    G.add_edge(r.player, r.recipient, n=r.n, pack=r.pack, dist=1 / (r.pack + 0.5))
t = touch[["player", "position", "touches", "pack", "xt"]].rename(columns={"pack": "packing", "xt": "xT"}).copy()
if G.number_of_edges():
    pr = nx.pagerank(G, weight="pack"); bt = nx.betweenness_centrality(G, weight="dist")
    t["packing_sent_via_links"] = t["player"].map(dict(G.out_degree(weight="pack"))).fillna(0)
    t["packing_received"] = t["player"].map(dict(G.in_degree(weight="pack"))).fillna(0)
    t["pagerank"] = t["player"].map(pr).round(3); t["betweenness"] = t["player"].map(bt).round(3)
st.subheader("Network metrics (NetworkX)")
st.dataframe(t.round(2).sort_values("touches", ascending=False), hide_index=True, use_container_width=True)
st.caption("PageRank and betweenness use packing as the link weight, so they highlight the players through whom line-breaking passes flow.")
