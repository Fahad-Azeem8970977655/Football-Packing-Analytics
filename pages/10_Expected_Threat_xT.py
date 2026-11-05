import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, add_player, lerp_color, new_pitch, role_group, ROLE_COLORS
from ui import setup
from utils import get_data, load_matches, load_minutes, load_xt_grid

setup("Expected Threat (xT)", "⚡", "A Markov-chain model gives every zone of the pitch a value - the chance of scoring soon from there. Each pass or carry is then scored by the threat it adds.")
base, df, opt = get_data()
matches = load_matches()
grid = load_xt_grid()
NY, NX = grid.shape
xc, yc = (np.arange(NX) + 0.5) * 120 / NX, (np.arange(NY) + 0.5) * 80 / NY
t1, t2, t3, t4 = st.tabs(["The xT surface", "Biggest threat actions", "Players: threat vs packing", "Teams"])

with t1:
    fig = new_pitch("Expected Threat of the ball in each zone (attacking left to right)", 640)
    fig.add_trace(go.Heatmap(x=xc, y=yc, z=grid, colorscale="Inferno", opacity=0.82, colorbar=dict(title="xT"), hovertemplate="xT %{z:.3f}<extra></extra>", zsmooth=False))
    st.plotly_chart(fig, use_container_width=True)
    c = st.columns(3)
    c[0].metric("Highest zone value", f"{grid.max():.3f}", help="Probability-like value: scoring chance from the best central zone near goal.")
    c[1].metric("Own-goal-area value", f"{grid[:, 0].mean():.4f}")
    c[2].metric("Grid", f"{NX} x {NY} zones")
    st.markdown("**How the model works.** For every zone we measure how often a possession there ends in a shot (s), how likely that shot is to score (g, from StatsBomb xG) and how often the ball moves "
                "to each other zone when a pass or carry succeeds. The value of a zone is `xT = s*g + (1-s) * sum(P(move to z') * xT(z'))`, solved by iterating until it stops changing. "
                "A pass is then worth `xT(end zone) - xT(start zone)` when completed (0 if it fails). It was trained on all freeze-frame competitions (men's and women's together).")

with t2:
    ml = matches[matches["match_id"].isin(df["match_id"].unique())].sort_values("match_date")
    label = st.selectbox("Match", ml["label"].tolist(), key="xt_match")
    mid = ml.loc[ml["label"] == label, "match_id"].iloc[0]
    md = df[df["match_id"] == mid]
    c = st.columns(3)
    team = c[0].selectbox("Team", sorted(md["team"].astype(str).unique()), key="xt_team")
    n = c[1].slider("Actions shown", 10, 80, 30, key="xt_n")
    md = md[(md["team"].astype(str) == team) & (md["xt"] > 0)].sort_values("xt", ascending=False)
    top = md.head(n)
    fig = new_pitch(f"{team}: the {len(top)} actions that added the most threat", 650)
    if len(top):
        mx = top["xt"].max()
        for r in top.itertuples():
            fig.add_annotation(x=r.ex, y=r.ey, ax=r.sx, ay=r.sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.2,
                               arrowwidth=2 + 3 * r.xt / mx, arrowcolor=lerp_color("#ffe066", "#ff006e", r.xt / mx), opacity=0.95)
        fig.add_trace(go.Scatter(x=(top["sx"] + top["ex"]) / 2, y=(top["sy"] + top["ey"]) / 2, mode="markers", showlegend=False, hoverinfo="text",
                                 text=top["player"].astype(str) + " (" + top["position"].astype(str) + ") " + top["minute"].astype(str) + "' " + top["action_type"].astype(str) + "<br>xT +" + top["xt"].round(3).astype(str)
                                 + " | packing " + top["pack_used"].astype(int).astype(str), marker=dict(size=10, color="rgba(0,0,0,0)")))
        pos_of = df[df["match_id"] == mid].groupby("player", observed=True)["position"].agg(lambda s: s.dropna().astype(str).mode().iat[0] if s.notna().any() else None)
        for p_, r in top.groupby("player", observed=True).agg(sx=("sx", "mean"), sy=("sy", "mean"), xt=("xt", "sum")).sort_values("xt", ascending=False).head(5).iterrows():
            pos = pos_of.get(p_)
            add_player(fig, r.sx, r.sy, abbr(pos), ROLE_COLORS[role_group(pos)], size=30, hover=f"{p_} ({pos}) - threat in these actions: {r.xt:.2f}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Arrow colour: yellow (small) to pink (large threat added). Circles = the five players behind most of this threat, with their position.")
    if len(top):
        st.dataframe(top[["minute", "player", "position", "action_type", "xt", "pack_used", "completed"]].round(3).head(40), hide_index=True, use_container_width=True)

with t3:
    mins = load_minutes()
    d = df.assign(player=df["player"].astype(str), team=df["team"].astype(str))
    g = d.groupby(["player", "team"]).agg(actions=("id", "size"), xt=("xt", "sum"), packing=("pack_used", "sum"),
                                          position=("position", lambda s: s.dropna().astype(str).mode().iat[0] if s.notna().any() else None)).reset_index()
    m = mins[mins["match_id"].isin(df["match_id"].unique())].groupby(["player", "team"])["minutes"].sum().reset_index()
    g = g.merge(m, on=["player", "team"], how="left")
    mm = st.slider("Minimum minutes", 200, 3000, 600, 100, key="xt_min")
    g = g[g["minutes"] >= mm].copy()
    if len(g) < 5:
        st.warning("Too few players - lower the minutes slider.")
    else:
        g["xt_p90"] = g["xt"] / g["minutes"] * 90
        g["packing_p90"] = g["packing"] / g["minutes"] * 90
        g["role"] = g["position"].map(role_group)
        fig = px.scatter(g, x="packing_p90", y="xt_p90", color="role", size="minutes", hover_name="player", hover_data=["team", "position"],
                         color_discrete_map=ROLE_COLORS, labels={"packing_p90": "packing per 90 (opponents bypassed)", "xt_p90": "threat added per 90 (xT)"})
        fig.add_vline(x=g["packing_p90"].median(), line_dash="dot", line_color="gray")
        fig.add_hline(y=g["xt_p90"].median(), line_dash="dot", line_color="gray")
        for q, (x, y) in {"Bypass AND threaten": (0.97, 0.97), "Bypass, little threat": (0.97, 0.03), "Threat without bypassing": (0.03, 0.97)}.items():
            fig.add_annotation(xref="paper", yref="paper", x=x, y=y, text=q, showarrow=False, xanchor="right" if x > 0.5 else "left", font=dict(color="#7d889c", size=11))
        st.plotly_chart(fig.update_layout(height=620), use_container_width=True)
        r_ = g[["packing_p90", "xt_p90"]].corr().iloc[0, 1]
        st.caption(f"Correlation between packing and threat per 90: {r_:.2f}. Players above the dotted lines on both axes are the most complete line-breakers.")
        cols = ["player", "team", "position", "minutes", "packing_p90", "xt_p90"]
        a, b = st.columns(2)
        a.markdown("**Most threat added per 90**")
        a.dataframe(g.sort_values("xt_p90", ascending=False)[cols].head(15).round(2), hide_index=True, use_container_width=True)
        b.markdown("**Most packing but least threat** (bypass defenders, add little danger)")
        g["gap"] = g["packing_p90"].rank(pct=True) - g["xt_p90"].rank(pct=True)
        b.dataframe(g.sort_values("gap", ascending=False)[cols].head(15).round(2), hide_index=True, use_container_width=True)

with t4:
    tm = df.assign(team=df["team"].astype(str)).groupby(["match_id", "team"]).agg(xt=("xt", "sum"), pack=("pack_used", "sum")).reset_index()
    tot = tm.groupby("match_id")["xt"].transform("sum")
    tm["xt_against"] = tot - tm["xt"]
    tm["n"] = tm.groupby("match_id")["team"].transform("nunique")
    tm = tm[tm["n"] == 2]
    t = tm.groupby("team").agg(matches=("match_id", "nunique"), xt_for=("xt", "mean"), xt_against=("xt_against", "mean"), packing=("pack", "mean")).reset_index()
    mmin = st.slider("Minimum matches", 1, int(max(t["matches"].max(), 2)), min(5, int(t["matches"].max())), key="xt_mm")
    t = t[t["matches"] >= mmin]
    t["net"] = t["xt_for"] - t["xt_against"]
    if len(t):
        fig = px.scatter(t, x="xt_for", y="xt_against", size="matches", color="net", color_continuous_scale="RdYlGn", hover_name="team", text="team",
                         labels={"xt_for": "threat created per match", "xt_against": "threat allowed per match"})
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig.update_traces(textposition="top center", textfont_size=9).update_layout(height=600), use_container_width=True)
        st.dataframe(t.sort_values("net", ascending=False).round(2), hide_index=True, use_container_width=True)
