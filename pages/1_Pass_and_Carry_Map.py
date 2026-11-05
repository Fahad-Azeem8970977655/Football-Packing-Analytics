import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, add_arrows, add_player, new_pitch, role_group, ROLE_COLORS
from ui import setup
from utils import get_data, load_matches

setup("Pass & Carry Map", "🗺️", "Real arrows on a real pitch: every pass and carry coloured by how many opponents it bypassed. The team attacks left to right.")
base, df, opt = get_data()
matches = load_matches()
tab1, tab2, tab3 = st.tabs(["Match explorer", "Where packing starts", "Where packing ends"])

with tab1:
    ml = matches[matches["match_id"].isin(df["match_id"].unique())].sort_values("match_date")
    label = st.selectbox("Match", ml["label"].tolist())
    mid = ml.loc[ml["label"] == label, "match_id"].iloc[0]
    md_all = df[df["match_id"] == mid]
    c = st.columns(5)
    team = c[0].selectbox("Team", sorted(md_all["team"].astype(str).unique()))
    md = md_all[md_all["team"].astype(str) == team]
    player = c[1].selectbox("Player", ["All"] + sorted(md["player"].astype(str).unique()))
    if player != "All":
        md = md[md["player"].astype(str) == player]
    min_pack = c[2].slider("Min packing", 0, 8, 1)
    kind = c[3].radio("Type", ["Both", "Pass", "Carry"], horizontal=True)
    max_n = c[4].slider("Max arrows", 30, 300, 150, 10)
    if kind != "Both":
        md = md[md["action_type"] == kind]
    show = md[md["pack_used"] >= min_pack].sort_values("pack_used", ascending=False).head(max_n)
    k = st.columns(5)
    k[0].metric("Arrows shown", len(show))
    k[1].metric("Total packing", int(md["pack_used"].sum()))
    k[2].metric("Line-breaking actions", int((md["pack_used"] >= 1).sum()))
    k[3].metric("Defensive-line breaks", int(md["breaks_def"].fillna(0).sum()))
    k[4].metric("Threat added (xT)", f"{md['xt'].sum():.2f}")
    fig = new_pitch(f"{team}" + ("" if player == "All" else f" - {player}"), 650)
    add_arrows(fig, show)
    # label the players behind the arrows with their position
    if len(show):
        pos_of = md_all.groupby("player", observed=True)["position"].agg(lambda s: s.dropna().astype(str).mode().iat[0] if s.notna().any() else None)
        top = show.groupby("player", observed=True).agg(sx=("sx", "mean"), sy=("sy", "mean"), pack=("pack_used", "sum"), n=("id", "size")).sort_values("pack", ascending=False).head(6)
        for p_, r in top.iterrows():
            pos = pos_of.get(p_)
            add_player(fig, r.sx, r.sy, abbr(pos), ROLE_COLORS[role_group(pos)], size=30, hover=f"{p_} ({pos})<br>packing {int(r.pack)} in {int(r.n)} arrows")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Solid arrow = completed, coloured by packing (grey = 0). Dotted line = failed pass. Circles = the six players behind most of the packing, "
               "placed at their average starting spot and labelled with their position (GK gold, DF blue, MD green, FW red).")
    if len(show):
        st.dataframe(show[["minute", "player", "position", "action_type", "recipient", "pack_used", "xt", "breaks_def", "breaks_mid", "n_opp_vis", "completed", "under_pressure"]]
                     .head(50).round(3), hide_index=True, use_container_width=True)


def zone_grid(d, xc, yc, how):
    nx_, ny_ = 12, 8
    x = d[d[xc].between(0, 120) & d[yc].between(0, 80)]
    xi = np.clip((x[xc] // (120 / nx_)).astype(int), 0, nx_ - 1)
    yi = np.clip((x[yc] // (80 / ny_)).astype(int), 0, ny_ - 1)
    t = x.assign(xi=xi, yi=yi).groupby(["yi", "xi"]).agg(avg=("pack_used", "mean"), tot=("pack_used", "sum"), n=("id", "size")).reset_index()
    z, n = np.full((ny_, nx_), np.nan), np.zeros((ny_, nx_))
    for r in t.itertuples():
        n[r.yi, r.xi] = r.n
        if how == "avg" and r.n >= 30:
            z[r.yi, r.xi] = r.avg
        if how == "share":
            z[r.yi, r.xi] = 100 * r.tot / max(x["pack_used"].sum(), 1)
    return (np.arange(nx_) + 0.5) * 120 / nx_, (np.arange(ny_) + 0.5) * 80 / ny_, z, n


with tab2:
    xc_, yc_, z, n = zone_grid(df, "sx", "sy", "avg")
    fig = new_pitch("Average packing of actions that START in each zone", 600)
    fig.add_trace(go.Heatmap(x=xc_, y=yc_, z=z, opacity=0.78, colorscale="YlOrRd", customdata=n, colorbar=dict(title="avg packing"),
                             hovertemplate="avg packing %{z:.2f}<br>actions %{customdata:.0f}<extra></extra>"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Zones with fewer than 30 actions are blank. Uses the sidebar filters.")

with tab3:
    xc_, yc_, z, n = zone_grid(df[df["pack_used"] >= 1], "ex", "ey", "share")
    fig = new_pitch("Where do line-breaking actions LAND? (share of all packing, %)", 600)
    fig.add_trace(go.Heatmap(x=xc_, y=yc_, z=z, opacity=0.78, colorscale="Viridis", customdata=n, colorbar=dict(title="% of packing"),
                             hovertemplate="%{z:.1f}% of packing<extra></extra>"))
    st.plotly_chart(fig, use_container_width=True)
