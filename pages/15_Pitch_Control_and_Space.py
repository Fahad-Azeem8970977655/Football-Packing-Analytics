import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, add_caption, add_player, new_pitch, role_group, ROLE_COLORS
from ui import card, setup
from utils import DATA, get_data, load_frames_top, load_matches

setup("Pitch Control & Space", "🧭", "Who owns the space at the moment of a pass? A nearest-player (Voronoi) control map for any freeze frame, and how much space receivers really have.")
try:
    from scipy.spatial import cKDTree
except ImportError:
    st.error("scipy missing: pip install scipy"); st.stop()
base, df, opt = get_data()
matches = load_matches()
t1, t2 = st.tabs(["Space map (one freeze frame)", "Receiver advantage (all passes)"])

with t1:
    ml = matches[matches["match_id"].isin(df["match_id"].unique())].sort_values("match_date")
    label = st.selectbox("Match", ml["label"].tolist(), key="pc_match")
    mid = ml.loc[ml["label"] == label, "match_id"].iloc[0]
    m_all = df[df["match_id"] == mid]
    md = m_all[m_all["has_frame"]]
    fp = DATA / "frames_by_match" / f"match_id={mid}"
    if not fp.exists():
        ft = load_frames_top()
        ft = ft[ft["match_id"] == mid]
        md = md[md["id"].isin(ft["event_uuid"].unique())]
    c = st.columns(3)
    team = c[0].selectbox("Team", sorted(md["team"].astype(str).unique()), key="pc_team")
    md = md[md["team"].astype(str) == team]
    min_pack = c[1].slider("Minimum packing of the action", 0, 10, 2, key="pc_pack")
    n = c[2].slider("Actions in list", 10, 100, 40, key="pc_n")
    cand = md[md["pack_used"] >= min_pack].sort_values("pack_used", ascending=False).head(n)
    if cand.empty:
        st.info("No actions with that much packing here. Lower the slider.")
    else:
        labels = {r.id: f"{r.minute}' {r.player} | {r.action_type} | packing {r.pack_used}" for r in cand.itertuples()}
        eid = st.selectbox("Action", list(labels), format_func=labels.get, key="pc_action")
        a = cand[cand["id"] == eid].iloc[0]
        fr = pd.read_parquet(fp) if fp.exists() else load_frames_top()
        fr = fr[fr["event_uuid"] == eid].reset_index(drop=True)
        if len(fr) < 6:
            st.warning("Frame too small to draw a control map.")
        else:
            gx, gy = np.meshgrid(np.arange(0.5, 120, 1.0), np.arange(0.5, 80, 1.0))
            pts = np.c_[gx.ravel(), gy.ravel()]
            tree = cKDTree(fr[["x", "y"]].to_numpy())
            _, nearest = tree.query(pts)
            ours = fr["teammate"].to_numpy()[nearest].reshape(gx.shape)
            share = ours.mean() * 100
            share_f3 = ours[:, 80:].mean() * 100
            dest_d, dest_i = tree.query([[a.ex, a.ey]])
            dest_ours = bool(fr["teammate"].iloc[int(dest_i[0])])
            adv = a.min_opp_dist_end - a.min_tm_dist_end if pd.notna(a.min_opp_dist_end) and pd.notna(a.min_tm_dist_end) else np.nan
            fig = new_pitch(f"{a.player} - space control at the moment of the {a.action_type.lower()}", 680)
            fig.add_trace(go.Heatmap(x=np.arange(0.5, 120, 1.0), y=np.arange(0.5, 80, 1.0), z=ours.astype(int), showscale=False, hoverinfo="skip", zsmooth=False,
                                     colorscale=[[0, "rgba(255,70,100,0.42)"], [1, "rgba(76,201,240,0.42)"]], zmin=0, zmax=1))
            opp = ~fr["teammate"]
            for mask, nm, col, txt, sz, sym in ((fr["teammate"] & ~fr["actor"] & ~fr["keeper"], "Teammates", "#4cc9f0", "T", 22, "circle"),
                                                (fr["teammate"] & fr["keeper"], "Own GK", "#4cc9f0", "GK", 26, "diamond"),
                                                (opp & ~fr["keeper"], "Opponents", "#ff6b81", "O", 22, "circle"),
                                                (opp & fr["keeper"], "Opponent GK", "#ff6b81", "GK", 26, "diamond")):
                s = fr[mask]
                if not s.empty:
                    fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="markers+text", text=[txt] * len(s), textfont=dict(size=10, color="#111", family="Arial Black"),
                                             name=f"{nm} ({len(s)})", hoverinfo="skip", marker=dict(color=col, size=sz, symbol=sym, line=dict(color="white", width=1))))
            ar = fr[fr["actor"]]
            if not ar.empty:
                add_player(fig, ar.iloc[0]["x"], ar.iloc[0]["y"], abbr(a.position), ROLE_COLORS[role_group(a.position)], symbol="star", size=40, name="Player on ball",
                           showlegend=True, hover=f"{a.player} ({a.position})")
            fig.add_annotation(x=a.ex, y=a.ey, ax=a.sx, ay=a.sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=4, arrowcolor="#ffd166")
            fig.add_trace(go.Scatter(x=[a.ex], y=[a.ey], mode="markers", name="Destination", marker=dict(size=30, symbol="circle-open", color="#ffd166", line=dict(width=4)),
                                     hoverinfo="skip"))
            if not np.isnan(adv):
                add_caption(fig, a.ex, a.ey - 6, f"space at destination: opponent {a.min_opp_dist_end:.1f} yd away, teammate {a.min_tm_dist_end:.1f} yd away", size=11)
            st.plotly_chart(fig, use_container_width=True)
            k = st.columns(4)
            k[0].metric("Our share of the pitch", f"{share:.0f}%")
            k[1].metric("Our share, final third", f"{share_f3:.0f}%")
            k[2].metric("Destination owned by", "us" if dest_ours else "the opponent")
            k[3].metric("Receiver advantage", "n/a" if np.isnan(adv) else f"{adv:+.1f} yd", help="Distance of the nearest opponent minus distance of the nearest teammate to the destination. Positive = the receiver is closer to a teammate.")
            st.caption("Blue = every point of the pitch that is closer to one of our visible players; red = closer to an opponent (a simple Voronoi map: no speeds or directions, and only players the 360 camera saw). "
                       "Large blue areas around the destination mean the pass lands in space we control.")

with t2:
    P = df[(df["action_type"] == "Pass") & df["min_tm_dist_end"].notna() & df["min_opp_dist_end"].notna()].copy()
    P["adv"] = P["min_opp_dist_end"] - P["min_tm_dist_end"]
    P["adv_bin"] = pd.cut(P["adv"], [-100, -6, -3, -1, 1, 3, 6, 100], labels=["<-6", "-6 to -3", "-3 to -1", "-1 to 1", "1 to 3", "3 to 6", ">6"])
    s = P.groupby("adv_bin", observed=True).agg(passes=("id", "size"), completion=("completed", "mean"), packing=("pack_used", "mean")).reset_index()
    s["completion"] = 100 * s["completion"]
    l, r = st.columns(2)
    l.plotly_chart(px.bar(s, x="adv_bin", y="completion", text=s["passes"].map("{:,}".format), labels={"adv_bin": "receiver advantage (yards)", "completion": "completion %"},
                          color_discrete_sequence=["#4cc9f0"]).update_layout(height=380), use_container_width=True)
    r.plotly_chart(px.bar(s, x="adv_bin", y="packing", labels={"adv_bin": "receiver advantage (yards)", "packing": "average packing"}, color_discrete_sequence=["#ff006e"])
                   .update_layout(height=380), use_container_width=True)
    st.caption("Receiver advantage = distance of the nearest opponent to the destination minus distance of the nearest teammate. Negative values are passes into contested space: "
               "completion collapses, but they are also where the most opponents are bypassed.")
    R = P[P["completed"] & P["recipient"].notna()].assign(recipient=lambda x: x["recipient"].astype(str))
    pos_of = df.groupby(df["player"].astype(str), observed=True)["position"].agg(lambda s_: s_.dropna().astype(str).mode().iat[0] if s_.notna().any() else None)
    g = R.groupby("recipient").agg(receptions=("id", "size"), avg_advantage=("adv", "mean"), contested=("adv", lambda x: (x < 0).mean() * 100), packing_received=("pack_used", "sum")).reset_index()
    mn = st.slider("Minimum receptions", 50, 600, 150, 25, key="pc_rec")
    g = g[g["receptions"] >= mn]
    g["position"] = g["recipient"].map(pos_of)
    g["role"] = g["position"].map(role_group)
    if len(g):
        st.subheader("Who receives the ball in contested space?")
        st.plotly_chart(px.scatter(g, x="avg_advantage", y="packing_received", size="receptions", color="role", color_discrete_map=ROLE_COLORS, hover_name="recipient",
                                   hover_data=["position", "contested"], labels={"avg_advantage": "average receiver advantage (yards; lower = receives under tighter marking)",
                                                                                "packing_received": "packing received"}).update_layout(height=520), use_container_width=True)
        a_, b_ = st.columns(2)
        a_.markdown("**Receive most often under tight marking** (lowest average advantage)")
        a_.dataframe(g.sort_values("avg_advantage").head(12)[["recipient", "position", "receptions", "avg_advantage", "contested"]].round(2), hide_index=True, use_container_width=True)
        b_.markdown("**Always find space** (highest average advantage)")
        b_.dataframe(g.sort_values("avg_advantage", ascending=False).head(12)[["recipient", "position", "receptions", "avg_advantage", "contested"]].round(2), hide_index=True, use_container_width=True)
