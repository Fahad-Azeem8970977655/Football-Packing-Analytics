import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from chains import get_chains
from pitch import abbr, add_player, lerp_color, new_pitch, role_group, short_name, ROLE_COLORS
from ui import setup
from utils import comp_filter, load_minutes

setup("Chain Value", "⛓️", "Credit everyone in the chain: every player who touched the ball in a possession that ended in a shot shares in its xG - not just the shooter and the last passer.")
M = comp_filter()
mids = set(M["match_id"])
sh, m = get_chains()
sh = sh[sh["match_id"].isin(mids)]
m = m[m["match_id"].isin(mids)]
if len(sh) < 100:
    st.warning("Too few shots - widen the filters."); st.stop()
part = m.groupby(["shot_id", "player"]).agg(is_key=("is_key", "max"), pack=("pack_cmp", "sum"), xt=("xt", "sum"), touches=("row", "size")).reset_index()
base_cols = ["shot_id", "shooter", "xg", "is_goal", "team", "match_id"]
part = part.merge(sh[base_cols], on="shot_id")
sr = sh[base_cols].rename(columns={"shooter": "player"}).assign(is_key=False, pack=0.0, xt=0.0, touches=0)
sr["shooter"] = sr["player"]
have = set(zip(part["shot_id"], part["player"]))
sr = sr[[(a, b) not in have for a, b in zip(sr["shot_id"], sr["player"])]]
part = pd.concat([part, sr], ignore_index=True)
part["is_shooter"] = part["player"] == part["shooter"]
part["xg_buildup"] = np.where(part["is_shooter"] | part["is_key"], 0.0, part["xg"])
g = part.groupby(["player", "team"]).agg(shots_in_chain=("shot_id", "nunique"), goals_in_chain=("is_goal", "sum"), xg_chain=("xg", "sum"), xg_buildup=("xg_buildup", "sum"), key_passes=("is_key", "sum"),
                                         shots_taken=("is_shooter", "sum"), pack=("pack", "sum")).reset_index()
xs = part[part["is_shooter"]].groupby(["player", "team"])["xg"].sum().rename("xg_shots").reset_index()
g = g.merge(xs, on=["player", "team"], how="left").fillna({"xg_shots": 0})
pos = m.dropna(subset=["position"]).groupby(["player", "team"])["position"].agg(lambda s: s.astype(str).mode().iat[0]).reset_index()
g = g.merge(pos, on=["player", "team"], how="left")
mins = load_minutes()
mins = mins[mins["match_id"].isin(mids)].groupby(["player", "team"])["minutes"].sum().reset_index()
mins["team"] = mins["team"].astype(str)
g = g.merge(mins, on=["player", "team"], how="left")
g["role"] = g["position"].map(role_group)
k = st.columns(4)
k[0].metric("Shots with a chain", f"{len(sh):,}")
k[1].metric("Average actions in a chain", f"{m.groupby('shot_id').size().mean():.1f}")
k[2].metric("Average players in a chain", f"{part.groupby('shot_id').size().mean():.1f}")
k[3].metric("xG in these shots", f"{sh['xg'].sum():,.0f}")
st.markdown("**xG chain** = the xG of every shot the player was involved in (any touch in the possession, or taking the shot). **xG buildup** = the same, excluding the shooter and the final (key) pass - credit for the play *before* the last pass. "
            "Penalties are excluded.")
t1, t2, t3, t4 = st.tabs(["Leaderboards", "Invisible contributors", "Key-pass duos", "Team chain map"])

with t1:
    mn = st.slider("Minimum minutes", 200, 3000, 900, 100, key="cv_min")
    G = g[g["minutes"] >= mn].copy()
    if len(G) < 5:
        st.warning("Too few players - lower the slider.")
    else:
        for c in ("xg_chain", "xg_buildup", "xg_shots", "key_passes", "shots_in_chain", "pack"):
            G[c + "_p90"] = G[c] / G["minutes"] * 90
        METR = {"xg_chain_p90": "xG chain / 90", "xg_buildup_p90": "xG buildup / 90", "key_passes_p90": "Key passes / 90", "xg_shots_p90": "xG of own shots / 90", "pack_p90": "Packing in shot sequences / 90"}
        l, r = st.columns([1, 2])
        mt = l.selectbox("Rank by", list(METR), format_func=METR.get, key="cv_metric")
        roles = l.multiselect("Role", ["GK", "DF", "MD", "FW"], default=["GK", "DF", "MD", "FW"], key="cv_role")
        n = l.slider("Players shown", 5, 40, 20, key="cv_n")
        G = G[G["role"].isin(roles)]
        top = G.sort_values(mt, ascending=False).head(n)
        r.plotly_chart(px.bar(top.iloc[::-1], x=mt, y=top.iloc[::-1]["player"] + " (" + top.iloc[::-1]["position"].astype(str).map(abbr) + ")", color="role", color_discrete_map=ROLE_COLORS, orientation="h",
                              hover_data=["team", "minutes"], labels={mt: METR[mt], "y": ""}).update_layout(height=max(380, 24 * n)), use_container_width=True)
        st.dataframe(G.sort_values(mt, ascending=False)[["player", "team", "position", "minutes", "shots_in_chain", "goals_in_chain", "xg_chain_p90", "xg_buildup_p90", "key_passes_p90", "xg_shots_p90"]].round(2).head(100),
                     hide_index=True, use_container_width=True)
        rs = part.merge(g[["player", "team", "role"]], on=["player", "team"], how="left").groupby("role").agg(xg_chain=("xg", "sum"), xg_buildup=("xg_buildup", "sum")).reset_index()
        rs[["xg_chain", "xg_buildup"]] = 100 * rs[["xg_chain", "xg_buildup"]] / rs[["xg_chain", "xg_buildup"]].sum()
        st.plotly_chart(px.bar(rs.melt(id_vars="role"), x="role", y="value", color="variable", barmode="group", labels={"value": "% of all credit", "role": "role", "variable": ""})
                        .update_layout(height=340, title="Who gets the chain credit? (by role)"), use_container_width=True)

with t2:
    G = g[g["minutes"] >= st.slider("Minimum minutes ", 200, 3000, 900, 100, key="cv_min2")].copy()
    if len(G) >= 5:
        for c in ("xg_buildup", "xg_shots"):
            G[c + "_p90"] = G[c] / G["minutes"] * 90
        fig = px.scatter(G, x="xg_shots_p90", y="xg_buildup_p90", size="minutes", color="role", color_discrete_map=ROLE_COLORS, hover_name="player", hover_data=["team", "position"],
                         labels={"xg_shots_p90": "xG of his own shots / 90", "xg_buildup_p90": "xG buildup / 90 (credit for the play before the last pass)"})
        fig.add_vline(x=G["xg_shots_p90"].median(), line_dash="dot", line_color="gray")
        fig.add_hline(y=G["xg_buildup_p90"].median(), line_dash="dot", line_color="gray")
        st.plotly_chart(fig.update_layout(height=600), use_container_width=True)
        G["gap"] = G["xg_buildup_p90"].rank(pct=True) - G["xg_shots_p90"].rank(pct=True)
        st.markdown("**Invisible contributors**: high buildup credit, few shots of their own")
        st.dataframe(G.sort_values("gap", ascending=False)[["player", "team", "position", "minutes", "xg_buildup_p90", "xg_shots_p90"]].round(2).head(20), hide_index=True, use_container_width=True)
        st.caption("Top-left = players who sit in the middle of chances without taking the shots. Classic examples are deep playmakers and full-backs who are invisible in goal and assist tables.")

with t3:
    key = m[m["is_key"]].copy()
    if len(key) < 50:
        st.info("Too few key passes in this selection.")
    else:
        d = key.groupby(["player", "shooter", "team"]).agg(shots=("shot_id", "nunique"), xg=("xg", "sum"), goals=("is_goal", "sum"), pack=("pack_cmp", "sum")).reset_index()
        mp = st.slider("Minimum shots created by the pair", 2, 15, 4, key="cv_duo")
        d = d[d["shots"] >= mp].copy()
        d["duo"] = d["player"] + " -> " + d["shooter"]
        d["xg_per_shot"] = d["xg"] / d["shots"]
        if len(d):
            top = d.sort_values("xg", ascending=False).head(20)
            st.plotly_chart(px.bar(top.iloc[::-1], x="xg", y="duo", color="xg_per_shot", color_continuous_scale="Magma", orientation="h", hover_data=["team", "shots", "goals", "pack"],
                                   labels={"xg": "xG created together", "duo": ""}).update_layout(height=620), use_container_width=True)
            st.dataframe(d.sort_values("xg_per_shot", ascending=False).head(20).round(2), hide_index=True, use_container_width=True)
            st.caption("Key passer -> shooter. Bars: total xG the pair created; table: pairs whose chances were best on average.")
        else:
            st.info("No pair reaches that many shots - lower the slider.")

with t4:
    team = st.selectbox("Team", sorted(g["team"].astype(str).unique()), key="cv_team")
    mt_ = m[m["team"] == team]
    gt = g[g["team"] == team].sort_values("xg_chain", ascending=False).head(14)
    if len(gt) < 3:
        st.info("Too few players for this team.")
    else:
        loc = mt_[mt_["player"].isin(gt["player"])].groupby("player").agg(x=("sx", "mean"), y=("sy", "mean")).reset_index()
        gt = gt.merge(loc, on="player")
        ed = mt_[(mt_["action_type"] == "Pass") & mt_["recipient"].notna() & mt_["completed"]].copy()
        ed["recipient"] = ed["recipient"].astype(str)
        ed = ed[ed["player"].isin(gt["player"]) & ed["recipient"].isin(gt["player"])].groupby(["player", "recipient"]).size().reset_index(name="n")
        ed = ed[ed["n"] >= 2].sort_values("n", ascending=False).head(30)
        P = gt.set_index("player")
        fig = new_pitch(f"{team}: players in shot chains - circle size = xG chain, arrows = passes inside the chains", 680)
        mx = max(ed["n"].max(), 1) if len(ed) else 1
        for r in ed.itertuples():
            p0, p1 = P.loc[r.player], P.loc[r.recipient]
            v = np.array([p1.x - p0.x, p1.y - p0.y]); L = float(np.hypot(*v))
            if L < 1:
                continue
            u = v / L; off = np.array([-u[1], u[0]]) * 0.9
            a0 = np.array([p0.x, p0.y]) + u * 3.6 + off; a1 = np.array([p1.x, p1.y]) - u * 3.6 + off
            fig.add_annotation(x=a1[0], y=a1[1], ax=a0[0], ay=a0[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.0,
                               arrowwidth=1 + 5 * r.n / mx, arrowcolor=lerp_color("#ffe066", "#ff006e", r.n / mx), opacity=0.9)
        for role, nm in (("GK", "Goalkeeper"), ("DF", "Defenders"), ("MD", "Midfielders"), ("FW", "Forwards")):
            s_ = gt[gt["role"] == role]
            if len(s_):
                fig.add_trace(go.Scatter(x=s_["x"], y=s_["y"], mode="markers+text", name=nm, text=s_["position"].astype(str).map(abbr), textfont=dict(size=10, color="#111", family="Arial Black"), hoverinfo="text",
                                         hovertext=s_["player"] + " (" + s_["position"].astype(str) + ")<br>xG chain " + s_["xg_chain"].round(2).astype(str) + "<br>xG buildup " + s_["xg_buildup"].round(2).astype(str),
                                         marker=dict(color=ROLE_COLORS[role], size=16 + 36 * s_["xg_chain"] / gt["xg_chain"].max(), line=dict(color="white", width=2))))
        fig.add_trace(go.Scatter(x=gt["x"], y=gt["y"] + 6.4, mode="text", text=gt["player"].map(short_name), textfont=dict(color="white", size=11), showlegend=False, hoverinfo="skip"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Positions are the average location of each player's touches inside shot chains, labelled with his position (GK, CB, CM ...). Arrow thickness = number of completed passes between the two inside chains that ended in a shot.")
