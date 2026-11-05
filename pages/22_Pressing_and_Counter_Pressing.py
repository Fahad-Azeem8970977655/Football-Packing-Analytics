import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, new_pitch, role_group, ROLE_COLORS
from ui import setup
from utils import comp_filter, load_counterpress, load_def_events, load_minutes, load_pressing_tm, load_regains

setup("Pressing & Counter-pressing", "🔥", "How high and how often teams press, what they do in the five seconds after losing the ball, and how dangerous their regains are.")
M = comp_filter()
mids = set(M["match_id"])
cs = M.set_index("match_id")["cs"]
D = load_def_events()
D = D[D["match_id"].isin(mids)]
T = load_pressing_tm()
T = T[T["match_id"].isin(mids)].copy()
L = load_counterpress()
L = L[L["match_id"].isin(mids)].copy()
R = load_regains()
R = R[R["match_id"].isin(mids)].copy()
for d in (T, L, R, D):
    d["team"] = d["team"].astype(str)
    d["cs"] = d["match_id"].map(cs)
    d["tc"] = d["team"] + " | " + d["cs"]
t1, t2, t3, t4 = st.tabs(["Pressing intensity", "Counter-pressing", "Regains & transitions", "Defensive actions & players"])

with t1:
    g = T.groupby("tc").agg(matches=("match_id", "nunique"), pressures=("pressures", "sum"), press_x=("press_x", "mean"), opp_half=("press_opp_half", "sum"), opp_passes_60=("opp_passes_60", "sum"),
                            da_60=("da_60", "sum"), high_regains=("high_regains", "sum"), f3_regains=("final_third_regains", "sum")).reset_index()
    g["pressures_per_match"] = g["pressures"] / g["matches"]
    g["opp_half_%"] = 100 * g["opp_half"] / g["pressures"]
    g["PPDA"] = g["opp_passes_60"] / g["da_60"].where(g["da_60"] > 0)
    g["high_regains_pm"] = g["high_regains"] / g["matches"]
    mm = st.slider("Minimum matches", 1, int(max(g["matches"].max(), 2)), min(4, int(g["matches"].max())), key="pr_mm")
    g = g[g["matches"] >= mm]
    if len(g) < 3:
        st.warning("Too few teams - lower the slider or widen the filters.")
    else:
        k = st.columns(4)
        k[0].metric("Average pressures per team-match", f"{T['pressures'].mean():.0f}")
        k[1].metric("Average pressing height", f"{g['press_x'].mean():.0f} yd", help="x of the pressure events; 120 = opponent goal line.")
        k[2].metric("PPDA-style (median)", f"{g['PPDA'].median():.1f}", help="Opponent passes in their own 60% per defensive action (duel, interception, foul) in that zone. Lower = more intense pressing.")
        k[3].metric("High regains per match", f"{g['high_regains_pm'].mean():.1f}")
        fig = px.scatter(g, x="PPDA", y="press_x", size="pressures_per_match", color="high_regains_pm", color_continuous_scale="YlOrRd", hover_name="tc", text="tc",
                         labels={"PPDA": "PPDA-style (lower = presses harder)", "press_x": "average pressing height (x)", "high_regains_pm": "high regains / match"})
        fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig.update_traces(textposition="top center", textfont_size=8).update_layout(height=600), use_container_width=True)
        st.caption("Top-right = high AND intense press (gegenpressing teams). Bubble size = pressures per match. Pressures are StatsBomb 'Pressure' events (the presser's location).")
        team = st.selectbox("Team pressure map", sorted(g["tc"]), key="pr_team")
        dd = D[(D["type"] == "Pressure") & (D["tc"] == team)]
        fig = new_pitch(f"{team} - where they press ({len(dd):,} pressures, attacking left to right)", 600)
        fig.add_trace(go.Histogram2d(x=dd["x"], y=dd["y"], xbins=dict(start=0, end=120, size=6), ybins=dict(start=0, end=80, size=6), colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,80,60,0.85)"]],
                                     showscale=False, hoverinfo="skip"))
        if len(dd):
            fig.add_shape(type="line", x0=dd["x"].mean(), x1=dd["x"].mean(), y0=0, y1=80, line=dict(color="#ffd166", width=3, dash="dash"))
            fig.add_annotation(x=dd["x"].mean(), y=-2.3, text=f"average pressing height {dd['x'].mean():.0f}", showarrow=False, font=dict(color="#ffd166", size=11))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(g.sort_values("PPDA")[["tc", "matches", "pressures_per_match", "press_x", "opp_half_%", "PPDA", "high_regains_pm"]].rename(columns={"tc": "team | competition"}).round(2), hide_index=True, use_container_width=True)

with t2:
    st.markdown("A **loss** = an incomplete pass, a dispossession or a miscontrol. **Counter-press** = the same team pressures within 25 yards in the next 5 seconds. **Regain** = the team wins the ball back (recovery, interception or new possession) within 8 seconds.")
    k = st.columns(4)
    k[0].metric("Ball losses analysed", f"{len(L):,}")
    k[1].metric("Pressed within 5 s", f"{L['pressed'].mean():.0%}")
    k[2].metric("Regained within 5 s", f"{L['regain5'].mean():.0%}")
    k[3].metric("Regained within 8 s", f"{L['regain8'].mean():.0%}")
    c = L.groupby("tc").agg(matches=("match_id", "nunique"), losses=("x", "size"), pressed=("pressed", "mean"), regain5=("regain5", "mean"), regain8=("regain8", "mean")).reset_index()
    c["losses_per_match"] = c["losses"] / c["matches"]
    for col in ("pressed", "regain5", "regain8"):
        c[col] = 100 * c[col]
    mm2 = st.slider("Minimum matches", 1, int(max(c["matches"].max(), 2)), min(4, int(c["matches"].max())), key="cp_mm")
    c = c[c["matches"] >= mm2]
    if len(c) >= 3:
        st.plotly_chart(px.scatter(c, x="pressed", y="regain8", size="losses_per_match", color="regain5", color_continuous_scale="Viridis", hover_name="tc", text="tc",
                                   labels={"pressed": "% of losses followed by a counter-press", "regain8": "% regained within 8 s", "regain5": "% regained in 5 s"})
                        .update_traces(textposition="top center", textfont_size=8).update_layout(height=580), use_container_width=True)
        st.caption("Do not read this as 'pressing causes regains': teams press hardest when the opponent has a clear hold, and loose balls get regained without any press. Compare teams, not cause and effect.")
        l, r = st.columns(2)
        team = l.selectbox("Team (loss map)", ["All teams"] + sorted(c["tc"]), key="cp_team")
        LL = L if team == "All teams" else L[L["tc"] == team]
        nx_, ny_ = 12, 8
        X = LL[LL["x"].between(0, 120) & LL["y"].between(0, 80)]
        xi = np.clip((X["x"] // (120 / nx_)).astype(int), 0, nx_ - 1)
        yi = np.clip((X["y"] // (80 / ny_)).astype(int), 0, ny_ - 1)
        gz = X.assign(xi=xi, yi=yi).groupby(["yi", "xi"]).agg(rate=("regain8", "mean"), n=("x", "size")).reset_index()
        z, n = np.full((ny_, nx_), np.nan), np.zeros((ny_, nx_))
        for q in gz.itertuples():
            n[q.yi, q.xi] = q.n
            if q.n >= 25:
                z[q.yi, q.xi] = 100 * q.rate
        fig = new_pitch(f"Where losses happen: % regained within 8 s ({team})", 540)
        fig.add_trace(go.Heatmap(x=(np.arange(nx_) + 0.5) * 120 / nx_, y=(np.arange(ny_) + 0.5) * 80 / ny_, z=z, customdata=n, opacity=0.8, colorscale="RdYlGn", colorbar=dict(title="% regained"),
                                 hovertemplate="%{z:.0f}% regained<br>%{customdata:.0f} losses<extra></extra>"))
        l.plotly_chart(fig, use_container_width=True)
        ts = np.linspace(0, 8, 17)
        curve = pd.DataFrame({"seconds": ts, "all teams": [100 * (L["regain_s"] <= t).mean() for t in ts]})
        if team != "All teams":
            curve[team] = [100 * (LL["regain_s"] <= t).mean() for t in ts]
        r.plotly_chart(px.line(curve.melt(id_vars="seconds", var_name="who", value_name="regained %"), x="seconds", y="regained %", color="who").update_layout(height=540, title="How fast is the ball won back? (cumulative)"),
                       use_container_width=True)
        lt = L.groupby("loss_type").agg(losses=("x", "size"), pressed=("pressed", "mean"), regain8=("regain8", "mean")).reset_index()
        lt[["pressed", "regain8"]] = 100 * lt[["pressed", "regain8"]]
        st.dataframe(lt.round(1), hide_index=True, use_container_width=True)
        st.dataframe(c.sort_values("regain8", ascending=False).rename(columns={"tc": "team | competition"}).round(1), hide_index=True, use_container_width=True)

with t3:
    R["zone"] = pd.cut(R["x"], [-1, 40, 60, 80, 121], labels=["own third", "own-half to halfway", "opponent half", "final third"])
    zs = R.groupby("zone", observed=True).agg(regains=("x", "size"), shot10=("shot10", "mean"), xg=("xg", "mean"), goals=("goals", "sum")).reset_index()
    zs["shot10"] = 100 * zs["shot10"]
    k = st.columns(3)
    k[0].metric("Regains analysed", f"{len(R):,}")
    k[1].metric("Shot within 10 s", f"{R['shot10'].mean():.1%}")
    k[2].metric("Shot within 10 s after a HIGH regain (x>=60)", f"{R[R['x'] >= 60]['shot10'].mean():.1%}")
    l, r = st.columns(2)
    l.plotly_chart(px.bar(zs, x="zone", y="shot10", text=zs["regains"].map("{:,}".format), labels={"shot10": "% followed by a shot within 10 s", "zone": "where the ball was won"},
                          color_discrete_sequence=["#ffb703"]).update_layout(height=380, title="Regain zone vs danger"), use_container_width=True)
    r.plotly_chart(px.bar(zs, x="zone", y="xg", labels={"xg": "xG of the possession per regain", "zone": ""}, color_discrete_sequence=["#ff006e"]).update_layout(height=380, title="xG per regain"), use_container_width=True)
    team = st.selectbox("Team (regain map)", ["All teams"] + sorted(R["tc"].unique()), key="rg_team")
    RR = R if team == "All teams" else R[R["tc"] == team]
    S_ = RR.sample(min(2500, len(RR)), random_state=0)
    fig = new_pitch(f"Ball recoveries and interceptions - {team}", 620)
    for flag, nm, col in ((False, "no shot within 10 s", "rgba(190,198,210,0.85)"), (True, "shot within 10 s", "#ff006e")):
        s_ = S_[S_["shot10"] == flag]
        fig.add_trace(go.Scatter(x=s_["x"], y=s_["y"], mode="markers", name=nm, hoverinfo="text", text=s_["player"].astype(str) + " | " + s_["minute"].astype(str) + "' | xG " + s_["xg"].astype(str),
                                 marker=dict(size=5 + (25 * s_["xg"].clip(upper=0.4) if flag else 0), color=col, opacity=0.8 if flag else 0.6)))
    st.plotly_chart(fig, use_container_width=True)
    gt = R.groupby("tc").agg(matches=("match_id", "nunique"), regains=("x", "size"), high=("x", lambda s: int((s >= 60).sum())), xg=("xg", "sum"), shot10=("shot10", "sum")).reset_index()
    gt["high_regains_pm"] = gt["high"] / gt["matches"]
    gt["xg_from_regains_pm"] = gt["xg"] / gt["matches"]
    gt["shots_within_10s_pm"] = gt["shot10"] / gt["matches"]
    gt = gt[gt["matches"] >= 4]
    st.markdown("**Teams that turn regains into chances** (4+ matches)")
    st.dataframe(gt.sort_values("xg_from_regains_pm", ascending=False).rename(columns={"tc": "team | competition"})[["team | competition", "matches", "high_regains_pm", "shots_within_10s_pm", "xg_from_regains_pm"]].round(2).head(25),
                 hide_index=True, use_container_width=True)

with t4:
    types = sorted(D["type"].astype(str).unique())
    pick = st.multiselect("Action types", types, default=["Interception", "Clearance", "Block", "Ball Recovery"], key="da_types")
    team = st.selectbox("Team", ["All teams"] + sorted(D["tc"].unique()), key="da_team")
    DD = D[D["type"].astype(str).isin(pick)]
    DD = DD if team == "All teams" else DD[DD["tc"] == team]
    if len(DD):
        fig = new_pitch(f"{team}: {', '.join(pick)} ({len(DD):,} actions)", 600)
        fig.add_trace(go.Histogram2d(x=DD["x"], y=DD["y"], xbins=dict(start=0, end=120, size=5), ybins=dict(start=0, end=80, size=5), colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(80,200,255,0.9)"]],
                                     showscale=False, hoverinfo="skip"))
        fig.add_shape(type="line", x0=DD["x"].mean(), x1=DD["x"].mean(), y0=0, y1=80, line=dict(color="#ffd166", width=3, dash="dash"))
        fig.add_annotation(x=DD["x"].mean(), y=-2.3, text=f"average height {DD['x'].mean():.0f}", showarrow=False, font=dict(color="#ffd166", size=11))
        st.plotly_chart(fig, use_container_width=True)
    mins = load_minutes()
    mins = mins[mins["match_id"].isin(mids)].groupby(["player", "team"])["minutes"].sum().reset_index()
    mins["team"] = mins["team"].astype(str)
    P = D.assign(player=D["player"].astype(str)).pivot_table(index=["player", "team"], columns="type", values="x", aggfunc="size", fill_value=0).reset_index()
    P = P.merge(mins, on=["player", "team"], how="inner")
    mn = st.slider("Minimum minutes", 200, 3000, 900, 100, key="da_min")
    P = P[P["minutes"] >= mn].copy()
    if len(P) >= 5:
        cols = [c for c in ["Pressure", "Ball Recovery", "Interception", "Duel", "Clearance", "Block", "Foul Committed", "Dribbled Past"] if c in P]
        for c in cols:
            P[c + " /90"] = P[c] / P["minutes"] * 90
        pos = D.dropna(subset=["position"]).assign(player=D["player"].astype(str)).groupby(["player", "team"])["position"].agg(lambda s: s.astype(str).mode().iat[0]).reset_index()
        P = P.merge(pos, on=["player", "team"], how="left")
        P["role"] = P["position"].map(role_group)
        metric = st.selectbox("Rank by", [c + " /90" for c in cols], key="da_metric")
        top = P.sort_values(metric, ascending=False).head(20)
        st.plotly_chart(px.bar(top.iloc[::-1], x=metric, y=top.iloc[::-1]["player"] + " (" + top.iloc[::-1]["position"].astype(str).map(abbr) + ")", color="role", color_discrete_map=ROLE_COLORS, orientation="h",
                               hover_data=["team", "minutes"], labels={"y": ""}).update_layout(height=560), use_container_width=True)
        st.dataframe(P.sort_values(metric, ascending=False)[["player", "team", "position", "minutes"] + [c + " /90" for c in cols]].round(2).head(100), hide_index=True, use_container_width=True)
