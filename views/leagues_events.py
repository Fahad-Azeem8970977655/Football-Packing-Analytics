import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import new_pitch
from utils import load_league_shots, load_team_matches


def render():
    st.markdown(
        "This page compares **Wyscout 2017/18** (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, plus Euro 2016 and World Cup 2018) with "
        "**StatsBomb** data: the full **2015/16** Premier League, La Liga, Serie A and Ligue 1 seasons, Copa America 2024, and all your freeze-frame competitions. "
        "These sources have **no player positions**, so true packing cannot be computed. Instead, event-based **proxies** are compared: progressive passes and carries, "
        "final-third and box entries, long-ball share, shots and xG.")
    st.info("Definitions are identical for both sources (open-play passes; progressive = moves the ball >= 10 m closer to goal; entries count completed passes). "
            "Volume metrics (passes, entries) and the long-ball share differ between providers because of how events are recorded and the different seasons, so "
            "compare **within a source** for volumes and be cautious across sources. "
            "StatsBomb xG is the provider's own; Wyscout xG is *estimated* from shot location and body part (model trained on Wyscout shots, held-out matches). "
            "Bundesliga 2015/16 covers only Bayer Leverkusen's 34 matches. Carries exist only in StatsBomb.")
    t = load_team_matches()
    wsh = load_league_shots()

    c = st.columns(2)
    src = c[0].multiselect("Source", ["Wyscout", "StatsBomb"], default=["Wyscout", "StatsBomb"])
    t = t[t["source"].isin(src)]
    comps = sorted(t["comp"].unique().tolist())
    sel = c[1].multiselect("Competitions", comps, default=comps)
    t = t[t["comp"].isin(sel)]
    if t.empty:
        st.warning("Select at least one competition."); return

    METRICS = {"passes": "Passes / match", "pass_completion_pct": "Pass completion %", "avg_pass_len": "Avg pass length (m)", "long_pct": "Long passes % (>=30 m)",
               "progressive": "Progressive passes / match", "progressive_completed": "Progressive completed / match",
               "final_third_entries": "Final-third entries / match", "box_entries": "Box entries / match",
               "progressive_carries": "Progressive carries / match (StatsBomb)", "shots": "Shots / match", "goals": "Goals / match", "xg": "xG / match"}
    agg = t.groupby("comp").agg(matches=("match_id", "nunique"), **{k: (k, "mean") for k in METRICS}).reset_index()
    agg["progressive_share_pct"] = 100 * agg["progressive"] / agg["passes"]
    agg["entries_per_100_passes"] = 100 * (agg["final_third_entries"]) / agg["passes"]
    agg["shots_per_100_passes"] = 100 * agg["shots"] / agg["passes"]

    st.subheader("1. Overview")
    st.dataframe(agg.round(2).rename(columns=METRICS).sort_values("matches", ascending=False), hide_index=True, use_container_width=True)

    st.subheader("2. Compare one metric")
    extra = {**METRICS, "progressive_share_pct": "Progressive share of passes %", "entries_per_100_passes": "Final-third entries per 100 passes",
             "shots_per_100_passes": "Shots per 100 passes"}
    m = st.selectbox("Metric", list(extra), index=list(extra).index("progressive_share_pct"), format_func=extra.get)
    a2 = agg.sort_values(m)
    a2["source"] = a2["comp"].str.extract(r"\[(.*)\]")
    st.plotly_chart(px.bar(a2, x=m, y="comp", color="source", orientation="h", text_auto=".2f", labels={m: extra[m], "comp": ""})
                    .update_layout(height=max(380, 34 * len(a2))), use_container_width=True)

    st.subheader("3. Style profile (radar)")
    radar_m = ["pass_completion_pct", "avg_pass_len", "long_pct", "progressive_share_pct", "entries_per_100_passes", "shots_per_100_passes", "xg"]
    pick = st.multiselect("Compare 2-4 competitions", agg["comp"].tolist(), default=agg["comp"].tolist()[:3], max_selections=4)
    if len(pick) >= 2:
        z = agg.set_index("comp")[radar_m]
        z = (z - z.min()) / (z.max() - z.min()).replace(0, np.nan)
        rad = go.Figure()
        labels = [extra.get(k, k) for k in radar_m]
        for p_ in pick:
            v = z.loc[p_].fillna(0.5).tolist()
            rad.add_trace(go.Scatterpolar(r=v + v[:1], theta=labels + labels[:1], fill="toself", name=p_, opacity=0.55))
        st.plotly_chart(rad.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), height=520,
                                          title="Each axis scaled 0-1 across the selected competitions (1 = highest)"), use_container_width=True)

    st.subheader("4. Teams inside one competition")
    comp = st.selectbox("Competition (team view)", sel, index=0)
    tt = t[t["comp"] == comp].groupby("team").agg(matches=("match_id", "nunique"), passes=("passes", "mean"), pass_completion=("pass_completion_pct", "mean"),
                                                   progressive=("progressive", "mean"), final_third=("final_third_entries", "mean"), box=("box_entries", "mean"),
                                                   shots=("shots", "mean"), xg=("xg", "mean"), goals=("goals", "mean")).reset_index()
    tt["finishing_goals_minus_xg_per_match"] = tt["goals"] - tt["xg"]
    mn = st.slider("Minimum matches", 1, int(tt["matches"].max()), min(5, int(tt["matches"].max())))
    tt = tt[tt["matches"] >= mn].round(2)
    if len(tt):
        st.plotly_chart(px.scatter(tt, x="progressive", y="xg", size="passes", color="finishing_goals_minus_xg_per_match", hover_name="team", text="team",
                                   color_continuous_scale="RdYlGn", labels={"progressive": "progressive passes per match", "xg": "xG per match"})
                        .update_traces(textposition="top center", textfont_size=9).update_layout(height=560), use_container_width=True)
        st.dataframe(tt.sort_values("progressive", ascending=False), hide_index=True, use_container_width=True)

    st.subheader("5. Does progression create chances? (team-match level)")
    rows = []
    for k, x in t.groupby("comp"):
        if len(x) >= 40 and x["progressive"].std() > 0:
            rows.append({"competition": k, "team_matches": len(x), "correlation (progressive, xG)": x[["progressive", "xg"]].corr().iloc[0, 1],
                         "xG per +10 progressive passes": 10 * np.polyfit(x["progressive"], x["xg"], 1)[0],
                         "correlation (box entries, xG)": x[["box_entries", "xg"]].corr().iloc[0, 1]})
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3).sort_values("xG per +10 progressive passes", ascending=False), hide_index=True, use_container_width=True)
    st.caption("Compare with the freeze-frame pages: there, packing (opponents actually bypassed) is measured directly; here only the ball's movement is known.")

    st.subheader("6. Shot map")
    wc = sorted(wsh["comp"].unique().tolist(), key=lambda x: ("Wyscout" in x, x))
    if wc:
        l, r = st.columns(2)
        w_comp = l.selectbox("Competition ", wc)
        ws = wsh[wsh["comp"] == w_comp]
        w_team = r.selectbox("Team", ["All teams"] + sorted(ws["team"].astype(str).unique().tolist()))
        no_pen = st.checkbox("Exclude penalties", True)
        if no_pen:
            ws = ws[~ws["penalty"].fillna(False).astype(bool)]
        if w_team != "All teams":
            ws = ws[ws["team"].astype(str) == w_team]
        fig = new_pitch(f"{w_team} - {len(ws):,} shots, {int(ws['goal'].sum())} goals, {ws['xg'].sum():.1f} xG", 600, half=True)
        for goal, nm, col, sym in ((False, "Shot", "#ffd166", "circle"), (True, "Goal", "#ff1744", "star")):
            s_ = ws[ws["goal"] == goal]
            fig.add_trace(go.Scatter(x=s_["sx"], y=s_["sy"], mode="markers", name=f"{nm} ({len(s_):,})",
                                     marker=dict(size=6 + 30 * s_["xg"], color=col, symbol=sym, opacity=0.65, line=dict(color="white", width=0.5)),
                                     text="xG " + s_["xg"].round(2).astype(str), hoverinfo="text"))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Marker size = xG (StatsBomb: provider xG; Wyscout: estimated). Wyscout locations are coarse (whole-percent grid), so treat individual shots as approximate.")

