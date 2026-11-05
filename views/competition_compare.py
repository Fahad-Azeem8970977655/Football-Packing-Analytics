import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import new_pitch
from utils import agg_players, get_data, load_shots


def render():
    st.markdown("Compare tournaments and leagues side by side: how much they bypass, how directly they play, how many chances they create, "
                "and where on the pitch the packing starts. Use the sidebar to choose which competitions are included.")
    base, df, opt = get_data()
    sh = load_shots()

    cs_map = (base[["match_id", "competition", "season"]].drop_duplicates("match_id")
              .assign(cs=lambda x: x["competition"].astype(str) + " " + x["season"].astype(str)).set_index("match_id")["cs"])
    d = agg_players(df)
    d["cs"] = d["match_id"].map(cs_map)
    d["team"] = d["team"].astype(str)
    gender = lambda s: "Women" if "Women" in s else "Men"

    tm = d.groupby(["cs", "match_id", "team"]).agg(pack=("pack_used", "sum"), actions=("id", "size"), lb=("lb", "sum"), prog=("prog", "sum"),
                                                  carries=("is_carry", "sum")).reset_index()
    s_ = sh[(sh["type"] == "Shot") & (sh["shot_type"] != "Penalty") & sh["match_id"].isin(tm["match_id"].unique())]
    s_ = s_.groupby(["match_id", "team"]).agg(xg=("xg", "sum"), shots=("id", "size"), goals=("is_goal", "sum")).reset_index()
    tm = tm.merge(s_, on=["match_id", "team"], how="left").fillna({"xg": 0, "shots": 0, "goals": 0})

    cov = base.assign(cs=base["match_id"].map(cs_map)).groupby("cs").agg(matches_total=("match_id", "nunique"), frame_coverage=("has_frame", "mean"))
    plen = d[d["action_type"] == "Pass"].groupby("cs")["length"].mean()
    t = tm.groupby("cs").agg(matches=("match_id", "nunique"), pack_per_team_match=("pack", "mean"), actions=("actions", "sum"), lb=("lb", "sum"),
                             carries=("carries", "sum"), pack=("pack", "sum"), prog_per_team_match=("prog", "mean"),
                             xg_per_team_match=("xg", "mean"), shots_per_team_match=("shots", "mean"), goals_per_team_match=("goals", "mean"))
    t["pack_per_action"] = t["pack"] / t["actions"]
    t["line_breaking_pct"] = 100 * t["lb"] / t["actions"]
    t["carry_share_pct"] = 100 * t["carries"] / t["actions"]
    t["avg_pass_length_m"] = plen
    t = t.join(cov["frame_coverage"].mul(100).rename("frame_coverage_pct")).drop(columns=["actions", "lb", "carries", "pack"]).reset_index()
    t["group"] = t["cs"].map(gender)
    min_m = st.slider("Minimum matches with freeze-frame data per competition", 1, 40, 5)
    excluded = t[(t["matches"] < min_m) | (t["frame_coverage_pct"] < 20)]
    if len(excluded):
        st.warning("Left out (too few matches with freeze frames): " + "; ".join(
            f"{r.cs} ({r.matches} matches, {r.frame_coverage_pct:.0f}% coverage)" for r in excluded.itertuples()))
    t = t[~t["cs"].isin(excluded["cs"])]
    tm = tm[tm["cs"].isin(t["cs"])]
    d = d[d["cs"].isin(t["cs"])]
    if t.empty:
        st.error("No competition has enough freeze-frame data - lower the slider."); return

    st.subheader("1. Overview table")
    st.dataframe(t.round(2).sort_values("pack_per_team_match", ascending=False), hide_index=True, use_container_width=True)
    st.caption("'Per team match' = one team's performance in one match. Packing counts only actions with a freeze frame, so competitions with low "
               "coverage (see the last column) are understated. Men's and women's competitions are labelled in 'group'.")

    st.subheader("2. Distribution of packing per team match")
    order = t.sort_values("pack_per_team_match")["cs"].tolist()
    fig = px.box(tm, x="pack", y="cs", color=tm["cs"].map(gender), category_orders={"cs": order}, orientation="h",
                 labels={"pack": "packing in one team's match", "cs": "", "color": "group"})
    st.plotly_chart(fig.update_layout(height=max(380, 38 * len(order))), use_container_width=True)

    st.subheader("3. Style profile (radar)")
    metrics = {"pack_per_team_match": "Packing / match", "pack_per_action": "Packing / action", "line_breaking_pct": "Line-breaking %",
               "carry_share_pct": "Carry share %", "avg_pass_length_m": "Pass length", "prog_per_team_match": "Progressive / match",
               "xg_per_team_match": "xG / match", "shots_per_team_match": "Shots / match"}
    pick = st.multiselect("Competitions to compare (2-4)", t["cs"].tolist(), default=t.sort_values("matches", ascending=False)["cs"].tolist()[:3], max_selections=4)
    if len(pick) >= 2:
        norm = (t.set_index("cs")[list(metrics)] - t.set_index("cs")[list(metrics)].min()) / (t.set_index("cs")[list(metrics)].max() - t.set_index("cs")[list(metrics)].min()).replace(0, np.nan)
        rad = go.Figure()
        for c in pick:
            v = norm.loc[c].fillna(0.5).tolist()
            rad.add_trace(go.Scatterpolar(r=v + v[:1], theta=list(metrics.values()) + [list(metrics.values())[0]], fill="toself", name=c, opacity=0.55))
        rad.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), height=540,
                          title="Each axis scaled 0-1 across all competitions in the table (1 = highest)")
        st.plotly_chart(rad, use_container_width=True)
    else:
        st.info("Pick at least two competitions.")

    st.subheader("4. Where does packing start? (head to head)")
    if len(pick) >= 2:
        l, r = st.columns(2)
        A = l.selectbox("Competition A", pick, index=0)
        B = r.selectbox("Competition B", pick, index=1)
        nx_, ny_ = 12, 8

        def grid(c):
            x = d[d["cs"] == c]
            x = x[x["sx"].between(0, 120) & x["sy"].between(0, 80)]
            xi = np.clip((x["sx"] // (120 / nx_)).astype(int), 0, nx_ - 1)
            yi = np.clip((x["sy"] // (80 / ny_)).astype(int), 0, ny_ - 1)
            z, n = np.full((ny_, nx_), np.nan), np.zeros((ny_, nx_))
            for rr in x.assign(xi=xi, yi=yi).groupby(["yi", "xi"]).agg(m=("pack_used", "mean"), n=("id", "size")).reset_index().itertuples():
                n[rr.yi, rr.xi] = rr.n
                if rr.n >= 30:
                    z[rr.yi, rr.xi] = rr.m
            return z, n
        zA, nA = grid(A)
        zB, nB = grid(B)
        diff = zA - zB
        xc, yc = (np.arange(nx_) + 0.5) * 120 / nx_, (np.arange(ny_) + 0.5) * 80 / ny_
        fig = new_pitch(f"Average packing by start zone: {A} minus {B}", 560)
        lim = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1
        fig.add_trace(go.Heatmap(x=xc, y=yc, z=diff, zmid=0, zmin=-lim, zmax=lim, colorscale="RdBu", opacity=0.8,
                                 hovertemplate="difference %{z:.2f}<extra></extra>", colorbar=dict(title="A - B")))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Blue = A bypasses more opponents from that zone; red = B does. Zones with fewer than 30 actions in either competition are blank.")

    st.subheader("5. Does packing turn into chances? (by competition)")
    rows = []
    for c, x in tm.groupby("cs"):
        if len(x) >= 20 and x["pack"].std() > 0:
            slope = np.polyfit(x["pack"], x["xg"], 1)[0]
            rows.append({"competition": c, "team_matches": len(x), "correlation": x[["pack", "xg"]].corr().iloc[0, 1], "xG_per_+100_packing": 100 * slope})
    if rows:
        cr = pd.DataFrame(rows).round(3)
        l, r = st.columns([2, 3])
        l.dataframe(cr.sort_values("xG_per_+100_packing", ascending=False), hide_index=True, use_container_width=True)
        r.plotly_chart(px.scatter(tm, x="pack", y="xg", color="cs", opacity=0.55, labels={"pack": "packing in the team's match", "xg": "open-play xG in the match"})
                       .update_layout(height=460), use_container_width=True)
    st.info("Competitions differ in the opposition and match context, and club samples (Bundesliga, La Liga, Ligue 1) are single clubs, "
            "so compare tournaments with tournaments for the cleanest reading.")

