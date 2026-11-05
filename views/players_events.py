import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import load_player_metrics


def render():
    st.markdown("Player-level comparison across **Wyscout 2017/18** (with ages, roles, foot, nationality), the full **StatsBomb 2015/16** league seasons, "
                "Copa America 2024 and your freeze-frame competitions. All metrics are **per 90 minutes** from event data (no player positions, so this is "
                "progression and chance creation, not packing).")
    st.info("Roles: GK / DF / MD / FW (Wyscout's role; for StatsBomb derived from the most-played position). Minutes for Wyscout are estimated from lineups and "
            "substitutions (90-minute matches). Carries exist only in StatsBomb, so progressive carries are blank for Wyscout players. "
            "Compare players within the same source and role when you can.")
    P = load_player_metrics()
    P = P[P["minutes"].fillna(0) > 0].copy()
    c = st.columns(4)
    src = c[0].multiselect("Source", ["Wyscout", "StatsBomb"], default=["Wyscout", "StatsBomb"])
    P = P[P["source"].isin(src)]
    comps = sorted(P["comp"].unique().tolist())
    sel = c[1].multiselect("Competitions", comps, default=comps)
    role = c[2].multiselect("Role", ["GK", "DF", "MD", "FW"], default=["DF", "MD", "FW"])
    min_min = c[3].slider("Minimum minutes", 90, 3000, 900, 90)
    P = P[P["comp"].isin(sel) & P["role"].isin(role) & (P["minutes"] >= min_min)].copy()
    if len(P) < 10:
        st.warning("Too few players - relax the filters."); return
    for col, name in (("passes", "passes_p90"), ("progressive_passes", "prog_passes_p90"), ("progressive_carries", "prog_carries_p90"), ("f3_pass", "f3_entries_p90"),
                      ("box_pass", "box_entries_p90"), ("shots", "shots_p90"), ("xg", "xg_p90"), ("goals", "goals_p90"), ("smart_passes", "smart_passes_p90")):
        P[name] = P[col] / P["minutes"] * 90
    P["xg_per_shot"] = P["xg"] / P["shots"].where(P["shots"] > 0)
    P["label"] = P["player"].astype(str) + " (" + P["team"].astype(str) + ", " + P["competition"].astype(str) + " " + P["season"].astype(str) + ")"
    METR = {"prog_passes_p90": "Progressive passes / 90", "prog_carries_p90": "Progressive carries / 90 (StatsBomb)", "f3_entries_p90": "Final-third entries by pass / 90",
            "box_entries_p90": "Box entries by pass / 90", "passes_p90": "Passes / 90", "pass_completion_pct": "Pass completion %", "long_pct": "Long passes %",
            "shots_p90": "Shots / 90", "xg_p90": "xG / 90", "goals_p90": "Goals / 90", "xg_per_shot": "xG per shot", "smart_passes_p90": "Smart passes / 90 (Wyscout)"}

    st.subheader("1. Leaderboard")
    l, r = st.columns([1, 2])
    m = l.selectbox("Rank by", list(METR), format_func=METR.get)
    n = l.slider("Players shown (events view)", 5, 40, 20)
    top = P.dropna(subset=[m]).sort_values(m, ascending=False).head(n)
    fig = px.bar(top.iloc[::-1], x=m, y="label", color="source", orientation="h", labels={m: METR[m], "label": ""}, hover_data=["minutes", "role"])
    r.plotly_chart(fig.update_layout(height=max(380, 24 * n)), use_container_width=True)
    show = ["player", "team", "competition", "season", "source", "role", "minutes", "age", m]
    st.dataframe(P.dropna(subset=[m]).sort_values(m, ascending=False)[show].round(2).head(200), hide_index=True, use_container_width=True)
    st.download_button("Download filtered players (CSV)", P.drop(columns=["label"]).to_csv(index=False).encode(), "players_filtered.csv")

    st.subheader("2. Relationships")
    l, r = st.columns(2)
    xm = l.selectbox("X axis", list(METR), index=0, format_func=METR.get)
    ym = r.selectbox("Y axis", list(METR), index=list(METR).index("xg_p90"), format_func=METR.get)
    d2 = P.dropna(subset=[xm, ym])
    st.plotly_chart(px.scatter(d2, x=xm, y=ym, color="role", size="minutes", hover_name="label", opacity=0.7, labels={xm: METR[xm], ym: METR[ym]})
                    .update_layout(height=560), use_container_width=True)

    st.subheader("3. Player profile and similar players")
    FE = ["prog_passes_p90", "f3_entries_p90", "box_entries_p90", "passes_p90", "pass_completion_pct", "long_pct", "shots_p90", "xg_p90"]
    pick = st.selectbox("Player", sorted(P["label"]))
    row = P[P["label"] == pick].iloc[0]
    same = st.checkbox("Compare only with players from the same source", True, help="Wyscout and StatsBomb record events differently.")
    peers = P[(P["role"] == row["role"]) & ((P["source"] == row["source"]) if same else True)]
    rk = peers[FE].rank(pct=True)
    pct = rk[peers["label"] == pick].iloc[0] * 100
    l, r = st.columns(2)
    fig = go.Figure(go.Scatterpolar(r=pct.tolist() + [pct.iloc[0]], theta=[METR[k] for k in FE] + [METR[FE[0]]], fill="toself", name=pick))
    l.plotly_chart(fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=500, title=f"Percentile vs {len(peers):,} {row['role']}s" + (f" ({row['source']})" if same else "")),
                   use_container_width=True)
    Z = peers[FE].fillna(peers[FE].mean())
    Z = (Z - Z.mean()) / Z.std().replace(0, 1)
    Zn = Z.to_numpy() / np.maximum(np.linalg.norm(Z.to_numpy(), axis=1, keepdims=True), 1e-9)
    i = int(np.where(peers["label"].to_numpy() == pick)[0][0])
    sim = peers.assign(similarity=Zn @ Zn[i]).drop(index=peers.index[i]).sort_values("similarity", ascending=False).head(12)
    r.markdown(f"**Most similar to {row['player']}** (same role" + (", same source" if same else "") + ")")
    r.dataframe(sim[["player", "team", "competition", "season", "minutes", "age", "similarity"]].round(2), hide_index=True, use_container_width=True)
    extra = [f"{x}: {row[x]}" for x in ("age", "foot", "height_cm", "nationality") if pd.notna(row.get(x))]
    if extra:
        st.caption(" | ".join(extra))

    st.subheader("4. Young talents (Wyscout, where age is known)")
    ymax = st.slider("Maximum age", 17, 25, 21)
    yt = P[(P["age"] <= ymax) & P["age"].notna()]
    if len(yt):
        ytm = st.selectbox("Sort young players by", list(METR), index=0, format_func=METR.get, key="yt")
        st.dataframe(yt.dropna(subset=[ytm]).sort_values(ytm, ascending=False)[["player", "player_full", "team", "competition", "age", "role", "foot", "nationality", "minutes", ytm]]
                     .head(30).round(2), hide_index=True, use_container_width=True)
        ab = P[P["age"].notna()].assign(age_group=lambda x: pd.cut(x["age"], [15, 21, 24, 28, 32, 45], labels=["<=21", "22-24", "25-28", "29-32", "33+"]))
        st.plotly_chart(px.box(ab, x="age_group", y="prog_passes_p90", color="role", labels={"prog_passes_p90": "progressive passes / 90", "age_group": "age"})
                        .update_layout(height=420), use_container_width=True)
    else:
        st.info("No players with a known age in this selection (ages come from Wyscout).")

