import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import setup
from utils import load_league_xpoints

setup("Expected League Tables", "🏟️", "What the table would look like if chances were converted at their expected rate - plus a season simulator that gives every team's title, top-4 and relegation odds.")
X = load_league_xpoints()
X["cs"] = X["competition"].astype(str) + " " + X["season"].astype(str) + " [" + X["source"] + "]"
sel = st.selectbox("League season", sorted(X["cs"].unique()), key="el_sel")
x = X[X["cs"] == sel].copy().reset_index(drop=True)
x["date"] = pd.to_datetime(x["date"], errors="coerce")
hs, as_ = x["home_score"].astype(int), x["away_score"].astype(int)
H = pd.DataFrame({"team": x["home_team"], "opp": x["away_team"], "gf": hs, "ga": as_, "xgf": x["xg_home"], "xga": x["xg_away"], "pts": np.where(hs > as_, 3, np.where(hs == as_, 1, 0)),
                  "xpts": 3 * x["p_home"] + x["p_draw"], "date": x["date"], "venue": "home", "mi": x.index})
A = pd.DataFrame({"team": x["away_team"], "opp": x["home_team"], "gf": as_, "ga": hs, "xgf": x["xg_away"], "xga": x["xg_home"], "pts": np.where(as_ > hs, 3, np.where(hs == as_, 1, 0)),
                  "xpts": 3 * x["p_away"] + x["p_draw"], "date": x["date"], "venue": "away", "mi": x.index})
TL = pd.concat([H, A], ignore_index=True)
t = TL.groupby("team").agg(played=("pts", "size"), points=("pts", "sum"), xpoints=("xpts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"), xgf=("xgf", "sum"), xga=("xga", "sum")).reset_index()
t["gd"], t["xgd"] = t["gf"] - t["ga"], t["xgf"] - t["xga"]
t["luck"] = t["points"] - t["xpoints"]
t = t.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True); t["rank"] = np.arange(1, len(t) + 1)
t = t.sort_values(["xpoints", "xgd"], ascending=False).reset_index(drop=True); t["xrank"] = np.arange(1, len(t) + 1)
t["places_above_expected"] = t["xrank"] - t["rank"]
t = t.sort_values("rank").reset_index(drop=True)
src = x["source"].iloc[0]
c = st.columns(4)
c[0].metric("Matches", len(x))
c[1].metric("Teams", len(t))
c[2].metric("Goals vs xG", f"{int((hs + as_).sum()):,} vs {(x['xg_home'] + x['xg_away']).sum():,.0f}")
c[3].metric("Luckiest team", t.sort_values("luck").iloc[-1]["team"], f"{t['luck'].max():+.1f} pts")
if src == "Wyscout":
    st.info("Wyscout xG is estimated from shot location (a model trained on these shots) and has no penalties, so every team's xPoints is a little low; compare teams with each other.")
if len(x) < len(t) * (len(t) - 1):
    st.info(f"Only {len(x)} of {len(t) * (len(t) - 1)} league matches are in the data, so points and tables are for the available matches.")
t1, t2, t3, t4 = st.tabs(["Actual vs expected table", "Luck & over-performance", "Season simulator", "Race over time"])

with t1:
    show = t[["rank", "team", "played", "points", "xpoints", "luck", "xrank", "places_above_expected", "gf", "ga", "gd", "xgf", "xga", "xgd"]].rename(columns={"rank": "pos", "xrank": "xPos", "luck": "pts - xPts"})
    st.dataframe(show.round(1), hide_index=True, use_container_width=True, height=min(900, 38 * len(show) + 40))
    st.caption("xPoints: each match is replayed from the exact xG of every shot (probability of each score line), then converted to expected points. 'places_above_expected' = expected position minus actual position.")

with t2:
    tt = t.sort_values("luck")
    l, r = st.columns(2)
    l.plotly_chart(px.bar(tt, x="luck", y="team", orientation="h", color="luck", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, labels={"luck": "points minus expected points", "team": ""})
                   .update_layout(height=max(420, 26 * len(tt)), coloraxis_showscale=False, title="Who got lucky (right) and unlucky (left)"), use_container_width=True)
    fig = px.scatter(t, x="xpoints", y="points", text="team", hover_name="team", color="luck", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, labels={"xpoints": "expected points", "points": "actual points"})
    mx = float(max(t["xpoints"].max(), t["points"].max()))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dot", color="gray"), name="points = xPoints"))
    r.plotly_chart(fig.update_traces(textposition="top center", textfont_size=8, selector=dict(mode="markers+text")).update_layout(height=max(420, 26 * len(tt)), title="Points vs expected points"), use_container_width=True)
    t["finishing"] = t["gf"] - t["xgf"]
    t["defending"] = t["xga"] - t["ga"]
    st.plotly_chart(px.scatter(t, x="finishing", y="defending", text="team", hover_name="team", labels={"finishing": "goals scored minus xG (finishing / luck at the front)", "defending": "xG conceded minus goals conceded (keeping / luck at the back)"})
                    .add_hline(y=0, line_dash="dot", line_color="gray").add_vline(x=0, line_dash="dot", line_color="gray")
                    .update_traces(textposition="top center", textfont_size=8).update_layout(height=520, title="Where did the luck come from?"), use_container_width=True)

with t3:
    n_sims = st.select_slider("Number of simulated seasons", options=[1000, 2000, 5000, 10000, 20000], value=5000, key="el_sims")
    teams = sorted(set(x["home_team"]) | set(x["away_team"]))
    idx = {tm: i for i, tm in enumerate(teams)}
    hi, ai = x["home_team"].map(idx).to_numpy(), x["away_team"].map(idx).to_numpy()

    @st.cache_data(show_spinner="Simulating seasons...")
    def simulate(ph, pdw, hi_, ai_, nt, ns):
        rng = np.random.default_rng(0)
        u = rng.random((ns, len(ph)))
        hp = np.where(u < ph, 3, np.where(u < ph + pdw, 1, 0)).astype(np.int16)
        ap = np.where(u < ph, 0, np.where(u < ph + pdw, 1, 3)).astype(np.int16)
        pts = np.zeros((ns, nt))
        for k in range(nt):
            pts[:, k] = hp[:, hi_ == k].sum(1) + ap[:, ai_ == k].sum(1)
        order = np.argsort(-(pts + rng.random(pts.shape) * 0.01), axis=1)
        counts = np.zeros((nt, nt))
        np.add.at(counts, (order.ravel(), np.tile(np.arange(nt), ns)), 1)
        return counts / ns, pts.mean(0), pts.std(0)

    prob, mp, sd = simulate(x["p_home"].to_numpy(), x["p_draw"].to_numpy(), hi, ai, len(teams), n_sims)
    nt = len(teams)
    res = pd.DataFrame({"team": teams, "expected_points": mp, "sd": sd, "title_%": 100 * prob[:, 0], "top4_%": 100 * prob[:, :4].sum(1), "relegation_%": 100 * prob[:, -3:].sum(1)})
    res = res.merge(t[["team", "points", "rank"]], on="team").sort_values("expected_points", ascending=False)
    order_t = res["team"].tolist()
    pm = pd.DataFrame(prob * 100, index=teams, columns=[str(i + 1) for i in range(nt)]).loc[order_t]
    st.plotly_chart(px.imshow(pm, text_auto=".0f", aspect="auto", color_continuous_scale="Viridis", labels=dict(x="final position", y="", color="% of simulations"))
                    .update_layout(height=max(500, 28 * nt), title="Probability of every final position (replaying the season from the xG of each shot)"), use_container_width=True)
    st.dataframe(res.round(1).rename(columns={"points": "actual_points", "rank": "actual_pos"}), hide_index=True, use_container_width=True)
    st.caption("Each simulated season draws every match result from its xG-based probabilities. Positions are by points (ties broken at random), so title/relegation odds ignore goal difference. "
               "A team whose actual points sit far outside its simulated range was unusually lucky or unlucky.")

with t4:
    pick = st.multiselect("Teams", t["team"].tolist(), default=t.sort_values("xpoints", ascending=False)["team"].tolist()[:4], key="el_teams")
    R = TL[TL["team"].isin(pick)].sort_values(["team", "date", "mi"]).copy()
    R["match_no"] = R.groupby("team").cumcount() + 1
    R["cum_luck"] = R.groupby("team").apply(lambda g: (g["pts"] - g["xpts"]).cumsum()).reset_index(level=0, drop=True)
    R["cum_pts"] = R.groupby("team")["pts"].cumsum()
    R["cum_xpts"] = R.groupby("team")["xpts"].cumsum()
    l, r = st.columns(2)
    l.plotly_chart(px.line(R, x="match_no", y="cum_luck", color="team", labels={"match_no": "matches played", "cum_luck": "cumulative points minus xPoints"}).update_layout(height=440, title="Luck builds up over the season"), use_container_width=True)
    r.plotly_chart(px.line(R.melt(id_vars=["team", "match_no"], value_vars=["cum_pts", "cum_xpts"]), x="match_no", y="value", color="team", line_dash="variable",
                           labels={"match_no": "matches played", "value": "points"}).update_layout(height=440, title="Actual (solid) vs expected (dashed) points"), use_container_width=True)
