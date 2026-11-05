import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import setup
from utils import get_data, load_actions, load_matches, load_shots

setup("Match Momentum & Tempo", "📊", "Who controlled each phase of a match, minute by minute - net threat, packing, cumulative xG and goals - plus the tempo curve across all matches.")
base, df, opt = get_data()
matches = load_matches()
mids = set(base["match_id"].unique())
A = load_actions()
sh = load_shots()
t1, t2 = st.tabs(["One match", "Tempo across matches"])


def eff_minute(d):
    """Minutes continue across halves; collapse first-half stoppage into minute 44 and second-half stoppage into 89."""
    return np.where(d["period"] == 1, np.minimum(d["minute"], 44), np.where(d["period"] == 2, np.minimum(d["minute"], 89), d["minute"]))


with t1:
    ml = matches[matches["match_id"].isin(mids)].sort_values("match_date")
    label = st.selectbox("Match", ml["label"].tolist(), key="mm_match")
    mid = ml.loc[ml["label"] == label, "match_id"].iloc[0]
    row = ml[ml["match_id"] == mid].iloc[0]
    home, away = row["home_team"], row["away_team"]
    width = st.select_slider("Window (minutes)", options=[3, 5, 10, 15], value=5, key="mm_w")
    a = A[A["match_id"] == mid].copy()
    a["team"] = a["team"].astype(str)
    a["bin"] = (eff_minute(a) // width * width).astype(int)
    pv = a.groupby(["bin", "team"]).agg(xt=("xt", "sum"), packing=("pack_cmp", "sum"), actions=("id", "size")).reset_index()
    bins = np.arange(0, int(a["bin"].max()) + width, width)
    def col(team, c):
        s = pv[pv["team"] == team].set_index("bin")[c].reindex(bins).fillna(0)
        return s.to_numpy()
    net = col(home, "xt") - col(away, "xt")
    fig = go.Figure(go.Bar(x=bins, y=net, marker_color=np.where(net >= 0, "#00d4a6", "#ff006e"), hovertext=[f"{b}-{b + width}'  net xT {v:+.2f}" for b, v in zip(bins, net)], hoverinfo="text"))
    s_ = sh[(sh["match_id"] == mid) & (sh["type"] == "Shot") & (sh["period"] < 5)].copy()
    g_ = s_[s_["is_goal"]]
    for r in g_.itertuples():
        mm = min(r.minute, 44) if r.period == 1 else min(r.minute, 89) if r.period == 2 else r.minute
        fig.add_annotation(x=mm, y=(1 if r.team == home else -1) * max(abs(net).max(), 0.1) * 1.05, text=f"⚽ {r.player.split()[-1]} {r.minute}'", showarrow=False,
                           font=dict(size=11, color="#d98f00"), yanchor="bottom" if r.team == home else "top")
    fig.add_vline(x=45, line_dash="dot", line_color="gray")
    fig.update_layout(height=430, xaxis_title="minute", yaxis_title=f"net threat (xT): {home} up, {away} down", title=f"Momentum: {home} vs {away}", bargap=0.08)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Each bar = {width} minutes. Green = {home} created more threat (xT) than {away}; pink = the reverse. Dotted line = half-time. First-half stoppage time is shown in the last first-half bar.")
    l, r = st.columns(2)
    cum = go.Figure()
    for team, colr in ((home, "#00d4a6"), (away, "#ff006e")):
        t_ = s_[s_["team"].astype(str) == team].sort_values(["period", "minute", "second"])
        x = [0] + [min(m_, 44) if p_ == 1 else min(m_, 89) if p_ == 2 else m_ for m_, p_ in zip(t_["minute"], t_["period"])] + [95]
        y = [0] + list(t_["xg"].fillna(0).cumsum()) + [t_["xg"].fillna(0).sum()]
        cum.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(shape="hv", color=colr, width=3), name=f"{team} ({t_['xg'].sum():.2f} xG, {int(t_['is_goal'].sum())} goals)"))
    cum.update_layout(height=380, xaxis_title="minute", yaxis_title="cumulative xG", title="Cumulative xG")
    l.plotly_chart(cum, use_container_width=True)
    pk = pd.DataFrame({"minute": np.r_[bins, bins], "packing": np.r_[col(home, "packing"), col(away, "packing")], "team": [home] * len(bins) + [away] * len(bins)})
    r.plotly_chart(px.bar(pk, x="minute", y="packing", color="team", barmode="group", color_discrete_sequence=["#00d4a6", "#ff006e"]).update_layout(height=380, title="Opponents bypassed per window"),
                   use_container_width=True)
    st.dataframe(pd.DataFrame({"window": [f"{b}-{b + width}'" for b in bins], f"{home} xT": col(home, "xt"), f"{away} xT": col(away, "xt"), f"{home} actions": col(home, "actions"),
                               f"{away} actions": col(away, "actions")}).round(2), hide_index=True, use_container_width=True)

with t2:
    a = A[A["match_id"].isin(mids)].copy()
    a["bin"] = (eff_minute(a) // 15 * 15).astype(int)
    a = a[a["bin"] < 90]
    n_match = a["match_id"].nunique()
    tt = a.groupby(["bin", "match_id", "team"], observed=True).agg(actions=("id", "size"), xt=("xt", "sum"), pack=("pack_cmp", "sum")).reset_index()
    cv = tt.groupby("bin").agg(actions=("actions", "mean"), a_sd=("actions", "std"), xt=("xt", "mean"), x_sd=("xt", "std"), n=("actions", "size")).reset_index()
    cv["label"] = cv["bin"].astype(str) + "-" + (cv["bin"] + 15).astype(str)
    l, r = st.columns(2)
    l.plotly_chart(px.bar(cv, x="label", y="actions", error_y=1.96 * cv["a_sd"] / np.sqrt(cv["n"]), labels={"actions": "passes + carries per team", "label": "minutes"},
                          color_discrete_sequence=["#4cc9f0"]).update_layout(height=380, title="Tempo: actions per team per 15 minutes"), use_container_width=True)
    r.plotly_chart(px.bar(cv, x="label", y="xt", error_y=1.96 * cv["x_sd"] / np.sqrt(cv["n"]), labels={"xt": "threat added (xT) per team", "label": "minutes"},
                          color_discrete_sequence=["#ffb703"]).update_layout(height=380, title="Threat: xT per team per 15 minutes"), use_container_width=True)
    st.caption(f"Averages over {n_match:,} matches (both teams), with 95% intervals. Packing is only measured where freeze frames exist, so it is not shown here; see Team Analysis.")
    g = sh[(sh["type"] == "Shot") & (sh["period"] < 5) & sh["match_id"].isin(mids)].copy()
    g["bin"] = (eff_minute(g) // 15 * 15).astype(int)
    g = g[g["bin"] < 90]
    gb = g.groupby("bin").agg(shots=("id", "size"), goals=("is_goal", "sum"), xg=("xg", "sum")).reset_index()
    gb["label"] = gb["bin"].astype(str) + "-" + (gb["bin"] + 15).astype(str)
    gb["goals_per_match"] = gb["goals"] / n_match
    st.plotly_chart(px.bar(gb, x="label", y=["goals_per_match"], labels={"label": "minutes", "value": "goals per match"}, color_discrete_sequence=["#ff006e"]).update_layout(height=340, title="When are goals scored?"),
                    use_container_width=True)
    h = a.assign(half=np.where(a["period"] == 1, "1st half", np.where(a["period"] == 2, "2nd half", "extra time"))).groupby("half").agg(actions=("id", "size"), xt=("xt", "sum"), progressive=("progressive", "sum")).reset_index()
    st.markdown("**Half by half**")
    st.dataframe(h.round(1), hide_index=True, use_container_width=True)
