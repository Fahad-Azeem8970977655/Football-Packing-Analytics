import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ui import setup
from utils import agg_players, get_data, load_game_states

setup('Game State', '⏱️', 'How packing changes with the scoreline - measured per 90 minutes spent trailing, level or leading.')
st.markdown("Do teams bypass more opponents when **trailing** than when **leading**? Every action is tagged with the score "
            "at that moment, and packing is expressed **per 90 minutes spent in that state** (so long spells of leading do not distort it).")
base, df, opt = get_data()
gs = load_game_states()

LAB = {-2: "Trailing by 2+", -1: "Trailing by 1", 0: "Level", 1: "Leading by 1", 2: "Leading by 2+"}
d = agg_players(df).assign(state=lambda x: x["goal_diff"].clip(-2, 2).astype(int), team=lambda x: x["team"].astype(str))
d["press"] = d["under_pressure"].astype(float)
d["pass_done"] = d["completed"].astype(float).where(d["action_type"] == "Pass")
d["pass_len"] = d["length"].where(d["action_type"] == "Pass")
g2 = gs.rename(columns={"gd_state": "state"})
g2 = g2[g2["match_id"].isin(d["match_id"].unique())]
pk = d.groupby(["match_id", "team", "state"]).agg(pack=("pack_used", "sum"), actions=("id", "size"), carries=("is_carry", "sum"),
                                                   lb=("lb", "sum"), prog=("prog", "sum"), press=("press", "sum")).reset_index()
tab = g2.merge(pk, on=["match_id", "team", "state"], how="left").fillna({"pack": 0, "actions": 0, "carries": 0, "lb": 0, "prog": 0, "press": 0})
tab["minutes"] = tab["seconds"] / 60

st.subheader("1. Packing and activity per 90 minutes in each game state")
s = tab.groupby("state").agg(minutes=("minutes", "sum"), pack=("pack", "sum"), actions=("actions", "sum"), carries=("carries", "sum"),
                              lb=("lb", "sum"), prog=("prog", "sum"), press=("press", "sum")).reset_index()
s = s[s["minutes"] > 0]
s["Game state"] = s["state"].map(LAB)
s["Packing per 90"] = s["pack"] / s["minutes"] * 90
s["Actions per 90"] = s["actions"] / s["minutes"] * 90
s["Packing per action"] = s["pack"] / s["actions"]
s["Line-breaking %"] = 100 * s["lb"] / s["actions"]
s["Carry share %"] = 100 * s["carries"] / s["actions"]
s["Progressive per 90"] = s["prog"] / s["minutes"] * 90
s["Under pressure %"] = 100 * s["press"] / s["actions"]
k = st.columns(len(s))
for col, r in zip(k, s.itertuples()):  # one metric per game state
    col.metric(LAB[r.state], f"{s.loc[r.Index, 'Packing per 90']:.0f}", help=f"{r.minutes:,.0f} team-minutes in this state")
l, r = st.columns(2)
l.plotly_chart(px.bar(s, x="Game state", y="Packing per 90", color="state", color_continuous_scale="RdBu", text_auto=".0f",
                      category_orders={"Game state": [LAB[i] for i in sorted(LAB)]}).update_layout(height=380, coloraxis_showscale=False),
               use_container_width=True)
r.plotly_chart(px.bar(s, x="Game state", y="Packing per action", color="state", color_continuous_scale="RdBu", text_auto=".2f",
                      category_orders={"Game state": [LAB[i] for i in sorted(LAB)]}).update_layout(height=380, coloraxis_showscale=False),
               use_container_width=True)
st.caption("Left: volume of bypassing per 90 minutes. Right: how much each action bypasses (quality / directness). States with little time have noisier values.")

st.subheader("2. How does the style change?")
cols = ["Game state", "minutes", "Actions per 90", "Line-breaking %", "Carry share %", "Progressive per 90", "Under pressure %"]
st.dataframe(s[cols].round(1), hide_index=True, use_container_width=True)
long = s.melt(id_vars="Game state", value_vars=["Line-breaking %", "Carry share %", "Under pressure %"])
st.plotly_chart(px.line(long, x="Game state", y="value", color="variable", markers=True,
                        category_orders={"Game state": [LAB[i] for i in sorted(LAB)]}).update_layout(height=360), use_container_width=True)

st.subheader("3. Which teams raise their packing when trailing?")
tt = tab.assign(side=np.where(tab["state"] < 0, "trailing", np.where(tab["state"] > 0, "leading", "level")))
tt = tt.groupby(["team", "side"]).agg(minutes=("minutes", "sum"), pack=("pack", "sum")).reset_index()
tt["p90"] = tt["pack"] / tt["minutes"] * 90
w = tt.pivot(index="team", columns="side", values="p90")
mn = tt.pivot(index="team", columns="side", values="minutes")
min_m = st.slider("Minimum minutes trailing AND leading", 30, 400, 120, 10)
ok = [t for t in w.index if all(c in mn.columns for c in ("trailing", "leading")) and mn.loc[t, "trailing"] >= min_m and mn.loc[t, "leading"] >= min_m]
if len(ok) >= 3:
    res = pd.DataFrame({"team": ok, "trailing": w.loc[ok, "trailing"].values, "leading": w.loc[ok, "leading"].values,
                        "min_trailing": mn.loc[ok, "trailing"].values, "min_leading": mn.loc[ok, "leading"].values})
    res["change_when_trailing"] = res["trailing"] - res["leading"]
    fig = px.scatter(res, x="leading", y="trailing", size="min_trailing", color="change_when_trailing", hover_name="team", text="team",
                     color_continuous_scale="RdBu_r", labels={"leading": "packing per 90 when leading", "trailing": "packing per 90 when trailing"})
    m_ = float(max(res["leading"].max(), res["trailing"].max()))
    fig.add_shape(type="line", x0=0, y0=0, x1=m_, y1=m_, line=dict(dash="dot"))
    st.plotly_chart(fig.update_traces(textposition="top center", textfont_size=9).update_layout(height=560), use_container_width=True)
    st.caption("Above the dotted line = the team bypasses more opponents per 90 when trailing than when leading.")
    st.dataframe(res.sort_values("change_when_trailing", ascending=False).round(1), hide_index=True, use_container_width=True)
else:
    st.info("Not enough teams with that many minutes in both states - lower the slider or widen the sidebar filters.")
st.info("Game-state effects mix several things: weaker teams trail more often, and teams chase games differently. Treat team comparisons as descriptive, "
        "and be careful with the 'by 2+' states, which have the least time.")
