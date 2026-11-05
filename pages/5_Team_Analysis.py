import plotly.express as px
import streamlit as st

from ui import setup
from utils import agg_players, get_data

setup('Team Analysis', '🛡️', 'Packing created vs conceded per match, and how each team’s packing builds through the match.')
base, df, opt = get_data()

d = agg_players(df)
mt = d.groupby(["match_id", "team"], observed=True).agg(pack=("pack_used", "sum"), lb=("lb", "sum"),
                                                        pack_pass=("pack_pass", "sum"), pack_carry=("pack_carry", "sum"),
                                                        actions=("id", "size")).reset_index()
mt["team"] = mt["team"].astype(str)
mt["pack_against"] = mt.groupby("match_id")["pack"].transform("sum") - mt["pack"]
mt["n_teams"] = mt.groupby("match_id")["team"].transform("nunique")
mt = mt[mt["n_teams"] == 2]          # need both teams to measure packing conceded
if mt.empty:
    st.warning("No matches with both teams' data."); st.stop()
t = mt.groupby("team").agg(matches=("match_id", "nunique"), pack_for=("pack", "mean"), pack_against=("pack_against", "mean"),
                           lb_actions=("lb", "mean"), pack_pass=("pack_pass", "mean"), pack_carry=("pack_carry", "mean")).reset_index()
t["net"] = t["pack_for"] - t["pack_against"]
mm = st.slider("Minimum matches", 1, int(t["matches"].max()), min(5, int(t["matches"].max())))
t = t[t["matches"] >= mm].round(2)

st.subheader("Packing created vs conceded (per match)")
fig = px.scatter(t, x="pack_for", y="pack_against", size="matches", hover_name="team", text="team", color="net",
                 color_continuous_scale="RdYlGn")
fig.update_traces(textposition="top center", textfont_size=9)
fig.update_yaxes(autorange="reversed", title="Packing conceded per match (lower = better)")
fig.update_xaxes(title="Packing created per match")
st.plotly_chart(fig.update_layout(height=600), use_container_width=True)

l, r = st.columns(2)
with l:
    st.subheader("How teams create packing")
    top = t.sort_values("pack_for", ascending=False).head(20)
    st.plotly_chart(px.bar(top.melt(id_vars="team", value_vars=["pack_pass", "pack_carry"]), y="team", x="value", color="variable",
                           orientation="h").update_layout(height=520, yaxis=dict(autorange="reversed")), use_container_width=True)
with r:
    st.subheader("Team table")
    st.dataframe(t.sort_values("net", ascending=False), hide_index=True, use_container_width=True, height=520)

st.subheader("Packing through the match")
team = st.selectbox("Team", sorted(t["team"]))
x = d[d["team"].astype(str) == team].copy()
x["window"] = (x["minute"] // 15 * 15).clip(upper=90)
n_m = x["match_id"].nunique()
w = (x.groupby("window")["pack_used"].sum() / n_m).reset_index()
w["window"] = w["window"].astype(int).astype(str) + "-" + (w["window"] + 15).astype(int).astype(str)
st.plotly_chart(px.bar(w, x="window", y="pack_used", labels={"pack_used": "packing per match"},
                       color_discrete_sequence=["#e76f51"]).update_layout(height=350), use_container_width=True)
