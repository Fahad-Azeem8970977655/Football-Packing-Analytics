import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import new_pitch
from ui import setup
from utils import get_data

setup("Defensive Lines", "🧱", "How high and compact is each defence - and does a high line get bypassed more? Built from the visible opponents in every freeze frame (the defending team is the opponent of the player on the ball).")
base, df, opt = get_data()
d = df[df["def_line_x"].notna()].copy()
d["opp_team"] = d["opp_team"].astype(str)
d["line_height"] = 120 - d["def_line_x"]                    # yards from the defending team's own goal
d["gap"] = d["def_line_x"] - d["mid_line_x"]                # distance between defensive and midfield line
d["def_state"] = (-d["goal_diff"]).clip(-2, 2)              # score from the DEFENDER's point of view
t = d.groupby("opp_team").agg(matches=("match_id", "nunique"), actions_faced=("id", "size"), line_height=("line_height", "mean"), line_width=("def_line_width", "mean"),
                              gap=("gap", "mean"), block_length=("opp_block_length", "mean"), packing_conceded=("pack_used", "mean"),
                              line_broken_pct=("breaks_def", lambda s: 100 * s.mean()), xt_conceded=("xt", "mean")).reset_index()
t["xt_conceded_per_100"] = 100 * t.pop("xt_conceded")
mm = st.slider("Minimum matches (as defending team)", 1, int(max(t["matches"].max(), 2)), min(4, int(t["matches"].max())), key="dl_mm")
t = t[t["matches"] >= mm]
if len(t) < 3:
    st.warning("Too few teams - lower the slider or widen the sidebar filters."); st.stop()
c = st.columns(4)
c[0].metric("Average line height", f"{t['line_height'].mean():.1f} yd", help="Distance of the defensive line from its own goal (mean of the 4 deepest visible outfield players).")
c[1].metric("Average gap to midfield line", f"{t['gap'].mean():.1f} yd")
c[2].metric("Defensive line broken", f"{t['line_broken_pct'].mean():.0f}% of actions")
c[3].metric("Teams", len(t))

st.subheader("1. Do high lines get bypassed more?")
corr = t[["line_height", "packing_conceded"]].corr().iloc[0, 1]
fig = px.scatter(t, x="line_height", y="packing_conceded", size="actions_faced", color="line_broken_pct", color_continuous_scale="RdYlGn_r", hover_name="opp_team", text="opp_team",
                 labels={"line_height": "defensive line height (yards from own goal)", "packing_conceded": "packing conceded per action", "line_broken_pct": "% line broken"})
b = np.polyfit(t["line_height"], t["packing_conceded"], 1)
xs = np.array([t["line_height"].min(), t["line_height"].max()])
fig.add_trace(go.Scatter(x=xs, y=b[0] * xs + b[1], mode="lines", name="trend", line=dict(color="gray", dash="dot")))
st.plotly_chart(fig.update_traces(textposition="top center", textfont_size=9, selector=dict(mode="markers+text")).update_layout(height=580), use_container_width=True)
st.caption(f"Correlation between line height and packing conceded across teams: {corr:+.2f}. Higher lines leave more space behind, but they also meet attackers further from goal.")

st.subheader("2. One defence in detail")
team = st.selectbox("Defending team", sorted(t["opp_team"]), key="dl_team")
x = d[d["opp_team"] == team]
fig = new_pitch(f"{team} defending - average shape (attackers move left to right)", 600)
for col, nm, colr in (("def_line_x", "Defensive line", "#ff4d6d"), ("mid_line_x", "Midfield line", "#ffb703")):
    v = x[col].dropna()
    if len(v):
        m_, s_ = float(v.mean()), float(v.std())
        fig.add_shape(type="rect", x0=m_ - s_ / 2, x1=m_ + s_ / 2, y0=0, y1=80, fillcolor=colr, opacity=0.18, line_width=0)
        fig.add_shape(type="line", x0=m_, x1=m_, y0=0, y1=80, line=dict(color=colr, width=3, dash="dash"))
        fig.add_annotation(x=m_, y=-2.3, text=f"{nm}: {120 - m_:.0f} yd from own goal", showarrow=False, font=dict(color=colr, size=11))
fig.add_annotation(x=112, y=40, text="their goal", showarrow=False, font=dict(color="white", size=12), textangle=90)
st.plotly_chart(fig, use_container_width=True)
st.caption("Dashed line = average position; shaded band = one standard deviation (how much the line moves). Everything is measured from the visible players only.")

a_, b_ = st.columns(2)
bins = np.arange(0, 121, 10)
x = x.assign(ball_bin=pd.cut(x["sx"], bins))
allx = d.assign(ball_bin=pd.cut(d["sx"], bins))
lt = x.groupby("ball_bin", observed=True)["def_line_x"].mean().rename(team)
la = allx.groupby("ball_bin", observed=True)["def_line_x"].mean().rename("all teams")
lb = pd.concat([lt, la], axis=1).reset_index()
lb["ball_x"] = lb["ball_bin"].map(lambda i: i.mid)
a_.plotly_chart(px.line(lb.melt(id_vars=["ball_bin", "ball_x"], var_name="who", value_name="line_x"), x="ball_x", y="line_x", color="who", markers=True,
                        labels={"ball_x": "ball position (x of the action)", "line_x": "defensive line x"}).update_layout(height=400, title="How the line follows the ball"), use_container_width=True)
st_ = x.groupby("def_state").agg(line_height=("line_height", "mean"), n=("id", "size"), sd=("line_height", "std")).reset_index()
st_["state"] = st_["def_state"].map({-2: "Trailing 2+", -1: "Trailing 1", 0: "Level", 1: "Leading 1", 2: "Leading 2+"})
b_.plotly_chart(px.bar(st_, x="state", y="line_height", text=st_["line_height"].round(1), error_y=1.96 * st_["sd"] / np.sqrt(st_["n"]),
                       category_orders={"state": ["Trailing 2+", "Trailing 1", "Level", "Leading 1", "Leading 2+"]}, labels={"line_height": "line height (yd)", "state": ""},
                       color_discrete_sequence=["#00d4a6"]).update_layout(height=400, title="Line height by score (defender's view)"), use_container_width=True)

st.subheader("3. Where is the line broken?")
nx_, ny_ = 12, 8
xx = x[x["sx"].between(0, 120) & x["sy"].between(0, 80)]
xi = np.clip((xx["sx"] // (120 / nx_)).astype(int), 0, nx_ - 1)
yi = np.clip((xx["sy"] // (80 / ny_)).astype(int), 0, ny_ - 1)
gr = xx.assign(xi=xi, yi=yi).groupby(["yi", "xi"]).agg(r=("breaks_def", "mean"), n=("id", "size")).reset_index()
z = np.full((ny_, nx_), np.nan)
for r in gr.itertuples():
    if r.n >= 20:
        z[r.yi, r.xi] = 100 * r.r
fig = new_pitch(f"% of actions against {team} that break the defensive line, by where the action starts", 560)
fig.add_trace(go.Heatmap(x=(np.arange(nx_) + 0.5) * 120 / nx_, y=(np.arange(ny_) + 0.5) * 80 / ny_, z=z, opacity=0.8, colorscale="YlOrRd", colorbar=dict(title="% broken"),
                         hovertemplate="%{z:.0f}% broken<extra></extra>"))
st.plotly_chart(fig, use_container_width=True)
st.subheader("4. Team table")
st.dataframe(t.sort_values("line_height", ascending=False).round(2), hide_index=True, use_container_width=True)
st.info("Only players visible to the 360 camera are in each frame, so 'line height' is the mean of the four deepest visible outfield players, not always the true back four.")
