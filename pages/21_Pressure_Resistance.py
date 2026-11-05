import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models import get_xpass
from pitch import new_pitch
from ui import setup
from utils import get_data

setup("Pressure Resistance", "🧯", "How does the nearest defender change passing - and who keeps completing passes under pressure, beyond what the xPass model expects?")
xp, stats = get_xpass()
base, df, opt = get_data()
P = df[(df["action_type"] == "Pass") & df["has_frame"]].copy()
P["xp"] = xp.reindex(P.index)
P = P.dropna(subset=["xp", "min_opp_dist_start"])
P["done"] = P["completed"].astype(int)
P["player"], P["team"] = P["player"].astype(str), P["team"].astype(str)
if P.empty:
    st.warning("No scored passes for these filters."); st.stop()
P["dist_bin"] = pd.cut(P["min_opp_dist_start"], [0, 2, 4, 6, 9, 500], labels=["<2 yd", "2-4", "4-6", "6-9", "9+"])
c = st.columns(4)
c[0].metric("Passes", f"{len(P):,}")
c[1].metric("Nearest defender < 3 yd", f"{(P['min_opp_dist_start'] < 3).mean():.0%}")
c[2].metric("Completion if < 3 yd", f"{P.loc[P['min_opp_dist_start'] < 3, 'done'].mean():.1%}")
c[3].metric("Completion if 9+ yd", f"{P.loc[P['min_opp_dist_start'] >= 9, 'done'].mean():.1%}")

st.subheader("1. What pressure does to passing")
b = P.groupby("dist_bin", observed=True).agg(passes=("id", "size"), actual=("done", "mean"), expected=("xp", "mean"), packing=("pack_used", "mean")).reset_index()
b[["actual", "expected"]] = 100 * b[["actual", "expected"]]
l, r = st.columns(2)
l.plotly_chart(px.bar(b.melt(id_vars="dist_bin", value_vars=["actual", "expected"]), x="dist_bin", y="value", color="variable", barmode="group",
                      labels={"dist_bin": "distance of the nearest defender to the passer", "value": "completion %"}).update_layout(height=380, title="Completion: actual vs model expectation"), use_container_width=True)
r.plotly_chart(px.bar(b, x="dist_bin", y="packing", text=b["passes"].map("{:,}".format), labels={"dist_bin": "distance of the nearest defender", "packing": "average packing"},
                      color_discrete_sequence=["#ff006e"]).update_layout(height=380, title="Packing per pass"), use_container_width=True)
P["lane_bin"] = P["n_opp_lane"].clip(upper=3).astype(int).astype(str).replace({"3": "3+"})
hm = P.groupby(["dist_bin", "lane_bin"], observed=True)["done"].mean().unstack() * 100
st.plotly_chart(px.imshow(hm, text_auto=".0f", aspect="auto", color_continuous_scale="RdYlGn", labels=dict(x="opponents in the passing lane", y="nearest defender to the passer", color="completion %"))
                .update_layout(height=360, title="Completion % by pressure and lane congestion"), use_container_width=True)

st.subheader("2. Where is the passer under most pressure?")
nx_, ny_ = 12, 8
Z = P[P["sx"].between(0, 120) & P["sy"].between(0, 80)]
xi = np.clip((Z["sx"] // (120 / nx_)).astype(int), 0, nx_ - 1)
yi = np.clip((Z["sy"] // (80 / ny_)).astype(int), 0, ny_ - 1)
gz = Z.assign(xi=xi, yi=yi).groupby(["yi", "xi"]).agg(d=("min_opp_dist_start", "mean"), n=("id", "size")).reset_index()
z = np.full((ny_, nx_), np.nan)
for r in gz.itertuples():
    if r.n >= 40:
        z[r.yi, r.xi] = r.d
fig = new_pitch("Average distance of the nearest defender when the pass starts (lower = tighter pressure)", 600)
fig.add_trace(go.Heatmap(x=(np.arange(nx_) + 0.5) * 120 / nx_, y=(np.arange(ny_) + 0.5) * 80 / ny_, z=z, opacity=0.8, colorscale="RdYlGn", colorbar=dict(title="yards"),
                         hovertemplate="%{z:.1f} yd<extra></extra>"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Who beats the model under pressure?")
thr = st.slider("Count a pass as 'pressured' if the nearest defender is closer than (yards)", 1.0, 6.0, 3.0, 0.5, key="pr_thr")
mn = st.slider("Minimum pressured passes", 30, 300, 80, 10, key="pr_min")
Q = P[P["min_opp_dist_start"] < thr]
pl = Q.groupby(["player", "team"]).agg(pressured=("id", "size"), completed=("done", "sum"), expected=("xp", "sum"), packing=("pack_used", "mean")).reset_index()
tot = P.groupby(["player", "team"]).size().rename("all_passes").reset_index()
pl = pl.merge(tot, on=["player", "team"])
pl = pl[pl["pressured"] >= mn].copy()
if pl.empty:
    st.info("No player reaches that many pressured passes - lower the minimum.")
else:
    pl["pressured_share_%"] = 100 * pl["pressured"] / pl["all_passes"]
    pl["completion_%"] = 100 * pl["completed"] / pl["pressured"]
    pl["over_expected_per_100"] = 100 * (pl["completed"] - pl["expected"]) / pl["pressured"]
    st.plotly_chart(px.scatter(pl, x="pressured_share_%", y="over_expected_per_100", size="pressured", color="packing", color_continuous_scale="Magma", hover_name="player", hover_data=["team", "completion_%"],
                               labels={"pressured_share_%": "share of his passes made under pressure %", "over_expected_per_100": "completion over expected per 100 pressured passes"})
                    .add_hline(y=0, line_dash="dot", line_color="gray").update_layout(height=560), use_container_width=True)
    cols = ["player", "team", "pressured", "pressured_share_%", "completion_%", "over_expected_per_100", "packing"]
    a_, b_ = st.columns(2)
    a_.markdown("**Best under pressure**")
    a_.dataframe(pl.sort_values("over_expected_per_100", ascending=False)[cols].head(15).round(2), hide_index=True, use_container_width=True)
    b_.markdown("**Struggle under pressure**")
    b_.dataframe(pl.sort_values("over_expected_per_100")[cols].head(15).round(2), hide_index=True, use_container_width=True)
    tm = Q.groupby("team").agg(pressured=("id", "size"), completed=("done", "sum"), expected=("xp", "sum")).reset_index()
    tm = tm[tm["pressured"] >= 5 * mn]
    if len(tm):
        tm["over_expected_per_100"] = 100 * (tm["completed"] - tm["expected"]) / tm["pressured"]
        st.plotly_chart(px.bar(tm.sort_values("over_expected_per_100").tail(25), x="over_expected_per_100", y="team", orientation="h", color="over_expected_per_100", color_continuous_scale="RdYlGn",
                               labels={"over_expected_per_100": "completion over expected per 100 pressured passes", "team": ""}).update_layout(height=640, coloraxis_showscale=False, title="Most press-resistant teams"),
                        use_container_width=True)
st.caption("The model sees the whole freeze frame, so it already expects pressured passes to fail more often; 'over expected' measures how far a player is above or below that.")
