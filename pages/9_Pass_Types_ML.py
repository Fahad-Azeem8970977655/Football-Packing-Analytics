import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import add_arrows, new_pitch
from ui import COLORWAY, setup
from utils import get_data, load_actions

setup("Pass Types (machine learning)", "🧩", "Unsupervised learning groups every pass by its geometry - then shows how often each type breaks lines, succeeds and adds threat.")
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    st.error("scikit-learn missing: pip install scikit-learn"); st.stop()


def make_X(d):
    return pd.DataFrame({"sx": d["sx"], "sy": d["sy"], "dx": d["ex"] - d["sx"], "dy": d["ey"] - d["sy"], "length": d["length"]})


@st.cache_resource(show_spinner="Learning pass types from ~200k passes...")
def fit(k):
    a = load_actions()
    p = a[a["action_type"] == "Pass"]
    X = make_X(p.sample(min(200_000, len(p)), random_state=0))
    sc = StandardScaler().fit(X)
    km = MiniBatchKMeans(k, n_init=5, random_state=0, batch_size=4096).fit(sc.transform(X))
    return sc, km


def name_of(c):
    third = "own third" if c.sx < 40 else "middle third" if c.sx < 80 else "final third"
    d = "forward" if c.dx >= 8 else "backward" if c.dx <= -8 else ("switch" if abs(c.dy) >= 18 else "sideways")
    size = "short" if c.length < 14 else "medium" if c.length < 28 else "long"
    lane = "wide" if abs(c.sy - 40) > 17 else "central"
    return f"{size} {d} ({third}, {lane})"


base, df, opt = get_data()
k = st.slider("Number of pass types (k)", 4, 12, 8)
sc, km = fit(k)
P = df[df["action_type"] == "Pass"].copy()
P["type_id"] = km.predict(sc.transform(make_X(P)))
cen = P.assign(dx=P["ex"] - P["sx"], dy=P["ey"] - P["sy"]).groupby("type_id").agg(
    sx=("sx", "mean"), sy=("sy", "mean"), ex=("ex", "mean"), ey=("ey", "mean"), dx=("dx", "mean"), dy=("dy", "mean"), length=("length", "mean"), n=("id", "size")).reset_index()
names, used = {}, {}
for r in cen.itertuples():
    nm = name_of(r)
    used[nm] = used.get(nm, 0) + 1
    names[r.type_id] = nm if used[nm] == 1 else f"{nm} #{used[nm]}"
P["pass_type_name"] = P["type_id"].map(names)
cen["name"] = cen["type_id"].map(names)
color_of = {n: COLORWAY[i % len(COLORWAY)] for i, n in enumerate(cen.sort_values("n", ascending=False)["name"])}

st.subheader("1. The pass-type map")
fig = new_pitch("Average arrow of each pass type (thickness = how often it is played)", 650)
mx = cen["n"].max()
for r in cen.itertuples():
    fig.add_annotation(x=r.ex, y=r.ey, ax=r.sx, ay=r.sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3,
                       arrowwidth=2 + 7 * r.n / mx, arrowcolor=color_of[r.name], opacity=0.95)
    fig.add_trace(go.Scatter(x=[r.sx], y=[r.sy], mode="markers+text", text=[str(r.type_id + 1)], textfont=dict(size=11, color="#111", family="Arial Black"),
                             marker=dict(size=22, color=color_of[r.name], line=dict(color="white", width=2)), name=f"{r.type_id + 1}. {r.name}",
                             hoverinfo="text", hovertext=f"{r.name}<br>{r.n:,} passes"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Each type is the centre of a cluster found by MiniBatchKMeans on start position, direction and length (no packing information is used for the clustering). "
           "The number marks where the average pass of that type starts.")

st.subheader("2. How good is each pass type?")
T = P.groupby("pass_type_name").agg(passes=("id", "size"), completion=("completed", "mean"), avg_packing=("pack_used", "mean"), line_breaking=("pack_used", lambda s: (s >= 1).mean()),
                                    xT_per_pass=("xt", "mean"), avg_length=("length", "mean")).reset_index()
T["share_%"] = 100 * T["passes"] / T["passes"].sum()
for c in ("completion", "line_breaking"):
    T[c] = 100 * T[c]
l, r = st.columns([3, 2])
l.dataframe(T.sort_values("passes", ascending=False).round(3), hide_index=True, use_container_width=True)
r.plotly_chart(px.scatter(T, x="completion", y="avg_packing", size="passes", color="pass_type_name", color_discrete_map=color_of, hover_name="pass_type_name",
                          labels={"completion": "completion %", "avg_packing": "average packing"}).update_layout(height=380, showlegend=False), use_container_width=True)
st.caption("Risk vs reward: types far to the upper-left bypass many opponents but are completed less often.")

st.subheader("3. Look at one type")
pick = st.selectbox("Pass type", T.sort_values("passes", ascending=False)["pass_type_name"].tolist())
S = P[P["pass_type_name"] == pick]
S = S.sample(min(220, len(S)), random_state=1) if len(S) else S
fig = new_pitch(f"{pick} - {len(S)} random passes", 620)
add_arrows(fig, S)
st.plotly_chart(fig, use_container_width=True)

st.subheader("4. Team fingerprints")
top_teams = P["team"].astype(str).value_counts().head(24).index
M = P[P["team"].astype(str).isin(top_teams)].groupby([P["team"].astype(str), "pass_type_name"]).size().unstack(fill_value=0)
M = (100 * M.div(M.sum(axis=1), axis=0)).round(1)
st.plotly_chart(px.imshow(M, aspect="auto", color_continuous_scale="Viridis", labels=dict(color="% of team's passes")).update_layout(height=max(420, 22 * len(M))),
                use_container_width=True)
with st.expander("Is this k a good choice? (silhouette check)"):
    sample = make_X(P.sample(min(4000, len(P)), random_state=0))
    Z = sc.transform(sample)
    ks = list(range(4, 13))
    sil = [silhouette_score(Z, MiniBatchKMeans(kk, n_init=3, random_state=0).fit_predict(Z)) for kk in ks]
    st.plotly_chart(px.line(x=ks, y=sil, markers=True, labels={"x": "k", "y": "silhouette"}).update_layout(height=300), use_container_width=True)
    st.caption("Higher = more clearly separated types. Pass geometry is continuous, so scores are modest; pick the k whose types are easiest to interpret.")
