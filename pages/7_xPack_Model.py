import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ui import setup
from utils import get_data, load_actions

setup('xPack: Expected Packing', '📈', 'A gradient-boosting model of how much packing a typical action creates - and who beats it.')
st.markdown("A gradient-boosting model predicts how much packing a **typical** action would create from its *situation* "
            "(start location, action type, pressure, play pattern, player position). **Packing over expected (PoE)** shows who "
            "creates more bypassing than the situation alone explains.")
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
except ImportError:
    st.error("scikit-learn missing: pip install scikit-learn"); st.stop()


def features(d):
    X = pd.DataFrame({"sx": d["sx"], "sy": d["sy"], "is_pass": (d["action_type"] == "Pass").astype(int),
                      "pressure": d["under_pressure"].astype(int)})
    for c in ("play_pattern", "position"):
        X[c] = pd.Categorical(d[c]).codes
    return X


@st.cache_resource(show_spinner="Training xPack model (once, ~1 min)...")
def train():
    a = load_actions()
    d = a[a["has_frame"]]
    X, y = features(d), d["pack_cmp"].astype(float)
    if len(d) > 500_000:
        idx = np.random.RandomState(0).choice(len(d), 500_000, replace=False)
    else:
        idx = np.arange(len(d))
    Xtr, Xte, ytr, yte = train_test_split(X.iloc[idx], y.iloc[idx], test_size=0.2, random_state=0)
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0,
                                      categorical_features=[X.columns.get_loc("play_pattern"), X.columns.get_loc("position")])
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    stats = {"r2": r2_score(yte, pred), "mae": mean_absolute_error(yte, pred), "n": len(d),
             "baseline_mae": mean_absolute_error(yte, np.full(len(yte), ytr.mean()))}
    return pd.Series(m.predict(X), index=d.index, name="xpack"), stats


xp, stats = train()
base, df, opt = get_data()
c = st.columns(4)
c[0].metric("Training actions", f"{stats['n']:,}")
c[1].metric("R² (hold-out)", f"{stats['r2']:.3f}")
c[2].metric("MAE", f"{stats['mae']:.3f}")
c[3].metric("MAE of 'always average'", f"{stats['baseline_mae']:.3f}")
st.caption("A modest R² is expected: where the ball goes (and which opponents are in the way) drives packing, not only the starting situation.")

d = df[df["has_frame"]].copy()
d["xpack"] = xp.reindex(d.index)
d = d.dropna(subset=["xpack", "pack_cmp"])

st.subheader("Actual vs expected packing by start position")
d["x_bin"] = (d["sx"] // 10 * 10).clip(0, 110)
b = d.groupby("x_bin").agg(actual=("pack_cmp", "mean"), expected=("xpack", "mean")).reset_index()
st.plotly_chart(px.line(b.melt(id_vars="x_bin"), x="x_bin", y="value", color="variable", markers=True,
                        labels={"x_bin": "start x (0 = own goal)", "value": "avg packing"}).update_layout(height=380),
                use_container_width=True)

st.subheader("Packing over expected - players")
min_act = st.slider("Minimum actions", 50, 2000, 300, 50)
p = d.groupby(["player", "team"], observed=True).agg(actions=("id", "size"), packing=("pack_cmp", "sum"),
                                                    expected=("xpack", "sum")).reset_index()
p = p[p["actions"] >= min_act]
p["poe"] = (p["packing"] - p["expected"]).round(1)
p["poe_per_100"] = (100 * p["poe"] / p["actions"]).round(2)
p[["packing", "expected"]] = p[["packing", "expected"]].round(1)
if p.empty:
    st.warning("No players meet the minimum."); st.stop()
l, r = st.columns(2)
top = p.sort_values("poe_per_100", ascending=False)
l.markdown("**Most packing above expectation (per 100 actions)**")
l.dataframe(top.head(15), hide_index=True, use_container_width=True)
r.markdown("**Least**")
r.dataframe(top.tail(15).iloc[::-1], hide_index=True, use_container_width=True)
st.plotly_chart(px.scatter(p, x="expected", y="packing", size="actions", color="poe_per_100", hover_name="player",
                           color_continuous_scale="RdYlGn", labels={"expected": "expected packing", "packing": "actual packing"})
                .add_shape(type="line", x0=0, y0=0, x1=p["expected"].max(), y1=p["expected"].max(), line=dict(dash="dot"))
                .update_layout(height=520), use_container_width=True)

st.subheader("Risk vs reward: pass completion by packing")
ps = df[(df["action_type"] == "Pass") & df["has_frame"]].copy()
ps["bucket"] = ps["pack"].clip(upper=6).astype(int).astype(str).replace({"6": "6+"})
rr = ps.groupby("bucket").agg(completion=("completed", "mean"), passes=("id", "size")).reset_index()
rr["completion"] = (rr["completion"] * 100).round(1)
st.plotly_chart(px.bar(rr, x="bucket", y="completion", text="passes", labels={"bucket": "opponents bypassed (if completed)", "completion": "completion %"},
                       color_discrete_sequence=["#4cc9f0"]).update_layout(height=360), use_container_width=True)
