import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from models import get_xpass
from ui import setup
from utils import agg_players, get_data, load_actions

setup('Pass Value (xPass)', '🎯', 'Completion probability from the freeze frame, combined with packing into risk-adjusted passing value.')
st.markdown(
    "An **xPass** model predicts the probability that each pass is completed, using the freeze frame "
    "(opponents in the passing lane, distance of the nearest defender to the passer and to the target, "
    "teammate support, pressure, geometry). Combining it with packing gives **risk-adjusted passing value**: "
    "who completes difficult, line-breaking passes - and who only plays safe ones.")
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
except ImportError:
    st.error("scikit-learn missing: pip install scikit-learn"); st.stop()


xp, stats = get_xpass()
base, df, opt = get_data()

c = st.columns(5)
c[0].metric("Passes modelled", f"{stats['n']:,}")
c[1].metric("ROC AUC (held-out matches)", f"{stats['auc']:.3f}")
c[2].metric("Brier score", f"{stats['brier']:.4f}", help="Lower is better")
c[3].metric("Brier of 'always average'", f"{stats['brier_base']:.4f}")
c[4].metric("Avg completion", f"{stats['rate']:.1%}")
st.caption("Predictions are out-of-fold: each pass is scored by a model that never saw its match. The model uses the pass destination, "
           "so it measures difficulty in hindsight (failed passes often end out of play or next to a defender); AUC is therefore optimistic "
           "if you read it as a forecast made before the pass.")

d = df[(df["action_type"] == "Pass") & df["has_frame"]].copy()
d["xp"] = xp.reindex(d.index)
d = d.dropna(subset=["xp"])
if d.empty:
    st.warning("No scored passes for these filters."); st.stop()
d["done"] = d["completed"].astype(int)
d["pack_potential"] = d["pack"]                                  # packing if the pass is completed
d["xpack_value"] = d["xp"] * d["pack_potential"]                 # risk-adjusted expected packing
d["pack_actual"] = d["pack_potential"] * d["done"]

l, r = st.columns(2)
with l:
    st.subheader("Calibration")
    d["pbin"] = pd.cut(d["xp"], np.linspace(0, 1, 11), include_lowest=True)
    cal = d.groupby("pbin", observed=True).agg(predicted=("xp", "mean"), actual=("done", "mean"), n=("id", "size")).reset_index()
    fig = px.scatter(cal, x="predicted", y="actual", size="n", labels={"predicted": "predicted completion", "actual": "actual completion"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
    st.plotly_chart(fig.update_layout(height=380), use_container_width=True)
with r:
    st.subheader("What makes a pass hard? (opponents in the lane)")
    lane = d.groupby(d["n_opp_lane"].clip(upper=4).astype(int).astype(str).replace({"4": "4+"}), observed=True).agg(
        completion=("done", "mean"), passes=("id", "size")).reset_index()
    lane["completion"] = (lane["completion"] * 100).round(1)
    st.plotly_chart(px.bar(lane, x="n_opp_lane", y="completion", text="passes",
                           labels={"n_opp_lane": "opponents in the passing lane", "completion": "completion %"},
                           color_discrete_sequence=["#4cc9f0"]).update_layout(height=380), use_container_width=True)

st.subheader("Bold passes: line-breaking passes that were completed against the odds")
thr = st.slider("Minimum packing", 1, 8, 3)
bold = d[(d["completed"]) & (d["pack_potential"] >= thr)].sort_values("xp").head(25)
st.dataframe(bold[["match_date", "competition", "team", "player", "recipient", "minute", "pack_potential", "xp", "n_opp_lane"]]
             .rename(columns={"xp": "completion prob.", "pack_potential": "packing"}).round(3),
             hide_index=True, use_container_width=True)
risk = d[d["pack_potential"] >= thr]
st.plotly_chart(px.histogram(risk, x="xp", color=risk["completed"].map({True: "completed", False: "failed"}), nbins=20,
                             barmode="overlay", opacity=0.65, labels={"xp": f"completion probability of passes bypassing >= {thr}"})
                .update_layout(height=340), use_container_width=True)

st.subheader("Players: completion & packing over expected")
min_p = st.slider("Minimum passes", 50, 1500, 300, 50)
pl = d.groupby(["player", "team"], observed=True).agg(
    passes=("id", "size"), completed=("done", "sum"), expected_completed=("xp", "sum"), packing=("pack_actual", "sum"),
    expected_packing=("xpack_value", "sum"), avg_difficulty=("xp", lambda s: 1 - s.mean())).reset_index()
pl = pl[pl["passes"] >= min_p]
if pl.empty:
    st.warning("No players reach that minimum."); st.stop()
pl["player"], pl["team"] = pl["player"].astype(str), pl["team"].astype(str)
pl["completion_over_exp_per100"] = (100 * (pl["completed"] - pl["expected_completed"]) / pl["passes"]).round(2)
pl["packing_over_exp_per100"] = (100 * (pl["packing"] - pl["expected_packing"]) / pl["passes"]).round(2)
pl["packing_per100"] = (100 * pl["packing"] / pl["passes"]).round(1)
pl["avg_difficulty"] = pl["avg_difficulty"].round(3)
st.plotly_chart(px.scatter(pl, x="avg_difficulty", y="completion_over_exp_per100", size="passes", color="packing_per100",
                           hover_name="player", hover_data=["team", "passes"], color_continuous_scale="YlOrRd",
                           labels={"avg_difficulty": "average pass difficulty (1 - completion prob.)",
                                   "completion_over_exp_per100": "completion over expected per 100 passes"})
                .update_layout(height=560), use_container_width=True)
st.caption("Top-right = attempts hard passes AND beats the model. Colour = packing per 100 passes.")
l, r = st.columns(2)
cols = ["player", "team", "passes", "avg_difficulty", "completion_over_exp_per100", "packing_per100", "packing_over_exp_per100"]
l.markdown("**Best completion over expected**")
l.dataframe(pl.sort_values("completion_over_exp_per100", ascending=False)[cols].head(15), hide_index=True, use_container_width=True)
r.markdown("**Most packing over expected (risk-adjusted)**")
r.dataframe(pl.sort_values("packing_over_exp_per100", ascending=False)[cols].head(15), hide_index=True, use_container_width=True)
