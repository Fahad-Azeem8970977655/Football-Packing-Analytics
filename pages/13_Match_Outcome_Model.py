import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import setup
from utils import get_data, load_matches, load_shots

setup("Match Outcome Model", "🏆", "Do packing and threat explain who wins - beyond shots and xG? Two machine-learning models compared with grouped cross-validation (whole matches held out).")
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
except ImportError:
    st.error("scikit-learn missing: pip install scikit-learn"); st.stop()

base, df, opt = get_data()
matches = load_matches()
sh = load_shots()
sh = sh[(sh["type"] == "Shot") & (sh["shot_type"] != "Penalty") & sh["match_id"].isin(df["match_id"].unique())]
d = df.assign(team=df["team"].astype(str))
tm = d.groupby(["match_id", "team"]).agg(actions=("id", "size"), pack=("pack_used", "sum"), lb=("pack_used", lambda s: int((s >= 1).sum())), xt=("xt", "sum"),
                                         prog=("progressive", "sum"), f3=("final_third_entry", "sum"), box=("box_entry", "sum")).reset_index()
sx = sh.assign(team=sh["team"].astype(str)).groupby(["match_id", "team"]).agg(xg=("xg", "sum"), shots=("id", "size")).reset_index()
tm = tm.merge(sx, on=["match_id", "team"], how="left").fillna({"xg": 0, "shots": 0})
tm["n"] = tm.groupby("match_id")["team"].transform("nunique")
tm = tm[tm["n"] == 2].drop(columns="n")
mt = matches.set_index("match_id")
A = tm.copy()
B = tm.rename(columns={c: c + "_o" for c in tm.columns if c not in ("match_id",)})
pair = A.merge(B, on="match_id")
pair = pair[pair["team"] != pair["team_o"]].copy()
hs, as_ = pair["match_id"].map(mt["home_score"]).astype(float), pair["match_id"].map(mt["away_score"]).astype(float)
is_home = pair["team"].to_numpy() == pair["match_id"].map(mt["home_team"]).to_numpy()
pair["gd"] = np.where(is_home, hs - as_, as_ - hs)
pair["y"] = np.select([pair["gd"] > 0, pair["gd"] == 0], [2, 1], 0)
for c in ("pack", "lb", "xt", "prog", "f3", "box", "xg", "shots"):
    pair[c + "_d"] = pair[c] - pair[c + "_o"]
pair["poss"] = pair["actions"] / (pair["actions"] + pair["actions_o"])
if len(pair) < 100:
    st.warning("Too few matches with both teams measured - widen the sidebar filters."); st.stop()

SETS = {"1. Shots & xG only": ["xg_d", "shots_d"],
        "2. + possession (share of actions)": ["xg_d", "shots_d", "poss"],
        "3. + packing & line-breaking": ["xg_d", "shots_d", "poss", "pack_d", "lb_d"],
        "4. + threat (xT) & progression": ["xg_d", "shots_d", "poss", "pack_d", "lb_d", "xt_d", "prog_d", "f3_d", "box_d"]}
y, g = pair["y"].to_numpy(), pair["match_id"].to_numpy()
rows, preds = [], {}
cv = GroupKFold(n_splits=5)
for name, cols in SETS.items():
    X = StandardScaler().fit_transform(pair[cols])
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv, groups=g, method="predict_proba")
    preds[name] = p
    rows.append({"features": name, "model": "Logistic regression", "log_loss": log_loss(y, p), "accuracy": accuracy_score(y, p.argmax(1)), "AUC (win vs not)": roc_auc_score(y == 2, p[:, 2])})
Xall = StandardScaler().fit_transform(pair[SETS["4. + threat (xT) & progression"]])
pg = cross_val_predict(HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=25, random_state=0), Xall, y, cv=cv, groups=g, method="predict_proba")
rows.append({"features": "4. + threat (xT) & progression", "model": "Gradient boosting", "log_loss": log_loss(y, pg), "accuracy": accuracy_score(y, pg.argmax(1)), "AUC (win vs not)": roc_auc_score(y == 2, pg[:, 2])})
prior = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
rows.insert(0, {"features": "0. Nothing (class frequencies)", "model": "Baseline", "log_loss": log_loss(y, prior), "accuracy": accuracy_score(y, prior.argmax(1)), "AUC (win vs not)": 0.5})
R = pd.DataFrame(rows)

c = st.columns(4)
c[0].metric("Matches", f"{pair['match_id'].nunique():,}")
c[1].metric("Team-match rows", f"{len(pair):,}", help="Each match appears twice (once from each team's side).")
c[2].metric("Best AUC (win vs not)", f"{R['AUC (win vs not)'].max():.3f}")
c[3].metric("Best accuracy", f"{R['accuracy'].max():.1%}")
st.markdown("Each match is described by **differences between the two teams** (A minus B). Outcome = win / draw / loss for team A. "
            "This *explains* results after the match - it is not a pre-match forecast.")
st.subheader("1. Does adding packing and threat help?")
l, r = st.columns(2)
l.plotly_chart(px.bar(R, x="AUC (win vs not)", y="features", color="model", orientation="h", barmode="group", text_auto=".3f", range_x=[0.45, max(0.9, R["AUC (win vs not)"].max() + 0.02)])
               .update_layout(height=420, yaxis=dict(autorange="reversed")), use_container_width=True)
r.plotly_chart(px.bar(R, x="log_loss", y="features", color="model", orientation="h", barmode="group", text_auto=".3f")
               .update_layout(height=420, yaxis=dict(autorange="reversed"), xaxis_title="log loss (lower is better)"), use_container_width=True)
st.dataframe(R.round(3), hide_index=True, use_container_width=True)
st.caption("Held-out matches, 5-fold grouped by match. If feature set 3 or 4 beats set 1-2, packing/threat carry information that shots and xG do not.")

st.subheader("2. What matters most? (standardised coefficients)")
cols4 = SETS["4. + threat (xT) & progression"]
bin_model = LogisticRegression(max_iter=2000).fit(Xall, (y == 2).astype(int))
coef = pd.DataFrame({"feature": cols4, "coefficient": bin_model.coef_[0]}).sort_values("coefficient")
LABEL = {"xg_d": "xG difference", "shots_d": "Shots difference", "poss": "Share of actions", "pack_d": "Packing difference", "lb_d": "Line-breaking actions difference",
         "xt_d": "Threat (xT) difference", "prog_d": "Progressive actions difference", "f3_d": "Final-third entries difference", "box_d": "Box entries difference"}
coef["feature"] = coef["feature"].map(LABEL)
st.plotly_chart(px.bar(coef, x="coefficient", y="feature", orientation="h", color="coefficient", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
                .update_layout(height=420, coloraxis_showscale=False), use_container_width=True)
st.caption("Effect on the odds of winning of one standard deviation more of each statistic, holding the others fixed. Correlated statistics share credit, so read size and sign, not exact values.")

st.subheader("3. Packing difference vs goal difference")
q = pair.assign(bin=pd.qcut(pair["pack_d"], 10, duplicates="drop")).groupby("bin", observed=True).agg(pack_d=("pack_d", "mean"), gd=("gd", "mean"), win=("y", lambda s: (s == 2).mean())).reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=pair["pack_d"], y=pair["gd"] + np.random.RandomState(0).normal(0, 0.06, len(pair)), mode="markers", name="team-matches", marker=dict(size=5, opacity=0.25)))
fig.add_trace(go.Scatter(x=q["pack_d"], y=q["gd"], mode="lines+markers", name="average per decile", line=dict(color="#ffb703", width=4)))
fig.update_layout(height=430, xaxis_title="packing difference (team A - team B)", yaxis_title="goal difference (team A - team B)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("4. Match simulator")
st.caption("Set the gaps between two teams and see the predicted result probabilities from the full model (logistic, all four feature groups).")
mdl = LogisticRegression(max_iter=2000).fit(Xall, y)
scaler = StandardScaler().fit(pair[cols4])
cc = st.columns(3)
vals = {}
vals["xg_d"] = cc[0].slider("xG difference", -3.0, 3.0, 0.0, 0.1)
vals["shots_d"] = cc[0].slider("Shots difference", -15, 15, 0)
vals["pack_d"] = cc[1].slider("Packing difference", -400, 400, 0, 10)
vals["lb_d"] = cc[1].slider("Line-breaking actions difference", -150, 150, 0, 5)
vals["xt_d"] = cc[2].slider("Threat (xT) difference", -2.0, 2.0, 0.0, 0.1)
vals["poss"] = cc[2].slider("Share of actions", 0.30, 0.70, 0.50, 0.01)
for k_ in ("prog_d", "f3_d", "box_d"):
    vals[k_] = float(pair[k_].mean())
row = pd.DataFrame([{k_: vals[k_] for k_ in cols4}])
prob = mdl.predict_proba(scaler.transform(row))[0]
st.plotly_chart(px.bar(x=["Loss", "Draw", "Win"], y=prob * 100, text=[f"{p_ * 100:.0f}%" for p_ in prob], color=["Loss", "Draw", "Win"],
                       color_discrete_map={"Loss": "#ff006e", "Draw": "#9aa5b1", "Win": "#00d4a6"}, labels={"x": "", "y": "probability %"}).update_layout(height=340, showlegend=False),
                use_container_width=True)
st.info("Progressive actions, final-third entries and box entries are held at their average. With only a few hundred matches the probabilities are rough guides.")
