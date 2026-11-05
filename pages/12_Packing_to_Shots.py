import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import setup
from utils import get_data, load_actions, load_matches, load_possessions, load_shots

setup('Packing → Shots, xG & Goals', '🥅', 'Does bypassing opponents lead to shots, chances and goals? Controlled regression with bootstrap intervals.')
st.markdown("Every **possession** is summarised by the packing its team created (opponents bypassed by passes and carries). "
            "We ask: *do possessions with more packing end in shots, higher-quality chances (xG) and goals - and does it still hold after "
            "controlling for possession length, where it started, and the team itself?* Penalties and shoot-outs are excluded.")
try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
except ImportError:
    st.error("scikit-learn missing: pip install scikit-learn"); st.stop()

base, df, opt = get_data()
matches = load_matches()
p = load_possessions()
p = p[p["match_id"].isin(base["match_id"].unique()) & (p["n_actions"] > 0)]
c = st.columns(4)
cov = c[0].slider("Min. freeze-frame coverage of the possession", 0.5, 1.0, 0.75, 0.05,
                  help="Share of the possession's passes/carries that have a freeze frame.")
excl = c[1].checkbox("Exclude corners, free kicks, kick-offs", True, help="Set pieces create shots without build-up.")
within = c[2].checkbox("Compare within team (team fixed effects)", False, help="Controls for strong teams both packing more and shooting more.")
outcome = c[3].radio("Binary outcome", ["Shot", "Goal"], horizontal=True)
p = p[p["coverage"] >= cov]
if excl:
    p = p[~p["play_pattern"].isin(["From Corner", "From Free Kick", "From Kick Off", "Other"])]
if len(p) < 500:
    st.warning("Too few possessions - relax the filters."); st.stop()
p = p.assign(y=((p["has_shot"] == 1) if outcome == "Shot" else (p["goals"] > 0)).astype(int))
k = st.columns(5)
k[0].metric("Possessions analysed", f"{len(p):,}")
k[1].metric("Ending in a shot", f"{p['has_shot'].mean():.1%}")
k[2].metric("Goals", f"{int(p['goals'].sum()):,}")
k[3].metric("Total xG", f"{p['xg'].sum():,.0f}")
k[4].metric("xG per possession", f"{p['xg'].mean():.3f}")


def wilson(kk, n, z=1.96):
    ph = kk / n
    den = 1 + z ** 2 / n
    ctr = (ph + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return ctr - half, ctr + half


st.subheader("1. Outcomes by packing in the possession")
p = p.assign(bucket=pd.cut(p["pack_total"], [-1, 0, 2, 5, 10, 20, 999], labels=["0", "1-2", "3-5", "6-10", "11-20", "21+"]))
b = p.groupby("bucket", observed=True).agg(n=("y", "size"), hits=("y", "sum"), xg=("xg", "mean"), xg_sd=("xg", "std")).reset_index()
b["rate"] = b["hits"] / b["n"]
ci = np.array([wilson(r.hits, r.n) for r in b.itertuples()])
b["lo"], b["hi"] = ci[:, 0], ci[:, 1]
l, r = st.columns(2)
fig = go.Figure(go.Bar(x=b["bucket"].astype(str), y=b["rate"] * 100, text=b["n"], textposition="outside", marker_color="#e76f51",
                       error_y=dict(type="data", symmetric=False, array=(b["hi"] - b["rate"]) * 100, arrayminus=(b["rate"] - b["lo"]) * 100)))
fig.update_layout(height=400, title=f"% of possessions ending in a {outcome.lower()}", xaxis_title="packing in the possession")
l.plotly_chart(fig, use_container_width=True)
fig2 = go.Figure(go.Bar(x=b["bucket"].astype(str), y=b["xg"], marker_color="#4cc9f0",
                        error_y=dict(type="data", array=1.96 * b["xg_sd"] / np.sqrt(b["n"]))))
fig2.update_layout(height=400, title="Average xG per possession (95% CI)", xaxis_title="packing in the possession")
r.plotly_chart(fig2, use_container_width=True)

st.subheader("2. Controlled effect")
X = pd.DataFrame({"pack_pass": p["pack_pass"], "pack_carry": p["pack_carry"], "n_actions": p["n_actions"], "start_x": p["start_x"]})
Xf = pd.concat([X, pd.get_dummies(p["possession_team"], drop_first=True).astype(float)], axis=1) if within else X
y, xg, groups = p["y"].to_numpy(), p["xg"].to_numpy(), p["match_id"].to_numpy()
sc = StandardScaler().fit(Xf)
Xs = sc.transform(Xf)
lr = LogisticRegression(max_iter=2000, C=1e6).fit(Xs, y)
or_ = np.exp(10 * lr.coef_[0][:2] / sc.scale_[:2])
lin = LinearRegression().fit(Xs, xg)
dxg = 10 * lin.coef_[:2] / sc.scale_[:2]


@st.cache_data(show_spinner="Bootstrapping confidence intervals over matches...")
def boot(Xv, yv, xgv, gv, n=60):
    rs = np.random.RandomState(0)
    idx = {g_: np.where(gv == g_)[0] for g_ in np.unique(gv)}
    keys = list(idx)
    res_or, res_x = [], []
    for _ in range(n):
        take = np.concatenate([idx[keys[j]] for j in rs.randint(0, len(keys), len(keys))])
        s_ = StandardScaler().fit(Xv[take])
        Z = s_.transform(Xv[take])
        res_or.append(np.exp(10 * LogisticRegression(max_iter=500, C=1e6).fit(Z, yv[take]).coef_[0][:2] / s_.scale_[:2]))
        res_x.append(10 * LinearRegression().fit(Z, xgv[take]).coef_[:2] / s_.scale_[:2])
    return np.array(res_or), np.array(res_x)


bo, bx = boot(Xf.to_numpy(float), y, xg, groups)
lo_o, hi_o = np.percentile(bo, 2.5, axis=0), np.percentile(bo, 97.5, axis=0)
lo_x, hi_x = np.percentile(bx, 2.5, axis=0), np.percentile(bx, 97.5, axis=0)
o = st.columns(4)
o[0].metric(f"Odds of a {outcome.lower()}: +10 packing (passes)", f"x{or_[0]:.2f}", help=f"95% CI x{lo_o[0]:.2f} to x{hi_o[0]:.2f}")
o[1].metric(f"Odds of a {outcome.lower()}: +10 packing (carries)", f"x{or_[1]:.2f}", help=f"95% CI x{lo_o[1]:.2f} to x{hi_o[1]:.2f}")
o[2].metric("xG per possession: +10 packing (passes)", f"{dxg[0]:+.3f}", help=f"95% CI {lo_x[0]:+.3f} to {hi_x[0]:+.3f}")
o[3].metric("xG per possession: +10 packing (carries)", f"{dxg[1]:+.3f}", help=f"95% CI {lo_x[1]:+.3f} to {hi_x[1]:+.3f}")
st.caption("95% bootstrap intervals resample whole matches. Controls: number of actions and start position" +
           (", and team." if within else ". Tick 'within team' to also control for team strength."))
gk = GroupKFold(n_splits=5)
pf = cross_val_predict(LogisticRegression(max_iter=2000), Xs, y, cv=gk, groups=groups, method="predict_proba")[:, 1]
keep = [i for i, c_ in enumerate(Xf.columns) if c_ not in ("pack_pass", "pack_carry")]
pn = cross_val_predict(LogisticRegression(max_iter=2000), Xs[:, keep], y, cv=gk, groups=groups, method="predict_proba")[:, 1]
a1, a0 = roc_auc_score(y, pf), roc_auc_score(y, pn)
q = st.columns(3)
q[0].metric("AUC without packing", f"{a0:.3f}")
q[1].metric("AUC with packing", f"{a1:.3f}")
q[2].metric("Gain from packing", f"{a1 - a0:+.3f}")
st.caption("AUC on held-out matches (5-fold, grouped by match).")

st.subheader("3. Teams: packing, chance quality and finishing")
t = p.groupby("possession_team").agg(possessions=("y", "size"), pack=("pack_total", "mean"), shots=("shots", "sum"), xg=("xg", "sum"),
                                     goals=("goals", "sum"), packing=("pack_total", "sum")).reset_index()
t["xg_per_100_packing"] = 100 * t["xg"] / t["packing"].where(t["packing"] > 0)
t["finishing_goals_minus_xg"] = t["goals"] - t["xg"]
mp = st.slider("Minimum possessions per team", 50, 1000, 300, 50)
t = t[t["possessions"] >= mp]
if not t.empty:
    t = t.round({"pack": 2, "xg": 1, "xg_per_100_packing": 2, "finishing_goals_minus_xg": 1})
    st.plotly_chart(px.scatter(t, x="pack", y="xg", size="possessions", color="finishing_goals_minus_xg", hover_name="possession_team",
                               text="possession_team", color_continuous_scale="RdYlGn",
                               labels={"pack": "average packing per possession", "xg": "total xG created (from analysed possessions)"})
                    .update_traces(textposition="top center", textfont_size=9).update_layout(height=560), use_container_width=True)
    st.dataframe(t.sort_values("xg_per_100_packing", ascending=False), hide_index=True, use_container_width=True)
else:
    st.info("No team reaches that many possessions - lower the slider.")

st.subheader("4. Key passes: does packing on the final pass create better chances?")
sh = load_shots()
sh = sh[(sh["type"] == "Shot") & (sh["shot_type"] != "Penalty") & sh["key_pass_id"].notna() & sh["match_id"].isin(base["match_id"].unique())]
a = load_actions()
kp = a.loc[a["id"].isin(sh["key_pass_id"]) & a["has_frame"], ["id", "pack", "n_opp_lane", "player", "recipient"]].rename(
    columns={"id": "key_pass_id", "player": "passer", "recipient": "shooter_rcv"})
kp = sh.merge(kp, on="key_pass_id", how="inner")
if len(kp) > 100:
    kp["bucket"] = pd.cut(kp["pack"], [-1, 0, 1, 2, 4, 99], labels=["0", "1", "2", "3-4", "5+"])
    kb = kp.groupby("bucket", observed=True).agg(shots=("id", "size"), xg=("xg", "mean"), goal_rate=("is_goal", "mean")).reset_index()
    kb["goal_rate"] = (kb["goal_rate"] * 100).round(1)
    l, r = st.columns(2)
    l.plotly_chart(px.bar(kb, x="bucket", y="xg", text="shots", labels={"bucket": "opponents bypassed by the key pass", "xg": "average xG of the resulting shot"},
                          color_discrete_sequence=["#4cc9f0"]).update_layout(height=360), use_container_width=True)
    r.plotly_chart(px.bar(kb, x="bucket", y="goal_rate", text="shots", labels={"bucket": "opponents bypassed by the key pass", "goal_rate": "% of shots that were goals"},
                          color_discrete_sequence=["#e76f51"]).update_layout(height=360), use_container_width=True)
    lab = matches.set_index("match_id")["label"]
    top = kp.sort_values(["is_goal", "pack"], ascending=False).head(15).copy()
    top["match"] = top["match_id"].map(lab)
    st.markdown("**Line-breaking assists** (goals first, then by packing of the key pass)")
    st.dataframe(top[["match", "team", "passer", "player", "minute", "pack", "xg", "is_goal"]].rename(columns={"player": "shooter", "pack": "key-pass packing"}).round(3),
                 hide_index=True, use_container_width=True)
else:
    st.info("Not enough shots with a measured key pass for these filters.")

st.subheader("5. The biggest line-breaking possessions that ended in a shot")
lab = matches.set_index("match_id")["label"]
top = p[p["has_shot"] == 1].sort_values("pack_total", ascending=False).head(15).copy()
top["match"] = top["match_id"].map(lab)
st.dataframe(top[["match", "possession_team", "start_min", "n_actions", "pack_total", "pack_pass", "pack_carry", "shots", "xg", "goals"]]
             .rename(columns={"start_min": "minute"}).round(3), hide_index=True, use_container_width=True)
st.info("Packing is also how a team gains territory, so part of the link is mechanical (possessions that reach the final third both bypass more opponents "
        "and shoot more). Read the numbers as a strong association, not proof of cause. Goals are rare (about 1 per 70 possessions), so goal results are noisier than shot or xG results.")
