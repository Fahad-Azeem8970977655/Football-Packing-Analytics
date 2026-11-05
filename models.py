"""Shared machine-learning models, cached once per session and reused by several pages."""
import numpy as np
import pandas as pd
import streamlit as st

from utils import load_actions


def xpass_features(d):
    return pd.DataFrame({
        "sx": d["sx"], "sy": d["sy"], "ex": d["ex"], "ey": d["ey"], "length": d["length"],
        "angle": np.arctan2(d["ey"] - d["sy"], d["ex"] - d["sx"]), "pressure": d["under_pressure"].astype(int),
        "n_opp_lane": d["n_opp_lane"], "min_opp_dist_start": d["min_opp_dist_start"],
        "min_opp_dist_end": d["min_opp_dist_end"], "n_opp_5m_end": d["n_opp_5m_end"],
        "n_tm_8m_end": d["n_tm_8m_end"], "n_opp_vis": d["n_opp_vis"],
        "height": pd.Categorical(d["pass_height"]).codes + 1, "body": pd.Categorical(d["pass_body_part"]).codes + 1})


@st.cache_resource(show_spinner="Training the xPass model with match-grouped 3-fold cross-validation (about a minute)...")
def get_xpass():
    """Out-of-fold completion probability for every pass that has a freeze frame.
    Returns (Series indexed like load_actions(), stats dict)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    a = load_actions()
    p = a[(a["action_type"] == "Pass") & a["has_frame"]]
    X, y, g = xpass_features(p), p["completed"].astype(int).to_numpy(), p["match_id"].to_numpy()
    oof = np.zeros(len(p))
    for tr, te in GroupKFold(n_splits=3).split(X, y, g):        # whole matches are held out
        if len(tr) > 250_000:
            tr = np.random.RandomState(0).choice(tr, 250_000, replace=False)
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    base = y.mean()
    stats = {"n": len(p), "auc": roc_auc_score(y, oof), "brier": brier_score_loss(y, oof),
             "brier_base": brier_score_loss(y, np.full(len(y), base)), "logloss": log_loss(y, oof), "rate": base}
    return pd.Series(oof, index=p.index, name="xp"), stats


ON_TARGET = ["Goal", "Saved", "Saved to Post"]


@st.cache_resource(show_spinner="Training the post-shot xG model (shot placement)...")
def get_psxg():
    """Post-shot xG: probability that an on-target shot scores, given its xG and where it was placed in the goal.
    Out-of-fold by match. Returns (Series indexed like load_shots_detail(), stats)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from utils import load_shots_detail
    d = load_shots_detail()
    p = d[d["outcome"].isin(ON_TARGET) & d["end_y"].notna() & d["end_z"].notna() & d["xg"].notna()]
    X = pd.DataFrame({"xg": p["xg"], "off_centre": (p["end_y"] - 40).abs(), "height": p["end_z"], "head": p["body_part"].eq("Head").astype(int),
                      "penalty": p["penalty"].astype(int), "free_kick": p["shot_type"].eq("Free Kick").astype(int)})
    y = p["is_goal"].astype(int).to_numpy()
    g = (p["competition"].astype(str) + p["match_id"].astype(str)).to_numpy()
    oof = np.zeros(len(p))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    stats = {"n": len(p), "auc": roc_auc_score(y, oof), "auc_xg_only": roc_auc_score(y, p["xg"]), "brier": brier_score_loss(y, oof), "brier_xg": brier_score_loss(y, p["xg"].clip(0, 1)), "rate": y.mean()}
    return pd.Series(oof, index=p.index, name="psxg"), stats
