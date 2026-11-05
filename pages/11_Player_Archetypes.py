import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import setup
from utils import agg_players, get_data, load_minutes

setup('Player Archetypes', '🧬', 'PCA + KMeans roles and cosine-similarity player search.')
st.markdown("Players are described by a per-90 passing/carrying profile (packing, line-breaking, depth, width, directness, risk). "
            "PCA + KMeans group them into **roles**, and cosine similarity finds **similar players**.")
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    st.error("scikit-learn missing: pip install scikit-learn"); st.stop()

base, df, opt = get_data()
mins = load_minutes()
d = agg_players(df)
d["width"] = (d["sy"] - 40).abs()
d["pass_len"] = d["length"].where(d["action_type"] == "Pass")
d["pass_fwd"] = d["forward_m"].where(d["action_type"] == "Pass")
d["pass_done"] = d["completed"].astype(float).where(d["action_type"] == "Pass")
d["press"] = d["under_pressure"].astype(float)
g = d.groupby(["player", "team"], observed=True).agg(
    actions=("id", "size"), packing=("pack_used", "sum"), lb=("lb", "mean"), carry_share=("is_carry", "mean"),
    pass_len=("pass_len", "mean"), pass_fwd=("pass_fwd", "mean"), pass_done=("pass_done", "mean"),
    depth=("sx", "mean"), width=("width", "mean"), press=("press", "mean"),
    prog=("prog", "sum"), f3=("f3", "sum"), box=("box", "sum"), pos=("position", lambda s: s.astype(str).mode().iat[0])).reset_index()
g["player"], g["team"] = g["player"].astype(str), g["team"].astype(str)
m = mins[mins["match_id"].isin(df["match_id"].unique())].groupby(["player", "team"])["minutes"].sum().reset_index()
g = g.merge(m, on=["player", "team"], how="inner")
min_min = st.slider("Minimum minutes", 200, 3000, 600, 100)
g = g[g["minutes"] >= min_min].copy()
if len(g) < 30:
    st.warning("Too few players - lower the minutes slider or widen the sidebar filters."); st.stop()
for c, o in (("actions", "actions_p90"), ("packing", "packing_p90"), ("prog", "progressive_p90"), ("f3", "final_third_p90"), ("box", "box_entries_p90")):
    g[o] = g[c] / g["minutes"] * 90
g["packing_per_action"] = g["packing"] / g["actions"]
FEATS = {"actions_p90": "Involvement", "packing_p90": "Packing", "packing_per_action": "Packing / action", "lb": "Line-breaking %",
         "carry_share": "Carry share", "pass_len": "Pass length", "pass_fwd": "Forward pass m", "pass_done": "Pass completion",
         "depth": "Pitch depth", "width": "Width", "press": "Under pressure %", "progressive_p90": "Progressive",
         "final_third_p90": "Final-third entries", "box_entries_p90": "Box entries"}
Z = StandardScaler().fit_transform(g[list(FEATS)].fillna(g[list(FEATS)].mean()))

st.subheader("1. How many roles?")
ks = list(range(2, 9))
sil = [silhouette_score(Z, KMeans(k, n_init=5, random_state=0).fit_predict(Z), sample_size=min(len(Z), 2000), random_state=0) for k in ks]
best = ks[int(np.argmax(sil))]
l, r = st.columns([2, 1])
l.plotly_chart(px.line(x=ks, y=sil, markers=True, labels={"x": "number of clusters k", "y": "silhouette score"}).update_layout(height=300),
               use_container_width=True)
k = r.slider("Number of roles (k)", 2, 8, min(max(best, 4), 8))
r.caption(f"Silhouette suggests k = {best}. Higher k gives finer roles.")
km = KMeans(k, n_init=10, random_state=0).fit(Z)
g["cluster"] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=list(FEATS))
names = {}
for i in range(k):
    z = cent.loc[i].sort_values(ascending=False)
    names[i] = f"R{i + 1}: high {FEATS[z.index[0]].lower()}, {FEATS[z.index[1]].lower()}" if z.iloc[0] > 0.3 else f"R{i + 1}: balanced"
g["role"] = g["cluster"].map(names)

st.subheader("2. Role map (PCA)")
pc = PCA(2, random_state=0).fit(Z)
P = pc.transform(Z)
g["pc1"], g["pc2"] = P[:, 0], P[:, 1]
st.plotly_chart(px.scatter(g, x="pc1", y="pc2", color="role", hover_name="player",
                           hover_data=["team", "pos", "minutes"], opacity=0.8,
                           labels={"pc1": f"PC1 ({pc.explained_variance_ratio_[0]:.0%} of variance)",
                                   "pc2": f"PC2 ({pc.explained_variance_ratio_[1]:.0%})"}).update_layout(height=560),
                use_container_width=True)
l, r = st.columns([3, 2])
heat = cent.rename(columns=FEATS)
heat.index = [names[i] for i in heat.index]
l.markdown("**What defines each role** (z-scores, red = above average)")
l.plotly_chart(px.imshow(heat, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto").update_layout(height=380),
               use_container_width=True)
cnt = g["role"].value_counts().reset_index()
cnt.columns = ["role", "players"]
r.markdown("**Players per role**")
r.dataframe(cnt, hide_index=True, use_container_width=True)

st.subheader("3. Player profile & similar players")
g["label"] = g["player"] + " (" + g["team"] + ")"
sel = st.selectbox("Player", sorted(g["label"]))
i = int(np.where(g["label"].to_numpy() == sel)[0][0])
pct = g[list(FEATS)].rank(pct=True).iloc[i] * 100
fig = go.Figure(go.Scatterpolar(r=pct.values.tolist() + [pct.values[0]], theta=[FEATS[c] for c in FEATS] + [FEATS[list(FEATS)[0]]],
                                fill="toself", name=sel))
fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=520, title=f"{sel} - percentile vs. all {len(g)} players")
l, r = st.columns(2)
l.plotly_chart(fig, use_container_width=True)
Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
sim = Zn @ Zn[i]
out = g.assign(similarity=sim).drop(index=g.index[i]).sort_values("similarity", ascending=False).head(12)
r.markdown(f"**Most similar to {sel}**")
r.dataframe(out[["player", "team", "pos", "minutes", "role", "similarity"]].round(3), hide_index=True, use_container_width=True)
st.caption("Profiles use freeze-frame actions only. Men's and women's data are mixed unless you filter competitions in the sidebar.")
st.download_button("Download roles (CSV)", g[["player", "team", "pos", "minutes", "role"] + list(FEATS)].to_csv(index=False).encode(), "player_roles.csv")
