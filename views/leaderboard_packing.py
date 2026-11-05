import plotly.express as px
import streamlit as st

from utils import agg_players, get_data, load_minutes


def render():
    base, df, opt = get_data()
    mins = load_minutes()

    d = agg_players(df)
    g = d.groupby(["player", "team"], observed=True).agg(
        actions=("id", "size"), passes=("is_pass", "sum"), carries=("is_carry", "sum"), packing=("pack_used", "sum"),
        pack_pass=("pack_pass", "sum"), pack_carry=("pack_carry", "sum"), lb_actions=("lb", "sum"),
        def_line_breaks=("db", "sum"), mid_line_breaks=("mb", "sum"), progressive=("prog", "sum"),
        final_third_entries=("f3", "sum"), box_entries=("box", "sum")).reset_index()
    g["player"], g["team"] = g["player"].astype(str), g["team"].astype(str)
    m = (mins[mins["match_id"].isin(df["match_id"].unique())].groupby(["player", "team"])["minutes"].sum().reset_index())
    g = g.merge(m, on=["player", "team"], how="left")
    g["minutes"] = g["minutes"].fillna(0)
    g["packing_p90"] = (g["packing"] / g["minutes"].where(g["minutes"] > 0) * 90).round(2)
    g["pack_per_action"] = (g["packing"] / g["actions"]).round(3)
    g["lb_pct"] = (100 * g["lb_actions"] / g["actions"]).round(1)

    min_min = st.slider("Minimum minutes played", 0, 3000, 450, 50)
    min_act = st.slider("Minimum actions", 0, 1000, 100, 10)
    g = g[(g["minutes"] >= min_min) & (g["actions"] >= min_act)]
    if g.empty:
        st.warning("No players meet these minimums."); return

    metrics = {"Packing per 90": "packing_p90", "Total packing": "packing", "Packing per action": "pack_per_action",
               "Line-breaking %": "lb_pct", "Defensive-line breaks": "def_line_breaks",
               "Progressive actions": "progressive", "Box entries": "box_entries"}
    l, r = st.columns([1, 2])
    sel = l.selectbox("Rank by", list(metrics))
    n = l.slider("Players shown", 5, 40, 20)
    pos_f = l.multiselect("Position contains", sorted(df["position"].dropna().astype(str).unique()))
    if pos_f:
        keep = df[df["position"].astype(str).isin(pos_f)]["player"].astype(str).unique()
        g = g[g["player"].isin(keep)]
    top = g.sort_values(metrics[sel], ascending=False).head(n)
    fig = px.bar(top.iloc[::-1], x=metrics[sel], y="player", color="team", orientation="h", hover_data=["minutes", "actions"])
    fig.update_layout(height=max(380, 24 * n), yaxis_title=None)
    r.plotly_chart(fig, use_container_width=True)

    st.subheader("Packing p90 vs line-breaking %")
    st.plotly_chart(px.scatter(g, x="lb_pct", y="packing_p90", size="actions", color="team", hover_name="player",
                               hover_data=["minutes", "packing"]).update_layout(height=520, showlegend=False),
                    use_container_width=True)
    st.subheader("Full table")
    st.dataframe(g.sort_values(metrics[sel], ascending=False), hide_index=True, use_container_width=True)
    st.download_button("Download CSV", g.to_csv(index=False).encode(), "player_packing.csv")

