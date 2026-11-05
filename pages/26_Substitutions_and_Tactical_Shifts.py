import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, role_group
from ui import setup
from utils import comp_filter, load_actions, load_cards, load_matches, load_shots, load_stints, load_tactical_shifts

setup("Substitutions & Tactical Shifts", "🔄", "When do coaches change the game, what does it do to the team's threat and packing, who are the best substitutes, and which positional switches are used most?")
M = comp_filter()
mids = set(M["match_id"])
cs = M.set_index("match_id")["cs"]
home, away = M.set_index("match_id")["home_team"].to_dict(), M.set_index("match_id")["away_team"].to_dict()
ST = load_stints()
ST = ST[ST["match_id"].isin(mids)].copy()
ST["team"], ST["player"] = ST["team"].astype(str), ST["player"].astype(str)
ST["cs"] = ST["match_id"].map(cs)
SUB = ST[ST["start_reason"].astype(str).str.startswith("Substitution - On")].copy()
SUB["reason"] = np.where(SUB["start_reason"].astype(str).str.contains("Injury"), "injury", "tactical")
TS = load_tactical_shifts()
TS = TS[TS["match_id"].isin(mids)].copy()
TS["team"] = TS["team"].astype(str)
CD = load_cards()
CD = CD[CD["match_id"].isin(mids)].copy()
CD["team"], CD["player"] = CD["team"].astype(str), CD["player"].astype(str)


@st.cache_resource(show_spinner="Building minute-by-minute team profiles...")
def minute_arrays(match_ids):
    A = load_actions()
    a = A.loc[A["match_id"].isin(match_ids) & A["period"].isin([1, 2]), ["match_id", "team", "period", "minute", "xt", "pack_cmp", "goal_diff"]].copy()
    a["team"] = a["team"].astype(str)
    g = a.groupby(["match_id", "team", "period", "minute"]).agg(xt=("xt", "sum"), pack=("pack_cmp", "sum"), actions=("xt", "size"), gd=("goal_diff", "mean")).reset_index()
    out = {}
    for (m_, t_, p_), d in g.groupby(["match_id", "team", "period"]):
        arr = np.zeros((4, 100))
        mi = d["minute"].clip(0, 99).astype(int).to_numpy()
        arr[0, mi], arr[1, mi], arr[2, mi], arr[3, mi] = d["xt"].to_numpy(), d["pack"].to_numpy(), d["actions"].to_numpy(), d["gd"].to_numpy()
        out[(m_, t_, p_)] = arr
    return out


ARR = minute_arrays(tuple(sorted(mids)))
BOUNDS = {1: (0, 44), 2: (45, 89)}


def effects(ev, w):
    """ev: DataFrame(match_id, team, period, minute). Before = w minutes before the change, after = w minutes from it, inside the same half."""
    rows = []
    for r in ev.itertuples():
        lo, hi = BOUNDS.get(r.period, (None, None))
        if lo is None or r.minute - w < lo or r.minute + w - 1 > hi:
            continue
        opp = away.get(r.match_id) if home.get(r.match_id) == r.team else home.get(r.match_id)
        a, b = ARR.get((r.match_id, r.team, r.period)), ARR.get((r.match_id, opp, r.period))
        if a is None or b is None:
            continue
        sl_b, sl_a = slice(r.minute - w, r.minute), slice(r.minute, r.minute + w)
        row = {"match_id": r.match_id, "team": r.team}
        for i, nm in enumerate(("xt", "pack", "actions")):
            row[nm + "_team"] = (a[i, sl_a].sum() - a[i, sl_b].sum()) / w * 10
            row[nm + "_opp"] = (b[i, sl_a].sum() - b[i, sl_b].sum()) / w * 10
            row[nm + "_net"] = row[nm + "_team"] - row[nm + "_opp"]
        gdv = a[3, max(r.minute - 1, lo)]
        row["state"] = "trailing" if gdv < -0.25 else "leading" if gdv > 0.25 else "level"
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(E):
    out = []
    for nm, lab in (("xt", "Threat (xT) per 10 min"), ("pack", "Packing per 10 min"), ("actions", "Actions per 10 min")):
        for who in ("team", "opp", "net"):
            v = E[f"{nm}_{who}"]
            out.append({"metric": lab, "who": {"team": "the team", "opp": "the opponent", "net": "net (team minus opponent)"}[who], "change": v.mean(), "ci": 1.96 * v.std() / np.sqrt(len(v))})
    return pd.DataFrame(out)


t1, t2, t3, t4, t5 = st.tabs(["Substitutions", "Do substitutions change the game?", "Best substitutes", "Tactical shifts", "Cards & discipline"])

with t1:
    k = st.columns(4)
    k[0].metric("Substitutions", f"{len(SUB):,}")
    k[1].metric("Tactical / injury", f"{(SUB['reason'] == 'tactical').mean():.0%} / {(SUB['reason'] == 'injury').mean():.0%}")
    k[2].metric("Median minute", f"{SUB['from_min'].median():.0f}'")
    k[3].metric("In the last 20 minutes", f"{(SUB['from_min'] >= 70).mean():.0%}")
    SUB["bin"] = (SUB["from_min"] // 5 * 5).clip(upper=120).astype(int)
    h = SUB.groupby(["bin", "reason"]).size().reset_index(name="n")
    st.plotly_chart(px.bar(h, x="bin", y="n", color="reason", color_discrete_map={"tactical": "#00d4a6", "injury": "#ff006e"}, labels={"bin": "minute (5-minute bins)", "n": "substitutions"})
                    .update_layout(height=380, title="When are substitutions made?"), use_container_width=True)
    first = SUB.groupby(["match_id", "team"])["from_min"].min().reset_index()
    first["tc"] = first["team"] + " | " + first["match_id"].map(cs)
    ft = first.groupby("tc").agg(matches=("match_id", "nunique"), median_first_sub=("from_min", "median")).reset_index()
    ft = ft[ft["matches"] >= 5].sort_values("median_first_sub")
    l, r = st.columns(2)
    l.markdown("**Earliest first substitution** (5+ matches)")
    l.dataframe(ft.head(15).round(1).rename(columns={"tc": "team | competition"}), hide_index=True, use_container_width=True)
    r.markdown("**Latest first substitution**")
    r.dataframe(ft.tail(15).iloc[::-1].round(1).rename(columns={"tc": "team | competition"}), hide_index=True, use_container_width=True)
    pc = SUB.assign(role=SUB["position"].map(role_group)).groupby("role").size().reset_index(name="n")
    st.plotly_chart(px.bar(pc, x="role", y="n", labels={"n": "players brought on", "role": "role of the substitute"}, color_discrete_sequence=["#4cc9f0"]).update_layout(height=320, title="Who comes on?"), use_container_width=True)

with t2:
    w = st.select_slider("Window (minutes before and after)", options=[8, 10, 12, 15], value=15, key="sx_w")
    first_t = SUB[(SUB["reason"] == "tactical") & (SUB["from_period"] == 2)].sort_values("from_min").groupby(["match_id", "team"]).head(1)
    ev = pd.DataFrame({"match_id": first_t["match_id"], "team": first_t["team"], "period": 2, "minute": first_t["from_min"].astype(int)})
    E = effects(ev, w)
    if len(E) < 30:
        st.warning("Too few substitutions with a clean window - widen the filters or shorten the window.")
    else:
        st.markdown(f"First tactical substitution of each team in the **second half**, between minute {45 + w} and {90 - w}, comparing the {w} minutes before with the {w} minutes after. "
                    f"The opponent's change over the same minutes is the control: a rise that the opponent also shows is just the match flow.")
        c = st.columns(3)
        c[0].metric("Substitutions analysed", f"{len(E):,}")
        S_ = summarise(E)
        net_x = S_[(S_["metric"] == "Threat (xT) per 10 min") & (S_["who"].str.startswith("net"))].iloc[0]
        c[1].metric("Net change in threat", f"{net_x['change']:+.3f} xT / 10 min", help=f"95% interval +/- {net_x['ci']:.3f}")
        net_p = S_[(S_["metric"] == "Packing per 10 min") & (S_["who"].str.startswith("net"))].iloc[0]
        c[2].metric("Net change in packing", f"{net_p['change']:+.1f} / 10 min", help=f"95% interval +/- {net_p['ci']:.1f}")
        st.plotly_chart(px.bar(S_, x="metric", y="change", color="who", barmode="group", error_y="ci", labels={"change": f"change after the substitution (per 10 minutes)", "metric": ""},
                               color_discrete_map={"the team": "#00d4a6", "the opponent": "#ff006e", "net (team minus opponent)": "#ffd166"}).update_layout(height=430), use_container_width=True)
        stt = E.groupby("state").agg(n=("xt_net", "size"), xt_net=("xt_net", "mean"), pack_net=("pack_net", "mean"), xt_se=("xt_net", lambda s: 1.96 * s.std() / np.sqrt(len(s)))).reset_index()
        st.plotly_chart(px.bar(stt, x="state", y="xt_net", error_y="xt_se", text=stt["n"], category_orders={"state": ["trailing", "level", "leading"]}, color_discrete_sequence=["#ffb703"],
                               labels={"xt_net": "net change in threat per 10 min", "state": "score when the substitution is made"}).update_layout(height=360, title="Does it depend on the score?"), use_container_width=True)
        st.caption("Intervals that cross zero mean the effect cannot be told apart from noise. Teams usually substitute when something is already changing (tiredness, a goal), so read this as what happens around substitutions, not what they cause.")

with t3:
    A = load_actions()
    sp = SUB["player"].unique()
    a = A.loc[A["match_id"].isin(mids) & A["player"].astype(str).isin(sp), ["match_id", "player", "minute", "second", "xt", "pack_cmp", "id"]].copy()
    a["player"] = a["player"].astype(str)
    a["clock"] = a["minute"] + a["second"] / 60
    m = SUB[["match_id", "team", "player", "position", "from_min", "to_min", "minutes", "cs"]].merge(a, on=["match_id", "player"])
    m = m[(m["clock"] >= m["from_min"]) & (m["clock"] < m["to_min"])]
    per = m.groupby(["match_id", "player"]).agg(xt=("xt", "sum"), pack=("pack_cmp", "sum"), actions=("id", "size")).reset_index()
    stint = SUB.groupby(["match_id", "player"]).agg(team=("team", "first"), position=("position", "first"), minutes=("minutes", "sum")).reset_index().merge(per, on=["match_id", "player"], how="left").fillna({"xt": 0, "pack": 0, "actions": 0})
    shots = load_shots()
    shots = shots[(shots["type"] == "Shot") & (shots["shot_type"] != "Penalty") & shots["match_id"].isin(mids)].copy()
    shots["player"], shots["clock"] = shots["player"].astype(str), shots["minute"] + shots["second"] / 60
    sm = SUB[["match_id", "player", "from_min", "to_min"]].merge(shots[["match_id", "player", "clock", "xg", "is_goal"]], on=["match_id", "player"])
    sm = sm[(sm["clock"] >= sm["from_min"]) & (sm["clock"] < sm["to_min"])]
    sg = sm.groupby("player").agg(goals=("is_goal", "sum"), xg=("xg", "sum"), shots=("xg", "size")).reset_index()
    pl = stint.groupby("player").agg(team=("team", lambda s: s.mode().iat[0]), position=("position", lambda s: s.dropna().astype(str).mode().iat[0] if s.notna().any() else None), appearances=("match_id", "nunique"),
                                     sub_minutes=("minutes", "sum"), xt=("xt", "sum"), pack=("pack", "sum"), actions=("actions", "sum")).reset_index().merge(sg, on="player", how="left").fillna({"goals": 0, "xg": 0, "shots": 0})
    tot_min = pl["sub_minutes"].sum()
    st.metric("Average threat (xT) per 90 for substitutes", f"{pl['xt'].sum() / tot_min * 90:.2f}")
    mn = st.slider("Minimum minutes as a substitute", 60, 600, 150, 30, key="sx_minsub")
    P = pl[pl["sub_minutes"] >= mn].copy()
    if len(P) < 5:
        st.info("No player reaches that many minutes - lower the slider.")
    else:
        for c_ in ("xt", "pack", "actions", "xg", "goals"):
            P[c_ + "_p90"] = P[c_] / P["sub_minutes"] * 90
        P["role"] = P["position"].map(role_group)
        METR = {"xt_p90": "Threat (xT) per 90", "pack_p90": "Packing per 90", "xg_p90": "xG per 90", "goals_p90": "Goals per 90", "actions_p90": "Passes + carries per 90"}
        mt = st.selectbox("Rank substitutes by", list(METR), format_func=METR.get, key="sx_metric")
        top = P.sort_values(mt, ascending=False).head(20)
        st.plotly_chart(px.bar(top.iloc[::-1], x=mt, y=top.iloc[::-1]["player"] + " (" + top.iloc[::-1]["position"].astype(str).map(abbr) + ")", orientation="h", color="role", hover_data=["team", "sub_minutes", "appearances"],
                               labels={mt: METR[mt], "y": ""}).update_layout(height=560), use_container_width=True)
        st.dataframe(P.sort_values(mt, ascending=False)[["player", "team", "position", "appearances", "sub_minutes", "xt_p90", "pack_p90", "xg_p90", "goals", "xg"]].round(2).head(60), hide_index=True, use_container_width=True)
        st.caption("Per-90 rates are measured only over the minutes the player was on the pitch as a substitute, so small samples are noisy; raise the minutes slider for steadier names.")

with t4:
    w2 = st.select_slider("Window (minutes before and after)", options=[6, 8, 10, 12], value=10, key="sx_w2")
    k = st.columns(3)
    k[0].metric("Tactical-shift events", f"{len(TS):,}")
    k[1].metric("Per match (both teams)", f"{len(TS) / max(len(mids), 1):.1f}")
    k[2].metric("Players switching position", f"{int((ST['start_reason'] == 'Tactical Shift').sum()):,}")
    evs = TS.rename(columns={"minute": "minute"})[["match_id", "team", "period", "minute"]]
    E2 = effects(evs, w2)
    if len(E2) >= 30:
        S2 = summarise(E2)
        st.plotly_chart(px.bar(S2, x="metric", y="change", color="who", barmode="group", error_y="ci", labels={"change": "change after the shift (per 10 minutes)", "metric": ""},
                               color_discrete_map={"the team": "#00d4a6", "the opponent": "#ff006e", "net (team minus opponent)": "#ffd166"}).update_layout(height=420, title=f"Before vs after a tactical shift ({len(E2):,} shifts, {w2} minutes either side)"),
                        use_container_width=True)
    else:
        st.info("Too few shifts with a clean window in this selection.")
    sw = ST[(ST["start_reason"] == "Tactical Shift") & ST["prev_position"].notna() & (ST["prev_position"] != ST["position"])].copy()
    sw["switch"] = sw["prev_position"].astype(str).map(abbr) + " -> " + sw["position"].astype(str).map(abbr)
    top = sw.groupby("switch").size().sort_values(ascending=False).head(20).reset_index(name="n")
    l, r = st.columns(2)
    l.plotly_chart(px.bar(top.iloc[::-1], x="n", y="switch", orientation="h", color_discrete_sequence=["#ffb703"], labels={"n": "times", "switch": ""}).update_layout(height=560, title="Most common positional switches"), use_container_width=True)
    pt = TS.groupby("team").size().rename("shifts").reset_index()
    nm = M.melt(id_vars=["match_id"], value_vars=["home_team", "away_team"], value_name="team").groupby("team")["match_id"].nunique().rename("matches").reset_index()
    pt = pt.merge(nm, on="team")
    pt = pt[pt["matches"] >= 4]
    pt["shifts_per_match"] = pt["shifts"] / pt["matches"]
    r.markdown("**Teams that change shape most** (4+ matches)")
    r.dataframe(pt.sort_values("shifts_per_match", ascending=False).head(20).round(2), hide_index=True, use_container_width=True)
    TS["bin"] = (TS["minute"] // 10 * 10).clip(upper=120)
    st.plotly_chart(px.histogram(TS, x="bin", nbins=13, labels={"bin": "minute"}, color_discrete_sequence=["#4cc9f0"]).update_layout(height=320, title="When do teams change shape?"), use_container_width=True)

with t5:
    k = st.columns(4)
    k[0].metric("Yellow cards", int((CD["card_type"] == "Yellow Card").sum()))
    k[1].metric("Second yellows", int((CD["card_type"] == "Second Yellow").sum()))
    k[2].metric("Red cards", int((CD["card_type"] == "Red Card").sum()))
    k[3].metric("Cards per match", f"{len(CD) / max(len(mids), 1):.2f}")
    CD["bin"] = (CD["minute"] // 5 * 5).clip(upper=120)
    st.plotly_chart(px.histogram(CD, x="bin", color="card_type", nbins=25, barmode="stack", labels={"bin": "minute (5-minute bins)"}, color_discrete_map={"Yellow Card": "#ffd166", "Second Yellow": "#fb5607", "Red Card": "#ff006e"})
                    .update_layout(height=380, title="When do players get booked?"), use_container_width=True)
    cm = CD.merge(ST[["match_id", "player", "position", "from_min", "to_min"]], on=["match_id", "player"], how="left")
    cm = cm[(cm["minute"] >= cm["from_min"] - 0.01) & (cm["minute"] <= cm["to_min"] + 0.01)].drop_duplicates(["match_id", "player", "minute"])
    cm["role"] = cm["position"].map(role_group)
    rr = cm.groupby("role").size().reset_index(name="cards")
    l, r = st.columns(2)
    l.plotly_chart(px.bar(rr, x="role", y="cards", labels={"role": "role at the moment of the card"}, color_discrete_sequence=["#ff6b81"]).update_layout(height=340, title="Cards by role"), use_container_width=True)
    reasons = CD.groupby("reason").size().sort_values(ascending=False).head(10).reset_index(name="cards")
    r.plotly_chart(px.bar(reasons.iloc[::-1], x="cards", y="reason", orientation="h", color_discrete_sequence=["#9aa5b1"], labels={"reason": ""}).update_layout(height=340, title="Reasons"), use_container_width=True)
    tc = CD.assign(tc=CD["team"] + " | " + CD["match_id"].map(cs)).groupby("tc").agg(cards=("minute", "size"), reds=("card_type", lambda s: int((s != "Yellow Card").sum())), matches=("match_id", "nunique")).reset_index()
    tc["cards_per_match"] = tc["cards"] / tc["matches"]
    tc = tc[tc["matches"] >= 4]
    st.markdown("**Most-booked teams** (4+ matches with a card)")
    st.dataframe(tc.sort_values("cards_per_match", ascending=False).head(20).round(2).rename(columns={"tc": "team | competition"}), hide_index=True, use_container_width=True)
    st.caption("Counts only the teams' matches in which at least one card was shown, so cards per match is a slight over-estimate for clean teams.")
