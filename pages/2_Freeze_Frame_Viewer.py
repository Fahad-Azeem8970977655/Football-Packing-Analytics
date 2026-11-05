import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pitch import abbr, add_caption, add_player, new_pitch, role_group, ROLE_COLORS
from ui import card, setup
from utils import DATA, get_data, load_frames_top, load_matches

setup("Freeze-Frame Viewer", "📸", "The exact moment of a pass or carry: who was bypassed, where the defensive lines were, who was on the ball and who received it.")
base, df, opt = get_data()
matches = load_matches()

ml = matches[matches["match_id"].isin(df["match_id"].unique())].sort_values("match_date")
label = st.selectbox("Match", ml["label"].tolist())
mid = ml.loc[ml["label"] == label, "match_id"].iloc[0]
m_all = df[df["match_id"] == mid]
md = m_all[m_all["has_frame"]]
fp = DATA / "frames_by_match" / f"match_id={mid}"
if not fp.exists():                       # lightweight dataset: only each match's top-packing actions have frames
    ft = load_frames_top()
    ft = ft[ft["match_id"] == mid]
    md = md[md["id"].isin(ft["event_uuid"].unique())]
c = st.columns(3)
team = c[0].selectbox("Team", sorted(md["team"].astype(str).unique()))
md = md[md["team"].astype(str) == team]
min_pack = c[1].slider("Minimum packing", 0, 10, 2)
n = c[2].slider("Actions in list", 10, 100, 40)
cand = md[md["pack_used"] >= min_pack].sort_values("pack_used", ascending=False).head(n)
if cand.empty:
    st.info("No actions with that much packing in this match/team. Lower the slider."); st.stop()
labels = {r.id: f"{r.minute}' {r.player} -> {r.recipient if r.action_type == 'Pass' and pd.notna(r.recipient) else r.action_type} | {r.action_type} | packing {r.pack_used}"
          for r in cand.itertuples()}
eid = st.selectbox("Action", list(labels), format_func=labels.get)
a = cand[cand["id"] == eid].iloc[0]
fr = pd.read_parquet(fp) if fp.exists() else load_frames_top()
fr = fr[fr["event_uuid"] == eid].reset_index(drop=True)
if fr.empty:
    st.warning("No freeze frame stored for this action."); st.stop()

pos_of = m_all.groupby("player", observed=True)["position"].agg(lambda s: s.dropna().astype(str).mode().iat[0] if s.notna().any() else None)
opp = ~fr["teammate"]
out_opp = opp & ~fr["keeper"]
byp = out_opp & (fr["x"] > a.sx) & (fr["x"] <= a.ex)
xs = np.sort(fr.loc[out_opp, "x"].to_numpy())[::-1]
def_line = float(xs[:4].mean()) if len(xs) >= 4 else None
mid_line = float(xs[4:8].mean()) if len(xs) >= 7 else None

fig = new_pitch(f"{a.player} - {a.action_type} - packing {int(a.pack_used)}", 680)
fig.add_shape(type="rect", x0=a.sx, x1=a.ex, y0=0, y1=80, fillcolor="rgba(255,255,255,0.14)", line_width=0)
for lx, nm, col in ((def_line, "Defensive line", "#ff4d6d"), (mid_line, "Midfield line", "#ffb703")):
    if lx is not None:
        fig.add_shape(type="line", x0=lx, x1=lx, y0=0, y1=80, line=dict(color=col, width=2, dash="dash"))
        fig.add_annotation(x=lx, y=-2.2, text=nm, showarrow=False, font=dict(color=col, size=11))
fig.add_annotation(x=a.ex, y=a.ey, ax=a.sx, ay=a.sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=4, arrowcolor="#ffd166")


def group(mask, name, color, text, size=22, symbol="circle", line=1):
    s = fr[mask]
    if not s.empty:
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="markers+text", text=[text] * len(s), textfont=dict(size=10, color="#111", family="Arial Black"),
                                 name=f"{name} ({len(s)})", hoverinfo="text", hovertext=[name] * len(s),
                                 marker=dict(color=color, size=size, symbol=symbol, line=dict(color="white", width=line))))


group(fr["teammate"] & ~fr["actor"] & ~fr["keeper"], "Teammates", "#4cc9f0", "T")
group(fr["teammate"] & fr["keeper"], "Own goalkeeper", "#4cc9f0", "GK", 26, "diamond")
group(opp & ~byp & ~fr["keeper"], "Opponents (not bypassed)", "#ff8fa3", "O")
group(byp, "Opponents BYPASSED", "#d00000", "X", 26, "circle", 3)
group(opp & fr["keeper"], "Opponent goalkeeper", "#ff8fa3", "GK", 26, "diamond")

actor = fr[fr["actor"]]
p_pos = a.position
if not actor.empty:
    ar = actor.iloc[0]
    add_player(fig, ar["x"], ar["y"], abbr(p_pos), ROLE_COLORS[role_group(p_pos)], symbol="star", size=40, name="Player on ball", showlegend=True,
               hover=f"{a.player} ({p_pos})")
    add_caption(fig, ar["x"], ar["y"] + 5.5, f"<b>{a.player}</b><br>{abbr(p_pos)} - on the ball", size=11)
r_pos = None
if a.action_type == "Pass" and pd.notna(a.recipient) and bool(a.completed):
    tm = fr[fr["teammate"] & ~fr["actor"] & ~fr["keeper"]].copy()
    if not tm.empty:
        tm["d"] = np.hypot(tm["x"] - a.ex, tm["y"] - a.ey)
        rc = tm.sort_values("d").iloc[0]
        r_pos = pos_of.get(str(a.recipient))
        fig.add_trace(go.Scatter(x=[rc["x"]], y=[rc["y"]], mode="markers", name="Receiver", marker=dict(size=34, symbol="circle-open", color="#06d6a0", line=dict(width=4)),
                                 hoverinfo="text", hovertext=f"{a.recipient} ({r_pos})"))
        add_caption(fig, rc["x"], rc["y"] - 5.5, f"<b>{a.recipient}</b><br>{abbr(r_pos)} - receives", size=11, color="#06d6a0")
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    card(f"<b>On the ball</b><br>{a.player}<br>Position: {p_pos or 'n/a'} ({abbr(p_pos)})<br>Team: {a.team}")
with c2:
    card(f"<b>Receiver</b><br>{a.recipient if pd.notna(a.recipient) else '-'}<br>Position: {r_pos or 'n/a'} ({abbr(r_pos)})<br>Pass {'completed' if bool(a.completed) else 'failed'}")
with c3:
    card(f"<b>Impact</b><br>Opponents bypassed: <b>{int(a.pack_used)}</b> of {int(a.n_opp_vis)} visible<br>Defensive line: {'broken' if a.breaks_def == 1 else 'intact'} | Midfield line: {'broken' if a.breaks_mid == 1 else 'intact'}<br>Threat added (xT): {a.xt:.3f}")
st.caption("Red X = opponents bypassed (between the start and end of the action). Dashed lines = estimated defensive line (4 deepest visible outfield opponents) and midfield line. "
           "StatsBomb 360 only identifies the player on the ball and the goalkeepers: teammates (T) and opponents (O) are anonymous; the receiver is the teammate nearest the pass destination, "
           "named from the event data. Positions come from the lineup.")
