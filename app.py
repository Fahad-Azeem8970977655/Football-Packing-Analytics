import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch import add_caption, add_player, new_pitch
from ui import card, pills, setup
from utils import load_actions, load_league_shots, load_matches, load_player_metrics

setup("Packing Lab", "⚽", "Packing, line-breaking and expected threat from StatsBomb 360 freeze frames - plus events-only analysis of Wyscout and StatsBomb leagues.")
a = load_actions()
m = load_matches()
pm = load_player_metrics()
ls = load_league_shots()

c = st.columns(6)
c[0].metric("Matches", f"{len(m):,}")
c[1].metric("Passes & carries", f"{len(a):,}")
c[2].metric("With freeze frame", f"{a['has_frame'].mean():.0%}", help="Share of actions where the positions of the players are known.")
c[3].metric("Competition-seasons", f"{pm['comp'].nunique()}")
c[4].metric("Player-seasons", f"{len(pm):,}")
c[5].metric("Shots with xG", f"{len(ls):,}")
pills("StatsBomb Open Data", "Wyscout (CC BY 4.0)", "360 freeze frames", "xT • xPass • xPack", "Pitch arrows & positions")
st.caption("Light or dark: open the app menu (three dots, top right) > Settings > Theme and choose Light, Dark or Use system setting. Charts, cards and tables adapt to both.")

t1, t2, t3, t4 = st.tabs(["Start here", "Packing explained", "Data & coverage", "Glossary"])

with t1:
    groups = [
        ("Look at the pitch", [("pages/1_Pass_and_Carry_Map.py", "Pass & Carry Map", "🗺️", "Arrows on a real pitch, coloured by packing; zone heatmaps."),
                               ("pages/2_Freeze_Frame_Viewer.py", "Freeze-Frame Viewer", "📸", "Every player's position at the moment of a pass; bypassed defenders and defensive lines."),
                               ("pages/3_Team_Shape_and_Network.py", "Team Shape & Network", "🕸️", "Average positions labelled by role, and who passes to whom.")]),
        ("Players & teams", [("pages/4_Player_Leaderboard.py", "Player Leaderboards", "🏅", "Packing per 90 and an events-only scouting view with ages and roles."),
                             ("pages/5_Team_Analysis.py", "Team Analysis", "🛡️", "Packing created vs conceded, and how it builds through a match."),
                             ("pages/6_Game_State.py", "Game State", "⏱️", "How teams change when trailing or leading.")]),
        ("Machine learning", [("pages/7_xPack_Model.py", "xPack", "📈", "Expected packing and who beats it."),
                              ("pages/8_Pass_Value_xPass.py", "Pass Value (xPass)", "🎯", "Completion probability x packing = risk-adjusted passing."),
                              ("pages/9_Pass_Types_ML.py", "Pass Types", "🧩", "Clustering every pass into types."),
                              ("pages/10_Expected_Threat_xT.py", "Expected Threat (xT)", "⚡", "A value for every zone, and for every pass and carry."),
                              ("pages/11_Player_Archetypes.py", "Player Archetypes", "🧬", "Roles from PCA + KMeans, similar-player search.")]),
        ("Results & context", [("pages/12_Packing_to_Shots.py", "Packing -> Shots & Goals", "🥅", "Does bypassing opponents create chances?"),
                               ("pages/13_Match_Outcome_Model.py", "Match Outcome Model", "🏆", "Does packing explain who wins beyond xG?"),
                               ("pages/14_Competitions_and_Leagues.py", "Competitions & Leagues", "🌍", "Tournaments and the top-5 leagues side by side.")]),
        ("Space, shape & pressure", [("pages/15_Pitch_Control_and_Space.py", "Pitch Control & Space", "🧭", "Voronoi control map and receiver advantage."),
                                     ("pages/16_Defensive_Lines.py", "Defensive Lines", "🧱", "How high and compact defences are, and when they break."),
                                     ("pages/20_Receivers_and_Targets.py", "Receivers & Targets", "🧲", "Who receives between the lines and in behind."),
                                     ("pages/21_Pressure_Resistance.py", "Pressure Resistance", "🧯", "Who beats the model under pressure.")]),
        ("Defending, shooting & game management", [("pages/22_Pressing_and_Counter_Pressing.py", "Pressing & Counter-pressing", "🔥", "Press height, PPDA, counter-press success and high regains."),
                                                   ("pages/23_Shooting_and_Finishing.py", "Shooting & Finishing", "🥅", "Shot maps, goal-mouth placement and post-shot xG."),
                                                   ("pages/24_Expected_League_Tables.py", "Expected League Tables", "🏟️", "xPoints, luck and a season simulator."),
                                                   ("pages/25_Chain_Value.py", "Chain Value", "⛓️", "xG chain and buildup for every player in the move."),
                                                   ("pages/26_Substitutions_and_Tactical_Shifts.py", "Substitutions & Shifts", "🔄", "Sub impact, best substitutes, positional switches, cards.")]),
        ("Sequences, set pieces & timing", [("pages/17_Possession_Sequences.py", "Possession Sequences", "🔗", "Numbered build-ups to shots and goals."),
                                            ("pages/18_Set_Pieces.py", "Set Pieces", "🚩", "Corners, free kicks, throw-ins and goal kicks."),
                                            ("pages/19_Match_Momentum.py", "Match Momentum & Tempo", "📊", "Minute-by-minute control and the tempo curve.")]),
    ]
    for title, items in groups:
        st.subheader(title)
        cols = st.columns(len(items))
        for col, (path, label, icon, desc) in zip(cols, items):
            with col:
                st.page_link(path, label=label, icon=icon)
                st.caption(desc)

with t2:
    l, r = st.columns([3, 2])
    fig = new_pitch("A pass that bypasses three defenders", 600)
    fig.add_shape(type="rect", x0=42, x1=74, y0=0, y1=80, fillcolor="rgba(255,255,255,0.14)", line_width=0)
    ours = [(34, 55, "T"), (30, 25, "T"), (76, 36, "T"), (88, 58, "T"), (60, 70, "T")]
    byp = [(50, 28), (58, 46), (66, 36)]
    out = [(84, 22), (92, 44), (96, 66), (40, 40)]
    fig.add_trace(go.Scatter(x=[o[0] for o in ours], y=[o[1] for o in ours], mode="markers+text", text=["T"] * len(ours), textfont=dict(size=10, color="#111", family="Arial Black"), name="Teammates",
                             marker=dict(color="#4cc9f0", size=22, line=dict(color="white", width=1))))
    fig.add_trace(go.Scatter(x=[o[0] for o in out], y=[o[1] for o in out], mode="markers+text", text=["O"] * len(out), textfont=dict(size=10, color="#111", family="Arial Black"), name="Opponents (not bypassed)",
                             marker=dict(color="#ff8fa3", size=22, line=dict(color="white", width=1))))
    fig.add_trace(go.Scatter(x=[o[0] for o in byp], y=[o[1] for o in byp], mode="markers+text", text=["X"] * 3, textfont=dict(size=11, color="white", family="Arial Black"), name="Opponents BYPASSED",
                             marker=dict(color="#d00000", size=26, line=dict(color="white", width=3))))
    add_player(fig, 36, 40, "CM", "#06d6a0", symbol="star", size=40, name="Passer", showlegend=True)
    fig.add_annotation(x=74, y=36, ax=36, ay=40, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=4, arrowcolor="#ffd166")
    add_caption(fig, 58, 8, "Opponents between the passer and the receiver = packing 3", size=13)
    fig.add_annotation(x=58, y=76, text="space bypassed", showarrow=False, font=dict(color="white", size=11))
    l.plotly_chart(fig, use_container_width=True)
    with r:
        card("<b>Packing</b> counts the outfield opponents a pass or carry takes out of the game: they were ahead of the ball when it started and level with or behind it when it ended.")
        card("<b>Line-breaking</b> means breaking a defensive or midfield line. The lines are estimated from the deepest visible opponents.")
        card("<b>Why freeze frames?</b> StatsBomb 360 records where visible players stood at the moment of each event, so we can count opponents instead of guessing from the ball alone.")
        card("<b>Caution:</b> only players the camera saw are in a frame, so packing is a <i>minimum</i>. Values are best used to compare players and teams with each other.")

with t3:
    cov = (a.assign(cs=a["competition"].astype(str) + " " + a["season"].astype(str)).groupby("cs")
           .agg(matches=("match_id", "nunique"), actions=("id", "size"), with_frame=("has_frame", "mean"), avg_visible_opponents=("n_opp_vis", "mean")).reset_index())
    cov["with_frame"] = (100 * cov["with_frame"]).round(1)
    l, r = st.columns([2, 3])
    l.dataframe(cov.sort_values("with_frame", ascending=False).rename(columns={"with_frame": "% with freeze frame"}).round(1), hide_index=True, use_container_width=True)
    r.plotly_chart(px.bar(cov.sort_values("with_frame"), x="with_frame", y="cs", orientation="h", text_auto=".0f", color="with_frame", color_continuous_scale="Tealgrn",
                          labels={"with_frame": "% of actions with a freeze frame", "cs": ""}).update_layout(height=430, coloraxis_showscale=False), use_container_width=True)
    st.info("AFCON 2023 has no usable freeze frames (51 of 52 matches are unscheduled for 360), so it has no packing. The 2015/16 league seasons, Copa America 2024 and all "
            "Wyscout competitions are analysed through event proxies on the events-only pages (no packing). See DATA_SOURCES.md for licences and attribution.")

with t4:
    for title, text in [
        ("Packing", "Number of outfield opponents bypassed by a pass or carry (ahead of the ball at the start, level or behind at the end)."),
        ("Line-breaking action", "An action with packing of at least 1. Defensive/midfield line breaks use the 4 deepest / next 4 visible opponents."),
        ("xT - Expected Threat", "Value of a zone = chance of scoring soon after the ball is there. Action xT = value gained by moving the ball."),
        ("xPass", "Model probability that a pass is completed, from geometry and the freeze frame."),
        ("xPack", "Model expectation of the packing a typical action creates from its situation. Packing over expected shows who beats it."),
        ("Pass types", "Clusters of passes found by KMeans on start location, direction and length."),
        ("Position labels", "GK goalkeeper, CB/LCB/RCB centre-backs, LB/RB full-backs, LWB/RWB wing-backs, CDM/CM/CAM midfielders, LW/RW wingers, CF/ST forwards."),
        ("Per 90", "A total divided by minutes played, times 90."),
    ]:
        with st.expander(title):
            st.write(text)
