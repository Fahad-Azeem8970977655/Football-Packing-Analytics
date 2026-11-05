import streamlit as st

from ui import setup
from views import leaderboard_packing, players_events

setup("Player Leaderboards", "🏅", "Packing, line-breaking and threat per 90 for freeze-frame competitions - plus an events-only scouting view across Wyscout and StatsBomb.")
t1, t2 = st.tabs(["Packing leaderboard (freeze-frame competitions)", "Events-only players (all sources, ages & roles)"])
with t1:
    leaderboard_packing.render()
with t2:
    players_events.render()
