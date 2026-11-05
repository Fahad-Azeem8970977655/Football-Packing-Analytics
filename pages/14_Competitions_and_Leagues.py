import streamlit as st

from ui import setup
from views import competition_compare, leagues_events

setup("Competitions & Leagues", "🌍", "Compare tournaments and top leagues: style, progression, chances created - with packing where freeze frames exist, event proxies everywhere else.")
t1, t2 = st.tabs(["Freeze-frame competitions (packing)", "Top-5 leagues & tournaments (events only)"])
with t1:
    competition_compare.render()
with t2:
    leagues_events.render()
