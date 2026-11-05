"""Shared look & feel: page setup, hero header, cards, chart theme."""
import plotly.io as pio
import streamlit as st

ACCENT, ACCENT2, WARM, HOT = "#00b894", "#4cc9f0", "#ffb703", "#ff006e"
COLORWAY = [ACCENT, ACCENT2, WARM, HOT, "#8338ec", "#fb5607", "#3a86ff", "#90be6d", "#f9c74f", "#f94144"]

CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px;}
/* hero banner: dark gradient with light text - readable on a light OR a dark page */
.hero {background: linear-gradient(120deg, #0f3d2e 0%, #14324f 55%, #2b1a4a 100%); border: 1px solid rgba(128,128,128,0.35);
       border-radius: 18px; padding: 22px 28px 16px 28px; margin-bottom: 18px; box-shadow: 0 6px 24px rgba(0,0,0,0.25);}
.hero h1 {margin: 0; font-size: 2.0rem; letter-spacing: 0.3px; color: #ffffff !important;}
.hero p {margin: 6px 0 0 0; color: #d4def0 !important; font-size: 1.0rem;}
/* neutral greys with transparency work on both themes */
[data-testid="stMetric"] {background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.28); border-radius: 14px; padding: 12px 16px;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
[data-testid="stMetricLabel"] p {opacity: 0.78; font-size: 0.82rem;}
h2, h3 {letter-spacing: 0.2px;}
h3 {border-left: 4px solid #00b894; padding-left: 10px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: rgba(128,128,128,0.12); border-radius: 10px 10px 0 0; padding: 8px 16px;}
.stTabs [aria-selected="true"] {background: rgba(0,184,148,0.22);}
.card {background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.28); border-radius: 14px; padding: 14px 18px; margin-bottom: 10px;}
.card b {color: #00a383;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; background: rgba(0,184,148,0.20); color:#008f74; font-weight:600; font-size: 0.78rem; margin-right: 6px;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,0.25);}
</style>
"""


def _neutral_template():
    """Transparent backgrounds + grey gridlines: the chart sits on the page and Streamlit's theme supplies the text colour,
    so the same figure reads correctly on a light or a dark page."""
    grid, axis = "rgba(128,128,128,0.25)", "rgba(128,128,128,0.45)"
    ax = dict(gridcolor=grid, zerolinecolor=axis, linecolor=axis)
    return dict(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORWAY, xaxis=ax, yaxis=ax,
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor=grid, linecolor=axis), angularaxis=dict(gridcolor=grid, linecolor=axis)),
        legend=dict(bgcolor="rgba(0,0,0,0)"), coloraxis=dict(colorbar=dict(outlinewidth=0))))


try:
    pio.templates["packing"] = _neutral_template()
    _ = pio.templates["packing"]                    # forces Plotly to validate the template now, not on the first chart
    pio.templates.default = "packing"
except Exception:                                   # never let styling break the app
    pio.templates.default = "plotly"


def setup(title, icon, subtitle=""):
    """Call first on every page: page config, CSS and a hero header."""
    st.set_page_config(page_title=f"{title} | Packing Lab", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(html):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


def pills(*items):
    st.markdown("".join(f'<span class="pill">{i}</span>' for i in items), unsafe_allow_html=True)
