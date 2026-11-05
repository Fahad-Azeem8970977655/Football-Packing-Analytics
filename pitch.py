"""Realistic football pitch (StatsBomb 120 x 80 yards) with goals, nets, arrows and labelled players.

geometry() is pure python (no plotting library) so it can be tested; new_pitch() turns it into a Plotly figure.
The acting team always attacks left -> right (x = 120 is the opponents' goal line).
"""
import numpy as np

GRASS = ("#2f8a3e", "#297a36")        # alternating mowing stripes
SURROUND = "#143d1f"
LINE = "rgba(255,255,255,0.95)"
NET = "rgba(255,255,255,0.40)"

POS_ABBR = {
    "Goalkeeper": "GK", "Right Back": "RB", "Left Back": "LB", "Right Center Back": "RCB", "Left Center Back": "LCB", "Center Back": "CB",
    "Right Wing Back": "RWB", "Left Wing Back": "LWB", "Right Defensive Midfield": "RDM", "Left Defensive Midfield": "LDM",
    "Center Defensive Midfield": "CDM", "Right Midfield": "RM", "Left Midfield": "LM", "Right Center Midfield": "RCM",
    "Left Center Midfield": "LCM", "Center Midfield": "CM", "Right Attacking Midfield": "RAM", "Left Attacking Midfield": "LAM",
    "Center Attacking Midfield": "CAM", "Right Wing": "RW", "Left Wing": "LW", "Right Center Forward": "RCF", "Left Center Forward": "LCF",
    "Center Forward": "CF", "Secondary Striker": "SS", "Striker": "ST"}


def abbr(pos):
    if not isinstance(pos, str):
        return "?"
    return POS_ABBR.get(pos, "".join(w[0] for w in pos.split()).upper())


def _arc(cx, cy, r, a0, a1, n=60):
    t = np.radians(np.linspace(a0, a1, n))
    return (cx + r * np.cos(t)).tolist(), (cy + r * np.sin(t)).tolist()


def _rect(x0, y0, x1, y1):
    return [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]


def geometry():
    """Return dict(lines=[(xs, ys, width)], net=[(xs, ys)], posts=[(x, y)], spots=[(x, y)], stripes=[(x0, x1, colour)])."""
    lines = []
    for r in (_rect(0, 0, 120, 80), _rect(0, 18, 18, 62), _rect(102, 18, 120, 62), _rect(0, 30, 6, 50), _rect(114, 30, 120, 50)):
        lines.append((r[0], r[1], 2.2))
    lines.append(([60, 60], [0, 80], 2.2))
    lines.append(_arc(60, 40, 10, 0, 360, 120) + (2.2,))
    lines.append(_arc(12, 40, 10, -53.13, 53.13) + (2.2,))          # penalty arcs ("D")
    lines.append(_arc(108, 40, 10, 126.87, 233.13) + (2.2,))
    for cx, cy, a0, a1 in ((0, 0, 0, 90), (0, 80, -90, 0), (120, 0, 90, 180), (120, 80, 180, 270)):
        lines.append(_arc(cx, cy, 1, a0, a1, 16) + (2.2,))
    # goals: 8 yards wide (y 36-44), 2.4 yards deep behind each goal line
    for gx0, gx1 in ((-2.4, 0), (120, 122.4)):
        lines.append((_rect(gx0, 36, gx1, 44)[0], _rect(gx0, 36, gx1, 44)[1], 3.2))
    net = []
    for gx0, gx1 in ((-2.4, 0), (120, 122.4)):
        for y in np.arange(37, 44, 1.0):
            net.append(([gx0, gx1], [float(y), float(y)]))
        for x in np.arange(gx0 + 0.6, gx1, 0.6):
            net.append(([float(x), float(x)], [36, 44]))
    posts = [(0, 36), (0, 44), (120, 36), (120, 44)]
    spots = [(60, 40), (12, 40), (108, 40)]
    stripes = [(i * 12, (i + 1) * 12, GRASS[i % 2]) for i in range(10)]
    return dict(lines=lines, net=net, posts=posts, spots=spots, stripes=stripes)


def new_pitch(title="", height=600, half=False, show_goals=True):
    """Plotly figure with the pitch drawn. half=True zooms on the attacking half."""
    import plotly.graph_objects as go
    g = geometry()
    fig = go.Figure()
    shapes = [dict(type="rect", x0=a, x1=b, y0=0, y1=80, fillcolor=c, line=dict(width=0), layer="below") for a, b, c in g["stripes"]]
    fig.update_layout(
        shapes=shapes, title=dict(text=title, x=0.01, font=dict(size=17)), height=height, plot_bgcolor=SURROUND, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=52 if title else 12, b=8),
        xaxis=dict(range=[56, 125] if half else [-5, 125], visible=False, fixedrange=False),
        yaxis=dict(range=[84, -4], visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-0.02, bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(font_size=13))
    for xs, ys, w in g["lines"]:
        if not show_goals and xs[0] in (-2.4, 120) and w == 3.2:
            continue
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=LINE, width=w), hoverinfo="skip", showlegend=False))
    if show_goals:
        nx, ny = [], []
        for xs, ys in g["net"]:
            nx += xs + [None]
            ny += ys + [None]
        fig.add_trace(go.Scatter(x=nx, y=ny, mode="lines", line=dict(color=NET, width=1), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[p[0] for p in g["posts"]], y=[p[1] for p in g["posts"]], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(color="white", size=8, line=dict(color="#222", width=1))))
    fig.add_trace(go.Scatter(x=[s[0] for s in g["spots"]], y=[s[1] for s in g["spots"]], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(color="white", size=5)))
    return fig


BUCKETS = [("0", 0, 0, "#9aa5b1"), ("1", 1, 1, "#ffc400"), ("2", 2, 2, "#ffb703"), ("3-4", 3, 4, "#fb5607"), ("5+", 5, 99, "#ff006e")]


def add_arrows(fig, d, value_col="pack_used", max_n=220, color_by="buckets", width=2.4, label_col=None):
    """Draw passes/carries as real arrows (start -> end). Completed = solid arrow; failed passes = dotted line.
    d needs sx, sy, ex, ey (+ value_col, 'completed', 'player', 'minute' for hover text)."""
    import plotly.graph_objects as go
    d = d.head(max_n)
    done = d["completed"].astype(bool) if "completed" in d else np.ones(len(d), bool)
    for r, ok in zip(d.itertuples(), done):
        v = getattr(r, value_col, 0) if value_col else 0
        v = 0 if v != v else v
        col = next((c for _, lo, hi, c in BUCKETS if lo <= v <= hi), BUCKETS[-1][3])
        if not ok:
            fig.add_trace(go.Scatter(x=[r.sx, r.ex], y=[r.sy, r.ey], mode="lines", line=dict(color="rgba(255,255,255,0.45)", width=1.4, dash="dot"),
                                     hoverinfo="skip", showlegend=False))
            continue
        fig.add_annotation(x=r.ex, y=r.ey, ax=r.sx, ay=r.sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                           arrowsize=1.1, arrowwidth=width if v else width - 1, arrowcolor=col, opacity=0.95)
    mid = d[done.to_numpy()] if len(d) else d
    if len(mid):
        txt = (mid["player"].astype(str) + " | " + mid["minute"].astype(str) + "' | " + value_col + " " + mid[value_col].astype(str)) if value_col in mid else mid["player"].astype(str)
        fig.add_trace(go.Scatter(x=(mid["sx"] + mid["ex"]) / 2, y=(mid["sy"] + mid["ey"]) / 2, mode="markers", showlegend=False, hoverinfo="text", text=txt,
                                 marker=dict(size=9, color="rgba(0,0,0,0)")))
    for name, lo, hi, c in BUCKETS:        # legend entries for the colour buckets
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=c, width=4), name=f"Packing {name}"))
    return fig


def add_player(fig, x, y, text, color="#ffd166", symbol="circle", size=26, name=None, hover=None, textcolor="#111", showlegend=False, outline="white"):
    """Marker with a short label inside (e.g. the position 'RCM'); hover shows the long text."""
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[text], textfont=dict(size=10, color=textcolor, family="Arial Black"),
                             textposition="middle center", name=name or text, showlegend=showlegend, hoverinfo="text" if hover else "skip", hovertext=hover,
                             marker=dict(color=color, size=size, symbol=symbol, line=dict(color=outline, width=2))))
    return fig


def add_caption(fig, x, y, text, color="white", size=12, bg="rgba(0,0,0,0.55)"):
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color=color, size=size), bgcolor=bg, bordercolor="rgba(255,255,255,0.4)", borderpad=3)
    return fig


ROLE_COLORS = {"GK": "#ffd166", "DF": "#4cc9f0", "MD": "#06d6a0", "FW": "#ff6b6b"}


def role_group(pos):
    if not isinstance(pos, str):
        return "MD"
    if "Goalkeeper" in pos:
        return "GK"
    if "Back" in pos:
        return "DF"
    if "Midfield" in pos:
        return "MD"
    return "FW"


def short_name(n):
    parts = str(n).split()
    return parts[-1] if parts else str(n)


def lerp_color(c0, c1, t):
    t = min(max(float(t), 0.0), 1.0)
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(a[k] + (b[k] - a[k]) * t):02x}" for k in range(3))


GOAL_ZONES_Y = (36.0, 38.667, 41.333, 44.0)      # left post -> right post, seen from behind the shooter
GOAL_ZONES_Z = (0.0, 0.889, 1.778, 2.667)        # ground -> crossbar (yards)


def new_goal(title="", height=430):
    """The goal frame seen from the shooter's side: posts at y = 36 and 44, crossbar at 2.67 yards."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=17)), height=height, plot_bgcolor="#143d1f", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=8, r=8, t=52 if title else 12, b=8), xaxis=dict(range=[32, 48], visible=False),
                      yaxis=dict(range=[-0.5, 3.6], visible=False, scaleanchor="x", scaleratio=1), legend=dict(orientation="h", y=-0.02, bgcolor="rgba(0,0,0,0)"))
    nx, ny = [], []
    for y in np.arange(36.5, 44, 0.5):
        nx += [float(y), float(y), None]
        ny += [0, 2.667, None]
    for z in np.arange(0.4, 2.667, 0.4):
        nx += [36, 44, None]
        ny += [float(z), float(z), None]
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="lines", line=dict(color=NET, width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[36, 36, 44, 44], y=[0, 2.667, 2.667, 0], mode="lines", line=dict(color="white", width=7), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[32, 48], y=[0, 0], mode="lines", line=dict(color="white", width=3), hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=32.4, y=0.25, text="left post is on the left, as the shooter sees the goal", showarrow=False, font=dict(color="#b9c7de", size=10), xanchor="left")
    return fig
