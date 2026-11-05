"""Packing / line-breaking maths (pure numpy + pandas, no Streamlit).

Coordinates are StatsBomb 120x80 with the *acting team always attacking left -> right*,
so the opponent goal line is at x = 120.

Packing  = number of OUTFIELD opponents that are "bypassed": ahead of the ball at the start
           of the action (x > start_x) and level with / behind it at the end (x <= end_x).
Lines    = opponents are ranked by depth (largest x = deepest). The 4 deepest outfield
           opponents approximate the DEFENSIVE line, ranks 5-8 the MIDFIELD line. A line is
           "broken" if start_x < line_x <= end_x (line_x = mean x of the players in the line;
           needs >= 3 visible players in that band, otherwise not measurable).
"""
import numpy as np
import pandas as pd


def packing_chunk(fr: pd.DataFrame, pos: np.ndarray, sx: np.ndarray, ex: np.ndarray) -> pd.DataFrame:
    """fr  : freeze-frame rows (event_uuid, x, y, teammate, keeper)
    pos : for each row of `fr`, the integer position of its action in sx/ex (-1 = ignore)
    sx, ex : start / end x of every action. Returns one row per action, indexed by position."""
    keep = pos >= 0
    if not keep.any():
        return pd.DataFrame()
    fr = fr.loc[keep]
    pos = pos[keep]
    opp = ~fr["teammate"].fillna(False).to_numpy(dtype=bool)
    gk = fr["keeper"].fillna(False).to_numpy(dtype=bool)
    x = fr["x"].to_numpy(dtype=float)
    s, e = sx[pos], ex[pos]
    out = opp & ~gk
    byp = (x > s) & (x <= e)
    df = pd.DataFrame({"pos": pos, "x": x, "opp_out": out,
                       "byp_out": out & byp, "byp_gk": opp & gk & byp})
    agg = df.groupby("pos").agg(n_opp_vis=("opp_out", "sum"),
                                pack=("byp_out", "sum"),
                                pack_gk=("byp_gk", "sum"))
    o = df[df["opp_out"]].copy()
    o["rank"] = o.groupby("pos")["x"].rank(method="first", ascending=False)
    s_a, e_a = sx[agg.index.to_numpy()], ex[agg.index.to_numpy()]
    for name, lo, hi in (("def", 1, 4), ("mid", 5, 8)):
        band = o[(o["rank"] >= lo) & (o["rank"] <= hi)].groupby("pos")["x"].agg(["mean", "count"])
        band = band[band["count"] >= 3]
        lx = pd.Series(np.nan, index=agg.index)
        lx.loc[band.index] = band["mean"]
        agg[f"breaks_{name}"] = ((s_a < lx.to_numpy()) & (e_a >= lx.to_numpy())).astype(np.int8)
    agg["lines_broken"] = agg["breaks_def"] + agg["breaks_mid"]
    for c in ("n_opp_vis", "pack", "pack_gk"):
        agg[c] = agg[c].astype(np.int16)
    return agg


def shape_features_chunk(fr, pos, ex, ey):
    """Opponent shape + receiver support from the freeze frame (visible players only).
    def_line_x   : mean x of the 4 deepest visible outfield opponents (>= 4 needed)
    def_line_width: y-spread of those 4
    mid_line_x   : mean x of the next 4 (>= 3 needed)
    opp_block_length / opp_block_width : x- and y-extent of all visible outfield opponents (>= 6 needed)
    min_tm_dist_end : distance from the action's destination to the nearest teammate (not the player on the ball)"""
    keep = pos >= 0
    if not keep.any():
        return pd.DataFrame()
    fr = fr.loc[keep]
    pos = pos[keep]
    opp = ~fr["teammate"].fillna(False).to_numpy(dtype=bool)
    gk = fr["keeper"].fillna(False).to_numpy(dtype=bool)
    act = fr["actor"].fillna(False).to_numpy(dtype=bool)
    x = fr["x"].to_numpy(dtype=float)
    y = fr["y"].to_numpy(dtype=float)
    df = pd.DataFrame({"pos": pos, "x": x, "y": y, "out": opp & ~gk})
    o = df[df["out"]].copy()
    o["rank"] = o.groupby("pos")["x"].rank(method="first", ascending=False)
    d4 = o[o["rank"] <= 4].groupby("pos").agg(def_line_x=("x", "mean"), n4=("x", "size"), y4min=("y", "min"), y4max=("y", "max"))
    d4 = d4[d4["n4"] >= 4]
    d4["def_line_width"] = d4["y4max"] - d4["y4min"]
    m4 = o[(o["rank"] >= 5) & (o["rank"] <= 8)].groupby("pos").agg(mid_line_x=("x", "mean"), nm=("x", "size"))
    m4 = m4[m4["nm"] >= 3]
    blk = o.groupby("pos").agg(xmin=("x", "min"), xmax=("x", "max"), ymin=("y", "min"), ymax=("y", "max"), n=("x", "size"))
    blk = blk[blk["n"] >= 6]
    blk["opp_block_length"] = blk["xmax"] - blk["xmin"]
    blk["opp_block_width"] = blk["ymax"] - blk["ymin"]
    tm = (~opp) & (~act)
    dd = np.where(tm, np.hypot(x - ex[pos], y - ey[pos]), 1e6)
    t = pd.DataFrame({"pos": pos, "d": dd}).groupby("pos")["d"].min()
    t = t.where(t < 1e6).rename("min_tm_dist_end")
    out = pd.concat([d4[["def_line_x", "def_line_width"]], m4[["mid_line_x"]], blk[["opp_block_length", "opp_block_width"]], t], axis=1)
    return out.astype(np.float32)
