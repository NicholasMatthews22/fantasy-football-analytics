"""Breakout detector.

Idea: usage leads, points follow. A breakout here is a RB, WR, or TE whose role is growing (more
targets and carries and a bigger share of snaps), not a backup QB who just got the starting job and
not a star who was already producing. Fantasy points swing a lot week to week
(touchdowns are random), but targets and carries are sticky. We estimate
"expected fantasy points" (xFP) from a player's opportunities alone, then flag
players whose xFP is rising and/or whose actual points haven't caught up yet.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import CURRENT_SEASON

XFP_INPUTS = {
    "QB": ["attempts", "carries"],
    "RB": ["carries", "targets", "receiving_air_yards"],
    "WR": ["targets", "receiving_air_yards", "carries"],
    "TE": ["targets", "receiving_air_yards"],
}


def add_xfp(pw):
    pw = pw.copy()
    pw["xfp"] = 0.0
    for pos, cols in XFP_INPUTS.items():
        m = pw["position"] == pos
        reg = LinearRegression().fit(pw.loc[m, cols], pw.loc[m, "ppr"])
        pw.loc[m, "xfp"] = reg.predict(pw.loc[m, cols]).clip(0)
    return pw


STAR_CUTOFF = {"RB": 36, "WR": 36, "TE": 12}   # top-N in points per game = already an established starter


def established_stars(pw, seasons_back=3):
    """Players who finished top-36 RB/WR or top-12 TE in points per game (6+ games) in any of the last
    few seasons. A star coming back from an injury or a slow start isn't a breakout."""
    stars = set()
    for yr in range(CURRENT_SEASON - seasons_back, CURRENT_SEASON):
        ppg = pw[pw["season"] == yr].groupby(["player_id", "position"])["ppr"].agg(["mean", "count"]).reset_index()
        ppg = ppg[ppg["count"] >= 6]
        for pos, n in STAR_CUTOFF.items():
            stars |= set(ppg[ppg["position"] == pos].nlargest(n, "mean")["player_id"])
    return stars


def find_breakouts(pw, active_ids, min_games=4):
    """RB/WR/TE whose role is growing: more targets and carries AND more snaps over their last 3 games
    than earlier this season. QBs are left out (a QB's usage only jumps when he takes over as the starter),
    and so are players who were already stars."""
    pw = add_xfp(pw)
    stars = established_stars(pw)
    cur = pw[(pw["season"] == CURRENT_SEASON) & pw["player_id"].isin(active_ids)]
    cur = cur.sort_values(["player_id", "week"])

    rows = []
    for pid, g in cur.groupby("player_id"):
        r = g.iloc[-1]
        if len(g) < min_games or r["position"] not in STAR_CUTOFF or pid in stars:
            continue
        last3, prior = g.tail(3), g.iloc[:-3]
        rows.append({
            "player_id": pid, "name": r["name"], "position": r["position"], "team": r["team"],
            "headshot_url": r["headshot_url"], "games": len(g),
            "xfp_last3": last3["xfp"].mean(), "xfp_before": prior["xfp"].mean(),
            "ppr_last3": last3["ppr"].mean(), "ppr_before": prior["ppr"].mean(),
            "opps_last3": last3["opportunities"].mean(), "opps_before": prior["opportunities"].mean(),
            "snap_last3": last3["snap_pct"].mean(), "snap_before": prior["snap_pct"].mean(),
            "tgt_share_last3": last3["target_share"].mean(), "tgt_share_before": prior["target_share"].mean(),
            "carry_share_last3": last3["carry_share"].mean(), "carry_share_before": prior["carry_share"].mean(),
        })
    b = pd.DataFrame(rows)
    if b.empty:
        return b, cur
    b["usage_change"] = b["xfp_last3"] - b["xfp_before"]
    b["snap_change"] = (b["snap_last3"] - b["snap_before"]).fillna(0)
    b["points_gap"] = b["xfp_last3"] - b["ppr_last3"]
    # a real breakout needs both: more of the offense running through him, and more time on the field
    b = b[(b["usage_change"] >= 1.5) & (b["snap_change"] >= 0.05) & (b["xfp_last3"] >= 7)]
    # skip guys already scoring like a weekly starter before the jump (a healthy star returning from injury)
    b = b[b["ppr_before"] < b["position"].map({"RB": 14, "WR": 14, "TE": 11})]
    b["breakout_score"] = b["usage_change"] + 15 * b["snap_change"] + 0.5 * b["points_gap"].clip(lower=0)
    b = b.sort_values("breakout_score", ascending=False)

    weekly = cur[["player_id", "week", "ppr", "xfp", "opportunities", "target_share", "carry_share", "snap_pct"]]
    return b.reset_index(drop=True), weekly
