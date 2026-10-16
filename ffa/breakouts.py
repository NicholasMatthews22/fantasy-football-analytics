"""Breakout detector.

Idea: usage leads, points follow. Fantasy points swing a lot week to week
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


def find_breakouts(pw, active_ids, min_games=4):
    pw = add_xfp(pw)
    cur = pw[(pw["season"] == CURRENT_SEASON) & pw["player_id"].isin(active_ids)]
    cur = cur.sort_values(["player_id", "week"])

    rows = []
    for pid, g in cur.groupby("player_id"):
        if len(g) < min_games:
            continue
        last3, prior = g.tail(3), g.iloc[:-3]
        r = g.iloc[-1]
        row = {
            "player_id": pid, "name": r["name"], "position": r["position"], "team": r["team"],
            "headshot_url": r["headshot_url"], "games": len(g),
            "xfp_last3": last3["xfp"].mean(), "xfp_before": prior["xfp"].mean(),
            "ppr_last3": last3["ppr"].mean(),
            "opps_last3": last3["opportunities"].mean(), "opps_before": prior["opportunities"].mean(),
            "tgt_share_last3": last3["target_share"].mean(), "tgt_share_before": prior["target_share"].mean(),
            "carry_share_last3": last3["carry_share"].mean(), "carry_share_before": prior["carry_share"].mean(),
        }
        rows.append(row)
    b = pd.DataFrame(rows)
    if b.empty:
        return b, cur
    b["usage_change"] = b["xfp_last3"] - b["xfp_before"]
    b["points_gap"] = b["xfp_last3"] - b["ppr_last3"]
    b["breakout_score"] = b["usage_change"] + 0.5 * b["points_gap"].clip(lower=0)
    b = b[b["xfp_last3"] >= 6].sort_values("breakout_score", ascending=False)

    weekly = cur[["player_id", "week", "ppr", "xfp", "opportunities", "target_share", "carry_share"]]
    return b.reset_index(drop=True), weekly
