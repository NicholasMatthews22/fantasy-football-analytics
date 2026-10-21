"""Head-to-head test against a real benchmark: Sleeper's own weekly projections (supplied to Sleeper by
Rotowire), the numbers millions of fantasy players see in the app.

Both sets of projections are scored on the exact same player-games, against the same actual full-PPR
points from nflverse:
  - last season: our model trained only on earlier seasons (a true holdout season)
  - this season: the projections we saved before each week, plus walk-forward backfills
"""
import json
import urllib.request

import pandas as pd

URL = ("https://api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular"
       "&position[]=QB&position[]=RB&position[]=WR&position[]=TE")
RELEVANT = 6.0      # same filter as the backtest: players with recent form of 6+ points


def fetch_week(season, week):
    """Sleeper's PPR projection for every QB/RB/WR/TE that week: DataFrame(sleeper_id, sleeper_proj)."""
    req = urllib.request.Request(URL.format(season=season, week=week), headers={"User-Agent": "RedZoneLab/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        rows = json.load(r)
    out = [{"sleeper_id": str(x["player_id"]), "sleeper_proj": float((x.get("stats") or {}).get("pts_ppr"))}
           for x in rows if x.get("player_id") and (x.get("stats") or {}).get("pts_ppr") is not None]
    return pd.DataFrame(out, columns=["sleeper_id", "sleeper_proj"]).drop_duplicates("sleeper_id")


def update_store(store, wanted, refresh=(), fetch=fetch_week):
    """store: DataFrame(season, week, sleeper_id, sleeper_proj). Fetches any (season, week) in `wanted`
    that isn't stored yet, and re-fetches the ones in `refresh` (the upcoming week, until kickoff)."""
    have = set(map(tuple, store[["season", "week"]].drop_duplicates().astype(int).values)) if len(store) else set()
    todo = [sw for sw in wanted if sw not in have or sw in refresh]
    new = []
    for season, week in todo:
        try:
            df = fetch(season, week)
        except Exception as e:      # Sleeper down: keep what we have, try again next run
            print(f"  Sleeper projections unavailable for {season} week {week}: {e}")
            continue
        if len(df):
            new.append(df.assign(season=season, week=week))
    if not new:
        return store
    new = pd.concat(new, ignore_index=True)
    keys = set(map(tuple, new[["season", "week"]].drop_duplicates().values))
    keep = store[[(s, w) not in keys for s, w in zip(store["season"], store["week"])]] if len(store) else store
    return pd.concat([keep, new], ignore_index=True)[["season", "week", "sleeper_id", "sleeper_proj"]]


def compare(ours, store, id_map):
    """ours: DataFrame(season, week, player_id, position, ewm_ppr, proj, ppr) of scored projections.
    Returns (summary rows by season+position, weekly rows, per-game rows) on games both sources projected."""
    sid = dict(zip(id_map["player_id"], id_map["sleeper_id"].astype(str)))
    o = ours[ours["ewm_ppr"] >= RELEVANT].copy()
    o["sleeper_id"] = o["player_id"].map(sid)
    g = o.merge(store, on=["season", "week", "sleeper_id"], how="inner")
    if g.empty:
        return [], [], g
    g["our_err"] = (g["proj"] - g["ppr"]).abs()
    g["sl_err"] = (g["sleeper_proj"] - g["ppr"]).abs()
    g["blend_err"] = ((g["proj"] + g["sleeper_proj"]) / 2 - g["ppr"]).abs()   # average of the two
    summary = []
    for season, s in g.groupby("season"):
        for pos in ["ALL", "QB", "RB", "WR", "TE"]:
            x = s if pos == "ALL" else s[s["position"] == pos]
            if len(x):
                summary.append({"season": int(season), "position": pos, "games": int(len(x)),
                                "our_mae": round(float(x["our_err"].mean()), 2),
                                "sleeper_mae": round(float(x["sl_err"].mean()), 2),
                                "blend_mae": round(float(x["blend_err"].mean()), 2),
                                "closer_pct": round(float((x["our_err"] < x["sl_err"]).mean()), 3)})
    weekly = g.groupby(["season", "week"]).agg(games=("ppr", "size"), our_mae=("our_err", "mean"),
                                                sleeper_mae=("sl_err", "mean")).reset_index()
    weekly[["our_mae", "sleeper_mae"]] = weekly[["our_mae", "sleeper_mae"]].astype(float).round(2)
    weekly[["season", "week", "games"]] = weekly[["season", "week", "games"]].astype(int)
    return summary, weekly.to_dict("records"), g
