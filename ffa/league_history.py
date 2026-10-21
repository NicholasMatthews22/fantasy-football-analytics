"""Everything that has happened in a Sleeper league since it started: every regular-season game,
each season's champion, all-time standings, head-to-head records, and a record book.

Teams are tracked by roster ID, which stays the same from season to season in a dynasty league.
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .trade_history import _safe, league_chain


def _season_games(lg, api, last_week):
    start = int((lg.get("settings") or {}).get("playoff_week_start") or 15)
    weeks = list(range(1, min(start - 1, last_week) + 1))
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda w: _safe(api.matchups, lg["league_id"], w), weeks))
    games = []
    for w, ms in zip(weeks, results):
        groups = {}
        for m in ms:
            if m.get("matchup_id") is not None:
                groups.setdefault(m["matchup_id"], []).append(m)
        for g in groups.values():
            if len(g) != 2:
                continue
            a, b = sorted(g, key=lambda m: m["roster_id"])
            pa, pb = float(a.get("points") or 0), float(b.get("points") or 0)
            if pa == 0 and pb == 0:          # not played yet
                continue
            games.append({"season": int(lg["season"]), "week": w, "a": int(a["roster_id"]), "b": int(b["roster_id"]),
                          "pa": round(pa, 2), "pb": round(pb, 2)})
    return games


def _playoffs(lg, api):
    br = _safe(api.winners_bracket, lg["league_id"])
    teams = {t for m in br for t in (m.get("t1"), m.get("t2")) if isinstance(t, int)}
    final = next((m for m in br if m.get("p") == 1 and m.get("w")), None)
    return {"season": int(lg["season"]), "playoff_teams": teams,
            "champion": final["w"] if final else None, "runner_up": final["l"] if final else None}


def collect(league_id, api, current_season, last_completed_week):
    """Returns (games DataFrame, seasons list). Past seasons use every regular-season week;
    the current season stops at the last fully completed NFL week."""
    chain = league_chain(league_id, api)
    games, seasons = [], []
    for season, lg in chain:
        last = last_completed_week if season == current_season else 18
        games += _season_games(lg, api, last)
        info = _playoffs(lg, api) if season < current_season or lg.get("status") == "complete" else \
            {"season": season, "playoff_teams": set(), "champion": None, "runner_up": None}
        info["in_progress"] = season == current_season and lg.get("status") != "complete"
        seasons.append(info)
    cols = ["season", "week", "a", "b", "pa", "pb"]
    return pd.DataFrame(games, columns=cols), sorted(seasons, key=lambda s: -s["season"])


def team_games(games):
    """One row per team per game: team, opp, pf, pa, result (W/L/T)."""
    if games.empty:
        return pd.DataFrame(columns=["season", "week", "team", "opp", "pf", "pa", "result"])
    a = games.rename(columns={"a": "team", "b": "opp", "pa": "pf", "pb": "pa_"})
    b = games.rename(columns={"b": "team", "a": "opp", "pb": "pf", "pa": "pa_"})
    t = pd.concat([a, b], ignore_index=True).rename(columns={"pa_": "pa"})
    t["result"] = ["W" if x > y else ("L" if x < y else "T") for x, y in zip(t["pf"], t["pa"])]
    return t[["season", "week", "team", "opp", "pf", "pa", "result"]]


def standings(games, seasons):
    t = team_games(games)
    out = t.groupby("team").agg(W=("result", lambda r: (r == "W").sum()), L=("result", lambda r: (r == "L").sum()),
                                T=("result", lambda r: (r == "T").sum()), PF=("pf", "sum"), PA=("pa", "sum"),
                                seasons=("season", "nunique")).reset_index()
    out["win_pct"] = (out["W"] + 0.5 * out["T"]) / (out["W"] + out["L"] + out["T"]).clip(lower=1)
    out["playoffs"] = out["team"].map(lambda r: sum(r in s["playoff_teams"] for s in seasons))
    out["titles"] = out["team"].map(lambda r: sum(s["champion"] == r for s in seasons))
    return out.sort_values(["titles", "win_pct", "PF"], ascending=False).reset_index(drop=True)


def head_to_head(games, team):
    t = team_games(games)
    t = t[t["team"] == team]
    if t.empty:
        return pd.DataFrame(columns=["opp", "W", "L", "T", "PF", "PA", "games"])
    return (t.groupby("opp").agg(W=("result", lambda r: (r == "W").sum()), L=("result", lambda r: (r == "L").sum()),
                                 T=("result", lambda r: (r == "T").sum()), PF=("pf", "sum"), PA=("pa", "sum"),
                                 games=("pf", "size")).reset_index()
             .sort_values(["games", "W"], ascending=False))


def record_book(games):
    """The league's single-game and single-season records."""
    t = team_games(games)
    if t.empty:
        return []
    recs = []
    hi = t.loc[t["pf"].idxmax()]
    recs.append(("Highest score", hi["team"], f'{hi["pf"]:.2f} pts', f'vs {{opp:{hi["opp"]}}}', hi["season"], hi["week"]))
    lo = t.loc[t["pf"].idxmin()]
    recs.append(("Lowest score", lo["team"], f'{lo["pf"]:.2f} pts', f'vs {{opp:{lo["opp"]}}}', lo["season"], lo["week"]))
    t["margin"] = t["pf"] - t["pa"]
    bl = t.loc[t["margin"].idxmax()]
    recs.append(("Biggest blowout", bl["team"], f'won by {bl["margin"]:.2f}', f'over {{opp:{bl["opp"]}}}',
                 bl["season"], bl["week"]))
    wins = t[t["margin"] > 0]
    if len(wins):
        cl = wins.loc[wins["margin"].idxmin()]
        recs.append(("Closest game", cl["team"], f'won by {cl["margin"]:.2f}', f'over {{opp:{cl["opp"]}}}',
                     cl["season"], cl["week"]))
    ss = t.groupby(["season", "team"]).agg(pf=("pf", "mean"), w=("result", lambda r: (r == "W").sum()),
                                           g=("result", "size")).reset_index()
    best = ss.loc[ss["pf"].idxmax()]
    recs.append(("Best scoring season", best["team"], f'{best["pf"]:.1f} pts per week', "", best["season"], None))
    ss["wp"] = ss["w"] / ss["g"]
    bw = ss.sort_values(["wp", "pf"], ascending=False).iloc[0]
    recs.append(("Best record", bw["team"], f'{int(bw["w"])}-{int(bw["g"] - bw["w"])}', "", bw["season"], None))
    return [{"record": r[0], "team": int(r[1]), "value": r[2], "detail": r[3], "season": int(r[4]),
             "week": None if r[5] is None else int(r[5])} for r in recs]
