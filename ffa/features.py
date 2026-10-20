"""Feature engineering.

Every feature is built only from information available *before* kickoff:
the player's form from previous games, how the opponent's defense has played,
and the betting line for the game.
"""
import pandas as pd

from .config import CURRENT_SEASON, DEFENSE_HALFLIFE, PLAYER_HALFLIFE, POS_CODE, POSITIONS

ROLL_COLS = [
    "ppr", "targets", "carries", "receptions", "opportunities", "target_share",
    "carry_share", "air_yards_share", "wopr", "attempts", "passing_yards",
    "rushing_yards", "receiving_yards", "total_tds", "receiving_air_yards",
]
STATE_COLS = [f"ewm_{c}" for c in ROLL_COLS] + [
    "l3_ppr", "l3_opportunities", "season_avg_ppr", "games_played"]
CONTEXT_COLS = ["def_factor", "implied_total", "spread", "home", "pos_code"]
FEATURES = STATE_COLS + CONTEXT_COLS

FEATURE_LABELS = {
    "ewm_ppr": "Recent fantasy points", "ewm_opportunities": "Recent targets + carries",
    "def_factor": "Opponent defense", "implied_total": "Team implied points",
    "l3_ppr": "Last 3 games points", "season_avg_ppr": "Season average",
}


# ---------- player form ----------
def player_state(pw):
    """Player form *after* each game, weighted toward recent games.

    Carries across seasons so Week 1 projections still use last year's data.
    """
    pw = pw.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = pw.groupby("player_id")
    st = pw[["player_id", "season", "week"]].copy()
    for c in ROLL_COLS:
        st[f"ewm_{c}"] = g[c].transform(lambda s: s.ewm(halflife=PLAYER_HALFLIFE).mean())
    st["l3_ppr"] = g["ppr"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    st["l3_opportunities"] = g["opportunities"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    st["season_avg_ppr"] = pw.groupby(["player_id", "season"])["ppr"].transform(lambda s: s.expanding().mean())
    st["games_played"] = (g.cumcount() + 1).clip(upper=40)
    return pw, st


# ---------- game context ----------
def team_games(games):
    """One row per team per game, with betting-line context from that team's side."""
    base = ["game_id", "season", "week", "gameday", "spread_line", "total_line", "home_score"]
    home = games[base + ["home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opp"})
    home["home"], home["spread"] = 1, home["spread_line"]
    away = games[base + ["away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opp"})
    away["home"], away["spread"] = 0, -away["spread_line"]
    tg = pd.concat([home, away], ignore_index=True)

    # nflverse spread_line is positive when the home team is favored.
    tg["implied_total"] = (tg["total_line"] + tg["spread"]) / 2
    team_avg = tg.groupby(["season", "team"])["implied_total"].transform("mean")
    tg["implied_total"] = tg["implied_total"].fillna(team_avg).fillna(22.5)
    tg["spread"] = tg["spread"].fillna(0)
    tg["played"] = tg["home_score"].notna()
    return tg[["game_id", "season", "week", "gameday", "team", "opp", "home", "spread",
               "implied_total", "played"]]


# ---------- defenses ----------
def defense_state(pw, tg):
    """How many PPR points each defense has allowed to each position, relative to average.

    1.20 means the defense gives up 20% more points than average to that position.
    """
    allowed = (pw.groupby(["season", "week", "opponent_team", "position"])["ppr"].sum()
                 .reset_index().rename(columns={"opponent_team": "team"}))
    played = tg[tg["played"]][["season", "week", "team"]]
    grid = played.merge(pd.DataFrame({"position": POSITIONS}), how="cross")
    d = grid.merge(allowed, on=["season", "week", "team", "position"], how="left").fillna({"ppr": 0})
    d = d.sort_values(["team", "position", "season", "week"]).reset_index(drop=True)
    d["ewm_allowed"] = d.groupby(["team", "position"])["ppr"].transform(
        lambda s: s.ewm(halflife=DEFENSE_HALFLIFE).mean())
    pos_avg = d.groupby("position")["ppr"].transform("mean")
    d["def_factor"] = d["ewm_allowed"] / pos_avg
    return d[["season", "week", "team", "position", "def_factor"]]


# ---------- assembled datasets ----------
def training_frame(pw, games):
    """Historical player-games with pre-game features and the actual result (ppr)."""
    tg = team_games(games)
    pw, st = player_state(pw)
    prev = st.groupby("player_id")[STATE_COLS].shift(1)          # form going INTO the game
    df = pd.concat([pw, prev], axis=1)
    df = df[df["games_played"].notna()]                          # need at least one prior game

    ds = defense_state(pw, tg)
    ds["def_factor"] = ds.groupby(["team", "position"])["def_factor"].shift(1)
    df = df.merge(ds.rename(columns={"team": "opponent_team"}),
                  on=["season", "week", "opponent_team", "position"], how="left")
    df = df.merge(tg[["season", "week", "team", "home", "spread", "implied_total"]],
                  on=["season", "week", "team"], how="left")
    df["def_factor"] = df["def_factor"].fillna(1.0)
    df["pos_code"] = df["position"].map(POS_CODE)
    return df


def completed_week(games):
    """Last week of the current season where every game has been played.

    Using the schedule (not the stats) keeps a Thursday night game from
    making the whole week look finished.
    """
    g = games[games["season"] == CURRENT_SEASON]
    done = g.groupby("week")["home_score"].apply(lambda s: s.notna().all()).sort_index()
    last = 0
    for week, finished in done.items():
        if not finished:
            break
        last = int(week)
    return last


def future_frame(pw, games):
    """Feature rows for every remaining game this season, for every active player."""
    tg = team_games(games)
    last = completed_week(games)
    pw, st = player_state(pw)

    info = pw.groupby("player_id").tail(1)[["player_id", "name", "position", "team",
                                            "headshot_url", "season", "week"]]
    active = info[(info["season"] == CURRENT_SEASON) & (info["week"] >= last - 2)]
    state = st.groupby("player_id").tail(1)[["player_id"] + STATE_COLS]
    players = active.drop(columns=["season", "week"]).merge(state, on="player_id")

    upcoming = tg[(tg["season"] == CURRENT_SEASON) & ~tg["played"]]
    df = players.merge(upcoming, on="team")

    ds = defense_state(pw, tg).groupby(["team", "position"]).tail(1)[["team", "position", "def_factor"]]
    df = df.merge(ds.rename(columns={"team": "opp"}), on=["opp", "position"], how="left")
    df["def_factor"] = df["def_factor"].fillna(1.0)
    df["pos_code"] = df["position"].map(POS_CODE)
    return df, last
