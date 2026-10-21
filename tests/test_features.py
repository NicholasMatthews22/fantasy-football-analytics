"""The projection model must only see information from before kickoff."""
import pandas as pd

from ffa import features


def make_weeks(points):
    """One WR with the given PPR scores in weeks 1..n of 2025."""
    n = len(points)
    zeros = {c: [0.0] * n for c in features.ROLL_COLS if c != "ppr"}
    return pd.DataFrame({"player_id": ["P1"] * n, "season": [2025] * n, "week": list(range(1, n + 1)),
                         "ppr": points, **zeros})


def test_form_going_into_a_game_ignores_that_game():
    pw, st = features.player_state(make_weeks([10.0, 20.0, 99.0]))
    prev = st.groupby("player_id")[features.STATE_COLS].shift(1)
    # going into week 3, the 99-point game hasn't happened yet
    assert prev.loc[2, "l3_ppr"] == 15.0
    assert prev.loc[2, "ewm_ppr"] < 20.0


def test_first_game_has_no_history():
    pw, st = features.player_state(make_weeks([12.0, 8.0]))
    prev = st.groupby("player_id")[features.STATE_COLS].shift(1)
    assert prev.loc[0].isna().all()


def test_recent_games_count_more():
    _, st = features.player_state(make_weeks([0.0, 0.0, 0.0, 30.0]))
    assert st.loc[3, "ewm_ppr"] > 30.0 / 4      # more than a plain average


def test_spread_sign_gives_favorite_more_implied_points():
    games = pd.DataFrame({"game_id": ["G"], "season": [2025], "week": [1], "gameday": ["2025-09-07"],
                          "spread_line": [7.0], "total_line": [45.0], "home_score": [None],
                          "home_team": ["KC"], "away_team": ["DEN"]})
    tg = features.team_games(games).set_index("team")
    assert tg.at["KC", "implied_total"] == 26.0       # home favored by 7: (45 + 7) / 2
    assert tg.at["DEN", "implied_total"] == 19.0
