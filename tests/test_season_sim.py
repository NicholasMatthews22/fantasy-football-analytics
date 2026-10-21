import numpy as np

from ffa import season_sim


def test_much_better_team_wins_most_titles():
    mu, sd = np.array([150.0, 100, 100, 100]), np.array([10.0, 10, 10, 10])
    weeks = [(mu, sd, [(0, 1), (2, 3)]), (mu, sd, [(0, 2), (1, 3)])]
    r = season_sim.simulate(["A", "B", "C", "D"], [0, 0, 0, 0], [0, 0, 0, 0], weeks, 4, mu, sd, n_sims=2000)
    assert r["title_pct"][0] > 0.9
    assert abs(r["title_pct"].sum() - 1) < 1e-9


def test_playoff_odds_are_probabilities_that_add_up():
    mu, sd = np.full(6, 110.0), np.full(6, 20.0)
    weeks = [(mu, sd, [(0, 1), (2, 3), (4, 5)])]
    r = season_sim.simulate(list("ABCDEF"), [3, 3, 2, 2, 1, 1], [0] * 6, weeks, 4, mu, sd, n_sims=1000)
    assert ((r["playoff_pct"] >= 0) & (r["playoff_pct"] <= 1)).all()
    assert abs(r["playoff_pct"].sum() - 4) < 1e-9     # exactly 4 playoff spots every simulation
    assert r["has_byes"] is False


def test_bracket_with_byes_still_crowns_one_team():
    rng = np.random.default_rng(1)
    mu, sd = np.full(6, 100.0), np.full(6, 1.0)
    assert season_sim.bracket_winner([0, 1, 2, 3, 4, 5], mu, sd, rng) in range(6)
