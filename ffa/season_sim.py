"""Monte Carlo season simulation: remaining regular-season matchups, then the playoff bracket."""
import math

import numpy as np


def bracket_winner(seeds, mu, sd, rng):
    """Single-elimination bracket. Top seeds get byes to fill it out; each round reseeds best vs worst."""
    n = len(seeds)
    size = 2 ** math.ceil(math.log2(n)) if n > 1 else 1
    byes = size - n
    alive = list(seeds[:byes])
    rest = list(seeds[byes:])
    first = [(rest[i], rest[-1 - i]) for i in range(len(rest) // 2)]
    for a, b in first:
        alive.append(a if rng.normal(mu[a], sd[a]) >= rng.normal(mu[b], sd[b]) else b)
    while len(alive) > 1:
        alive.sort(key=lambda t: seeds.index(t))
        nxt = []
        for i in range(len(alive) // 2):
            a, b = alive[i], alive[-1 - i]
            nxt.append(a if rng.normal(mu[a], sd[a]) >= rng.normal(mu[b], sd[b]) else b)
        alive = nxt
    return alive[0]


def simulate(teams, wins, pf, weeks, playoff_teams, playoff_mu, playoff_sd, n_sims=5000, seed=7):
    """
    teams: list of team keys. wins/pf: current wins and points for (arrays aligned with teams).
    weeks: list of (mu[T], sd[T], pairs[(i, j)]) for each remaining regular-season week.
    Returns dict of arrays: avg_wins, playoff_pct, bye_pct, title_pct.
    """
    rng = np.random.default_rng(seed)
    T = len(teams)
    W = np.tile(np.asarray(wins, float), (n_sims, 1))
    PF = np.tile(np.asarray(pf, float), (n_sims, 1))
    for mu, sd, pairs in weeks:
        scores = rng.normal(mu, sd, size=(n_sims, T))
        PF += scores
        for i, j in pairs:
            won = scores[:, i] > scores[:, j]
            W[:, i] += won
            W[:, j] += ~won
    order = np.argsort(-(W + PF * 1e-6), axis=1)          # wins first, points for breaks ties
    p = min(playoff_teams, T)
    size = 2 ** math.ceil(math.log2(p)) if p > 1 else 1
    n_byes = size - p
    made = np.zeros(T)
    byes = np.zeros(T)
    titles = np.zeros(T)
    for s in range(n_sims):
        seeds = list(order[s, :p])
        made[seeds] += 1
        byes[seeds[:n_byes]] += 1
        titles[bracket_winner(seeds, playoff_mu, playoff_sd, rng)] += 1
    return {"avg_wins": W.mean(axis=0), "playoff_pct": made / n_sims, "bye_pct": byes / n_sims,
            "title_pct": titles / n_sims, "has_byes": n_byes > 0}
