import pandas as pd

from ffa import benchmark


def test_both_sources_scored_on_the_same_games():
    ours = pd.DataFrame({"season": 2025, "week": [1, 1, 2], "player_id": ["a", "b", "a"], "position": "WR",
                         "ewm_ppr": [10, 10, 10], "proj": [10.0, 10.0, 10.0], "ppr": [12.0, 4.0, 10.0]})
    store = pd.DataFrame({"season": 2025, "week": [1, 2], "sleeper_id": ["1", "1"], "sleeper_proj": [14.0, 13.0]})
    ids = pd.DataFrame({"player_id": ["a", "b"], "sleeper_id": ["1", "2"]})
    summary, weekly, games = benchmark.compare(ours, store, ids)
    assert len(games) == 2                       # player b has no Sleeper projection, so he's left out of both
    allrow = next(r for r in summary if r["position"] == "ALL")
    assert allrow["our_mae"] == 1.0 and allrow["sleeper_mae"] == 2.5


def test_update_store_only_fetches_whats_missing():
    calls = []

    def fetch(s, w):
        calls.append((s, w))
        return pd.DataFrame({"sleeper_id": ["1"], "sleeper_proj": [5.0]})

    store = benchmark.update_store(pd.DataFrame(columns=["season", "week", "sleeper_id", "sleeper_proj"]),
                                   [(2025, 1), (2025, 2)], fetch=fetch)
    calls.clear()
    benchmark.update_store(store, [(2025, 1), (2025, 2), (2025, 3)], refresh={(2025, 2)}, fetch=fetch)
    assert calls == [(2025, 2), (2025, 3)]
