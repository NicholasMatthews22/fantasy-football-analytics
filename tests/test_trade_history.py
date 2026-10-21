from types import SimpleNamespace

from ffa import trade_history


def fake_api():
    leagues = {"L2": {"league_id": "L2", "season": "2026", "previous_league_id": "L1"},
               "L1": {"league_id": "L1", "season": "2025", "previous_league_id": None}}
    tx = {("L1", 3): [{"transaction_id": "t1", "type": "trade", "status": "complete", "leg": 3,
                       "status_updated": 1727000000000, "roster_ids": [1, 2], "adds": {"100": 2}, "drops": {"100": 1},
                       "draft_picks": [{"season": "2026", "round": 1, "roster_id": 2, "owner_id": 1,
                                        "previous_owner_id": 2}],
                       "waiver_budget": [{"sender": 2, "receiver": 1, "amount": 10}]},
                      {"transaction_id": "w1", "type": "waiver", "status": "complete"}]}
    return SimpleNamespace(
        league=lambda lid: leagues.get(lid, {}),
        transactions=lambda lid, w: tx.get((lid, w), []),
        drafts=lambda lid: [{"draft_id": "D", "status": "complete", "settings": {"rounds": 4}}] if lid == "L2" else [],
        draft=lambda did: {"season": "2026", "slot_to_roster_id": {"1": 2, "2": 1}},
        draft_picks=lambda did: [{"round": 1, "draft_slot": 1, "pick_no": 1, "player_id": "555",
                                  "metadata": {"first_name": "Rookie", "last_name": "Back", "position": "RB"}}],
        league_users=lambda lid: [], league_rosters=lambda lid: [{"roster_id": 1}, {"roster_id": 2}],
        team_name=lambda u: "x", avatar_url=lambda u: None)


def test_finds_trades_across_seasons_and_resolves_picks():
    trades, teams = trade_history.collect("L2", fake_api(), rounds=4)
    assert len(trades) == 1                               # the waiver claim is ignored
    t = trades[0]
    assert t["season"] == 2025
    assert t["sides"][2]["players"] == ["100"]
    pick = t["sides"][1]["picks"][0]
    assert pick["became"]["name"] == "Rookie Back" and pick["became"]["slot"] == "1.01"
    assert t["sides"][1]["faab"] == 10
    assert set(teams) == {1, 2}
