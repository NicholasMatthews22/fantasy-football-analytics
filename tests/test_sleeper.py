from ffa import sleeper


def test_traded_picks_move_to_their_new_team():
    rosters = [{"roster_id": 1}, {"roster_id": 2}]
    traded = [{"season": "2027", "round": 1, "roster_id": 2, "owner_id": 1}]
    own = sleeper.pick_ownership(rosters, traded, [2027], 2)
    assert (2027, 1, 2) in own[1] and (2027, 1, 1) in own[1]
    assert own[2] == [(2027, 2, 2)]


def test_team_name_falls_back_to_display_name():
    assert sleeper.team_name({"display_name": "nick", "metadata": {}}) == "nick"
    assert sleeper.team_name({"display_name": "nick", "metadata": {"team_name": "Squad"}}) == "Squad"
