from ffa import fantasycalc


def rows(*pairs):
    return [{"player": {"name": n, "position": "PICK"}, "value": v} for n, v in pairs]


def test_reads_tiers_and_fills_missing_rounds_and_years():
    pv = fantasycalc.pick_values(rows(("2027 1st", 3000), ("2027 2nd", 1200), ("2027 1st (Early)", 5000),
                                      ("2028 1st", 2800), ("2028 2nd", 1100)), [2027, 2028, 2029], 5)
    assert pv[(2027, 1, "Early")] == 5000
    assert pv[(2027, 1, "Late")] == 3000                 # tier not listed: falls back to the round value
    assert pv[(2027, 3, None)] < pv[(2027, 2, None)] < pv[(2027, 1, None)]
    assert pv[(2027, 5, None)] > 0
    assert pv[(2029, 1, None)] == pv[(2028, 1, None)]    # year not listed: uses the latest listed year


def test_player_values_skip_picks():
    data = [{"player": {"sleeperId": "4034", "position": "RB"}, "value": 7000},
            {"player": {"name": "2027 1st", "position": "PICK"}, "value": 3000}]
    assert fantasycalc.player_values(data) == {"4034": 7000.0}
