import pandas as pd

from ffa import breakouts
from ffa.config import CURRENT_SEASON


def season_rows(pid, pos, ppr, snaps, opps, season=CURRENT_SEASON):
    n = len(ppr)
    return pd.DataFrame({"player_id": pid, "name": pid, "position": pos, "team": "NYJ", "headshot_url": "",
                         "season": season, "week": range(1, n + 1), "ppr": ppr, "snap_pct": snaps,
                         "targets": [o / 2 for o in opps], "carries": [o / 2 for o in opps], "opportunities": opps,
                         "receiving_air_yards": 0.0, "attempts": 0.0, "target_share": 0.1, "carry_share": 0.1})


def test_rising_backup_counts_but_qbs_and_stars_dont():
    rising = dict(ppr=[2, 2, 3, 12, 14, 15], snaps=[.2, .2, .25, .7, .8, .85], opps=[3, 3, 4, 18, 20, 21])
    pw = pd.concat([
        season_rows("backup_rb", "RB", **rising),
        season_rows("star_rb", "RB", **rising),
        season_rows("star_rb", "RB", [25] * 10, [.9] * 10, [25] * 10, season=CURRENT_SEASON - 1),
        season_rows("new_qb", "QB", **rising),
        # filler so the xFP regression has something to fit for every position
        *[season_rows(f"f{i}", p, [8 + i] * 6, [.5] * 6, [10 + i] * 6) for i in range(3) for p in ["QB", "RB", "WR", "TE"]],
    ], ignore_index=True)
    pw["player_id"] = pw["player_id"].astype(str) + pw["position"].where(pw["player_id"].str.startswith("f"), "")
    b, _ = breakouts.find_breakouts(pw, set(pw["player_id"]))
    found = set(b["player_id"])
    assert "backup_rb" in found
    assert "star_rb" not in found and "new_qb" not in found
