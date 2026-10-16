"""Run the whole pipeline: download data -> build database -> train model -> write projections.

Usage:  python scripts/update.py            (refreshes the current season)
        python scripts/update.py --offline  (uses cached files in data/raw)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ffa import data, features, model, breakouts  # noqa: E402
from ffa.config import CURRENT_SEASON  # noqa: E402


def main(offline=False):
    print("1/4 Building database")
    pw, games = data.build_database(refresh_current=not offline)

    print("2/4 Training model + backtesting")
    train = features.training_frame(pw, games)
    models, metrics = model.train_and_save(train)
    for label, rows in metrics.items():
        a = rows[0]
        print(f"  {label}: model MAE {a['model_mae']} vs baseline {a['baseline_mae']} "
              f"({a['games']:,} player-games)")

    print("3/4 Projecting remaining games")
    fut, last_week = features.future_frame(pw, games)
    fut = fut.join(model.predict(models, fut))
    cols = ["player_id", "name", "position", "team", "headshot_url", "week", "gameday", "opp",
            "home", "implied_total", "spread", "def_factor", "ewm_ppr", "proj", "floor", "ceiling"]
    proj = fut[cols].round(2)
    proj["last_completed_week"] = last_week

    # Injury report: only applies if it's for the upcoming week (reports come out Wed-Fri).
    inj = data.read_table("injuries")
    next_week = int(proj["week"].min())
    proj["status"] = None
    if len(inj) and int(inj["week"].max()) == next_week:
        st = inj.dropna(subset=["status"]).set_index("player_id")["status"]
        m = proj["week"] == next_week
        proj.loc[m, "status"] = proj.loc[m, "player_id"].map(st)
        out = m & (proj["status"] == "Out")
        proj.loc[out, ["proj", "floor", "ceiling"]] = 0.0
        print(f"  Week {next_week} injury report applied: {int(out.sum())} players ruled out")
    else:
        print(f"  Week {next_week} injury report not out yet")
    data.write_table(proj, "projections")
    print(f"  {proj['player_id'].nunique()} players, weeks {proj['week'].min()}-{proj['week'].max()}")

    print("4/4 Finding breakouts")
    b, weekly = breakouts.find_breakouts(pw, set(proj["player_id"]))
    data.write_table(b.round(3), "breakouts")
    data.write_table(weekly.round(3), "usage_weekly")
    print(f"  {len(b)} candidates | season {CURRENT_SEASON}, through week {last_week}")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
