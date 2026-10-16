"""Run the whole pipeline: download data -> build database -> train model -> write projections.

Usage:  python scripts/update.py            (refreshes the current season)
        python scripts/update.py --offline  (uses cached files in data/raw)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3  # noqa: E402

import pandas as pd  # noqa: E402

from ffa import data, features, model, breakouts  # noqa: E402
from ffa.config import DB_PATH  # noqa: E402
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
    cols = ["player_id", "name", "position", "team", "headshot_url", "week", "gameday", "game_id", "opp",
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

    print("   Game-day weather for upcoming outdoor games")
    from ffa.weather import game_weather
    raw_games = pd.read_csv(data.RAW_DIR / "games.csv")
    wx = game_weather(raw_games, CURRENT_SEASON)
    data.write_table(wx, "weather")
    print(f"  forecasts for {int(wx['wind'].notna().sum())} games")

    print("4/5 Saving projections for the accuracy tracker")
    track_accuracy(proj, train, pw, games, last_week)

    print("5/5 Finding breakouts")
    b, weekly = breakouts.find_breakouts(pw, set(proj["player_id"]))
    data.write_table(b.round(3), "breakouts")
    data.write_table(weekly.round(3), "usage_weekly")
    print(f"  {len(b)} candidates | season {CURRENT_SEASON}, through week {last_week}")


HIST_COLS = ["season", "week", "player_id", "name", "position", "team", "ewm_ppr",
             "proj", "floor", "ceiling", "source"]


def track_accuracy(proj, train, pw, games, last_week):
    """Keep a snapshot of every week's pre-game projections, then score them once games are played."""
    with sqlite3.connect(DB_PATH) as con:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        hist = pd.read_sql("SELECT * FROM projection_history", con) if "projection_history" in tables \
            else pd.DataFrame(columns=HIST_COLS)

    # 1) Live snapshot for the upcoming week. Each run overwrites it until kickoff, so the final
    #    saved version is the last one before each player's game (including injury news).
    next_week = int(proj["week"].min())
    snap = proj[proj["week"] == next_week].assign(season=CURRENT_SEASON, source="live")[HIST_COLS]
    key = ["season", "week", "player_id"]
    hist = hist.merge(snap[key], on=key, how="left", indicator=True)
    hist = hist[hist["_merge"] == "left_only"].drop(columns="_merge")
    hist = pd.concat([hist, snap], ignore_index=True)

    # 2) Backfill finished weeks that were never tracked live, using walk-forward predictions.
    tracked = set(hist.loc[hist["season"] == CURRENT_SEASON, "week"].astype(int))
    missing = [w for w in range(1, last_week + 1) if w not in tracked]
    if missing:
        print(f"  backfilling weeks {missing} with walk-forward predictions")
        bf = model.walk_forward(train, CURRENT_SEASON, missing).assign(source="backfill")
        hist = pd.concat([hist, bf[HIST_COLS]], ignore_index=True)

    data.write_table(hist, "projection_history")

    # 3) Score every tracked player-game that has been played.
    actual = pw[pw["season"] == CURRENT_SEASON][["player_id", "week", "ppr"]]
    acc = hist[hist["season"] == CURRENT_SEASON].merge(actual, on=["player_id", "week"], how="inner")
    acc = acc[acc["week"] <= last_week]
    data.write_table(acc, "accuracy")
    print(f"  {len(snap)} projections saved for Week {next_week}; {len(acc):,} scored player-games")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
