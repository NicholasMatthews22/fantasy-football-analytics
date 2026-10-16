"""Download nflverse data and load it into a SQLite database."""
import sqlite3
import urllib.request

import pandas as pd

from .config import CURRENT_SEASON, DB_PATH, FIRST_SEASON, POSITIONS, RAW_DIR

STATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "stats_player/stats_player_week_{season}.parquet")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
INJURY_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
              "injuries/injuries_{season}.parquet")
SNAP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
            "snap_counts/snap_counts_{season}.parquet")
PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
           "pbp/play_by_play_{season}.parquet")
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
IDS_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
TEAMS_URL = "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/teams_colors_logos.csv"

KEEP_COLS = [
    "player_id", "player_display_name", "position", "headshot_url", "season", "week",
    "game_id", "team", "opponent_team", "attempts", "completions", "passing_yards",
    "passing_tds", "passing_interceptions", "passing_air_yards", "carries", "rushing_yards",
    "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "target_share", "air_yards_share", "wopr", "fantasy_points_ppr",
]


def _download(url, dest, refresh=False):
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def download_raw(refresh_current=True):
    """Fetch weekly player stats for each season plus the full schedule.

    Past seasons are cached. The current season and schedule are re-downloaded
    so each run picks up the latest week.
    """
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        refresh = refresh_current and season == CURRENT_SEASON
        _download(STATS_URL.format(season=season), RAW_DIR / f"stats_player_week_{season}.parquet", refresh)
        _download(SNAP_URL.format(season=season), RAW_DIR / f"snap_counts_{season}.parquet", refresh)
        _download(PBP_URL.format(season=season), RAW_DIR / f"pbp_{season}.parquet", refresh)
    _download(PLAYERS_URL, RAW_DIR / "players.parquet", refresh_current)
    _download(GAMES_URL, RAW_DIR / "games.csv", refresh_current)
    _download(TEAMS_URL, RAW_DIR / "teams.csv", False)
    _download(IDS_URL, RAW_DIR / "db_playerids.csv", refresh_current)
    try:
        _download(INJURY_URL.format(season=CURRENT_SEASON),
                  RAW_DIR / f"injuries_{CURRENT_SEASON}.parquet", refresh_current)
    except Exception as e:  # injury file can lag at the very start of a season
        print(f"  no injury report yet ({e})")


def load_usage_extras():
    """Snap share (from snap counts) and red zone opportunities (from play-by-play) per player-game."""
    seasons = range(FIRST_SEASON, CURRENT_SEASON + 1)
    ids = pd.read_parquet(RAW_DIR / "players.parquet", columns=["gsis_id", "pfr_id"]).dropna()

    snaps = pd.concat([pd.read_parquet(RAW_DIR / f"snap_counts_{s}.parquet") for s in seasons])
    snaps = snaps[snaps["game_type"] == "REG"].merge(ids, left_on="pfr_player_id", right_on="pfr_id")
    snaps = (snaps.rename(columns={"gsis_id": "player_id", "offense_pct": "snap_pct"})
                  .groupby(["season", "week", "player_id"])["snap_pct"].max().reset_index())

    cols = ["season", "week", "season_type", "posteam", "play_type", "yardline_100",
            "receiver_player_id", "rusher_player_id"]
    pbp = pd.concat([pd.read_parquet(RAW_DIR / f"pbp_{s}.parquet", columns=cols) for s in seasons])
    rz = pbp[(pbp["season_type"] == "REG") & (pbp["yardline_100"] <= 20)
             & pbp["play_type"].isin(["pass", "run"])]
    team_rz = rz.groupby(["season", "week", "posteam"]).size().rename("team_rz").reset_index()
    tgt = rz.dropna(subset=["receiver_player_id"]).groupby(
        ["season", "week", "posteam", "receiver_player_id"]).size().rename("rz_opps")
    car = rz.dropna(subset=["rusher_player_id"]).groupby(
        ["season", "week", "posteam", "rusher_player_id"]).size().rename("rz_opps")
    tgt.index.names = car.index.names = ["season", "week", "posteam", "player_id"]
    rzp = pd.concat([tgt, car]).groupby(level=[0, 1, 2, 3]).sum().reset_index()
    rzp = rzp.merge(team_rz, on=["season", "week", "posteam"])
    rzp["rz_share"] = rzp["rz_opps"] / rzp["team_rz"]
    rzp = rzp.groupby(["season", "week", "player_id"])[["rz_opps", "rz_share"]].sum().reset_index()

    return snaps.merge(rzp, on=["season", "week", "player_id"], how="outer")


def load_player_weeks():
    frames = [pd.read_parquet(RAW_DIR / f"stats_player_week_{s}.parquet")
              for s in range(FIRST_SEASON, CURRENT_SEASON + 1)]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"]

    # Team carries per game, so we can compute each back's share of the backfield.
    team_carries = (df.groupby(["game_id", "team"])["carries"].sum()
                      .rename("team_carries").reset_index())

    df = df[df["position"].isin(POSITIONS)][KEEP_COLS].copy()
    df = df.merge(team_carries, on=["game_id", "team"], how="left")
    df["carry_share"] = (df["carries"] / df["team_carries"]).fillna(0)
    df = df.rename(columns={"player_display_name": "name", "fantasy_points_ppr": "ppr"})

    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(0)
    df = df.merge(load_usage_extras(), on=["season", "week", "player_id"], how="left")
    df[["rz_opps", "rz_share"]] = df[["rz_opps", "rz_share"]].fillna(0)   # no red zone touches = 0
    # snap_pct stays missing when snap counts couldn't be matched; the model handles gaps
    df["total_tds"] = df["passing_tds"] + df["rushing_tds"] + df["receiving_tds"]
    df["opportunities"] = df["targets"] + df["carries"]
    return df.drop(columns="team_carries")


def load_games():
    g = pd.read_csv(RAW_DIR / "games.csv")
    g = g[(g["season"] >= FIRST_SEASON) & (g["game_type"] == "REG")]
    return g[["game_id", "season", "week", "gameday", "home_team", "away_team",
              "home_score", "away_score", "spread_line", "total_line"]].copy()


def load_teams():
    t = pd.read_csv(RAW_DIR / "teams.csv")
    return t[["team_abbr", "team_name", "team_nick", "team_color", "team_color2",
              "team_logo_espn"]].rename(columns={"team_abbr": "team", "team_logo_espn": "logo"})


def load_injuries():
    """Latest game-status designation for each player (Out, Doubtful, Questionable)."""
    path = RAW_DIR / f"injuries_{CURRENT_SEASON}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "week", "status", "injury", "practice"])
    i = pd.read_parquet(path)
    i = i[(i["season_type"] == "REG") & i["position"].isin(POSITIONS)]
    i = i[i["week"] == i["week"].max()]
    i = i.rename(columns={"gsis_id": "player_id", "report_status": "status",
                          "report_primary_injury": "injury", "practice_status": "practice"})
    return i[["player_id", "week", "status", "injury", "practice"]].drop_duplicates("player_id", keep="last")


def load_id_map():
    """Sleeper player IDs -> nflverse (GSIS) IDs, for importing Sleeper rosters."""
    d = pd.read_csv(RAW_DIR / "db_playerids.csv", usecols=["sleeper_id", "gsis_id"], low_memory=False).dropna()
    d["sleeper_id"] = d["sleeper_id"].astype("int64").astype(str)
    return d.rename(columns={"gsis_id": "player_id"}).drop_duplicates("sleeper_id")


def build_database(refresh_current=True):
    download_raw(refresh_current)
    pw = load_player_weeks()
    games = load_games()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        pw.to_sql("player_weeks", con, if_exists="replace", index=False)
        games.to_sql("games", con, if_exists="replace", index=False)
        load_teams().to_sql("teams", con, if_exists="replace", index=False)
        load_injuries().to_sql("injuries", con, if_exists="replace", index=False)
        load_id_map().to_sql("id_map", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_pw_player ON player_weeks(player_id, season, week)")
    print(f"  player_weeks: {len(pw):,} rows | games: {len(games):,} rows")
    return pw, games


def read_table(name):
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(f"SELECT * FROM {name}", con)


def write_table(df, name):
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql(name, con, if_exists="replace", index=False)
