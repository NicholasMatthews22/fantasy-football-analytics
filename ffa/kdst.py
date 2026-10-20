"""Kicker and team defense (D/ST) projections.

Both use Sleeper's default scoring. The model is deliberately simple: a player's (or
defense's) recent weighted average plus the Vegas context that drives these positions,
the kicker's team implied total and the points the defense's opponent is expected to score.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import CURRENT_SEASON, FIRST_SEASON, PLAYER_HALFLIFE, RAW_DIR

PTS_ALLOWED_TIERS = [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1), (999, -4)]


def _raw_stats():
    frames = [pd.read_parquet(RAW_DIR / f"stats_player_week_{s}.parquet")
              for s in range(FIRST_SEASON, CURRENT_SEASON + 1)]
    df = pd.concat(frames, ignore_index=True)
    return df[df["season_type"] == "REG"]


def kicker_games(raw):
    k = raw[raw["position"] == "K"].copy().fillna(0)
    k["pts"] = (3 * (k["fg_made_0_19"] + k["fg_made_20_29"] + k["fg_made_30_39"]) + 4 * k["fg_made_40_49"]
                + 5 * (k["fg_made_50_59"] + k["fg_made_60_"]) + k["pat_made"] - k["fg_missed"] - k["pat_missed"])
    return k[["player_id", "player_display_name", "headshot_url", "team", "opponent_team", "season", "week", "pts"]] \
        .rename(columns={"player_display_name": "name"})


def _pa_points(pa):
    for limit, pts in PTS_ALLOWED_TIERS:
        if pa <= limit:
            return pts
    return -4


def defense_games(raw, games):
    num = ["def_sacks", "def_interceptions", "fumble_recovery_opp", "def_tds", "fumble_recovery_tds",
           "special_teams_tds", "def_safeties", "def_fg_blocks", "def_punt_blocks"]
    d = raw.groupby(["season", "week", "team"])[num].sum().reset_index()
    played = games[games["home_score"].notna()]
    home = played[["season", "week", "home_team", "away_team", "away_score"]].rename(
        columns={"home_team": "team", "away_team": "opp", "away_score": "pa"})
    away = played[["season", "week", "away_team", "home_team", "home_score"]].rename(
        columns={"away_team": "team", "home_team": "opp", "home_score": "pa"})
    d = pd.concat([home, away]).merge(d, on=["season", "week", "team"], how="left").fillna(0)
    d["pts"] = (d["def_sacks"] + 2 * d["def_interceptions"] + 2 * d["fumble_recovery_opp"]
                + 6 * (d["def_tds"] + d["fumble_recovery_tds"] + d["special_teams_tds"])
                + 2 * d["def_safeties"] + 2 * (d["def_fg_blocks"] + d["def_punt_blocks"])
                + d["pa"].map(_pa_points))
    return d[["team", "opp", "season", "week", "pts"]]


def _with_form(df, key):
    df = df.sort_values([key, "season", "week"]).reset_index(drop=True)
    df["ewm_after"] = df.groupby(key)["pts"].transform(lambda s: s.ewm(halflife=PLAYER_HALFLIFE).mean())
    df["ewm_before"] = df.groupby(key)["ewm_after"].shift(1)
    return df


def build(games, tg, completed_week, injuries):
    """Returns (projection rows for remaining games, backtest metrics, weekly history)."""
    raw = _raw_stats()
    ctx = tg[["season", "week", "team", "home", "spread", "implied_total"]]
    opp_ctx = tg[["season", "week", "team", "implied_total"]].rename(
        columns={"team": "opp", "implied_total": "opp_implied"})

    k = _with_form(kicker_games(raw), "player_id").merge(ctx, on=["season", "week", "team"], how="left")
    d = _with_form(defense_games(raw, games), "team").merge(ctx, on=["season", "week", "team"], how="left") \
        .merge(opp_ctx, on=["season", "week", "opp"], how="left")
    specs = {"K": (k, ["ewm_before", "implied_total"]), "DST": (d, ["ewm_before", "opp_implied"])}

    models, metrics = {}, {}
    for pos, (df, feats) in specs.items():
        hist = df.dropna(subset=feats)
        # backtest: train on earlier seasons, test on last season
        tr, te = hist[hist["season"] < CURRENT_SEASON - 1], hist[hist["season"] == CURRENT_SEASON - 1]
        m = LinearRegression().fit(tr[feats], tr["pts"])
        resid = te["pts"] - m.predict(te[feats])
        metrics[pos] = {"model_mae": round(float(resid.abs().mean()), 2),
                        "baseline_mae": round(float((te["pts"] - te["ewm_before"]).abs().mean()), 2),
                        "games": int(len(te))}
        final = LinearRegression().fit(hist[feats], hist["pts"])
        models[pos] = (final, feats, float(resid.quantile(0.15)), float(resid.quantile(0.85)))

    upcoming = tg[(tg["season"] == CURRENT_SEASON) & ~tg["played"]]
    rows = []

    # kickers: last kicker to play for each team recently
    kl = k.groupby("player_id").tail(1)
    kl = kl[(kl["season"] == CURRENT_SEASON) & (kl["week"] >= completed_week - 2)]
    kl = kl.sort_values("week").groupby("team").tail(1)
    kf = kl[["player_id", "name", "headshot_url", "team", "ewm_after"]].merge(upcoming, on="team")
    kf = kf.rename(columns={"ewm_after": "ewm_before"})
    rows.append(_predict(kf.assign(position="K"), models["K"]))

    # defenses: every team
    dl = d.groupby("team").tail(1)[["team", "ewm_after"]].rename(columns={"ewm_after": "ewm_before"})
    teams = pd.read_csv(RAW_DIR / "teams.csv").set_index("team_abbr")
    df_ = dl.merge(upcoming, on="team").merge(opp_ctx, on=["season", "week", "opp"], how="left")
    df_["opp_implied"] = df_["opp_implied"].fillna(22.5)
    df_["player_id"] = "DST_" + df_["team"]
    df_["name"] = df_["team"].map(teams["team_nick"]).fillna(df_["team"]) + " D/ST"
    df_["headshot_url"] = df_["team"].map(teams["team_logo_espn"])
    rows.append(_predict(df_.assign(position="DST"), models["DST"]))

    out = pd.concat(rows, ignore_index=True)

    # games already played this week (e.g. Thursday night): keep them, scored with the actual result
    cur_week = int(upcoming["week"].min()) if len(upcoming) else None
    done = []
    if cur_week:
        kd_ = k[(k["season"] == CURRENT_SEASON) & (k["week"] == cur_week)].assign(position="K")
        dd_ = d[(d["season"] == CURRENT_SEASON) & (d["week"] == cur_week)].copy()
        dd_["player_id"] = "DST_" + dd_["team"]
        dd_["name"] = dd_["team"].map(teams["team_nick"]).fillna(dd_["team"]) + " D/ST"
        dd_["headshot_url"] = dd_["team"].map(teams["team_logo_espn"])
        dd_ = dd_.assign(position="DST")
        for x in (kd_, dd_):
            if len(x):
                x = x.merge(tg[["season", "week", "team", "game_id", "gameday", "opp", "home", "spread",
                                "implied_total"]], on=["season", "week", "team"], how="left", suffixes=("_x", ""))
                x = x.assign(proj=x["pts"], floor=x["pts"], ceiling=x["pts"], ewm_ppr=x["ewm_after"])
                done.append(x)
    if done:
        out = pd.concat([out] + done, ignore_index=True)
    status = injuries.dropna(subset=["status"]).set_index("player_id")["status"] if len(injuries) else pd.Series(dtype=str)
    out["status"] = out["player_id"].map(status)
    if "pts" in out:
        out.loc[out["pts"].notna(), "status"] = "Final"
    out["def_factor"] = 1.0
    hist = pd.concat([k.assign(position="K")[["player_id", "season", "week", "opponent_team", "pts"]]
                      .rename(columns={"opponent_team": "opp"}),
                      d.assign(player_id="DST_" + d["team"], position="DST")[["player_id", "season", "week", "opp", "pts"]]])
    hist = hist[hist["season"] == CURRENT_SEASON]
    return out, metrics, hist


def _predict(df, spec):
    model, feats, lo, hi = spec
    df = df.copy()
    df["proj"] = np.clip(model.predict(df[feats]), 0 if df["position"].iloc[0] == "K" else -2, None)
    df["floor"] = df["proj"] + lo
    df["ceiling"] = df["proj"] + hi
    df["ewm_ppr"] = df["ewm_before"]
    return df
