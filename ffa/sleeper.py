"""Read-only access to Sleeper's free public API (no login or API key needed).

Docs: https://docs.sleeper.com
"""
import json
import urllib.error
import urllib.request

BASE = "https://api.sleeper.app/v1"


class SleeperError(Exception):
    pass


def _get(path):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=15) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SleeperError(f"Sleeper returned an error ({e.code}).") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise SleeperError("Couldn't reach Sleeper. Try again in a minute.") from e


def find_user(username):
    """Returns the user dict, or None if the username doesn't exist."""
    user = _get(f"/user/{username.strip()}")
    return user or None


def leagues_for(user_id, season):
    return _get(f"/user/{user_id}/leagues/nfl/{season}") or []


def league_rosters(league_id):
    return _get(f"/league/{league_id}/rosters") or []


def my_roster(rosters, user_id):
    """The roster owned (or co-owned) by this user."""
    for r in rosters:
        if r.get("owner_id") == user_id or user_id in (r.get("co_owners") or []):
            return r
    return None


def rostered_ids(rosters):
    """Every Sleeper player ID on any roster in the league (starters, bench, IR, taxi)."""
    ids = set()
    for r in rosters:
        for key in ("players", "reserve", "taxi"):
            ids.update(r.get(key) or [])
    return ids


def ppr_value(league):
    """Points per reception in this league's scoring (1.0 = full PPR)."""
    return (league.get("scoring_settings") or {}).get("rec", 1.0)


def league_users(league_id):
    return _get(f"/league/{league_id}/users") or []


def matchups(league_id, week):
    return _get(f"/league/{league_id}/matchups/{week}") or []


def team_name(user):
    """A league member's team name, falling back to their display name."""
    meta = user.get("metadata") or {}
    return meta.get("team_name") or user.get("display_name") or "Team"
