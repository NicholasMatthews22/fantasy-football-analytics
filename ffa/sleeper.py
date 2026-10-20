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


def avatar_url(user):
    """A manager's team avatar if they set one, otherwise their profile picture."""
    meta = user.get("metadata") or {}
    if str(meta.get("avatar", "")).startswith("http"):
        return meta["avatar"]
    return f"https://sleepercdn.com/avatars/thumbs/{user['avatar']}" if user.get("avatar") else None


def traded_picks(league_id):
    """Every draft pick that has changed hands: season, round, roster_id (original team), owner_id (current team)."""
    return _get(f"/league/{league_id}/traded_picks") or []


def pick_ownership(rosters, traded, seasons, rounds):
    """{current roster_id: [(season, round, original roster_id), ...]} starting from every team owning its own picks."""
    owner = {}
    for r in rosters:
        rid = r.get("roster_id")
        for s in seasons:
            for rd in range(1, rounds + 1):
                owner[(s, rd, rid)] = rid
    for t in traded:
        key = (int(t.get("season", 0)), int(t.get("round", 0)), t.get("roster_id"))
        if key in owner and t.get("owner_id") is not None:
            owner[key] = t["owner_id"]
    out = {}
    for (s, rd, orig), cur in owner.items():
        out.setdefault(cur, []).append((s, rd, orig))
    for v in out.values():
        v.sort()
    return out
