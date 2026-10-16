"""Trade market values from FantasyCalc (free, no key), built from thousands of real fantasy trades.

Values depend on league format, so we request the version that matches the user's league:
dynasty or redraft, 1QB or superflex, league size, and PPR setting.
"""
import json
import urllib.request

URL = ("https://api.fantasycalc.com/values/current?isDynasty={dynasty}&numQbs={qbs}"
       "&numTeams={teams}&ppr={ppr}")


def fetch_values(dynasty=False, superflex=False, teams=12, ppr=1.0):
    """Returns {sleeper_id: value}. Raises on network errors so the caller can fall back."""
    ppr = 1 if ppr >= 0.75 else (0.5 if ppr >= 0.25 else 0)
    url = URL.format(dynasty=str(bool(dynasty)).lower(), qbs=2 if superflex else 1, teams=int(teams), ppr=ppr)
    req = urllib.request.Request(url, headers={"User-Agent": "RedZoneLab/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        rows = json.load(r)
    out = {}
    for row in rows:
        player = row.get("player") or {}
        sid = player.get("sleeperId")
        if sid and row.get("value") is not None:
            out[str(sid)] = float(row["value"])
    return out
