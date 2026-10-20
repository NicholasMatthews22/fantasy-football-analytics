"""Trade market values from FantasyCalc (free, no key), built from thousands of real fantasy trades.

Values depend on league format, so we request the version that matches the user's league:
dynasty or redraft, 1QB or superflex, league size, and PPR setting. Dynasty values also include
rookie draft picks, named like "2027 1st", "2027 1st (Early)", "2028 2nd".
"""
import json
import re
import urllib.request

URL = ("https://api.fantasycalc.com/values/current?isDynasty={dynasty}&numQbs={qbs}"
       "&numTeams={teams}&ppr={ppr}")
PICK_RE = re.compile(r"^(\d{4}) (\d)(?:st|nd|rd|th)(?: \((Early|Mid|Late)\))?$")


def fetch_rows(dynasty=False, superflex=False, teams=12, ppr=1.0):
    """Raw FantasyCalc rows. Raises on network errors so the caller can fall back."""
    ppr = 1 if ppr >= 0.75 else (0.5 if ppr >= 0.25 else 0)
    url = URL.format(dynasty=str(bool(dynasty)).lower(), qbs=2 if superflex else 1, teams=int(teams), ppr=ppr)
    req = urllib.request.Request(url, headers={"User-Agent": "RedZoneLab/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def player_values(rows):
    """{sleeper_id: value} for players."""
    out = {}
    for row in rows:
        p = row.get("player") or {}
        if p.get("position") != "PICK" and p.get("sleeperId") and row.get("value") is not None:
            out[str(p["sleeperId"])] = float(row["value"])
    return out


def pick_values(rows, seasons, rounds):
    """{(season, round, tier): value} where tier is "Early", "Mid", "Late", or None (unknown slot).

    FantasyCalc usually lists only the first few rounds and the next two or three draft years.
    Missing rounds are extended from the drop-off between the last two listed rounds; a missing
    year uses the latest listed year's values (future picks trade at roughly the same price).
    """
    raw = {}
    for row in rows:
        p = row.get("player") or {}
        m = PICK_RE.match(p.get("name") or "")
        if p.get("position") == "PICK" and m and row.get("value") is not None:
            raw[(int(m.group(1)), int(m.group(2)), m.group(3))] = float(row["value"])
    if not raw:
        return {}
    out = {}
    have_years = sorted({k[0] for k in raw})
    for yr in seasons:
        src = yr if yr in have_years else max(y for y in have_years if y <= yr) if any(y <= yr for y in have_years) else have_years[0]
        base = {rd: raw.get((src, rd, None)) for rd in range(1, rounds + 1)}
        listed = [rd for rd, v in base.items() if v]
        for rd in range(1, rounds + 1):
            if base[rd] is None and listed:
                last = [r_ for r_ in listed if r_ < rd]
                if len(last) >= 2:
                    ratio = max(0.25, min(0.8, base[last[-1]] / base[last[-2]]))
                    base[rd] = base[last[-1]] * ratio ** (rd - last[-1])
                elif last:
                    base[rd] = base[last[-1]] * 0.45 ** (rd - last[-1])
        for rd in range(1, rounds + 1):
            if base[rd]:
                out[(yr, rd, None)] = round(base[rd])
                for tier in ("Early", "Mid", "Late"):
                    out[(yr, rd, tier)] = round(raw.get((src, rd, tier), base[rd]))
    return out
