"""Every trade in a Sleeper league since it was created, with traded draft picks resolved to the
player they turned into once that draft has happened.

Sleeper starts a new league ID each season and links it to the last one with previous_league_id,
so we walk that chain back to the first season and pull every week's transactions from each.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

WEEKS = range(0, 19)   # offseason moves are filed under week 0 or 1


def league_chain(league_id, api):
    """[(season, league dict), ...] newest first, following previous_league_id back to the start."""
    out, seen = [], set()
    lid = league_id
    while lid and lid not in seen and lid != "0":
        seen.add(lid)
        lg = api.league(lid)
        if not lg:
            break
        out.append((int(lg.get("season") or 0), lg))
        lid = lg.get("previous_league_id")
    return out


def _season_trades(lg, api):
    lid = lg["league_id"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        weeks = list(ex.map(lambda w: _safe(api.transactions, lid, w), WEEKS))
    trades, ids = [], set()
    for w, txs in zip(WEEKS, weeks):
        for t in txs:
            if t.get("type") == "trade" and t.get("status") == "complete" and t["transaction_id"] not in ids:
                ids.add(t["transaction_id"])
                trades.append({**t, "_week": w})
    return trades


def _safe(fn, *a):
    try:
        return fn(*a) or []
    except Exception:
        return []


def _draft_results(lg, api, rounds):
    """{(season, round, original roster_id): pick dict} for the league's rookie draft that season."""
    out = {}
    ds = [d for d in _safe(api.drafts, lg["league_id"]) if d.get("status") in ("complete", "drafting")
          and int((d.get("settings") or {}).get("rounds", 0)) <= 10]      # a big startup draft isn't a rookie draft
    if not ds:
        return out
    # skip a startup draft: prefer the draft whose round count matches the rookie draft
    ds.sort(key=lambda d: (int((d.get("settings") or {}).get("rounds", 0)) != rounds, -(d.get("start_time") or 0)))
    d = api.draft(ds[0]["draft_id"]) or ds[0]
    slot_to_roster = {int(k): v for k, v in (d.get("slot_to_roster_id") or {}).items()}
    n = len(slot_to_roster) or 1
    season = int(d.get("season") or lg.get("season") or 0)
    for p in _safe(api.draft_picks, ds[0]["draft_id"]):
        orig = slot_to_roster.get(int(p.get("draft_slot") or 0))
        if orig is None or not p.get("player_id"):
            continue
        meta = p.get("metadata") or {}
        out[(season, int(p["round"]), orig)] = {
            "player_id": str(p["player_id"]),
            "name": f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip(),
            "position": meta.get("position"), "team": meta.get("team"),
            "slot": f"{p['round']}.{(int(p.get('pick_no', 1)) - 1) % n + 1:02d}",
        }
    return out


def collect(league_id, api, rounds=5):
    """Returns (trades, teams).

    trades: list of dicts, newest first, each with season, week, time, and sides:
        {roster_id: {"players": [sleeper ids], "picks": [pick dicts], "faab": dollars,
                     "sent_players": [...], "sent_picks": [...]}}
        A pick dict has season, round, orig (original roster), and "became" when its draft happened.
    teams: {roster_id: {"owner", "avatar"}} from the newest season, so names match what the league uses now.
    """
    chain = league_chain(league_id, api)
    drafted = {}
    raw = []
    for season, lg in chain:
        drafted.update(_draft_results(lg, api, rounds))
        raw += [(season, t) for t in _season_trades(lg, api)]

    trades = []
    for season, t in raw:
        rids = [int(r) for r in (t.get("roster_ids") or [])]
        sides = {r: {"players": [], "picks": [], "faab": 0, "sent_players": [], "sent_picks": []} for r in rids}
        for pid, rid in (t.get("adds") or {}).items():
            sides.setdefault(int(rid), {"players": [], "picks": [], "faab": 0, "sent_players": [], "sent_picks": []})
            sides[int(rid)]["players"].append(str(pid))
        for pid, rid in (t.get("drops") or {}).items():
            if int(rid) in sides:
                sides[int(rid)]["sent_players"].append(str(pid))
        for p in t.get("draft_picks") or []:
            pk = {"season": int(p["season"]), "round": int(p["round"]), "orig": int(p["roster_id"])}
            pk["became"] = drafted.get((pk["season"], pk["round"], pk["orig"]))
            if int(p["owner_id"]) in sides:
                sides[int(p["owner_id"])]["picks"].append(pk)
            if p.get("previous_owner_id") is not None and int(p["previous_owner_id"]) in sides:
                sides[int(p["previous_owner_id"])]["sent_picks"].append(pk)
        for b in t.get("waiver_budget") or []:
            if int(b["receiver"]) in sides:
                sides[int(b["receiver"])]["faab"] += int(b.get("amount", 0))
        ts = (t.get("status_updated") or t.get("created") or 0) / 1000
        trades.append({"id": t["transaction_id"], "season": season, "week": t.get("leg", t["_week"]),
                       "time": datetime.fromtimestamp(ts, timezone.utc) if ts else None, "sides": sides})
    trades.sort(key=lambda x: x["time"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    teams = {}
    if chain:
        lg = chain[0][1]
        users = {u["user_id"]: u for u in _safe(api.league_users, lg["league_id"])}
        for r in _safe(api.league_rosters, lg["league_id"]):
            u = users.get(r.get("owner_id"), {})
            teams[int(r["roster_id"])] = {"owner": api.team_name(u) if u else f"Team {r['roster_id']}",
                                          "avatar": api.avatar_url(u) if u else None}
    return trades, teams
