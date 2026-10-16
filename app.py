"""Fantasy Football Analytics: projections, start/sit, breakouts, trades, and player profiles (full PPR)."""
import json
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ffa.config import DB_PATH, FANTASY_LAST_WEEK, METRICS_PATH
from ffa.features import FEATURE_LABELS

st.set_page_config(page_title="Fantasy Football Analytics", page_icon="🏈", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&display=swap');
h1, h2, h3, .stTabs button p, [data-testid="stMetricValue"] {
  font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif !important; letter-spacing: .3px; }
h1 { font-weight: 700 !important; text-transform: uppercase; }
.stTabs button p { font-size: 1.15rem !important; font-weight: 600; text-transform: uppercase; }
.cond { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; }

/* at-a-glance cards */
.gl-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 10px; }
.gl-grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
@media (max-width: 900px) { .gl-grid, .gl-grid.three { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.gl-card { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-radius: 10px;
  border: 1px solid rgba(128,128,128,.25); border-left: 6px solid var(--tc); background: rgba(128,128,128,.06);
  min-width: 0; }
.gl-card .hs { width: 54px; height: 54px; border-radius: 50%; object-fit: cover; object-position: top;
  background: rgba(128,128,128,.18); flex: none; }
.gl-card .logo { width: 40px; height: 40px; flex: none; }
.gl-txt { min-width: 0; }
.gl-lbl { font-size: .72rem; text-transform: uppercase; letter-spacing: .6px; opacity: .65; }
.gl-name { font-weight: 700; font-size: 1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.gl-meta { font-size: .8rem; opacity: .75; }
.gl-pts { margin-left: auto; font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 700; line-height: 1; }
.gl-pts small { display: block; font-size: .7rem; font-weight: 500; opacity: .6; text-align: right; }

/* badges */
.badge { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: .72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .4px; vertical-align: middle; }
.b-Out { background: #d62728; color: #fff; }
.b-Doubtful { background: #ff7f0e; color: #fff; }
.b-Questionable { background: #f2c94c; color: #222; }

/* start/sit + profile cards */
.ss-grid { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 14px; }
@media (max-width: 900px) { .ss-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.ss-card { border-radius: 12px; overflow: hidden; border: 1px solid rgba(128,128,128,.25);
  background: rgba(128,128,128,.05); position: relative; }
.ss-card.best { box-shadow: 0 0 0 3px #2ca02c; }
.ss-band { background: var(--tc); color: #fff; display: flex; align-items: center; gap: 8px; padding: 6px 10px;
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.ss-band img { width: 26px; height: 26px; background: rgba(255,255,255,.9); border-radius: 50%; padding: 2px; }
.ss-band .badge { margin-left: auto; }
.ss-body { padding: 12px 14px 14px; text-align: center; }
.ss-body .hs { width: 92px; height: 92px; border-radius: 50%; object-fit: cover; object-position: top;
  background: rgba(128,128,128,.18); }
.ss-name { font-weight: 700; font-size: 1.1rem; margin-top: 4px; }
.ss-meta { font-size: .82rem; opacity: .75; }
.ss-proj { font-family: 'Barlow Condensed', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1.05; margin-top: 6px; }
.ss-proj span { font-size: .8rem; font-weight: 500; opacity: .6; margin-left: 4px; }
.range { position: relative; height: 10px; border-radius: 5px; background: rgba(128,128,128,.2); margin: 10px 4px 4px; }
.range-bar { position: absolute; top: 0; bottom: 0; border-radius: 5px; background: var(--tc); opacity: .55; }
.range-dot { position: absolute; top: -4px; width: 18px; height: 18px; margin-left: -9px; border-radius: 50%;
  background: var(--tc); border: 3px solid #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.2); }
.range-lbl { display: flex; justify-content: space-between; font-size: .75rem; opacity: .7; }
.pick { position: absolute; top: 44px; right: 10px; background: #2ca02c; color: #fff; font-weight: 700;
  font-size: .75rem; padding: 2px 8px; border-radius: 4px; letter-spacing: .5px; }

/* profile header */
.pf { display: flex; gap: 18px; align-items: center; border-radius: 12px; padding: 16px 20px; color: #fff;
  background: linear-gradient(100deg, var(--tc) 0%, var(--tc) 55%, var(--tc2) 140%); flex-wrap: wrap; }
.pf .hs { width: 110px; height: 110px; border-radius: 50%; object-fit: cover; object-position: top;
  background: rgba(255,255,255,.2); border: 3px solid rgba(255,255,255,.85); }
.pf-name { font-family: 'Barlow Condensed', sans-serif; font-size: 2.2rem; font-weight: 700; line-height: 1;
  text-transform: uppercase; }
.pf-sub { opacity: .9; margin-top: 4px; }
.pf-logo { width: 80px; height: 80px; margin-left: auto; filter: drop-shadow(0 1px 2px rgba(0,0,0,.4)); }
.pf-stats { display: flex; gap: 22px; margin-top: 10px; flex-wrap: wrap; }
.pf-stats div { font-family: 'Barlow Condensed', sans-serif; font-size: 1.6rem; font-weight: 700; line-height: 1; }
.pf-stats small { display: block; font-family: inherit; font-size: .7rem; font-weight: 500; opacity: .8;
  text-transform: uppercase; letter-spacing: .5px; }
</style>
""", unsafe_allow_html=True)

POS_COLORS = {"QB": "#e15759", "RB": "#59a14f", "WR": "#4e79a7", "TE": "#f28e2b"}
STATUS_STYLE = {"Out": "background-color:#d62728;color:white", "Doubtful": "background-color:#ff7f0e;color:white",
                "Questionable": "background-color:#f2c94c;color:#222"}


# ---------- data ----------
@st.cache_data
def query(sql):
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


teams = query("SELECT * FROM teams").set_index("team")
proj = query("SELECT * FROM projections")
breakouts = query("SELECT * FROM breakouts")
usage = query("SELECT * FROM usage_weekly")
season = int(proj["gameday"].str[:4].min())
games_cur = query(f"SELECT * FROM player_weeks WHERE season = {season}")
metrics = load_metrics()

last_week = int(proj["last_completed_week"].iloc[0])
next_week = int(proj["week"].min())
injury_posted = proj.loc[proj["week"] == next_week, "status"].notna().any()


def team_attr(abbr, col, default=""):
    return teams[col].get(abbr, default) if abbr in teams.index else default


for df in (proj, breakouts):
    df["color"] = df["team"].map(teams["team_color"]).fillna("#555")
    df["logo"] = df["team"].map(teams["logo"])

proj["label"] = proj["name"] + " (" + proj["position"] + ", " + proj["team"] + ")"
proj["matchup"] = proj.apply(lambda r: ("vs " if r["home"] else "@ ") + r["opp"], axis=1)
proj["matchup_grade"] = pd.cut(proj["def_factor"], [0, 0.85, 1.15, 99],
                               labels=["Tough", "Neutral", "Good"]).astype(str)
proj["status"] = proj["status"].fillna("")
players = (proj.sort_values("ewm_ppr", ascending=False)
               .drop_duplicates("player_id")[["player_id", "label", "name", "position", "team"]])
labels = players["label"].tolist()
id_by_label = dict(zip(players["label"], players["player_id"]))
label_by_id = dict(zip(players["player_id"], players["label"]))


def html(s):
    """Render an HTML snippet. Lines are flattened so Streamlit doesn't read indentation as a code block."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def badge(status):
    return f'<span class="badge b-{status}">{status}</span>' if status else ""


def week_row(pid, week):
    r = proj[(proj["player_id"] == pid) & (proj["week"] == week)]
    return r.iloc[0] if len(r) else None


# ---------- header ----------
st.title("Fantasy Football Analytics")
st.caption(f"Full PPR · {season} season · stats through Week {last_week} · built on nflverse data")

wk = proj[(proj["week"] == next_week) & (proj["status"] != "Out")]
cards = []
for pos in ["QB", "RB", "WR", "TE"]:
    r = wk[wk["position"] == pos].nlargest(1, "proj").iloc[0]
    cards.append(f"""<div class="gl-card" style="--tc:{r['color']}">
      <img class="hs" src="{r['headshot_url']}"><div class="gl-txt">
      <div class="gl-lbl">Top {pos} · Week {next_week}</div><div class="gl-name">{r['name']}</div>
      <div class="gl-meta">{r['team']} {r['matchup']} {badge(r['status'])}</div></div>
      <div class="gl-pts">{r['proj']:.1f}<small>proj</small></div></div>""")
html(f'<div class="gl-grid">{"".join(cards)}</div>')

smash = wk[wk["proj"] >= 12].nlargest(1, "def_factor").iloc[0]
bo = breakouts.iloc[0]
hi = wk.drop_duplicates("team").nlargest(1, "implied_total").iloc[0]
html(f"""<div class="gl-grid three">
  <div class="gl-card" style="--tc:{smash['color']}"><img class="hs" src="{smash['headshot_url']}"><div class="gl-txt">
    <div class="gl-lbl">Best matchup</div><div class="gl-name">{smash['name']}</div>
    <div class="gl-meta">{smash['matchup']} allows {(smash['def_factor'] - 1) * 100:+.0f}% to {smash['position']}s</div></div></div>
  <div class="gl-card" style="--tc:{bo['color']}"><img class="hs" src="{bo['headshot_url']}"><div class="gl-txt">
    <div class="gl-lbl">Top breakout candidate</div><div class="gl-name">{bo['name']}</div>
    <div class="gl-meta">Targets + carries {bo['opps_before']:.1f} → {bo['opps_last3']:.1f} per game</div></div></div>
  <div class="gl-card" style="--tc:{hi['color']}"><img class="logo" src="{hi['logo']}"><div class="gl-txt">
    <div class="gl-lbl">Highest-scoring offense</div><div class="gl-name">{team_attr(hi['team'], 'team_name', hi['team'])}</div>
    <div class="gl-meta">Vegas expects {hi['implied_total']:.1f} pts {hi['matchup']}</div></div></div>
</div>""")

if not injury_posted:
    st.caption(f"🩺 The Week {next_week} injury report isn't final yet. Statuses update automatically "
               "Saturday and Sunday mornings.")


# ---------- player profile (used in two places) ----------
def render_profile(pid, key):
    info = proj[proj["player_id"] == pid].iloc[0]
    tc = info["color"]
    tc2 = team_attr(info["team"], "team_color2", tc)
    log = games_cur[games_cur["player_id"] == pid].sort_values("week")
    u = usage[usage["player_id"] == pid].sort_values("week")
    fut = proj[proj["player_id"] == pid].sort_values("week")

    # season rank by points per game among players at the position with 3+ games
    pos_ppg = (games_cur[games_cur["position"] == info["position"]].groupby("player_id")["ppr"]
               .agg(["mean", "size"]).query("size >= 3")["mean"].sort_values(ascending=False))
    rank = (list(pos_ppg.index).index(pid) + 1) if pid in pos_ppg.index else None
    ppg = log["ppr"].mean() if len(log) else 0
    nxt = fut.iloc[0]

    html(f"""<div class="pf" style="--tc:{tc};--tc2:{tc2}">
      <img class="hs" src="{info['headshot_url']}">
      <div><div class="pf-name">{info['name']} {badge(nxt['status'] if nxt['week'] == next_week else '')}</div>
        <div class="pf-sub">{info['position']} · {team_attr(info['team'], 'team_name', info['team'])}</div>
        <div class="pf-stats">
          <div>{ppg:.1f}<small>PPR per game</small></div>
          <div>{info['position']}{rank if rank else '–'}<small>Rank (per game)</small></div>
          <div>{log['ppr'].sum():.0f}<small>Season points</small></div>
          <div>{nxt['proj']:.1f}<small>Week {int(nxt['week'])} {nxt['matchup']}</small></div>
          <div>{fut[fut['week'] <= FANTASY_LAST_WEEK]['proj'].sum():.0f}<small>Rest of season</small></div>
        </div></div>
      <img class="pf-logo" src="{info['logo']}">
    </div>""")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=log["week"], y=log["ppr"], name="Actual", marker_color=tc,
                             customdata=log["opponent_team"],
                             hovertemplate="Week %{x} vs %{customdata}: %{y:.1f} pts<extra></extra>"))
        if len(u):
            fig.add_trace(go.Scatter(x=u["week"], y=u["xfp"], name="Expected (usage)", mode="lines+markers",
                                     line=dict(color="#888", width=2, dash="dot")))
        fig.add_trace(go.Scatter(
            x=fut["week"], y=fut["proj"], name="Projected", mode="markers",
            marker=dict(color=tc, size=11, symbol="diamond", line=dict(color="white", width=1)),
            error_y=dict(type="data", symmetric=False, array=fut["ceiling"] - fut["proj"],
                         arrayminus=fut["proj"] - fut["floor"], color="rgba(128,128,128,.35)",
                         thickness=6, width=0),
            customdata=fut[["matchup", "floor", "ceiling"]],
            hovertemplate="Week %{x} %{customdata[0]}: %{y:.1f} proj (%{customdata[1]:.1f} to "
                          "%{customdata[2]:.1f})<extra></extra>"))
        if ppg:
            fig.add_hline(y=ppg, line=dict(color="rgba(128,128,128,.6)", dash="dash"),
                          annotation_text=f"Season avg {ppg:.1f}", annotation_position="top right")
        fig.update_layout(title="Season so far and projections ahead", height=360,
                          margin=dict(l=10, r=10, t=50, b=60),
                          xaxis=dict(dtick=1, title="Week", range=[0.4, max(fut["week"].max(), 18) + 0.6]),
                          yaxis_title="PPR points", legend=dict(orientation="h", y=-0.22, x=0))
        st.plotly_chart(fig, width="stretch", key=f"{key}_pts")
    with c2:
        fig = go.Figure()
        if info["position"] != "QB":
            fig.add_trace(go.Scatter(x=u["week"], y=u["target_share"] * 100, name="Target share",
                                     mode="lines+markers", line=dict(color=tc, width=3)))
        if info["position"] in ("RB", "QB"):
            fig.add_trace(go.Scatter(x=u["week"], y=u["carry_share"] * 100, name="Carry share",
                                     mode="lines+markers", line=dict(color="#888", width=3)))
        fig.update_layout(title="Share of team targets and carries", height=360, margin=dict(l=10, r=10, t=50, b=60),
                          xaxis=dict(dtick=1, title="Week"), yaxis=dict(title="% of team", range=[0, 105]),
                          legend=dict(orientation="h", y=-0.22, x=0))
        st.plotly_chart(fig, width="stretch", key=f"{key}_use")

    pos = info["position"]
    cols = {"week": "Wk", "opponent_team": "Opp", "ppr": "PPR"}
    if pos == "QB":
        cols.update({"completions": "Cmp", "attempts": "Att", "passing_yards": "Pass Yds",
                     "passing_tds": "Pass TD", "passing_interceptions": "INT", "carries": "Rush",
                     "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD"})
    elif pos == "RB":
        cols.update({"carries": "Rush", "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
                     "targets": "Tgt", "receptions": "Rec", "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD"})
    else:
        cols.update({"targets": "Tgt", "receptions": "Rec", "receiving_yards": "Rec Yds",
                     "receiving_tds": "Rec TD", "target_share": "Tgt Share"})
    gl = log[list(cols)].rename(columns=cols).sort_values("Wk", ascending=False)
    if "Tgt Share" in gl:
        gl["Tgt Share"] = (gl["Tgt Share"] * 100).round(0).astype(int).astype(str) + "%"
    st.markdown("**Game log**")
    st.dataframe(gl, hide_index=True, width="stretch",
                 column_config={"PPR": st.column_config.NumberColumn(format="%.1f")})


# ---------- tabs ----------
tab_proj, tab_player, tab_ss, tab_bo, tab_trade, tab_model = st.tabs(
    ["Projections", "Players", "Start / Sit", "Breakouts", "Trade Analyzer", "Model"])

with tab_proj:
    c1, c2, c3 = st.columns([1, 2, 2])
    week = c1.selectbox("Week", sorted(proj["week"].unique()), key="pw")
    pos = c2.multiselect("Position", ["QB", "RB", "WR", "TE"], default=["QB", "RB", "WR", "TE"])
    search = c3.text_input("Search player", "")

    t = proj[(proj["week"] == week) & proj["position"].isin(pos)]
    if search:
        t = t[t["name"].str.contains(search, case=False)]
    t = t.sort_values("proj", ascending=False).reset_index(drop=True)
    t["rank"] = t["position"] + (t.groupby("position").cumcount() + 1).astype(str)
    show = t[["headshot_url", "logo", "rank", "name", "status", "matchup", "matchup_grade",
              "implied_total", "floor", "proj", "ceiling"]]
    styled = show.style.map(lambda s: STATUS_STYLE.get(s, ""), subset=["status"]).map(
        lambda g: {"Good": "color:#2ca02c;font-weight:600", "Tough": "color:#d62728;font-weight:600"}.get(g, ""),
        subset=["matchup_grade"])

    st.caption("Click a row to open that player's profile below the table.")
    sel = st.dataframe(
        styled, hide_index=True, width="stretch", height=520, on_select="rerun",
        selection_mode="single-row", key="proj_table",
        column_config={
            "headshot_url": st.column_config.ImageColumn("", width="small"),
            "logo": st.column_config.ImageColumn("", width="small"),
            "rank": "Rank", "name": "Player", "status": "Status", "matchup": "Opp",
            "matchup_grade": "Matchup",
            "implied_total": st.column_config.NumberColumn("Team implied", format="%.1f",
                                                           help="Points Vegas expects this team to score"),
            "floor": st.column_config.NumberColumn("Floor", format="%.1f", help="15th percentile outcome"),
            "proj": st.column_config.ProgressColumn("Projection", format="%.1f", min_value=0,
                                                    max_value=float(proj["proj"].max())),
            "ceiling": st.column_config.NumberColumn("Ceiling", format="%.1f", help="85th percentile outcome"),
        })
    rows = sel.selection.rows if sel else []
    if rows:
        st.divider()
        render_profile(t.iloc[rows[0]]["player_id"], "proj_pf")

with tab_player:
    pick = st.selectbox("Find a player", labels, index=0, key="pf_pick")
    render_profile(id_by_label[pick], "tab_pf")

with tab_ss:
    st.subheader("Who should I start?")
    c1, c2 = st.columns([1, 3])
    ss_week = c1.selectbox("Week", sorted(proj["week"].unique()), key="ssw")
    picks = c2.multiselect("Pick 2 to 4 players", labels, max_selections=4, placeholder="Start typing a name...")

    if len(picks) < 2:
        st.info("Choose at least two players to compare.")
    else:
        rows = [(lab, week_row(id_by_label[lab], ss_week)) for lab in picks]
        playing = [r for _, r in rows if r is not None]
        on_bye = [lab for lab, r in rows if r is None]
        if on_bye:
            st.warning(f"No game in Week {ss_week} for: {', '.join(on_bye)}")
        if playing:
            df = pd.DataFrame(playing)
            live = df[df["status"] != "Out"]
            best_id = live.loc[live["proj"].idxmax(), "player_id"] if len(live) else None
            scale = max(df["ceiling"].max(), 1)
            cards_html = []
            for _, r in df.sort_values("proj", ascending=False).iterrows():
                f, p, c = r["floor"] / scale * 100, r["proj"] / scale * 100, r["ceiling"] / scale * 100
                cards_html.append(f"""<div class="ss-card {'best' if r['player_id'] == best_id else ''}" style="--tc:{r['color']}">
                  <div class="ss-band"><img src="{r['logo']}">{team_attr(r['team'], 'team_nick', r['team'])}{badge(r['status'])}</div>
                  {'<div class="pick">START</div>' if r['player_id'] == best_id else ''}
                  <div class="ss-body"><img class="hs" src="{r['headshot_url']}">
                    <div class="ss-name">{r['name']}</div>
                    <div class="ss-meta">{r['position']} · {r['matchup']} · {r['matchup_grade']} matchup</div>
                    <div class="ss-proj">{r['proj']:.1f}<span>proj</span></div>
                    <div class="range"><div class="range-bar" style="left:{f}%;width:{c - f}%"></div>
                      <div class="range-dot" style="left:{p}%"></div></div>
                    <div class="range-lbl"><span>Floor {r['floor']:.1f}</span><span>Ceiling {r['ceiling']:.1f}</span></div>
                  </div></div>""")
            html(f'<div class="ss-grid" style="--n:{len(df)}">{"".join(cards_html)}</div>')

            if len(live) >= 2:
                safe = live.loc[live["floor"].idxmax()]
                boom = live.loc[live["ceiling"].idxmax()]
                top2 = live["proj"].nlargest(2)
                gap = top2.iloc[0] - top2.iloc[1]
                st.write("")
                if gap < 1.5 and safe["player_id"] != boom["player_id"]:
                    st.info(f"**It's close** ({gap:.1f} points apart). Favored to win your matchup? "
                            f"Lean **{safe['name']}** for the safer floor. Underdog? Lean **{boom['name']}** "
                            "for the higher ceiling.")
                else:
                    st.success(f"**Start {live.loc[live['proj'].idxmax(), 'name']}.** "
                               f"Projected {gap:.1f} points ahead of the next option.")
            if (df["status"] == "Questionable").any():
                st.caption("Questionable players usually play, but check news before kickoff.")

with tab_bo:
    st.subheader("Breakout detector")
    st.write("Fantasy points bounce around week to week, but **targets and carries are sticky**. "
             "This ranks players whose usage jumped over their last 3 games, plus players whose "
             "points haven't caught up to their usage yet. **xFP** = expected fantasy points from "
             "opportunities alone.")
    bpos = st.radio("Position", ["All", "QB", "RB", "WR", "TE"], horizontal=True, key="bpos")
    b = breakouts if bpos == "All" else breakouts[breakouts["position"] == bpos]
    b = b.head(25).copy()
    b["usage"] = b.apply(lambda r: f"{r['opps_before']:.1f} → {r['opps_last3']:.1f}", axis=1)
    b["signal"] = b.apply(lambda r: " · ".join(
        s for s, ok in [("Usage spike", r["usage_change"] >= 3), ("Due for points", r["points_gap"] >= 3)] if ok),
        axis=1)
    sel = st.dataframe(
        b[["headshot_url", "logo", "name", "position", "usage", "xfp_before", "xfp_last3", "ppr_last3",
           "signal", "breakout_score"]],
        hide_index=True, width="stretch", height=480, on_select="rerun", selection_mode="single-row",
        key="bo_table",
        column_config={
            "headshot_url": st.column_config.ImageColumn("", width="small"),
            "logo": st.column_config.ImageColumn("", width="small"),
            "name": "Player", "position": "Pos",
            "usage": st.column_config.TextColumn("Targets + carries / game", help="Earlier this season → last 3 games"),
            "xfp_before": st.column_config.NumberColumn("xFP before", format="%.1f"),
            "xfp_last3": st.column_config.NumberColumn("xFP last 3", format="%.1f"),
            "ppr_last3": st.column_config.NumberColumn("Actual last 3", format="%.1f"),
            "signal": "Signal",
            "breakout_score": st.column_config.ProgressColumn(
                "Breakout score", format="%.1f", min_value=0,
                max_value=float(max(breakouts["breakout_score"].max(), 1))),
        })
    rows = sel.selection.rows if sel else []
    st.caption("Click a row to see that player's full profile.")
    if rows and b.iloc[rows[0]]["player_id"] in label_by_id:
        st.divider()
        render_profile(b.iloc[rows[0]]["player_id"], "bo_pf")

with tab_trade:
    st.subheader("Trade analyzer")
    st.write("Compares rest-of-season projected points, including each player's remaining schedule and bye week.")
    weeks = sorted(proj["week"].unique())
    end_default = FANTASY_LAST_WEEK if FANTASY_LAST_WEEK in weeks else weeks[-1]
    w0, w1 = st.slider("Weeks to count", min_value=weeks[0], max_value=weeks[-1], value=(weeks[0], end_default))
    c1, c2 = st.columns(2)
    give = c1.multiselect("You give", labels, key="give", placeholder="Players you're sending")
    get = c2.multiselect("You get", labels, key="get", placeholder="Players you're receiving")

    def ros(labs):
        out = []
        for lab in labs:
            r = proj[(proj["player_id"] == id_by_label[lab]) & proj["week"].between(w0, w1)]
            out.append({"Player": lab, "Games": len(r), "Pts per game": r["proj"].mean() if len(r) else 0,
                        "ROS points": r["proj"].sum()})
        return pd.DataFrame(out)

    if give and get:
        g, r = ros(give), ros(get)
        gt, rt = g["ROS points"].sum(), r["ROS points"].sum()
        diff = rt - gt
        pct = diff / gt * 100 if gt else 0
        verdict, fn = (("Fair trade", st.info) if abs(pct) < 5 else
                       ("You win this trade", st.success) if pct > 0 else ("You lose this trade", st.error))
        fn(f"**{verdict}** · you {'gain' if diff >= 0 else 'lose'} {abs(diff):.1f} projected points "
           f"({pct:+.0f}%) from Week {w0} to {w1}")
        cfg = {"Pts per game": st.column_config.NumberColumn(format="%.1f"),
               "ROS points": st.column_config.NumberColumn(format="%.1f")}
        a, bcol = st.columns(2)
        a.metric("You give", f"{gt:.1f} pts")
        a.dataframe(g, hide_index=True, width="stretch", column_config=cfg)
        bcol.metric("You get", f"{rt:.1f} pts", f"{diff:+.1f}")
        bcol.dataframe(r, hide_index=True, width="stretch", column_config=cfg)

        def weekly(labs):
            ids = [id_by_label[l] for l in labs]
            w = proj[proj["player_id"].isin(ids) & proj["week"].between(w0, w1)]
            return w.groupby("week")["proj"].sum().reindex(range(w0, w1 + 1), fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(w0, w1 + 1)), y=weekly(give), name="You give", marker_color="#e15759"))
        fig.add_trace(go.Bar(x=list(range(w0, w1 + 1)), y=weekly(get), name="You get", marker_color="#59a14f"))
        fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=30, b=30), xaxis_title="Week",
                          yaxis_title="Projected points", xaxis=dict(dtick=1), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("Points aren't everything: getting 2 players for 1 only helps if they'd crack your starting lineup.")
    else:
        st.info("Add at least one player to each side.")

with tab_model:
    st.subheader("How the projections work")
    st.markdown(
        "- **Data:** weekly stats for every QB, RB, WR, and TE since 2021 from "
        "[nflverse](https://github.com/nflverse), stored in a SQLite database, plus weekly injury reports.\n"
        "- **Features (all known before kickoff):** recent form weighted toward the latest games "
        "(points, targets, carries, target share, air yards), the opponent's points allowed to that "
        "position, and the Vegas implied team total.\n"
        "- **Model:** gradient-boosted trees (scikit-learn). Floor and ceiling come from how actual "
        "scores spread around projections in a held-out season. Players ruled Out are set to 0.\n"
        "- **Baseline:** each player's recent weighted average. A useful model has to beat it.")
    st.markdown("#### Backtest: predicting games the model never saw")
    for label, rows_ in metrics["backtest"].items():
        m = pd.DataFrame(rows_)
        a = m[m["position"] == "ALL"].iloc[0]
        better = (1 - a["model_mae"] / a["baseline_mae"]) * 100
        st.markdown(f"**{label}:** average miss of **{a['model_mae']:.2f} pts** vs {a['baseline_mae']:.2f} for the "
                    f"baseline (**{better:.1f}% more accurate**), across {a['games']:,} player-games. "
                    f"{a['range_hit_rate'] * 100:.0f}% of results landed between floor and ceiling.")
        st.dataframe(m.rename(columns={"position": "Position", "games": "Player-games",
                                       "model_mae": "Model avg miss", "baseline_mae": "Baseline avg miss",
                                       "range_hit_rate": "In floor-ceiling range", "rank_corr": "Rank correlation"}),
                     hide_index=True, width="stretch")
    st.markdown("#### What drives the projections")
    imp = pd.DataFrame(metrics["importance"]).head(10)
    imp["feature"] = imp["feature"].map(
        lambda f: FEATURE_LABELS.get(f, f.replace("ewm_", "Recent ").replace("_", " ")))
    fig = go.Figure(go.Bar(x=imp["importance"][::-1], y=imp["feature"][::-1], orientation="h", marker_color="#4e79a7"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="Increase in average miss when the feature is scrambled (pts)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Evaluated on fantasy-relevant players (recent form of 6+ PPR points per game).")
