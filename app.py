"""Fantasy Football Analytics: projections, start/sit, breakouts, and trades (full PPR)."""
import json
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ffa.config import DB_PATH, FANTASY_LAST_WEEK, METRICS_PATH
from ffa.features import FEATURE_LABELS

st.set_page_config(page_title="Fantasy Football Analytics", page_icon="🏈", layout="wide")

POS_COLORS = {"QB": "#e15759", "RB": "#59a14f", "WR": "#4e79a7", "TE": "#f28e2b"}


# ---------- data ----------
@st.cache_data
def load(table):
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(f"SELECT * FROM {table}", con)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


proj = load("projections")
breakouts = load("breakouts")
usage = load("usage_weekly")
metrics = load_metrics()

last_week = int(proj["last_completed_week"].iloc[0])
next_week = int(proj["week"].min())
season = int(proj["gameday"].str[:4].min())

proj["label"] = proj["name"] + " (" + proj["position"] + ", " + proj["team"] + ")"
proj["matchup"] = proj.apply(lambda r: ("vs " if r["home"] else "@ ") + r["opp"], axis=1)
proj["matchup_grade"] = pd.cut(proj["def_factor"], [0, 0.85, 1.15, 99], labels=["Tough", "Neutral", "Good"])
players = (proj.sort_values("ewm_ppr", ascending=False)
               .drop_duplicates("player_id")[["player_id", "label", "position", "team", "headshot_url"]])
labels = players["label"].tolist()
id_by_label = dict(zip(players["label"], players["player_id"]))


def fmt_pts(x):
    return f"{x:.1f}"


# ---------- header ----------
st.title("Fantasy Football Analytics")
st.caption(f"Full PPR · {season} season · data through Week {last_week} · "
           "Projections update weekly from nflverse play-by-play stats")

tab_proj, tab_ss, tab_bo, tab_trade, tab_model = st.tabs(
    ["📊 Projections", "⚖️ Start / Sit", "📈 Breakouts", "🔁 Trade Analyzer", "🧪 Model"])

# ---------- projections ----------
with tab_proj:
    c1, c2, c3 = st.columns([1, 2, 2])
    week = c1.selectbox("Week", sorted(proj["week"].unique()), key="pw")
    pos = c2.multiselect("Position", ["QB", "RB", "WR", "TE"], default=["QB", "RB", "WR", "TE"])
    search = c3.text_input("Search player", "")

    t = proj[(proj["week"] == week) & proj["position"].isin(pos)]
    if search:
        t = t[t["name"].str.contains(search, case=False)]
    t = t.sort_values("proj", ascending=False)
    t["pos_rank"] = t.groupby("position").cumcount() + 1
    t["rank"] = t["position"] + t["pos_rank"].astype(str)

    st.dataframe(
        t[["headshot_url", "rank", "name", "team", "matchup", "matchup_grade", "implied_total",
           "floor", "proj", "ceiling"]],
        hide_index=True, width="stretch", height=560,
        column_config={
            "headshot_url": st.column_config.ImageColumn("", width="small"),
            "rank": "Rank", "name": "Player", "team": "Team", "matchup": "Opp",
            "matchup_grade": "Matchup",
            "implied_total": st.column_config.NumberColumn("Team implied pts", format="%.1f",
                                                           help="Points Vegas expects this team to score"),
            "floor": st.column_config.NumberColumn("Floor", format="%.1f", help="15th percentile outcome"),
            "proj": st.column_config.ProgressColumn("Projection", format="%.1f", min_value=0,
                                                    max_value=float(proj["proj"].max())),
            "ceiling": st.column_config.NumberColumn("Ceiling", format="%.1f", help="85th percentile outcome"),
        })
    st.caption("Players on bye don't appear for that week. Projections don't account for injuries, "
               "so check the injury report before setting lineups.")

# ---------- start / sit ----------
with tab_ss:
    st.subheader("Who should I start?")
    c1, c2 = st.columns([1, 3])
    ss_week = c1.selectbox("Week", sorted(proj["week"].unique()), key="ssw")
    picks = c2.multiselect("Pick 2 to 4 players", labels, max_selections=4,
                           default=labels[:0], placeholder="Start typing a name...")

    if len(picks) < 2:
        st.info("Choose at least two players to compare.")
    else:
        rows = []
        for lab in picks:
            r = proj[(proj["player_id"] == id_by_label[lab]) & (proj["week"] == ss_week)]
            rows.append(r.iloc[0] if len(r) else pd.Series({"label": lab, "bye": True}))
        playing = [r for r in rows if not r.get("bye", False)]
        on_bye = [r["label"] for r in rows if r.get("bye", False)]

        if on_bye:
            st.warning("No game in Week {} for: {}".format(ss_week, ", ".join(on_bye)))
        if playing:
            df = pd.DataFrame(playing).sort_values("proj")
            fig = go.Figure()
            for _, r in df.iterrows():
                color = POS_COLORS[r["position"]]
                fig.add_trace(go.Scatter(x=[r["floor"], r["ceiling"]], y=[r["label"]] * 2, mode="lines",
                                         line=dict(color=color, width=14), opacity=0.35, showlegend=False,
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[r["proj"]], y=[r["label"]], mode="markers+text",
                                         marker=dict(color=color, size=16, line=dict(color="white", width=2)),
                                         text=[fmt_pts(r["proj"])], textposition="top center",
                                         showlegend=False,
                                         hovertemplate=f"Floor {r['floor']:.1f} · Proj {r['proj']:.1f} · "
                                                       f"Ceiling {r['ceiling']:.1f}<extra></extra>"))
            fig.update_layout(height=120 + 70 * len(df), margin=dict(l=10, r=10, t=10, b=30),
                              xaxis_title="Projected PPR points (bar = floor to ceiling)")
            st.plotly_chart(fig, width="stretch")

            best = df.loc[df["proj"].idxmax()]
            safe = df.loc[df["floor"].idxmax()]
            boom = df.loc[df["ceiling"].idxmax()]
            a, b, c = st.columns(3)
            a.metric("Start", best["name"], f"{best['proj']:.1f} projected")
            b.metric("Safest floor", safe["name"], f"{safe['floor']:.1f} floor")
            c.metric("Highest ceiling", boom["name"], f"{boom['ceiling']:.1f} ceiling")

            gap = df["proj"].max() - df["proj"].nlargest(2).iloc[-1]
            if gap < 1.5:
                st.write(f"**It's close** (within {gap:.1f} points). If you're a big favorite this week, "
                         f"lean {safe['name']} for the safer floor. If you're an underdog, "
                         f"lean {boom['name']} for upside.")

            st.dataframe(df.sort_values("proj", ascending=False)[
                ["name", "matchup", "matchup_grade", "implied_total", "floor", "proj", "ceiling"]],
                hide_index=True, width="stretch",
                column_config={"name": "Player", "matchup": "Opp", "matchup_grade": "Matchup",
                               "implied_total": st.column_config.NumberColumn("Team implied pts", format="%.1f"),
                               "floor": st.column_config.NumberColumn("Floor", format="%.1f"),
                               "proj": st.column_config.NumberColumn("Projection", format="%.1f"),
                               "ceiling": st.column_config.NumberColumn("Ceiling", format="%.1f")})

# ---------- breakouts ----------
with tab_bo:
    st.subheader("Breakout detector")
    st.write("Fantasy points bounce around week to week, but **targets and carries are sticky**. "
             "This ranks players whose usage has jumped over their last 3 games, plus players whose "
             "points haven't caught up to their usage yet. **xFP** = expected fantasy points based "
             "only on a player's opportunities.")

    bpos = st.radio("Position", ["All", "QB", "RB", "WR", "TE"], horizontal=True, key="bpos")
    b = breakouts if bpos == "All" else breakouts[breakouts["position"] == bpos]
    b = b.head(25).copy()
    b["usage"] = b.apply(lambda r: f"{r['opps_before']:.1f} → {r['opps_last3']:.1f}", axis=1)

    def tag(r):
        tags = []
        if r["usage_change"] >= 3:
            tags.append("Usage spike")
        if r["points_gap"] >= 3:
            tags.append("Due for points")
        return " · ".join(tags)
    b["signal"] = b.apply(tag, axis=1)

    st.dataframe(
        b[["headshot_url", "name", "position", "team", "usage", "xfp_before", "xfp_last3",
           "ppr_last3", "signal", "breakout_score"]],
        hide_index=True, width="stretch", height=480,
        column_config={
            "headshot_url": st.column_config.ImageColumn("", width="small"),
            "name": "Player", "position": "Pos", "team": "Team",
            "usage": st.column_config.TextColumn("Targets + carries per game",
                                                 help="Earlier this season → last 3 games"),
            "xfp_before": st.column_config.NumberColumn("xFP before", format="%.1f"),
            "xfp_last3": st.column_config.NumberColumn("xFP last 3", format="%.1f"),
            "ppr_last3": st.column_config.NumberColumn("Actual pts last 3", format="%.1f"),
            "signal": "Signal",
            "breakout_score": st.column_config.ProgressColumn(
                "Breakout score", format="%.1f", min_value=0,
                max_value=float(max(breakouts["breakout_score"].max(), 1))),
        })

    pick = st.selectbox("See weekly trend", b["name"].tolist(), key="bopick")
    if pick:
        pid = b.loc[b["name"] == pick, "player_id"].iloc[0]
        u = usage[usage["player_id"] == pid].sort_values("week")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=u["week"], y=u["ppr"], name="Actual points", marker_color="#bab0ac"))
        fig.add_trace(go.Scatter(x=u["week"], y=u["xfp"], name="Expected points (xFP)", mode="lines+markers",
                                 line=dict(color="#4e79a7", width=3)))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=30), xaxis_title="Week",
                          yaxis_title="PPR points", legend=dict(orientation="h", y=1.12),
                          xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")

# ---------- trade analyzer ----------
with tab_trade:
    st.subheader("Trade analyzer")
    st.write("Compares rest-of-season projected points, including each player's remaining "
             "schedule and bye week.")
    weeks = sorted(proj["week"].unique())
    end_default = FANTASY_LAST_WEEK if FANTASY_LAST_WEEK in weeks else weeks[-1]
    w0, w1 = st.slider("Weeks to count", min_value=weeks[0], max_value=weeks[-1],
                       value=(weeks[0], end_default))
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

        if abs(pct) < 5:
            verdict, fn = "Fair trade", st.info
        elif pct > 0:
            verdict, fn = "You win this trade", st.success
        else:
            verdict, fn = "You lose this trade", st.error
        fn(f"**{verdict}** · you {'gain' if diff >= 0 else 'lose'} {abs(diff):.1f} projected points "
           f"({pct:+.0f}%) from Week {w0} to {w1}")

        a, b = st.columns(2)
        a.metric("You give", f"{gt:.1f} pts")
        a.dataframe(g, hide_index=True, width="stretch",
                    column_config={"Pts per game": st.column_config.NumberColumn(format="%.1f"),
                                   "ROS points": st.column_config.NumberColumn(format="%.1f")})
        b.metric("You get", f"{rt:.1f} pts", f"{diff:+.1f}")
        b.dataframe(r, hide_index=True, width="stretch",
                    column_config={"Pts per game": st.column_config.NumberColumn(format="%.1f"),
                                   "ROS points": st.column_config.NumberColumn(format="%.1f")})

        # weekly view
        def weekly(labs):
            ids = [id_by_label[l] for l in labs]
            w = proj[proj["player_id"].isin(ids) & proj["week"].between(w0, w1)]
            return w.groupby("week")["proj"].sum().reindex(range(w0, w1 + 1), fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(w0, w1 + 1)), y=weekly(give), name="You give", marker_color="#e15759"))
        fig.add_trace(go.Bar(x=list(range(w0, w1 + 1)), y=weekly(get), name="You get", marker_color="#59a14f"))
        fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=30, b=30),
                          xaxis_title="Week", yaxis_title="Projected points", xaxis=dict(dtick=1),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("Points aren't everything: also think about your roster needs. Getting 2 players for 1 "
                   "only helps if they'd actually crack your starting lineup.")
    else:
        st.info("Add at least one player to each side.")

# ---------- model ----------
with tab_model:
    st.subheader("How the projections work")
    st.markdown(
        "- **Data:** weekly stats for every QB, RB, WR, and TE since 2021 from "
        "[nflverse](https://github.com/nflverse), stored in a SQLite database.\n"
        "- **Features (all known before kickoff):** recent form weighted toward the latest games "
        "(points, targets, carries, target share, air yards), the opponent's points allowed to that "
        "position, and the Vegas implied team total.\n"
        "- **Model:** gradient-boosted trees (scikit-learn). Floor and ceiling come from how actual "
        "scores spread around projections in a held-out season.\n"
        "- **Baseline:** each player's recent weighted average. A useful model has to beat it.")

    st.markdown("#### Backtest: predicting games the model never saw")
    for label, rows in metrics["backtest"].items():
        m = pd.DataFrame(rows)
        all_row = m[m["position"] == "ALL"].iloc[0]
        better = (1 - all_row["model_mae"] / all_row["baseline_mae"]) * 100
        st.markdown(f"**{label}:** average miss of **{all_row['model_mae']:.2f} pts** vs "
                    f"{all_row['baseline_mae']:.2f} for the baseline (**{better:.1f}% more accurate**), "
                    f"across {all_row['games']:,} player-games. "
                    f"{all_row['range_hit_rate'] * 100:.0f}% of results landed between floor and ceiling.")
        st.dataframe(m.rename(columns={"position": "Position", "games": "Player-games",
                                       "model_mae": "Model avg miss", "baseline_mae": "Baseline avg miss",
                                       "range_hit_rate": "In floor-ceiling range",
                                       "rank_corr": "Rank correlation"}),
                     hide_index=True, width="stretch")

    st.markdown("#### What drives the projections")
    imp = pd.DataFrame(metrics["importance"]).head(10)
    imp["feature"] = imp["feature"].map(lambda f: FEATURE_LABELS.get(f, f.replace("ewm_", "Recent ").replace("_", " ")))
    fig = go.Figure(go.Bar(x=imp["importance"][::-1], y=imp["feature"][::-1], orientation="h",
                           marker_color="#4e79a7"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="Increase in average miss when the feature is scrambled (pts)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Evaluated on fantasy-relevant players (recent form of 6+ PPR points per game).")
