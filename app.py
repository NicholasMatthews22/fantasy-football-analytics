"""Fantasy Football Analytics: projections, rankings, my team, start/sit, breakouts, trades (full PPR)."""
import json
import math
import sqlite3

import numpy as np

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ffa import sleeper
from ffa.config import DB_PATH, FANTASY_LAST_WEEK, METRICS_PATH
from ffa.features import FEATURE_LABELS

st.set_page_config(page_title="Red Zone Lab · Fantasy Football Analytics", page_icon="assets/favicon.png",
                   layout="wide")
LOGO = ('<svg viewBox="0 0 32 32" width="30" height="30"><rect width="32" height="32" rx="8" fill="#7c5cff"/>'
        '<path d="M16 26.5V17M9 17h14M9 17V6.5M23 17V6.5" stroke="#fff" stroke-width="2.8" stroke-linecap="round" '
        'stroke-linejoin="round" fill="none"/></svg>')

# ---------- look: dark, Sleeper-style ----------
POS_COLORS = {"QB": "#ff4f81", "RB": "#22d3b6", "WR": "#58b6ff", "TE": "#ffb35c", "K": "#b794ff", "DST": "#9aa3b8"}
ALL_POS = ["QB", "RB", "WR", "TE", "K", "DST"]
# Sleeper roster slots -> positions that can fill them
SLOT_ELIG = {"QB": ["QB"], "RB": ["RB"], "WR": ["WR"], "TE": ["TE"], "K": ["K"], "DEF": ["DST"],
             "FLEX": ["RB", "WR", "TE"], "WRRB_FLEX": ["RB", "WR"], "REC_FLEX": ["WR", "TE"],
             "SUPER_FLEX": ["QB", "RB", "WR", "TE"]}
SLOT_LABEL = {"DEF": "DST", "SUPER_FLEX": "SF", "WRRB_FLEX": "W/R", "REC_FLEX": "W/T"}
DEFAULT_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
STD_SCORING = {"rec": 1.0, "rec_yd": 0.1, "rec_td": 6, "rush_yd": 0.1, "rush_td": 6, "pass_yd": 0.04,
               "pass_td": 4, "pass_int": -1, "bonus_rec_te": 0}
SCORING_PRESETS = {"Full PPR": {}, "Half PPR": {"rec": 0.5}, "Standard (no PPR)": {"rec": 0.0}}
ACCENT, MUTED = "#9b6bff", "#8b93a7"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --surface: #1a1f2c; --surface2: #222838; --line: #2b3244; --muted: #8b93a7; --accent: #9b6bff;
  --qb: #ff4f81; --rb: #22d3b6; --wr: #58b6ff; --te: #ffb35c; }
.block-container { padding-top: 2rem !important; max-width: 1280px; }
h1, h2, h3, .stTabs button p, [data-testid="stMetricValue"], .num {
  font-family: 'Inter', 'Segoe UI', sans-serif !important; }
h1 { font-weight: 800 !important; letter-spacing: -.5px; font-size: 2.1rem !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p, .stApp input, .stApp button p,
.gl-card, .lu-row, .pc-row, .ss-card, .note, .chip { font-family: 'Inter', 'Segoe UI', sans-serif; }
.gl-pts, .lu-pts, .pc-pts, .ss-proj, .mu-pts, .mu-wp, .pf-stats div { font-variant-numeric: tabular-nums; letter-spacing: -.3px; }

/* page header */
.ph { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  border-bottom: 1px solid var(--line); padding-bottom: 12px; margin: 0 0 16px; }
.ph h2 { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 1.6rem; letter-spacing: -.4px; margin: 0; padding: 0; }
.ph .info { display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; margin-left: 8px;
  border-radius: 50%; border: 1px solid var(--muted); color: var(--muted); font-size: .7rem; font-weight: 700;
  cursor: help; vertical-align: middle; position: relative; top: -3px; }
.ph.sub { border: 0; padding: 0; margin: 4px 0 10px; }
.ph.sub h3 { font-size: 1.15rem; font-weight: 700; margin: 0; padding: 0; color: #e8ebf2; }
.ph .meta { color: var(--muted); font-size: .82rem; }
.ph .meta b { color: #cfd5e3; font-weight: 600; }

/* footer */
.foot { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: .78rem;
  display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.foot a { color: #cfd5e3; text-decoration: none; } .foot a:hover { color: var(--accent); }
h1 span.hl { color: var(--accent); }
h3 { font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; overflow-x: auto; scrollbar-width: none; }
.stTabs button { border-radius: 999px !important; padding: 6px 14px !important; background: var(--surface) !important; }
.stTabs button[aria-selected="true"] { background: var(--accent) !important; }
.stTabs button[aria-selected="true"] p { color: #fff !important; }
.stTabs button p { font-size: .9rem !important; font-weight: 600; white-space: nowrap; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px; }
[data-testid="stMetricValue"] { font-weight: 700; }
.sub { color: var(--muted); font-size: .9rem; margin-top: -6px; }
/* hide Streamlit's own header buttons and badges */
header[data-testid="stHeader"] { background: transparent !important; height: 2.6rem; }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"], .stAppDeployButton, [class*="viewerBadge"], [data-testid="manage-app-button"] {
  display: none !important; }

/* sidebar menu */
section[data-testid="stSidebar"] { background: #141824 !important; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .brand-row { display: flex; align-items: center; gap: 10px; margin: -8px 0 18px; }
section[data-testid="stSidebar"] .brand { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 1.1rem;
  line-height: 1.1; letter-spacing: -.3px; }
section[data-testid="stSidebar"] .brand span { color: var(--accent); }
section[data-testid="stSidebar"] .side-sub { color: var(--muted); font-size: .74rem; margin-top: 2px; }
section[data-testid="stSidebar"] [data-testid="stRadio"], section[data-testid="stSidebar"] [data-testid="stRadio"] > div,
section[data-testid="stSidebar"] [data-testid="stRadio"] label + div { width: 100% !important; }
section[data-testid="stSidebar"] [role="radiogroup"] { display: flex !important; flex-direction: column; align-items: stretch !important; gap: 4px; width: 100% !important; }
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
section[data-testid="stSidebar"] [data-testid="stRadio"] * { max-width: none !important; }
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]) { width: 100% !important; }
section[data-testid="stSidebar"] [role="radiogroup"] > * { width: 100% !important; max-width: none !important; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] { width: 100% !important; max-width: none !important;
  display: flex !important; box-sizing: border-box; padding: 10px 14px; border-radius: 10px; margin: 0; cursor: pointer;
  transition: background .15s; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div { width: 100%; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none !important; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: var(--surface2); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] { background: var(--accent); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { color: #fff !important; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p { font-weight: 600; font-size: .95rem; }
.b-Final { background: #3a4256; color: #e8ebf2; }
span[data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] > div {
  background: #232a3a !important; border: 1px solid var(--accent) !important; }
[data-testid="stMultiSelectTagsContainer"] > div, [data-testid="stMultiSelectTagsContainer"] > div * { color: #e8ebf2 !important; }
span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg { color: #e8ebf2 !important; }
.flabel { font-size: .875rem; margin-bottom: .25rem; min-height: 1.4rem; display: flex; align-items: flex-end; }
.intro { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px;
  margin: 4px 0 14px; }
.intro-t { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.25rem; }
.intro-w { color: #cfd5e3; margin-top: 2px; }
.intro-h { color: var(--muted); font-size: .82rem; margin-top: 6px; }
.chip { display: inline-flex; align-items: center; gap: 6px; background: var(--surface); border: 1px solid var(--line);
  border-radius: 999px; padding: 3px 10px; font-size: .8rem; color: #cfd5e3; margin: 2px 6px 2px 0; }
.chip b { color: var(--accent); }

/* position pills and badges */
.pos { display: inline-block; min-width: 30px; text-align: center; padding: 1px 6px; border-radius: 6px;
  font-size: .7rem; font-weight: 800; letter-spacing: .3px; color: #10131c; vertical-align: middle; }
.pos-QB { background: var(--qb); } .pos-RB { background: var(--rb); } .pos-WR { background: var(--wr); }
.pos-TE { background: var(--te); } .pos-K { background: #b794ff; } .pos-DST { background: #9aa3b8; }
.pos-FLEX { background: linear-gradient(90deg, var(--rb), var(--wr), var(--te)); }
.badge { display: inline-block; padding: 1px 7px; border-radius: 6px; font-size: .68rem; font-weight: 800;
  text-transform: uppercase; letter-spacing: .4px; vertical-align: middle; }
.b-Out { background: #ff4d4d; color: #fff; } .b-Doubtful { background: #ff8a3d; color: #fff; }
.b-Questionable { background: #ffd24d; color: #222; } .b-wind { background: #7aa7ff; color: #0e1424; }

/* cards */
.gl-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 6px 0 10px; }
.gl-grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.gl-card { display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 12px; min-width: 0;
  background: var(--surface); border: 1px solid var(--line); position: relative; overflow: hidden; }
.gl-card::before { content: none; }
.gl-card .hs, .avatar { width: 44px; height: 44px; border-radius: 50%; object-fit: cover; object-position: top;
  background: var(--surface2); flex: none; }
.gl-card .logo { width: 44px; height: 44px; flex: none; }
.gl-txt { min-width: 0; }
.gl-lbl { font-size: .68rem; text-transform: uppercase; letter-spacing: .7px; color: var(--muted); font-weight: 600; }
.gl-name { font-weight: 700; font-size: 1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.gl-meta { font-size: .8rem; color: var(--muted); }
.gl-pts { margin-left: auto; font-family: 'Inter', sans-serif; font-size: 1.45rem; font-weight: 700; line-height: 1; }
.gl-pts small { display: block; font-size: .65rem; font-weight: 500; color: var(--muted); text-align: right; }
.note { font-size: .85rem; color: #cfd5e3; margin: 0 0 8px; }

/* start/sit */
.ss-grid { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 12px; }
.ss-card { border-radius: 14px; overflow: hidden; background: var(--surface); border: 1px solid var(--line); position: relative; }
.ss-card.best { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), 0 8px 30px rgba(155,107,255,.15); }
.ss-band { background: var(--tc); color: #fff; display: flex; align-items: center; gap: 8px; padding: 7px 12px;
  font-family: 'Inter', sans-serif; font-weight: 700; font-size: .85rem; text-transform: uppercase; letter-spacing: .5px; }
.ss-band img { width: 24px; height: 24px; background: rgba(255,255,255,.92); border-radius: 50%; padding: 2px; }
.ss-band .badge { margin-left: auto; }
.ss-body { padding: 14px; text-align: center; }
.ss-body .hs { width: 86px; height: 86px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2); }
.ss-name { font-weight: 700; font-size: 1.05rem; margin-top: 6px; }
.ss-meta { font-size: .8rem; color: var(--muted); }
.ss-proj { font-family: 'Inter', sans-serif; font-size: 2.4rem; font-weight: 700; line-height: 1.05; margin-top: 6px; }
.ss-proj span { font-size: .75rem; font-weight: 500; color: var(--muted); margin-left: 4px; }
.range { position: relative; height: 8px; border-radius: 4px; background: var(--surface2); margin: 12px 4px 4px; }
.range-bar { position: absolute; top: 0; bottom: 0; border-radius: 4px; background: var(--accent); opacity: .45; }
.range-dot { position: absolute; top: -5px; width: 18px; height: 18px; margin-left: -9px; border-radius: 50%;
  background: var(--accent); border: 3px solid #10131c; }
.range-lbl { display: flex; justify-content: space-between; font-size: .72rem; color: var(--muted); }
.pick { position: absolute; top: 44px; right: 10px; background: var(--accent); color: #fff; font-weight: 800;
  font-size: .7rem; padding: 2px 8px; border-radius: 6px; letter-spacing: .5px; }

/* profile */
.pf { display: flex; gap: 18px; align-items: center; border-radius: 14px; padding: 18px 22px; color: #fff; flex-wrap: wrap;
  background: linear-gradient(110deg, var(--tc) 0%, color-mix(in srgb, var(--tc) 55%, #10131c) 70%, #10131c 100%); }
.pf .hs { width: 104px; height: 104px; border-radius: 50%; object-fit: cover; object-position: top;
  background: rgba(255,255,255,.15); border: 3px solid rgba(255,255,255,.85); }
.pf-name { font-family: 'Inter', sans-serif; font-size: 1.9rem; font-weight: 800; line-height: 1.05; }
.pf-sub { opacity: .9; margin-top: 4px; }
.pf-logo { width: 76px; height: 76px; margin-left: auto; filter: drop-shadow(0 1px 3px rgba(0,0,0,.5)); }
.pf-stats { display: flex; gap: 22px; margin-top: 10px; flex-wrap: wrap; }
.pf-stats div { font-family: 'Inter', sans-serif; font-size: 1.4rem; font-weight: 700; line-height: 1; }
.pf-stats small { display: block; font-family: inherit; font-size: .64rem; font-weight: 500; opacity: .8;
  text-transform: uppercase; letter-spacing: .5px; margin-top: 3px; }

/* lineup */
.lu { display: grid; gap: 4px; margin: 6px 0 14px; }
.lu-row { display: flex; align-items: center; gap: 10px; padding: 5px 12px; border-radius: 10px;
  background: var(--surface); border: 1px solid var(--line); min-height: 48px; }
.lu-row.empty { opacity: .6; }
.lu-row .slot { width: 46px; flex: none; }
.lu-row .hs { width: 34px; height: 34px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2); flex: none; }
.lu-row .logo { width: 22px; height: 22px; flex: none; }
.lu-name { font-weight: 700; font-size: .92rem; min-width: 0; line-height: 1.2; }
.lu-name small { display: block; font-weight: 400; color: var(--muted); font-size: .76rem; }
.lu-pts { margin-left: auto; text-align: right; font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 700; line-height: 1; }
.lu-pts small { display: block; font-size: .62rem; font-weight: 500; color: var(--muted); margin-top: 2px; }

/* projection cards */
.pc { display: grid; gap: 6px; }
.pc-row { display: grid; grid-template-columns: 44px 46px minmax(0, 1fr) minmax(0, 150px) 86px; align-items: center;
  gap: 12px; padding: 8px 14px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); }
.pc-rank { font-family: 'Inter', sans-serif; font-weight: 700; color: var(--muted); font-size: .85rem; }
.pc-row .hs { width: 44px; height: 44px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2); }
.pc-main { min-width: 0; }
.pc-name { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pc-meta { color: var(--muted); font-size: .78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pc-meta .good { color: var(--rb); font-weight: 700; } .pc-meta .tough { color: var(--qb); font-weight: 700; }
.pc-range .range { margin: 4px 0 3px; }
.pc-range .range-lbl { font-size: .66rem; }
.pc-pts { text-align: right; font-family: 'Inter', sans-serif; font-size: 1.45rem; font-weight: 700; line-height: 1; }
.pc-pts small { display: block; font-size: .62rem; color: var(--muted); font-weight: 500; margin-top: 2px; }

/* matchup */
.mu { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 12px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; }
.mu-side.right { text-align: right; }
.mu-name { font-weight: 700; color: #cfd5e3; }
.mu-pts { font-family: 'Inter', sans-serif; font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
.mu-mid { text-align: center; }
.mu-wp { font-family: 'Inter', sans-serif; font-size: 1.6rem; font-weight: 800; color: var(--accent); }
.mu-mid small { color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .5px; }
.wp-bar { height: 8px; border-radius: 4px; background: var(--qb); margin: 10px 0 14px; overflow: hidden; }
.wp-bar div { height: 100%; background: var(--accent); }

/* phones */
@media (max-width: 900px) {
  .gl-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .gl-grid.three { grid-template-columns: 1fr; }
  .ss-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 600px) {
  .block-container { padding-left: 12px !important; padding-right: 12px !important; padding-top: 1.2rem !important; }
  h1 { font-size: 1.5rem !important; }
  .gl-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .gl-grid.three { grid-template-columns: 1fr; }
  .gl-card { padding: 9px 10px; gap: 8px; }
  .gl-card .hs, .gl-card .logo { display: none; }
  .gl-grid:not(.three) .gl-card { flex-wrap: wrap; }
  .gl-grid:not(.three) .gl-pts { margin-left: 0; font-size: 1.35rem; flex-basis: 100%; }
  .gl-grid:not(.three) .gl-pts small { display: inline; margin-left: 4px; }
  .gl-name { font-size: .92rem; }
  .gl-meta { font-size: .75rem; }
  .ss-grid { grid-template-columns: 1fr; }
  .pf { padding: 14px; gap: 12px; }
  .pf .hs { width: 72px; height: 72px; }
  .pf-name { font-size: 1.4rem; }
  .pf-logo { display: none; }
  .pf-stats { gap: 14px; }
  .pf-stats div { font-size: 1.1rem; }
  .lu-row { padding: 5px 8px; gap: 7px; }
  .pc-row { grid-template-columns: 30px 38px minmax(0, 1fr) 62px; gap: 8px; padding: 7px 10px; }
  .pc-row .hs { width: 38px; height: 38px; }
  .pc-range { display: none; }
  .pc-pts { font-size: 1.2rem; }
  .lu-row .logo { display: none; }
  .lu-row .slot { width: 40px; }
  .lu-pts { font-size: 1.25rem; }
}
</style>
""", unsafe_allow_html=True)

STATUS_STYLE = {"Final": "background-color:#3a4256;color:#e8ebf2", "Out": "background-color:#ff4d4d;color:white", "Doubtful": "background-color:#ff8a3d;color:white",
                "Questionable": "background-color:#ffd24d;color:#222"}
POS_STYLE = {p: f"color:{c};font-weight:800" for p, c in POS_COLORS.items()}


def plot_style(fig, height=340, legend_below=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=46, b=60 if legend_below else 30),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#cfd5e3"), title_font=dict(family="Inter", size=14))
    fig.update_xaxes(gridcolor="#232a3a", zerolinecolor="#2b3244")
    fig.update_yaxes(gridcolor="#232a3a", zerolinecolor="#2b3244")
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", y=-0.24, x=0))
    return fig


# ---------- data ----------
@st.cache_data
def query(sql):
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con)


teams = query("SELECT * FROM teams").set_index("team")
proj = query("SELECT * FROM projections")
breakouts = query("SELECT * FROM breakouts")
usage = query("SELECT * FROM usage_weekly")
weather = query("SELECT * FROM weather")
id_map = query("SELECT * FROM id_map")
acc = query("SELECT * FROM accuracy")
season = int(proj["gameday"].str[:4].min())
games_cur = query(f"SELECT * FROM player_weeks WHERE season = {season}")
metrics = json.loads(METRICS_PATH.read_text())
kdst_hist = query("SELECT * FROM kdst_history")
experiments = json.loads((METRICS_PATH.parent / "experiments.json").read_text())

last_week = int(proj["last_completed_week"].iloc[0])
next_week = int(proj["week"].min())
injury_posted = proj.loc[proj["week"] == next_week, "status"].notna().any()


def team_attr(abbr, col, default=""):
    return teams[col].get(abbr, default) if abbr in teams.index else default


def weather_text(roof, temp, wind):
    if roof in ("dome", "closed"):
        return "Indoors"
    if pd.notna(wind):
        return f"{temp:.0f}°F · {wind:.0f} mph"
    return "Outdoors" if roof in ("outdoors", "open") else ""


proj = proj.merge(weather, on="game_id", how="left")
proj["weather"] = [weather_text(r, t, w) for r, t, w in zip(proj["roof"], proj["temp"], proj["wind"])]
proj["windy"] = proj["wind"].fillna(0) >= 15
for df in (proj, breakouts):
    df["color"] = df["team"].map(teams["team_color"]).fillna("#555")
    df["logo"] = df["team"].map(teams["logo"])
proj["label"] = proj["name"] + " (" + proj["position"] + ", " + proj["team"] + ")"
proj["matchup"] = proj.apply(lambda r: ("vs " if r["home"] else "@ ") + r["opp"], axis=1)
proj["matchup_grade"] = pd.cut(proj["def_factor"], [0, 0.85, 1.15, 99],
                               labels=["Tough", "Neutral", "Good"]).astype(str)
proj["status"] = proj["status"].fillna("")
players = (proj.sort_values("ewm_ppr", ascending=False)
               .drop_duplicates("player_id")[["player_id", "label", "name", "position", "team",
                                              "headshot_url", "logo", "color"]])
labels = players["label"].tolist()
id_by_label = dict(zip(players["label"], players["player_id"]))
label_by_id = dict(zip(players["player_id"], players["label"]))
gsis_by_sleeper = dict(zip(id_map["sleeper_id"], id_map["player_id"]))


# ---------- helpers ----------
def html(s):
    """Render an HTML snippet. Lines are flattened so Streamlit doesn't read indentation as a code block."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def badge(status):
    return f'<span class="badge b-{status}">{status}</span>' if status else ""


def wind_badge(r):
    return f'<span class="badge b-wind">Wind {r["wind"]:.0f} mph</span>' if r.get("windy", False) else ""


def pill(pos_):
    label = SLOT_LABEL.get(pos_, pos_)
    cls = pos_ if pos_ in POS_COLORS else ("DST" if pos_ == "DEF" else "FLEX")
    return f'<span class="pos pos-{cls}">{label}</span>'


def intro(title, what, how):
    """Page header: title, a tooltip with what the page does and how to use it, and current context."""
    tip = f"{what} {how}".replace('"', "'")
    if st.session_state.get("_nested"):
        html(f'<div class="ph sub"><h3>{title}<span class="info" title="{tip}">i</span></h3></div>')
        return
    meta = f"Week <b>{next_week}</b> · <b>{st.session_state.get('score_label', 'Full PPR')}</b>"
    if league():
        meta += f" · <b>{league()['name']}</b>"
    html(f'<div class="ph"><h2>{title}<span class="info" title="{tip}">i</span></h2><div class="meta">{meta}</div></div>')


def position_filter(key):
    """Dropdown with checkboxes: pick one position, several, or all."""
    opts = ALL_POS
    ss = st.session_state
    for p_ in opts:
        ss.setdefault(f"{key}_{p_}", True)
    chosen = [p_ for p_ in opts if ss[f"{key}_{p_}"]]
    ss[f"{key}_all"] = len(chosen) == len(opts)
    summary = "All positions" if len(chosen) == len(opts) else (", ".join(chosen) if chosen else "None selected")

    def toggle_all():
        for p_ in opts:
            ss[f"{key}_{p_}"] = ss[f"{key}_all"]

    st.markdown('<div class="flabel">Position</div>', unsafe_allow_html=True)
    with st.popover(summary, width="stretch"):
        st.checkbox("All positions", key=f"{key}_all", on_change=toggle_all)
        st.markdown('<hr style="margin:2px 0 6px;border-color:#2b3244">', unsafe_allow_html=True)
        for p_ in opts:
            st.checkbox(p_, key=f"{key}_{p_}")
    return chosen


def week_row(pid, week):
    r = proj[(proj["player_id"] == pid) & (proj["week"] == week)]
    return r.iloc[0] if len(r) else None


def load_example_team():
    """A realistic mid-draft roster: one strong starter at each spot plus depth, not all stars."""
    picks_by_pos = {"QB": [3, 14], "RB": [2, 9, 18, 27, 35], "WR": [4, 11, 20, 30, 38], "TE": [5, 15]}
    team = []
    for pos_, ranks in picks_by_pos.items():
        pool = players[players["position"] == pos_]["label"].tolist()
        team += [pool[r - 1] for r in ranks if r <= len(pool)]
    st.session_state["team"] = team


def init_from_url(key, param):
    """Seed a multiselect from the page URL the first time it renders (for shareable links)."""
    if key not in st.session_state:
        ids = [i for i in st.query_params.get(param, "").split(",") if i in label_by_id]
        st.session_state[key] = [label_by_id[i] for i in ids]


def sync_url(param, chosen_labels):
    ids = [id_by_label[l] for l in chosen_labels if l in id_by_label]
    if ids:
        st.query_params[param] = ",".join(ids)
    elif param in st.query_params:
        del st.query_params[param]


@st.cache_data
def ros_values(w0, w1, teams_n, scoring_key=""):
    """Rest-of-season points and value over a replacement-level starter for every active player."""
    r = (proj[proj["week"].between(w0, w1)].groupby("player_id")
         .agg(ros=("proj", "sum"), games=("proj", "size")).reset_index())
    r = players.merge(r, on="player_id", how="left").fillna({"ros": 0, "games": 0})
    r["ppg"] = r["ros"] / r["games"].where(r["games"] > 0)
    repl_rank = {"QB": teams_n, "RB": int(teams_n * 2.5), "WR": int(teams_n * 2.5), "TE": teams_n,
                 "K": teams_n, "DST": teams_n}
    r["repl"] = 0.0
    for pos_, n in repl_rank.items():
        vals = r.loc[r["position"] == pos_, "ros"].sort_values(ascending=False).reset_index(drop=True)
        r.loc[r["position"] == pos_, "repl"] = float(vals.iloc[min(n, len(vals) - 1)]) if len(vals) else 0.0
    r["value"] = (r["ros"] - r["repl"]).clip(lower=0)
    r = r.sort_values(["value", "ros"], ascending=False).reset_index(drop=True)
    r["rank"] = r.index + 1
    r["pos_rank"] = r["position"] + (r.groupby("position").cumcount() + 1).astype(str)
    return r


# ---------- Sleeper league connection ----------
@st.cache_data(ttl=600, show_spinner=False)
def sl_user(username):
    return sleeper.find_user(username)


@st.cache_data(ttl=600, show_spinner=False)
def sl_leagues(user_id, yr):
    return sleeper.leagues_for(user_id, yr)


@st.cache_data(ttl=300, show_spinner=False)
def sl_rosters(league_id):
    return sleeper.league_rosters(league_id)


@st.cache_data(ttl=600, show_spinner=False)
def sl_users(league_id):
    return sleeper.league_users(league_id)


@st.cache_data(ttl=120, show_spinner=False)
def sl_matchups(league_id, week):
    return sleeper.matchups(league_id, week)


def to_gsis(sleeper_id):
    """Sleeper player ID -> our player ID. Sleeper uses team abbreviations for defenses."""
    sid = str(sleeper_id)
    if sid.isalpha():
        return "DST_" + {"LAR": "LA"}.get(sid, sid)
    return gsis_by_sleeper.get(sid)


def connect_league(league, user_id):
    """Pull every roster, team name, scoring setting, and lineup slot from a Sleeper league."""
    rosters = sl_rosters(league["league_id"])
    try:
        names = {u["user_id"]: sleeper.team_name(u) for u in sl_users(league["league_id"])}
    except sleeper.SleeperError:
        names = {}
    mine = sleeper.my_roster(rosters, user_id) or {}
    my_ids = [to_gsis(p) for p in (mine.get("players") or [])]
    st.session_state["team"] = [label_by_id[i] for i in my_ids if i in label_by_id]
    st.session_state["league_info"] = {
        "name": league.get("name", "My league"), "id": league["league_id"],
        "teams": int(league.get("total_rosters") or 12), "ppr": sleeper.ppr_value(league),
        "scoring": league.get("scoring_settings") or {},
        "slots": [x for x in (league.get("roster_positions") or DEFAULT_SLOTS) if x in SLOT_ELIG] or DEFAULT_SLOTS,
        "rostered": {to_gsis(p) for p in sleeper.rostered_ids(rosters)} - {None},
        "mine": set(i for i in my_ids if i), "unmatched": sum(1 for i in my_ids if i not in label_by_id),
        "my_roster_id": mine.get("roster_id"),
        "rosters": {r.get("roster_id"): {"owner": names.get(r.get("owner_id"), f"Team {r.get('roster_id')}"),
                                         "ids": [g for g in (to_gsis(p) for p in (r.get("players") or [])) if g]}
                    for r in rosters},
    }
    st.session_state["scoring"] = "My league"


def league():
    return st.session_state.get("league_info")


# reconnect automatically when the page is opened from a saved link
if league() is None and st.query_params.get("sleeper") and st.query_params.get("league"):
    try:
        u = sl_user(st.query_params["sleeper"])
        lg = next((x for x in sl_leagues(u["user_id"], season) if x["league_id"] == st.query_params["league"]), None)
        if u and lg:
            connect_league(lg, u["user_id"])
    except Exception:
        pass


# ---------- scoring ----------
rates = query("SELECT * FROM player_rates").set_index("player_id")


def rescore(df, settings):
    """Convert full-PPR projections to another scoring system using each player's stat mix."""
    sc = {**STD_SCORING, **{k: float(v) for k, v in settings.items() if k in STD_SCORING}}
    r = rates.reindex(df["player_id"]).fillna(0)
    te = (df["position"].values == "TE").astype(float)

    def pts(w):
        return (r["receptions"].values * (w["rec"] + te * w["bonus_rec_te"]) + r["receiving_yards"].values * w["rec_yd"]
                + r["receiving_tds"].values * w["rec_td"] + r["rushing_yards"].values * w["rush_yd"]
                + r["rushing_tds"].values * w["rush_td"] + r["passing_yards"].values * w["pass_yd"]
                + r["passing_tds"].values * w["pass_td"] + r["passing_interceptions"].values * w["pass_int"])
    base, new = pts(STD_SCORING), pts(sc)
    ratio = np.where((base > 1) & df["position"].isin(["QB", "RB", "WR", "TE"]).values, new / np.maximum(base, 1), 1.0)
    out = df.copy()
    for c in ["proj", "floor", "ceiling"]:
        out[c] = out[c] * ratio
    return out


# ---------- sidebar ----------
with st.sidebar:
    st.markdown(f'<div class="brand-row">{LOGO}<div><div class="brand">Red Zone Lab</div>'
                f'<div class="side-sub">Fantasy football analytics</div></div></div>', unsafe_allow_html=True)
    nav_box = st.container()
    st.divider()

# ---------- scoring choice ----------
score_opts = (["My league"] if league() else []) + list(SCORING_PRESETS)
if st.session_state.get("scoring") not in score_opts:
    st.session_state["scoring"] = score_opts[0]
scoring_name = st.sidebar.selectbox("Scoring", score_opts, key="scoring",
                            help="Projections are converted to this scoring. 'My league' uses your Sleeper settings.")
proj = rescore(proj, league()["scoring"] if scoring_name == "My league" else SCORING_PRESETS[scoring_name])
score_label = f"{league()['name']} scoring" if scoring_name == "My league" else scoring_name
st.session_state["score_label"] = score_label


def section_header(title, sub=""):
    meta = f"Week <b>{next_week}</b> · <b>{score_label}</b>" + (f" · <b>{league()['name']}</b>" if league() else "")
    html(f'<div class="ph"><div><h2>{title}</h2>{f"<div class=meta>{sub}</div>" if sub else ""}</div>'
         f'<div class="meta">{meta}</div></div>')


def render_home_summary():

    wk = proj[(proj["week"] == next_week) & ~proj["status"].isin(["Out", "Final"])]
    cards = []
    for pos_ in ["QB", "RB", "WR", "TE"]:
        r = wk[wk["position"] == pos_].nlargest(1, "proj").iloc[0]
        cards.append(f"""<div class="gl-card" style="--tc:{POS_COLORS[pos_]}">
          <img class="hs" src="{r['headshot_url']}"><div class="gl-txt">
          <div class="gl-lbl">{pill(pos_)} Top</div><div class="gl-name">{r['name']}</div>
          <div class="gl-meta">{r['team']} {r['matchup']} {badge(r['status'])}</div></div>
          <div class="gl-pts">{r['proj']:.1f}<small>proj</small></div></div>""")
    html(f'<div class="gl-grid">{"".join(cards)}</div>')

    smash = wk[wk["proj"] >= 12].nlargest(1, "def_factor").iloc[0]
    bo = breakouts.iloc[0]
    hi = wk.drop_duplicates("team").nlargest(1, "implied_total").iloc[0]
    html(f"""<div class="gl-grid three">
      <div class="gl-card" style="--tc:{ACCENT}"><img class="hs" src="{smash['headshot_url']}"><div class="gl-txt">
        <div class="gl-lbl">Best matchup</div><div class="gl-name">{smash['name']}</div>
        <div class="gl-meta">{smash['matchup']} allows {(smash['def_factor'] - 1) * 100:+.0f}% to {smash['position']}s</div></div></div>
      <div class="gl-card" style="--tc:{ACCENT}"><img class="hs" src="{bo['headshot_url']}"><div class="gl-txt">
        <div class="gl-lbl">Top breakout candidate</div><div class="gl-name">{bo['name']}</div>
        <div class="gl-meta">Targets + carries {bo['opps_before']:.1f} → {bo['opps_last3']:.1f} per game</div></div></div>
      <div class="gl-card" style="--tc:{ACCENT}"><img class="logo" src="{hi['logo']}"><div class="gl-txt">
        <div class="gl-lbl">Highest-scoring offense</div><div class="gl-name">{team_attr(hi['team'], 'team_name', hi['team'])}</div>
        <div class="gl-meta">Vegas expects {hi['implied_total']:.1f} pts {hi['matchup']}</div></div></div>
    </div>""")

    notes = []
    if not injury_posted:
        notes.append(f'<span class="badge b-Questionable">Injuries</span> The Week {next_week} injury report isn\'t final '
                     "yet. Statuses update automatically Saturday and Sunday mornings.")
    windy = wk[wk["windy"]].drop_duplicates("game_id")
    if len(windy):
        games_txt = ", ".join(f"{r['team']} {r['matchup']} ({r['wind']:.0f} mph)" for _, r in windy.iterrows())
        notes.append(f'<span class="badge b-wind">Wind watch</span> {games_txt}. In backtests, QBs in 15+ mph wind '
                     "scored about 3 points below projection.")
    for n in notes:
        html(f'<div class="note">{n}</div>')



# ---------- player profile ----------
def render_profile(pid, key):
    info = players[players["player_id"] == pid].iloc[0]
    if info["position"] in ("K", "DST"):
        return render_kdst_profile(pid, info, key)
    tc = info["color"]
    log = games_cur[games_cur["player_id"] == pid].sort_values("week")
    u = usage[usage["player_id"] == pid].sort_values("week")
    fut = proj[proj["player_id"] == pid].sort_values("week")
    pos_ppg = (games_cur[games_cur["position"] == info["position"]].groupby("player_id")["ppr"]
               .agg(["mean", "size"]).query("size >= 3")["mean"].sort_values(ascending=False))
    rank = (list(pos_ppg.index).index(pid) + 1) if pid in pos_ppg.index else None
    ppg = log["ppr"].mean() if len(log) else 0
    nxt = fut.iloc[0]
    tags = (badge(nxt["status"]) if nxt["week"] == next_week else "") + wind_badge(nxt)

    html(f"""<div class="pf" style="--tc:{tc}">
      <img class="hs" src="{info['headshot_url']}">
      <div><div class="pf-name">{info['name']} {tags}</div>
        <div class="pf-sub">{pill(info['position'])} {team_attr(info['team'], 'team_name', info['team'])}</div>
        <div class="pf-stats">
          <div>{ppg:.1f}<small>PPR / game</small></div>
          <div>{info['position']}{rank if rank else '–'}<small>Rank (per game)</small></div>
          <div>{log['ppr'].sum():.0f}<small>Season pts</small></div>
          <div>{nxt['proj']:.1f}<small>Wk {int(nxt['week'])} {nxt['matchup']}</small></div>
          <div>{fut[fut['week'] <= FANTASY_LAST_WEEK]['proj'].sum():.0f}<small>Rest of season</small></div>
        </div></div>
      <img class="pf-logo" src="{info['logo']}">
    </div>""")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=log["week"], y=log["ppr"], name="Actual", marker_color=POS_COLORS[info["position"]],
                             customdata=log["opponent_team"],
                             hovertemplate="Week %{x} vs %{customdata}: %{y:.1f} pts<extra></extra>"))
        if len(u):
            fig.add_trace(go.Scatter(x=u["week"], y=u["xfp"], name="Expected (usage)", mode="lines+markers",
                                     line=dict(color="#cfd5e3", width=2, dash="dot")))
        fig.add_trace(go.Scatter(
            x=fut["week"], y=fut["proj"], name="Projected", mode="markers",
            marker=dict(color=ACCENT, size=10, symbol="diamond"),
            error_y=dict(type="data", symmetric=False, array=fut["ceiling"] - fut["proj"],
                         arrayminus=fut["proj"] - fut["floor"], color="rgba(155,107,255,.25)", thickness=6, width=0),
            customdata=fut[["matchup", "floor", "ceiling"]],
            hovertemplate="Week %{x} %{customdata[0]}: %{y:.1f} proj (%{customdata[1]:.1f} to "
                          "%{customdata[2]:.1f})<extra></extra>"))
        if ppg:
            fig.add_hline(y=ppg, line=dict(color="rgba(207,213,227,.4)", dash="dash"),
                          annotation_text=f"Season avg {ppg:.1f}", annotation_position="top right")
        fig.update_layout(title="Season so far and projections ahead", yaxis_title="PPR points",
                          xaxis=dict(dtick=1, title="Week", range=[0.4, max(fut["week"].max(), 18) + 0.6]))
        st.plotly_chart(plot_style(fig, 360), width="stretch", key=f"{key}_pts")
    with c2:
        fig = go.Figure()
        if info["position"] != "QB":
            fig.add_trace(go.Scatter(x=u["week"], y=u["target_share"] * 100, name="Target share",
                                     mode="lines+markers", line=dict(color=POS_COLORS["WR"], width=3)))
        if info["position"] in ("RB", "QB"):
            fig.add_trace(go.Scatter(x=u["week"], y=u["carry_share"] * 100, name="Carry share",
                                     mode="lines+markers", line=dict(color=POS_COLORS["RB"], width=3)))
        fig.add_trace(go.Scatter(x=log["week"], y=log["snap_pct"] * 100, name="Snap share",
                                 mode="lines", line=dict(color="#8b93a7", width=2, dash="dot")))
        fig.update_layout(title="Usage: share of team snaps, targets, carries",
                          xaxis=dict(dtick=1, title="Week"), yaxis=dict(title="% of team", range=[0, 105]))
        st.plotly_chart(plot_style(fig, 360), width="stretch", key=f"{key}_use")

    pos_ = info["position"]
    cols = {"week": "Wk", "opponent_team": "Opp", "ppr": "PPR", "snap_pct": "Snap %", "rz_opps": "RZ opps"}
    if pos_ == "QB":
        cols.update({"completions": "Cmp", "attempts": "Att", "passing_yards": "Pass Yds",
                     "passing_tds": "Pass TD", "passing_interceptions": "INT", "carries": "Rush",
                     "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD"})
    elif pos_ == "RB":
        cols.update({"carries": "Rush", "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
                     "targets": "Tgt", "receptions": "Rec", "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD"})
    else:
        cols.update({"targets": "Tgt", "receptions": "Rec", "receiving_yards": "Rec Yds",
                     "receiving_tds": "Rec TD", "target_share": "Tgt Share"})
    gl = log[list(cols)].rename(columns=cols).sort_values("Wk", ascending=False)
    gl["Snap %"] = (gl["Snap %"] * 100).round(0).fillna(0).astype(int).astype(str) + "%"
    gl["RZ opps"] = gl["RZ opps"].astype(int)
    if "Tgt Share" in gl:
        gl["Tgt Share"] = (gl["Tgt Share"] * 100).round(0).astype(int).astype(str) + "%"
    st.markdown("**Game log**")
    st.dataframe(gl, hide_index=True, width="stretch", column_config={"PPR": st.column_config.NumberColumn(format="%.1f")})


def render_kdst_profile(pid, info, key):
    fut = proj[proj["player_id"] == pid].sort_values("week")
    log = kdst_hist[kdst_hist["player_id"] == pid].sort_values("week")
    nxt = fut.iloc[0]
    html(f"""<div class="pf" style="--tc:{info['color']}">
      <img class="hs" src="{info['headshot_url']}">
      <div><div class="pf-name">{info['name']}</div>
        <div class="pf-sub">{pill(info['position'])} {team_attr(info['team'], 'team_name', info['team'])}</div>
        <div class="pf-stats">
          <div>{log['pts'].mean() if len(log) else 0:.1f}<small>Pts / game</small></div>
          <div>{nxt['proj']:.1f}<small>Wk {int(nxt['week'])} {nxt['matchup']}</small></div>
          <div>{fut[fut['week'] <= FANTASY_LAST_WEEK]['proj'].sum():.0f}<small>Rest of season</small></div>
        </div></div>
      <img class="pf-logo" src="{info['logo']}"></div>""")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=log["week"], y=log["pts"], name="Actual", marker_color=POS_COLORS[info["position"]],
                         customdata=log["opp"], hovertemplate="Week %{x} vs %{customdata}: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fut["week"], y=fut["proj"], name="Projected", mode="markers",
                             marker=dict(color=ACCENT, size=10, symbol="diamond"),
                             error_y=dict(type="data", symmetric=False, array=fut["ceiling"] - fut["proj"],
                                          arrayminus=fut["proj"] - fut["floor"], color="rgba(155,107,255,.25)",
                                          thickness=6, width=0)))
    fig.update_layout(title="Season so far and projections ahead", yaxis_title="Fantasy points",
                      xaxis=dict(dtick=1, title="Week", range=[0.4, 18.6]))
    st.plotly_chart(plot_style(fig, 360), width="stretch", key=f"{key}_kd")
    st.caption("Kickers and defenses use Sleeper's default scoring.")


def optimize(cands, slots, col="proj"):
    """Best lineup for these slots: fill the most restrictive slots first, highest projection wins."""
    order = sorted(range(len(slots)), key=lambda i: len(SLOT_ELIG[slots[i]]))
    pool = cands.sort_values(col, ascending=False)
    used, result = set(), [None] * len(slots)
    for i in order:
        c = pool[pool["position"].isin(SLOT_ELIG[slots[i]]) & ~pool["player_id"].isin(used)]
        if len(c):
            result[i] = c.iloc[0]
            used.add(result[i]["player_id"])
    return list(zip(slots, result))


def lineup_total(players_, slots):
    """Fast version of optimize() that only returns the total, for scanning thousands of trades."""
    order = sorted(slots, key=lambda x: len(SLOT_ELIG[x]))
    pool = sorted(players_, key=lambda x: -x[2])
    used, tot = set(), 0.0
    for sl in order:
        for p_ in pool:
            if p_[0] not in used and p_[1] in SLOT_ELIG[sl]:
                used.add(p_[0])
                tot += p_[2]
                break
    return tot


def my_slots():
    return league()["slots"] if league() else DEFAULT_SLOTS


def row_html(slot, r):
    return f"""<div class="lu-row"><span class="slot">{pill(slot)}</span>
      <img class="hs" src="{r['headshot_url']}"><img class="logo" src="{r['logo']}">
      <div class="lu-name">{r['name']} {badge(r['status'])}{wind_badge(r)}<small>{r['position']} · {r['matchup']} ·
      {r['matchup_grade']} matchup</small></div>
      <div class="lu-pts">{r['proj']:.1f}<small>{r['floor']:.1f} to {r['ceiling']:.1f}</small></div></div>"""


def empty_html(slot, text="Empty"):
    return f'<div class="lu-row empty"><span class="slot">{pill(slot)}</span><span>{text}</span></div>'


def lineup_rows(lineup, unavailable=None):
    """unavailable: {position: [names]} of rostered players with no game (bye) or ruled out."""
    unavailable = {k: list(v) for k, v in (unavailable or {}).items()}
    parts = []
    for sl, r in lineup:
        if r is not None:
            parts.append(row_html(sl, r))
            continue
        names_ = [n for p_ in SLOT_ELIG[sl] for n in unavailable.get(p_, [])]
        if names_:
            n = names_[0]
            for p_ in SLOT_ELIG[sl]:
                if n in unavailable.get(p_, []):
                    unavailable[p_].remove(n)
            parts.append(empty_html(sl, f"{n} · no game this week"))
        else:
            parts.append(empty_html(sl, "Nobody on your roster for this spot"))
    return '<div class="lu">' + "".join(parts) + "</div>"


def win_probability(a, b):
    """P(team A outscores team B), treating each player's floor-to-ceiling range as roughly normal."""
    def stats(rows):
        mu = sum(r["proj"] for r in rows)
        var = sum(((r["ceiling"] - r["floor"]) / 2.07) ** 2 for r in rows)
        return mu, var
    (ma, va), (mb, vb) = stats(a), stats(b)
    if va + vb == 0:
        return 0.5
    return 0.5 * (1 + math.erf((ma - mb) / math.sqrt(2 * (va + vb))))


def view_matchup(week):
    lg = league()
    try:
        ms = sl_matchups(lg["id"], week)
    except sleeper.SleeperError as e:
        st.error(str(e))
        return
    mine = next((m for m in ms if m.get("roster_id") == lg["my_roster_id"]), None)
    if not mine or mine.get("matchup_id") is None:
        st.info(f"No Week {week} matchup found in {lg['name']}. It may be a bye, or the playoff bracket isn't set yet.")
        return
    opp = next((m for m in ms if m.get("matchup_id") == mine["matchup_id"]
                and m.get("roster_id") != mine["roster_id"]), None)
    slots = my_slots()
    wk_proj = proj[(proj["week"] == week)].drop_duplicates("player_id").set_index("player_id", drop=False)

    def side(m):
        rows, html_rows = [], []
        for slot, sid in zip(slots, (m.get("starters") or [])):
            gid = to_gsis(sid) if sid and str(sid) != "0" else None
            if gid in wk_proj.index:
                r = wk_proj.loc[gid]
                rows.append(r)
                html_rows.append(row_html(slot, r))
            else:
                name = label_by_id.get(gid, "Empty slot" if gid is None else "Player not projected")
                html_rows.append(empty_html(slot, f"{name} · no game" if gid in label_by_id else name))
        return rows, '<div class="lu">' + "".join(html_rows) + "</div>"

    my_rows, my_html = side(mine)
    me_name = lg["rosters"].get(lg["my_roster_id"], {}).get("owner", "You")
    if opp is None:
        st.info("Your opponent for this week isn't set yet.")
        return
    op_rows, op_html = side(opp)
    op_name = lg["rosters"].get(opp["roster_id"], {}).get("owner", "Opponent")
    p = win_probability(my_rows, op_rows)
    my_tot, op_tot = sum(r["proj"] for r in my_rows), sum(r["proj"] for r in op_rows)
    html(f"""<div class="mu">
      <div class="mu-side"><div class="mu-name">{me_name}</div><div class="mu-pts">{my_tot:.1f}</div></div>
      <div class="mu-mid"><div class="mu-wp">{p * 100:.0f}%</div><small>win probability</small></div>
      <div class="mu-side right"><div class="mu-name">{op_name}</div><div class="mu-pts">{op_tot:.1f}</div></div>
    </div>
    <div class="wp-bar"><div style="width:{p * 100:.1f}%"></div></div>""")

    # could a different lineup from my roster do better?
    roster_ids = lg["rosters"].get(lg["my_roster_id"], {}).get("ids", [])
    avail = wk_proj[wk_proj["player_id"].isin(roster_ids) & (wk_proj["status"] != "Out")].reset_index(drop=True)
    best = optimize(avail, slots)
    best_tot = sum(r["proj"] for _, r in best if r is not None)
    if best_tot - my_tot > 0.5:
        cur_ids = {r["player_id"] for r in my_rows}
        best_ids = {r["player_id"] for _, r in best if r is not None}
        start = [label_by_id[i].split(" (")[0] for i in best_ids - cur_ids if i in label_by_id]
        sit = [label_by_id[i].split(" (")[0] for i in cur_ids - best_ids if i in label_by_id]
        new_p = win_probability([r for _, r in best if r is not None], op_rows)
        st.warning(f"**Lineup tip:** start {', '.join(start) or 'nobody new'}"
                   f"{' instead of ' + ', '.join(sit) if sit else ''}. That's +{best_tot - my_tot:.1f} projected points "
                   f"and raises your win probability to {new_p * 100:.0f}%. Set it in the Sleeper app.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{me_name}** · your starters in Sleeper")
        html(my_html)
    with c2:
        st.markdown(f"**{op_name}**")
        html(op_html)
    st.caption("Win probability compares both lineups' projected ranges.")


def use_trade(give_label, get_label):
    st.session_state["give"] = [give_label]
    st.session_state["get"] = [get_label]
    st.session_state["tr_view"] = "Trade analyzer"


def view_trade_finder():
    intro("Trade finder", "Scans every team in your league for 1-for-1 trades that make both starting lineups better, "
          "so they're trades the other manager has a reason to accept.",
          "Connect your Sleeper league in My Team, pick the weeks to count, then send an idea to the analyzer.")
    lg = league()
    if not lg:
        st.info("Connect your Sleeper league in the My Team tab to find trades.")
        return
    weeks = sorted(proj["week"].unique())
    end_default = FANTASY_LAST_WEEK if FANTASY_LAST_WEEK in weeks else weeks[-1]
    w0, w1 = st.slider("Weeks to count", min_value=weeks[0], max_value=weeks[-1], value=(weeks[0], end_default),
                       key="tf_w")
    ros = ros_values(w0, w1, lg["teams"], scoring_name).set_index("player_id")
    slots = my_slots()

    def plist(ids):
        return [(i, ros.at[i, "position"], float(ros.at[i, "ros"])) for i in ids if i in ros.index]

    mine = plist(lg["rosters"].get(lg["my_roster_id"], {}).get("ids", []))
    base_me = lineup_total(mine, slots)
    ideas = []
    for rid, r in lg["rosters"].items():
        if rid == lg["my_roster_id"]:
            continue
        theirs = plist(r["ids"])
        base_them = lineup_total(theirs, slots)
        for a in mine:
            for b in theirs:
                if a[2] < 15 or b[2] < 15:
                    continue
                gm = lineup_total([x for x in mine if x[0] != a[0]] + [b], slots) - base_me
                gt = lineup_total([x for x in theirs if x[0] != b[0]] + [a], slots) - base_them
                if gm > 3 and gt > 3:
                    ideas.append({"team": r["owner"], "give": a[0], "get": b[0], "gm": gm, "gt": gt})
    if not ideas:
        st.info("No trades found that help both lineups. Your roster may already be well balanced, "
                "or try a different week range.")
        return
    # best ideas first, but no player shows up more than twice so the list stays varied
    picked, seen = [], {}
    for t in sorted(ideas, key=lambda x: (-min(x["gm"], x["gt"]), -x["gm"])):
        if seen.get(t["give"], 0) < 2 and seen.get(t["get"], 0) < 2:
            picked.append(t)
            seen[t["give"]] = seen.get(t["give"], 0) + 1
            seen[t["get"]] = seen.get(t["get"], 0) + 1
        if len(picked) == 12:
            break
    ideas = picked
    st.markdown(f"**{len(ideas)} win-win trade ideas** · gains are rest-of-season points from your starting lineup")
    for k, t in enumerate(ideas):
        g, r = players.set_index("player_id").loc[t["give"]], players.set_index("player_id").loc[t["get"]]
        c1, c2, c3, c4 = st.columns([3, 3, 2, 1.3], vertical_alignment="center")
        c1.markdown(f"Give {pill(g['position'])} **{g['name']}**", unsafe_allow_html=True)
        c2.markdown(f"Get {pill(r['position'])} **{r['name']}** from {t['team']}", unsafe_allow_html=True)
        c3.markdown(f"You **+{t['gm']:.0f}** · them +{t['gt']:.0f}")
        c4.button("Analyze", key=f"tf_{k}", on_click=use_trade, args=(label_by_id[t["give"]], label_by_id[t["get"]]),
                  width="stretch")
    st.caption("Uses your league's roster slots and scoring. Value is what each player adds to a starting lineup, "
               "so depth that wouldn't start counts for nothing.")



# ---------- tabs ----------

def view_projections():
    intro("Weekly projections", "Every player's projected points for the week, with a floor (a bad day) and a ceiling (a great day).", "Pick a week and positions, or search a name. Bars show each player's floor-to-ceiling range.")
    c1, c2, c3 = st.columns([1, 2, 2])
    week = c1.selectbox("Week", sorted(proj["week"].unique()), key="pw")
    with c2:
        pos_sel = position_filter("pf_pos")
    search = c3.text_input("Search player", "", placeholder="Name")
    t = proj[(proj["week"] == week) & proj["position"].isin(pos_sel)]
    if search:
        t = t[t["name"].str.contains(search, case=False)]
    t = t.sort_values("proj", ascending=False).reset_index(drop=True)
    t["rank"] = t["position"] + (t.groupby("position").cumcount() + 1).astype(str)
    if t.empty:
        st.info("No players match those filters.")
        return
    pick = st.selectbox("Open a player's profile", ["(choose a player)"] + t["label"].tolist(), key="pc_pick")
    if pick != "(choose a player)":
        render_profile(id_by_label[pick], "proj_pf")
        st.divider()
    n_show = st.session_state.get("pc_n", 40)
    scale = max(float(t["ceiling"].max()), 1.0)
    cards = []
    for _, r in t.head(n_show).iterrows():
        grade = {"Good": '<span class="good">Good</span>', "Tough": '<span class="tough">Tough</span>'}.get(
            r["matchup_grade"], r["matchup_grade"])
        wx = f" · {r['weather']}" if r["weather"] else ""
        final = r["status"] == "Final"
        f_, p_, c_ = r["floor"] / scale * 100, r["proj"] / scale * 100, r["ceiling"] / scale * 100
        rng = ("" if final else f'''<div class="pc-range"><div class="range"><div class="range-bar" style="left:{f_}%;width:{c_ - f_}%"></div>
              <div class="range-dot" style="left:{p_}%"></div></div>
              <div class="range-lbl"><span>{r['floor']:.1f}</span><span>{r['ceiling']:.1f}</span></div></div>''')
        cards.append(f'''<div class="pc-row"><div class="pc-rank">{r['rank']}</div>
          <img class="hs" src="{r['headshot_url']}">
          <div class="pc-main"><div class="pc-name">{r['name']} {badge(r['status'])}{wind_badge(r)}</div>
            <div class="pc-meta">{pill(r['position'])} {r['team']} {r['matchup']} · {grade}{wx}</div></div>
          {rng if not final else '<div class="pc-range"></div>'}
          <div class="pc-pts">{r['proj']:.1f}<small>{'final' if final else 'proj'}</small></div></div>''')
    html('<div class="pc">' + "".join(cards) + "</div>")
    if len(t) > n_show:
        if st.button(f"Show more ({len(t) - n_show} left)", width="stretch"):
            st.session_state["pc_n"] = n_show + 40
            st.rerun()

def view_rankings():
    intro("Rest-of-season rankings", "Every player ranked by value: projected points above a replacement-level starter at his position.", "Set your league size and the weeks you care about. Connect Sleeper in My Team to see who is available.")
    weeks = sorted(proj["week"].unique())
    end_default = FANTASY_LAST_WEEK if FANTASY_LAST_WEEK in weeks else weeks[-1]
    c1, c2, c3 = st.columns([3, 1, 2])
    rw0, rw1 = c1.slider("Weeks", min_value=weeks[0], max_value=weeks[-1], value=(weeks[0], end_default), key="rk_w")
    sizes = [8, 10, 12, 14]
    default_n = league()["teams"] if league() and league()["teams"] in sizes else 12
    rk_n = c2.selectbox("League size", sizes, index=sizes.index(default_n), key="rk_n")
    with c3:
        rk_pos = position_filter("rk_pos")
    rk = ros_values(rw0, rw1, rk_n, scoring_name)
    rk = rk[rk["position"].isin(rk_pos)].copy()
    if league():
        rk["owner"] = rk["player_id"].map(
            lambda i: "Mine" if i in league()["mine"] else ("Taken" if i in league()["rostered"] else "Available"))
        if st.toggle(f"Only show players available in {league()['name']}", key="rk_avail"):
            rk = rk[rk["owner"] == "Available"]
    cols = ["headshot_url", "rank", "pos_rank", "name", "position", "team", "games", "ppg", "ros", "value"]
    if league():
        cols.append("owner")
    styled = (rk[cols].head(300).style.map(lambda p: POS_STYLE.get(p, ""), subset=["position"])
              .map(lambda o: {"Available": "color:#22d3b6;font-weight:700", "Mine": "color:#ffb35c;font-weight:700"}.get(o, ""),
                   subset=["owner"] if league() else []))
    st.dataframe(styled, hide_index=True, width="stretch", height=620, column_config={
        "headshot_url": st.column_config.ImageColumn("", width="small"),
        "rank": "Rank", "pos_rank": "Pos rank", "name": "Player", "position": "Pos", "team": "Team",
        "games": st.column_config.NumberColumn("Games left", format="%d"),
        "ppg": st.column_config.NumberColumn("Pts / game", format="%.1f"),
        "ros": st.column_config.NumberColumn("ROS points", format="%.1f"),
        "value": st.column_config.ProgressColumn("Value", format="%.1f", min_value=0,
                                                 max_value=float(max(rk["value"].max(), 1)) if len(rk) else 1.0,
                                                 color="#9b6bff",
                                                 help="Rest-of-season points above a replacement-level starter"),
        "owner": "League status",
    })
    if not league():
        st.caption("Connect Sleeper in My Team to see who's available.")

def view_team():
    intro("My Team", "Your best lineup for the week, close calls, alerts, and the best pickups on your waiver wire.", "Import your roster from Sleeper or add players by hand. Bookmark the page and your team is saved.")
    with st.container(border=True):
        st.markdown("**Import from Sleeper** · pulls your roster and your league's waiver wire")
        c1, c2 = st.columns([2, 3])
        if "sl_user" not in st.session_state:
            st.session_state["sl_user"] = st.query_params.get("sleeper", "")
        uname = c1.text_input("Sleeper username", key="sl_user", placeholder="Your Sleeper username")
        if uname:
            try:
                user = sl_user(uname)
                if not user:
                    c2.warning(f"No Sleeper account named \"{uname}\". Check the spelling.")
                else:
                    lgs = sl_leagues(user["user_id"], season)
                    if not lgs:
                        c2.info(f"No {season} NFL leagues found for {uname}.")
                    else:
                        names = {lg["league_id"]: lg.get("name", "League") for lg in lgs}
                        ids = list(names)
                        start = ids.index(st.query_params["league"]) if st.query_params.get("league") in ids else 0
                        lid = c2.selectbox("League", ids, index=start, format_func=names.get, key="sl_league")
                        chosen = next(lg for lg in lgs if lg["league_id"] == lid)
                        if c2.button("Import roster", type="primary", on_click=connect_league,
                                     args=(chosen, user["user_id"])):
                            st.query_params["sleeper"] = uname
                            st.query_params["league"] = lid
            except sleeper.SleeperError as e:
                c2.error(str(e))
        if league():
            lg = league()
            extra = f" {lg['unmatched']} roster spots (IDP or brand-new players) aren't projected here." \
                if lg["unmatched"] else ""
            st.success(f"Connected to **{lg['name']}** ({lg['teams']} teams).{extra}")
            st.caption("Using your league's scoring and lineup slots.")

    init_from_url("team", "team")
    c1, c2 = st.columns([4, 1])
    roster = c1.multiselect("Your roster", labels, key="team", placeholder="Add your players...")
    c2.button("Load example team", width="stretch", on_click=load_example_team)
    sync_url("team", roster)
    t_week = st.selectbox("Week", sorted(proj["week"].unique()), key="team_week")

    if league():
        st.session_state.setdefault("team_view", "My lineup")
        mode = st.segmented_control("Show", ["My lineup", "This week's matchup"], required=True, key="team_view",
                                    label_visibility="collapsed")
        if mode == "This week's matchup":
            return view_matchup(t_week)

    if not roster:
        st.info("Import from Sleeper, add players by hand, or load the example team to see your optimal lineup.")
    else:
        ids = [id_by_label[l] for l in roster]
        avail = proj[proj["player_id"].isin(ids) & (proj["week"] == t_week)].copy()
        on_bye = [label_by_id[i] for i in ids if i not in set(avail["player_id"])]
        out = avail[avail["status"] == "Out"]
        avail = avail[avail["status"] != "Out"].sort_values("proj", ascending=False)

        lineup = optimize(avail, my_slots())
        used = {r["player_id"] for _, r in lineup if r is not None}
        bench = avail[~avail["player_id"].isin(used)]

        a, b, c = st.columns(3)
        a.metric("Projected points", f"{sum(r['proj'] for _, r in lineup if r is not None):.1f}")
        b.metric("Floor", f"{sum(r['floor'] for _, r in lineup if r is not None):.1f}")
        c.metric("Ceiling", f"{sum(r['ceiling'] for _, r in lineup if r is not None):.1f}")

        unavailable = {}
        for i in ids:
            if i not in set(avail["player_id"]) and i in label_by_id:
                pos_i = players.loc[players["player_id"] == i, "position"].iloc[0]
                unavailable.setdefault(pos_i, []).append(label_by_id[i].split(" (")[0])
        html(lineup_rows(lineup, unavailable))

        calls = []
        for slot, r in lineup:
            if r is None:
                continue
            alt = bench[bench["position"].isin(SLOT_ELIG[slot])]
            if len(alt) and r["proj"] - alt.iloc[0]["proj"] < 1.5:
                calls.append(f"**{SLOT_LABEL.get(slot, slot)}:** {r['name']} ({r['proj']:.1f}) over {alt.iloc[0]['name']} "
                             f"({alt.iloc[0]['proj']:.1f})")
        alerts = []
        if on_bye:
            alerts.append("No game this week: " + ", ".join(on_bye))
        if len(out):
            alerts.append("Ruled out: " + ", ".join(out["name"]))
        q = avail[avail["status"].isin(["Questionable", "Doubtful"])]
        if len(q):
            alerts.append("Check before kickoff: " + ", ".join(q["name"] + " (" + q["status"] + ")"))
        w_ = avail[avail["windy"] & avail["position"].isin(["QB", "WR", "TE"])]
        if len(w_):
            alerts.append("Windy game forecast: " + ", ".join(w_["name"] + f" ({w_['wind'].iloc[0]:.0f} mph)"))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Close calls**")
            if calls:
                for x in calls:
                    st.write(x)
                st.caption("Compare them in Start / Sit.")
            else:
                st.caption("None this week.")
            if alerts:
                st.markdown("**Alerts**")
                for x in alerts:
                    st.write(x)
        with c2:
            st.markdown("**Bench**")
            if len(bench):
                st.dataframe(bench[["name", "position", "matchup", "status", "proj"]], hide_index=True, width="stretch",
                             column_config={"name": "Player", "position": "Pos", "matchup": "Opp", "status": "Status",
                                            "proj": st.column_config.NumberColumn("Proj", format="%.1f")})
            else:
                st.caption("Nobody on your bench this week.")

        # waiver targets
        weakest = {}
        for slot, r in lineup:
            if r is not None:
                for p_ in SLOT_ELIG[slot]:
                    if p_ not in weakest or r["proj"] < weakest[p_][1]:
                        weakest[p_] = (r["name"], r["proj"])
        pool = proj[(proj["week"] == t_week) & ~proj["player_id"].isin(ids) & (proj["status"] != "Out")]
        if league():
            st.markdown(f"**Best available in {league()['name']}**")
            pool = pool[~pool["player_id"].isin(league()["rostered"])]
        else:
            st.markdown("**Waiver targets that would start for you**")
            pool = pool[pool["player_id"].isin(breakouts["player_id"])]
        pool = pool.merge(breakouts[["player_id", "breakout_score"]], on="player_id", how="left")
        rows_w = []
        for _, r in pool.sort_values("proj", ascending=False).iterrows():
            w = weakest.get(r["position"])
            if w is None or r["proj"] > w[1]:
                rows_w.append({"Player": f"{r['name']} ({r['position']}, {r['team']})", "Week proj": r["proj"],
                               "Would replace": f"{w[0]} ({w[1]:.1f})" if w else "Open slot",
                               "Breakout score": r["breakout_score"]})
            if len(rows_w) == 8:
                break
        if rows_w:
            st.dataframe(pd.DataFrame(rows_w), hide_index=True, width="stretch",
                         column_config={"Week proj": st.column_config.NumberColumn(format="%.1f"),
                                        "Breakout score": st.column_config.NumberColumn(format="%.1f")})
            if not league():
                st.caption("Breakout candidates not on your roster. Import your Sleeper league to see only players "
                           "who are actually available.")
        else:
            st.caption("Nobody available projects higher than your current starters this week.")

def view_players():
    intro("Player profiles", "Season stats, game log, usage trends, and projections for every remaining week.", "Search for any player.")
    pick = st.selectbox("Find a player", labels, index=0, key="pf_pick")
    render_profile(id_by_label[pick], "tab_pf")

def view_startsit():
    intro("Start / Sit", "Compare 2 to 4 players for a lineup decision.", "Pick the week and players. The highlighted card is the start. Close calls get a tiebreaker tip.")
    c1, c2 = st.columns([1, 3])
    ss_week = c1.selectbox("Week", sorted(proj["week"].unique()), key="ssw")
    init_from_url("ss_picks", "ss")
    picks = c2.multiselect("Pick 2 to 4 players", labels, max_selections=4, key="ss_picks",
                           placeholder="Start typing a name...")
    sync_url("ss", picks)
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
                best = r["player_id"] == best_id
                cards_html.append(f"""<div class="ss-card {'best' if best else ''}" style="--tc:{r['color']}">
                  <div class="ss-band"><img src="{r['logo']}">{team_attr(r['team'], 'team_nick', r['team'])}{badge(r['status'])}</div>
                  {'<div class="pick">START</div>' if best else ''}
                  <div class="ss-body"><img class="hs" src="{r['headshot_url']}">
                    <div class="ss-name">{r['name']}</div>
                    <div class="ss-meta">{pill(r['position'])} {r['matchup']} · {r['matchup_grade']} matchup</div>
                    <div class="ss-meta">{r['weather']} {wind_badge(r)}</div>
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
                    st.info(f"**It's close** ({gap:.1f} points apart). Favored to win your matchup? Lean "
                            f"**{safe['name']}** for the safer floor. Underdog? Lean **{boom['name']}** for upside.")
                else:
                    st.success(f"**Start {live.loc[live['proj'].idxmax(), 'name']}.** "
                               f"Projected {gap:.1f} points ahead of the next option.")
            if (df["status"] == "Questionable").any():
                st.caption("Questionable players usually play, but check news before kickoff.")
            st.caption("Share it by copying the page link.")

def view_breakouts():
    intro("Breakouts", "Players whose usage is climbing before their points catch up. Targets and carries predict future points better than last week's score.", "Look for the Usage spike and Due for points tags. Tap a row for the full profile.")
    c1, c2 = st.columns([3, 2])
    with c1:
        bpos = position_filter("bo_pos")
    b = breakouts[breakouts["position"].isin(bpos)]
    if league() and c2.toggle(f"Only available in {league()['name']}", key="bo_avail"):
        b = b[~b["player_id"].isin(league()["rostered"])]
    b = b.head(25).copy()
    b["usage"] = b.apply(lambda r: f"{r['opps_before']:.1f} → {r['opps_last3']:.1f}", axis=1)
    b["signal"] = b.apply(lambda r: " · ".join(
        s for s, ok in [("Usage spike", r["usage_change"] >= 3), ("Due for points", r["points_gap"] >= 3)] if ok), axis=1)
    sel = st.dataframe(
        b[["headshot_url", "name", "position", "team", "usage", "xfp_before", "xfp_last3", "ppr_last3",
           "signal", "breakout_score"]].style.map(lambda p: POS_STYLE.get(p, ""), subset=["position"]),
        hide_index=True, width="stretch", height=480, on_select="rerun", selection_mode="single-row", key="bo_table",
        column_config={
            "headshot_url": st.column_config.ImageColumn("", width="small"),
            "name": "Player", "position": "Pos", "team": "Team",
            "usage": st.column_config.TextColumn("Targets + carries / game", help="Earlier this season → last 3 games"),
            "xfp_before": st.column_config.NumberColumn("xFP before", format="%.1f"),
            "xfp_last3": st.column_config.NumberColumn("xFP last 3", format="%.1f"),
            "ppr_last3": st.column_config.NumberColumn("Actual last 3", format="%.1f"),
            "signal": "Signal",
            "breakout_score": st.column_config.ProgressColumn("Breakout score", format="%.1f", min_value=0,
                                                              max_value=float(max(breakouts["breakout_score"].max(), 1)),
                                                              color="#9b6bff"),
        })
    rows = sel.selection.rows if sel else []
    st.caption("Select a row for the full profile.")
    if rows and b.iloc[rows[0]]["player_id"] in label_by_id:
        st.divider()
        render_profile(b.iloc[rows[0]]["player_id"], "bo_pf")

def view_trades():
    intro("Trade analyzer", "Judges a trade by value above replacement, so a 2-for-1 doesn't win just by adding bodies.", "Add players to each side. Copy the page link to share the exact trade.")
    weeks = sorted(proj["week"].unique())
    end_default = FANTASY_LAST_WEEK if FANTASY_LAST_WEEK in weeks else weeks[-1]
    c0, c00 = st.columns([3, 1])
    w0, w1 = c0.slider("Weeks to count", min_value=weeks[0], max_value=weeks[-1], value=(weeks[0], end_default))
    sizes = [8, 10, 12, 14]
    default_n = league()["teams"] if league() and league()["teams"] in sizes else 12
    teams_n = c00.selectbox("League size", sizes, index=sizes.index(default_n), key="league_size")
    init_from_url("give", "give")
    init_from_url("get", "get")
    c1, c2 = st.columns(2)
    give = c1.multiselect("You give", labels, key="give", placeholder="Players you're sending")
    get = c2.multiselect("You get", labels, key="get", placeholder="Players you're receiving")
    sync_url("give", give)
    sync_url("get", get)
    vals = ros_values(w0, w1, teams_n, scoring_name).set_index("player_id")

    def side(labs):
        rows_ = []
        for lab in labs:
            v = vals.loc[id_by_label[lab]]
            rows_.append({"Player": lab, "Games": int(v["games"]), "ROS points": v["ros"],
                          "Replacement": v["repl"], "Value": v["value"]})
        return pd.DataFrame(rows_)

    if give and get:
        g, r = side(give), side(get)
        gv, rv = g["Value"].sum(), r["Value"].sum()
        diff = rv - gv
        if abs(diff) < max(5, 0.08 * max(gv, rv, 1)):
            verdict, fn = "Fair trade", st.info
        elif diff > 0:
            verdict, fn = "You win this trade", st.success
        else:
            verdict, fn = "You lose this trade", st.error
        fn(f"**{verdict}** · you {'gain' if diff >= 0 else 'give up'} {abs(diff):.1f} points of value above "
           f"replacement (Weeks {w0}-{w1}, {teams_n}-team league)")
        cfg = {c: st.column_config.NumberColumn(format="%.1f") for c in ["ROS points", "Value"]}
        cfg["Replacement"] = st.column_config.NumberColumn(
            "Replacement level", format="%.1f", help="Rest-of-season points of the last startable player at that position")
        a, bcol = st.columns(2)
        a.metric("You give", f"{gv:.1f} value")
        a.caption(f"{g['ROS points'].sum():.0f} raw projected points")
        a.dataframe(g, hide_index=True, width="stretch", column_config=cfg)
        bcol.metric("You get", f"{rv:.1f} value")
        bcol.caption(f"{r['ROS points'].sum():.0f} raw projected points")
        bcol.dataframe(r, hide_index=True, width="stretch", column_config=cfg)
        st.caption("Share it by copying the page link.")
    else:
        st.info("Add at least one player to each side.")

def view_accuracy():
    intro("Accuracy", "How the projections have actually performed, scored against real results every week.", "Pick a week to see the best calls and the biggest misses.")
    if acc.empty:
        st.info("No scored weeks yet. Accuracy appears here once games are played.")
    else:
        rel = acc[acc["proj"] >= 8].copy()
        rel["err"] = (rel["proj"] - rel["ppr"]).abs()
        rel["base_err"] = (rel["ewm_ppr"] - rel["ppr"]).abs()
        rel["in_range"] = rel["ppr"].between(rel["floor"], rel["ceiling"])
        wk_stats = rel.groupby("week").agg(model=("err", "mean"), baseline=("base_err", "mean")).reset_index()
        src = acc.groupby("week")["source"].first()
        a, b, c, d = st.columns(4)
        a.metric("Weeks scored", len(wk_stats))
        b.metric("Average miss", f"{rel['err'].mean():.2f} pts",
                 f"{rel['err'].mean() - rel['base_err'].mean():+.2f} vs baseline", delta_color="inverse")
        c.metric("In floor-ceiling range", f"{rel['in_range'].mean() * 100:.0f}%")
        d.metric("Player-games scored", f"{len(rel):,}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=wk_stats["week"], y=wk_stats["model"], name="Model avg miss", marker_color=ACCENT,
                             text=wk_stats["model"].round(2), textposition="inside", insidetextanchor="start",
                             textfont=dict(color="#ffffff")))
        fig.add_trace(go.Scatter(x=wk_stats["week"], y=wk_stats["baseline"], name="Baseline avg miss",
                                 mode="lines+markers", line=dict(color=POS_COLORS["QB"], width=3)))
        fig.update_layout(xaxis=dict(dtick=1, title="Week"), yaxis_title="Average miss (pts)")
        st.plotly_chart(plot_style(fig), width="stretch", key="acc_chart")
        bf = [int(w) for w, s_ in src.items() if s_ == "backfill"]
        if bf:
            st.caption(f"Weeks {bf[0]}-{bf[-1]} are walk-forward backfills: the model was retrained before each week "
                       "using only earlier games. Later weeks use projections saved live before kickoff. "
                       "Scored on players projected for 8+ points.")
        aw = st.selectbox("Look at a week", sorted(rel["week"].unique(), reverse=True), key="acc_week")
        w = rel[rel["week"] == aw].copy()
        w["miss"] = w["ppr"] - w["proj"]
        cols_ = {"name": "Player", "position": "Pos", "team": "Team", "proj": "Projected", "ppr": "Actual",
                 "miss": "Difference"}
        cfg = {k: st.column_config.NumberColumn(format="%.1f") for k in ["Projected", "Actual", "Difference"]}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Best calls** (projected 12+, closest to actual)")
            st.dataframe(w[w["proj"] >= 12].nsmallest(8, "err")[list(cols_)].rename(columns=cols_),
                         hide_index=True, width="stretch", column_config=cfg)
        with c2:
            st.markdown("**Biggest misses**")
            st.dataframe(w.nlargest(8, "err")[list(cols_)].rename(columns=cols_),
                         hide_index=True, width="stretch", column_config=cfg)
        fig = go.Figure()
        for pos_, color in POS_COLORS.items():
            s_ = w[w["position"] == pos_]
            fig.add_trace(go.Scatter(x=s_["proj"], y=s_["ppr"], mode="markers", name=pos_,
                                     marker=dict(color=color, size=8, opacity=.8), text=s_["name"],
                                     hovertemplate="%{text}: proj %{x:.1f}, actual %{y:.1f}<extra></extra>"))
        mx = max(w["proj"].max(), w["ppr"].max()) + 2
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect projection",
                                 line=dict(color="rgba(207,213,227,.45)", dash="dash")))
        fig.update_layout(title=f"Week {aw}: projected vs actual", xaxis_title="Projected PPR", yaxis_title="Actual PPR")
        st.plotly_chart(plot_style(fig, 420), width="stretch", key="acc_scatter")

def view_model():
    intro("How the model works", "The data, model, backtest results, and experiments behind every number in the app.", "Read top to bottom.")
    st.markdown(
        "- **Data:** weekly stats for every QB, RB, WR, and TE since 2021 from "
        "[nflverse](https://github.com/nflverse), stored in a SQLite database, plus injury reports, snap counts, "
        "play-by-play, and game-day weather forecasts.\n"
        "- **Features (all known before kickoff):** recent form weighted toward the latest games "
        "(points, targets, carries, target share, air yards), the opponent's points allowed to that "
        "position, and the Vegas implied team total.\n"
        "- **Model:** one gradient-boosted tree model (scikit-learn) with position as a feature. Floor and ceiling "
        "come from how actual scores spread around projections in a held-out season. Players ruled Out are set to 0.\n"
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
    imp["feature"] = imp["feature"].map(lambda f: FEATURE_LABELS.get(f, f.replace("ewm_", "Recent ").replace("_", " ")))
    fig = go.Figure(go.Bar(x=imp["importance"][::-1], y=imp["feature"][::-1], orientation="h", marker_color=ACCENT))
    fig.update_layout(xaxis_title="Increase in average miss when the feature is scrambled (pts)")
    st.plotly_chart(plot_style(fig, 360, legend_below=False), width="stretch")
    st.caption("Evaluated on fantasy-relevant players (recent form of 6+ PPR points per game).")
    st.markdown("#### Experiments")
    st.write("Ideas I tested that didn't make the cut. A change was kept only if it lowered the average miss in both "
             "backtests. Splitting into one model per position left each model with too little data. Weather didn't "
             "move the overall average because strong wind is rare, but QBs in 15+ mph wind did score about 3 points "
             "below projection, so the app flags windy games instead.")
    ex = pd.DataFrame(experiments).rename(columns={"change": "Change", "mae_2025": "2025 avg miss",
                                                   "mae_2026": "2026 avg miss", "kept": "Kept"})
    st.dataframe(ex, hide_index=True, width="stretch",
                 column_config={"2025 avg miss": st.column_config.NumberColumn(format="%.2f"),
                                "2026 avg miss": st.column_config.NumberColumn(format="%.2f")})


# ---------- layout: six sections in the sidebar menu, each with one clear job ----------
qp = st.query_params
if "give" in qp or "get" in qp:
    default_tab = "Trades"
elif league():
    default_tab = "My Team"
else:
    default_tab = "This Week"
names = ["This Week", "My Team", "Rankings", "Players", "Trades", "How It Works"]
st.session_state.setdefault("nav", default_tab)
st.session_state["_nested"] = False
with nav_box:
    icons = {"This Week": ":material/today:", "My Team": ":material/groups:", "Rankings": ":material/leaderboard:",
             "Players": ":material/person_search:", "Trades": ":material/swap_horiz:", "How It Works": ":material/insights:"}
    section = st.radio("Menu", names, key="nav", label_visibility="collapsed", format_func=lambda n: f"{icons[n]} {n}")
tabs = {n: st.container() if n == section else None for n in names}

if section == "This Week":
    section_header("This week", f"Stats through Week {last_week}")
    render_home_summary()
    st.write("")
    st.session_state["_nested"] = True
    st.session_state.setdefault("tw_view", "Start / Sit" if "ss" in qp else "Projections")
    view = st.segmented_control("View", ["Projections", "Start / Sit"], required=True, key="tw_view",
                                label_visibility="collapsed")
    view_projections() if view == "Projections" else view_startsit()
if section == "My Team":
    view_team()
if section == "Rankings":
    section_header("Rankings")
    st.session_state["_nested"] = True
    st.session_state.setdefault("rk_view", "Rest of season")
    view = st.segmented_control("View", ["Rest of season", "Breakouts"], required=True, key="rk_view",
                                label_visibility="collapsed")
    view_rankings() if view == "Rest of season" else view_breakouts()
if section == "Players":
    view_players()
if section == "Trades":
    section_header("Trades")
    st.session_state["_nested"] = True
    st.session_state.setdefault("tr_view", "Trade analyzer")
    view = st.segmented_control("View", ["Trade analyzer", "Trade finder"], required=True, key="tr_view",
                                label_visibility="collapsed")
    view_trades() if view == "Trade analyzer" else view_trade_finder()
if section == "How It Works":
    section_header("How it works")
    st.session_state["_nested"] = True
    st.session_state.setdefault("hw_view", "Accuracy")
    view = st.segmented_control("View", ["Accuracy", "The model"], required=True, key="hw_view",
                                label_visibility="collapsed")
    view_accuracy() if view == "Accuracy" else view_model()

updated = metrics.get("updated_at", "")
html(f"""<div class="foot"><div>Red Zone Lab · projections update Tue, Sat and Sun mornings{' · last updated ' + updated if updated else ''}</div>
<div>Data: <a href="https://github.com/nflverse" target="_blank">nflverse</a> · <a href="https://sleeper.com" target="_blank">Sleeper</a> ·
<a href="https://open-meteo.com" target="_blank">Open-Meteo</a> · <a href="https://github.com/NicholasMatthews22/fantasy-football-analytics" target="_blank">Source code</a></div></div>""")
