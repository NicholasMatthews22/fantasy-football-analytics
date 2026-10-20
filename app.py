"""Fantasy Football Analytics: projections, rankings, my team, start/sit, breakouts, trades (full PPR)."""
import json
import math
import sqlite3

import numpy as np

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ffa import fantasycalc, season_sim, sleeper
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
ACCENT, MUTED = "#9b6bff", "#a3abbd"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap');
:root { --surface: #1a1f2c; --surface2: #222838; --line: #2b3244; --muted: #8b93a7; --accent: #9b6bff;
  --qb: #ff4f81; --rb: #22d3b6; --wr: #58b6ff; --te: #ffb35c; }
.block-container { padding-top: 2rem !important; max-width: 1280px; }
h1, h2, h3, .stTabs button p, [data-testid="stMetricValue"], .num {
  font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif !important; }
h1 { font-weight: 800 !important; letter-spacing: -.5px; font-size: 2.1rem !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p, .stApp input, .stApp button p,
.gl-card, .lu-row, .pc-row, .ss-card, .note, .chip { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; }
.gl-pts, .lu-pts, .pc-pts, .ss-proj, .mu-pts, .mu-wp, .pf-stats div { font-variant-numeric: tabular-nums; letter-spacing: -.3px; }

/* page header */
.ph { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  border-bottom: 1px solid var(--line); padding-bottom: 12px; margin: 0 0 16px; }
.ph h2 { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-weight: 800; font-size: 1.6rem; letter-spacing: -.4px; margin: 0; padding: 0; }
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
section[data-testid="stSidebar"] .brand { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-weight: 800; font-size: 1.1rem;
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
.intro-t { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-weight: 700; font-size: 1.25rem; }
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
.gl-pts { margin-left: auto; font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-size: 1.45rem; font-weight: 700; line-height: 1; }
.gl-pts small { display: block; font-size: .65rem; font-weight: 500; color: var(--muted); text-align: right; }
.note { font-size: .85rem; color: #cfd5e3; margin: 0 0 8px; }

/* start/sit */
.ss-grid { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 12px; }
.ss-card { border-radius: 14px; overflow: hidden; background: var(--surface); border: 1px solid var(--line); position: relative; }
.ss-card.best { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), 0 8px 30px rgba(155,107,255,.15); }
.ss-band { background: var(--tc); color: #fff; display: flex; align-items: center; gap: 8px; padding: 7px 12px;
  font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-weight: 700; font-size: .85rem; text-transform: uppercase; letter-spacing: .5px; }
.ss-band img { width: 24px; height: 24px; background: rgba(255,255,255,.92); border-radius: 50%; padding: 2px; }
.ss-band .badge { margin-left: auto; }
.ss-body { padding: 14px; text-align: center; }
.ss-body .hs { width: 86px; height: 86px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2); }
.ss-name { font-weight: 700; font-size: 1.05rem; margin-top: 6px; }
.ss-meta { font-size: .8rem; color: var(--muted); }
.ss-proj { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-size: 2.4rem; font-weight: 700; line-height: 1.05; margin-top: 6px; }
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
.pf-name { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-size: 1.9rem; font-weight: 800; line-height: 1.05; }
.pf-sub { opacity: .9; margin-top: 4px; }
.pf-logo { width: 76px; height: 76px; margin-left: auto; filter: drop-shadow(0 1px 3px rgba(0,0,0,.5)); }
.pf-stats { display: flex; gap: 22px; margin-top: 10px; flex-wrap: wrap; }
.pf-stats div { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-size: 1.4rem; font-weight: 700; line-height: 1; }
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
.lu-pts { margin-left: auto; text-align: right; font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-size: 1.2rem; font-weight: 700; line-height: 1; }
.lu-pts small { display: block; font-size: .62rem; font-weight: 500; color: var(--muted); margin-top: 2px; }

/* projection cards */
.pc { display: grid; gap: 6px; }
.pc-row { display: grid; grid-template-columns: 44px 46px minmax(0, 1fr) minmax(0, 150px) 86px; align-items: center;
  gap: 12px; padding: 8px 14px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); }
.pc-rank { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-weight: 700; color: var(--muted); font-size: .85rem; }
.pc-row .hs { width: 44px; height: 44px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2); }
.pc-main { min-width: 0; }
.pc-name { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pc-meta { color: var(--muted); font-size: .78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pc-meta .good { color: var(--rb); font-weight: 700; } .pc-meta .tough { color: var(--qb); font-weight: 700; }
.pc-range .range { margin: 4px 0 3px; }
.pc-range .range-lbl { font-size: .66rem; }
.pc-pts { text-align: right; font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-size: 1.45rem; font-weight: 700; line-height: 1; }
.pc-pts small { display: block; font-size: .62rem; color: var(--muted); font-weight: 500; margin-top: 2px; }

/* matchup */
.mu { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 12px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; }
.mu-side.right { text-align: right; }
.mu-name { font-weight: 700; color: #cfd5e3; }
.mu-pts { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
.mu-mid { text-align: center; }
.mu-wp { font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; font-size: 1.6rem; font-weight: 800; color: var(--accent); }
.mu-mid small { color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .5px; }
.wp-bar { height: 8px; border-radius: 4px; background: var(--qb); margin: 10px 0 14px; overflow: hidden; }
.wp-bar div { height: 100%; background: var(--accent); }

/* avatars */
.av { border-radius: 50%; object-fit: cover; flex: none; background: var(--surface2); vertical-align: middle; }
.av-txt { display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: .7rem;
  color: #e8ebf2; background: #2d3448; }

/* scoreboard */
.sb { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 14px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px 12px; }
.sb-team { display: flex; align-items: center; gap: 12px; min-width: 0; }
.sb-team.right { justify-content: flex-end; text-align: right; }
.sb-name { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sb-rec { color: var(--muted); font-size: .76rem; }
.sb-pts { font-size: 2rem; font-weight: 800; margin-left: auto; font-variant-numeric: tabular-nums; letter-spacing: -.5px; }
.sb-team.right .sb-pts { margin: 0 auto 0 0; }
.sb-mid { text-align: center; }
.sb-wk { font-size: .68rem; font-weight: 700; letter-spacing: 1px; color: var(--muted); }
.sb-vs { font-size: .72rem; color: var(--muted); }
.sb-wp { grid-column: 1 / -1; display: flex; align-items: center; gap: 10px; font-size: .8rem; font-weight: 700;
  font-variant-numeric: tabular-nums; }
.sb-wp .wp-bar { flex: 1; margin: 0; }

/* power rankings */
.pr { display: grid; gap: 4px; }
.pr-head, .pr-row { display: grid; grid-template-columns: 34px minmax(0, 2.2fr) 1fr .8fr .8fr 1.4fr 1.4fr; align-items: center;
  gap: 12px; padding: 8px 14px; }
.pr-head { font-size: .68rem; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); font-weight: 600; }
.pr-row { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
.pr-row.mine { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
.pr-rank { font-weight: 800; color: var(--muted); }
.pr-team { display: flex; align-items: center; gap: 10px; min-width: 0; }
.pr-name { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pr-num { font-variant-numeric: tabular-nums; font-weight: 600; }
.pr-num small, .pr-bar small { display: block; color: var(--muted); font-weight: 500; font-size: .7rem; }
.pr-bar b { font-variant-numeric: tabular-nums; font-size: .9rem; }
.pr-bar i { display: block; height: 6px; border-radius: 3px; background: var(--surface2); margin-top: 4px; overflow: hidden; }
.pr-bar em { display: block; height: 100%; background: var(--accent); }
.pr-bar em.t { background: #ffb35c; }
.tf-who { display: inline-flex; align-items: center; gap: 8px; }
.tf-rec { color: var(--muted); font-size: .76rem; }

/* trade finder */
.tf-count { color: var(--muted); font-size: .85rem; margin: 4px 0 12px; }
.tf-card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px;
  margin-bottom: 8px; }
.tf-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: .85rem;
  color: #cfd5e3; margin-bottom: 10px; }
.tf-g { display: inline-block; font-size: .72rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; margin-left: 6px;
  font-variant-numeric: tabular-nums; }
.tf-g.me { background: rgba(155,107,255,.18); color: #c9b3ff; }
.tf-g.them { background: var(--surface2); color: var(--muted); }
.tf-sides { display: grid; grid-template-columns: minmax(0, 1fr) 24px minmax(0, 1fr); align-items: center; gap: 8px; }
.tf-lbl { font-size: .66rem; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); font-weight: 600;
  margin-bottom: 6px; }
.tf-pl { display: flex; align-items: center; gap: 10px; min-width: 0; }
.tf-pl .hs { width: 40px; height: 40px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2);
  flex: none; }
.tf-txt { min-width: 0; }
.tf-txt .nm { font-weight: 700; font-size: .92rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tf-txt .mt { font-size: .74rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pk { width: 36px; height: 36px; border-radius: 8px; background: var(--surface2); border: 1px solid var(--line);
  display: inline-flex; align-items: center; justify-content: center; text-align: center; font-size: .62rem;
  font-weight: 800; line-height: 1.1; color: #e8ebf2; flex: none; }
.tf-arrow { text-align: center; color: var(--muted); font-size: 1.1rem; }

/* avatars */
.av { border-radius: 50%; object-fit: cover; flex: none; background: var(--surface2); vertical-align: middle; }
.av-txt { display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: .7rem;
  color: #e8ebf2; background: #2d3448; }

/* scoreboard */
.sb { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 14px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px 12px; }
.sb-team { display: flex; align-items: center; gap: 12px; min-width: 0; }
.sb-team.right { justify-content: flex-end; text-align: right; }
.sb-name { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sb-rec { color: var(--muted); font-size: .76rem; }
.sb-pts { font-size: 2rem; font-weight: 800; margin-left: auto; font-variant-numeric: tabular-nums; letter-spacing: -.5px; }
.sb-team.right .sb-pts { margin: 0 auto 0 0; }
.sb-mid { text-align: center; }
.sb-wk { font-size: .68rem; font-weight: 700; letter-spacing: 1px; color: var(--muted); }
.sb-vs { font-size: .72rem; color: var(--muted); }
.sb-wp { grid-column: 1 / -1; display: flex; align-items: center; gap: 10px; font-size: .8rem; font-weight: 700;
  font-variant-numeric: tabular-nums; }
.sb-wp .wp-bar { flex: 1; margin: 0; }

/* power rankings */
.pr { display: grid; gap: 4px; }
.pr-head, .pr-row { display: grid; grid-template-columns: 34px minmax(0, 2.2fr) 1fr .8fr .8fr 1.4fr 1.4fr; align-items: center;
  gap: 12px; padding: 8px 14px; }
.pr-head { font-size: .68rem; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); font-weight: 600; }
.pr-row { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
.pr-row.mine { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
.pr-rank { font-weight: 800; color: var(--muted); }
.pr-team { display: flex; align-items: center; gap: 10px; min-width: 0; }
.pr-name { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pr-num { font-variant-numeric: tabular-nums; font-weight: 600; }
.pr-num small, .pr-bar small { display: block; color: var(--muted); font-weight: 500; font-size: .7rem; }
.pr-bar b { font-variant-numeric: tabular-nums; font-size: .9rem; }
.pr-bar i { display: block; height: 6px; border-radius: 3px; background: var(--surface2); margin-top: 4px; overflow: hidden; }
.pr-bar em { display: block; height: 100%; background: var(--accent); }
.pr-bar em.t { background: #ffb35c; }
.tf-who { display: inline-flex; align-items: center; gap: 8px; }
.tf-rec { color: var(--muted); font-size: .76rem; }

/* trade finder */
.tf-count { color: var(--muted); font-size: .82rem; margin: 4px 0 12px; }
.tf-count b { color: #e8ebf2; }
.tf-card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; margin-bottom: 8px;
  min-height: 210px; display: flex; flex-direction: column; }
.tf-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: .85rem; color: #cfd5e3;
  margin-bottom: 10px; }
.tf-fair { font-size: .72rem; font-weight: 700; color: var(--muted); }
.tf-sides { display: grid; grid-template-columns: minmax(0, 1fr) 22px minmax(0, 1fr); align-items: start; gap: 8px; flex: 1; }
.tf-lbl { font-size: .66rem; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); font-weight: 600;
  margin-bottom: 6px; display: flex; justify-content: space-between; }
.tf-lbl span { color: #cfd5e3; font-variant-numeric: tabular-nums; }
.tf-pl { display: flex; align-items: center; gap: 9px; min-width: 0; margin-bottom: 6px; }
.tf-pl .hs { width: 36px; height: 36px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2); flex: none; }
.tf-txt { min-width: 0; }
.tf-txt .nm { font-weight: 700; font-size: .88rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tf-txt .mt { font-size: .72rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pk { width: 36px; height: 36px; border-radius: 8px; background: var(--surface2); border: 1px solid var(--line);
  display: inline-flex; align-items: center; justify-content: center; text-align: center; font-size: .62rem;
  font-weight: 800; line-height: 1.1; color: #e8ebf2; flex: none; }
.tf-arrow { text-align: center; color: var(--muted); font-size: 1rem; padding-top: 26px; }
.tf-foot { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
.tf-g { display: inline-block; font-size: .72rem; font-weight: 700; padding: 3px 8px; border-radius: 6px; font-variant-numeric: tabular-nums; }
.tf-g.me { background: rgba(155,107,255,.18); color: #c9b3ff; }
.tf-g.them { background: var(--surface2); color: var(--muted); }

/* ---------- cleanup pass: flatter, higher contrast, fewer decorations ---------- */
:root { --muted: #a3abbd; --line: #30384c; --surface: #191e2a; --surface2: #232a3a; }
.gl-pts, .lu-pts, .pc-pts, .ss-proj, .mu-pts, .sb-pts, .pf-stats div, .hl-pts, .pr-num, .pr-bar b, .tf-lbl span {
  font-stretch: 88%; font-variant-numeric: tabular-nums; letter-spacing: 0; }
h2, h3, .ph h2 { letter-spacing: -.2px !important; }
.pos-FLEX { background: #c7cdda; }
.badge { text-transform: none; letter-spacing: 0; font-weight: 700; font-size: .7rem; }
.pf { background: var(--surface); border: 1px solid var(--line); color: #e8ebf2; }
.pf .hs { border: 0; background: var(--surface2); }
.pf-stats small { text-transform: none; letter-spacing: 0; font-size: .72rem; opacity: 1; color: var(--muted); }
.pf-logo { filter: none; }
.ss-card.best { border-color: var(--line); box-shadow: none; }
.ss-band { background: var(--surface2); color: #e8ebf2; text-transform: none; letter-spacing: 0; font-weight: 600; }
.ss-band img { background: transparent; padding: 0; }
.pick { top: 46px; background: #e8ebf2; color: #10131c; letter-spacing: 0; }
.range-bar { opacity: .35; }
.range-dot { border-width: 2px; }
.lu-row.empty { opacity: 1; color: var(--muted); }
.tf-lbl, .pr-head { text-transform: none; letter-spacing: 0; font-size: .76rem; }
.pk { width: 36px; height: 36px; border-radius: 8px; background: var(--surface2); border: 1px solid var(--line);
  display: inline-flex; align-items: center; justify-content: center; text-align: center; font-size: .62rem;
  font-weight: 800; line-height: 1.1; color: #e8ebf2; flex: none; }
.tf-arrow { font-size: .75rem; padding-top: 30px; }
.pr-row.mine { border-color: var(--line); box-shadow: none; background: #222a3c; }
.pr-row.mine .pr-name::after { content: "You"; margin-left: 8px; font-size: .68rem; font-weight: 700; padding: 1px 6px;
  border-radius: 4px; background: #e8ebf2; color: #10131c; vertical-align: middle; }
.sb-wk { letter-spacing: 0; font-size: .78rem; }
.mu-mid small { text-transform: none; letter-spacing: 0; }
span[data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] > div { border-color: var(--line) !important; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] { transition: none; padding: 8px 12px; border-radius: 8px; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] { background: var(--surface2); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { color: #fff !important; font-weight: 700; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before { content: ""; width: 3px;
  align-self: stretch; margin: 2px 9px 2px -4px; border-radius: 2px; background: var(--accent); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p { color: #c3c9d6; font-weight: 500; }
.stApp button { transition: none !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* home summary: two plain lists */
.home { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr); gap: 24px; margin: 0 0 24px; }
.home-h { font-weight: 700; font-size: .95rem; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.hl-row { display: grid; grid-template-columns: 32px 32px minmax(0, 1fr) auto; align-items: center; gap: 12px;
  padding: 8px 0; border-bottom: 1px solid var(--line); }
.hl-pos { font-size: .78rem; font-weight: 700; color: var(--muted); }
.hl-row .hs { width: 32px; height: 32px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2); }
.hl-main { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hl-main small { color: var(--muted); font-size: .8rem; margin-left: 6px; }
.hl-pts { font-weight: 700; font-size: 1.1rem; }
.nt-row { display: grid; grid-template-columns: 140px minmax(0, 1fr); gap: 12px; padding: 10px 0;
  border-bottom: 1px solid var(--line); font-size: .9rem; line-height: 1.45; }
.nt-k { color: var(--muted); }
.nt-v { color: #d4d9e4; }
.nt-v b { color: #fff; }

/* consistent spacing: 8 / 12 / 16 / 24 */
.lu { gap: 6px; margin: 8px 0 16px; }
.pc { gap: 6px; }
.tf-card { margin-bottom: 8px; }
.ph { margin-bottom: 16px; }
.ph.sub { margin: 8px 0 12px; }
[data-testid="stMetric"] { border-radius: 12px; padding: 12px 16px; }

/* home: top projected tiles */
.tp-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 16px; }
.tp { position: relative; overflow: hidden; border-radius: 12px; padding: 12px 14px; min-width: 0;
  background: color-mix(in srgb, var(--tc) 38%, #151a25); display: flex; flex-direction: column; gap: 10px; }
.tp-logo { position: absolute; right: -18px; bottom: -18px; width: 110px; height: 110px; opacity: .13; pointer-events: none; }
.tp-top { display: flex; justify-content: space-between; gap: 8px; font-size: .76rem; font-weight: 600;
  color: rgba(255,255,255,.78); }
.tp-mid { display: flex; align-items: center; gap: 12px; min-width: 0; }
.tp-hs { width: 60px; height: 60px; border-radius: 50%; object-fit: cover; object-position: top;
  background: rgba(0,0,0,.25); flex: none; }
.tp-name { font-weight: 800; font-size: 1.1rem; line-height: 1.2; color: #fff; min-width: 0; }
.tp-pts { font-weight: 800; font-size: 2rem; color: #fff; line-height: 1; font-stretch: 88%;
  font-variant-numeric: tabular-nums; }
.tp-pts small { font-size: .72rem; font-weight: 600; color: rgba(255,255,255,.72); margin-left: 6px; }

/* home: slate + notes */
.home2 { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 16px; margin: 0 0 24px; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 4px 16px 8px; }
.panel-h { display: flex; justify-content: space-between; align-items: baseline; font-weight: 700; padding: 12px 0 10px;
  border-bottom: 1px solid var(--line); }
.panel-h span { font-weight: 500; font-size: .78rem; color: var(--muted); }
.gm { display: grid; grid-template-columns: 34px minmax(0, 1fr) 16px minmax(0, 1fr) 76px 70px; align-items: center; gap: 8px;
  padding: 8px 0; border-bottom: 1px solid var(--line); font-size: .86rem; }
.gm:last-child { border-bottom: 0; }
.gm.done { opacity: .55; }
.gm-day { color: var(--muted); font-size: .76rem; font-weight: 600; }
.gm-t { display: flex; align-items: center; gap: 6px; font-weight: 600; min-width: 0; }
.gm-t img { width: 22px; height: 22px; }
.gm-t b { margin-left: auto; font-weight: 700; font-variant-numeric: tabular-nums; color: #e8ebf2; }
.gm-at { color: var(--muted); font-size: .75rem; text-align: center; }
.gm-line { color: var(--muted); font-size: .78rem; text-align: right; white-space: nowrap; }
.gm-tot { text-align: right; font-size: .78rem; font-weight: 700; color: #c3c9d6; font-variant-numeric: tabular-nums; }
.gm-tot.hot { color: #ffb35c; }
.nt-row2 { display: grid; grid-template-columns: 40px minmax(0, 1fr); gap: 12px; align-items: start; padding: 12px 0;
  border-bottom: 1px solid var(--line); }
.nt-row2:last-child { border-bottom: 0; }
.nt-row2 img { width: 40px; height: 40px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface2); }
.nt-dot { width: 40px; height: 40px; border-radius: 50%; background: var(--surface2); display: block; }
.nt-row2 .nt-k { color: var(--muted); font-size: .78rem; font-weight: 600; margin-bottom: 2px; }
.nt-row2 .nt-v { color: #d4d9e4; font-size: .9rem; line-height: 1.45; }
.nt-row2 .nt-v b { color: #fff; }

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
  .home { grid-template-columns: 1fr; gap: 16px; }
  .tp-grid { grid-template-columns: 1fr 1fr; gap: 8px; }
  .tp { padding: 10px 12px; gap: 8px; }
  .tp-top span:last-child { display: none; }
  .tp-hs { width: 44px; height: 44px; }
  .tp-name { font-size: .95rem; }
  .tp-pts { font-size: 1.5rem; }
  .home2 { grid-template-columns: 1fr; }
  .gm { grid-template-columns: 30px minmax(0, 1fr) 12px minmax(0, 1fr) 58px; font-size: .8rem; }
  .gm-tot { display: none; }
  .nt-row { grid-template-columns: 1fr; gap: 2px; }
  .tf-sides { grid-template-columns: 1fr; }
  .pr-head { display: none; }
  .pr-row { grid-template-columns: 18px minmax(0, 1fr) 40px 62px 48px; gap: 7px; padding: 8px 10px; font-size: .85rem; }
  .pr-row .av { width: 26px !important; height: 26px !important; }
  .pr-num small { display: none; }
  .pr-bar small { display: none; }
  .pr-bar b { font-size: .8rem; }
  .pr-row > span:nth-child(4), .pr-row > span:nth-child(5) { display: none; }
  .sb { grid-template-columns: 1fr 1fr; }
  .sb-mid { display: none; }
  .sb-pts { font-size: 1.5rem; }
  .sb-team .av { width: 36px !important; height: 36px !important; }
  .pk { width: 36px; height: 36px; border-radius: 8px; background: var(--surface2); border: 1px solid var(--line);
  display: inline-flex; align-items: center; justify-content: center; text-align: center; font-size: .62rem;
  font-weight: 800; line-height: 1.1; color: #e8ebf2; flex: none; }
.tf-arrow { padding: 0; transform: rotate(90deg); }
  .tf-card { min-height: 0; }
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
                      font=dict(color="#d4d9e4", family="Archivo, Helvetica Neue, Arial, sans-serif"), title_font=dict(family="Archivo", size=14))
    fig.update_xaxes(gridcolor="#232a3a", zerolinecolor="#2b3244")
    fig.update_yaxes(gridcolor="#232a3a", zerolinecolor="#2b3244")
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", y=-0.24, x=0))
    return fig


# ---------- data ----------
@st.cache_data
def _query(sql, db_version):
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con)


def query(sql):
    """Cached read that refreshes automatically whenever the database file changes (each data update)."""
    return _query(sql, DB_PATH.stat().st_mtime)


DB_VERSION = DB_PATH.stat().st_mtime
games_tbl = query("SELECT * FROM games")
games_tbl["kick"] = games_tbl["gameday"].astype(str) + " " + (
    games_tbl["gametime"].astype(str) if "gametime" in games_tbl else "")
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
def ros_values(w0, w1, teams_n, scoring_key="", db_version=None):
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
        users = sl_users(league["league_id"])
    except sleeper.SleeperError:
        users = []
    names = {u["user_id"]: sleeper.team_name(u) for u in users}
    avatars = {u["user_id"]: sleeper.avatar_url(u) for u in users}
    lsettings = league.get("settings") or {}
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
        "dynasty": (league.get("settings") or {}).get("type") == 2,
        "superflex": "SUPER_FLEX" in (league.get("roster_positions") or [])
                     or (league.get("roster_positions") or []).count("QB") >= 2,
        "rosters": {r.get("roster_id"): {
            "owner": names.get(r.get("owner_id"), f"Team {r.get('roster_id')}"),
            "avatar": avatars.get(r.get("owner_id")),
            "wins": int((r.get("settings") or {}).get("wins", 0)),
            "losses": int((r.get("settings") or {}).get("losses", 0)),
            "ties": int((r.get("settings") or {}).get("ties", 0)),
            "pf": float((r.get("settings") or {}).get("fpts", 0)) + float((r.get("settings") or {}).get("fpts_decimal", 0)) / 100,
            "ids": [g for g in (to_gsis(p) for p in (r.get("players") or [])) if g]} for r in rosters},
        "playoff_teams": int(lsettings.get("playoff_teams") or 6),
        "playoff_start": int(lsettings.get("playoff_week_start") or 15),
        "pick_seasons": PICK_SEASONS, "pick_rounds": int(lsettings.get("draft_rounds") or 5),
    }
    if st.session_state["league_info"]["dynasty"]:
        try:
            traded = sleeper.traded_picks(league["league_id"])
        except sleeper.SleeperError:
            traded = []
        st.session_state["league_info"]["picks"] = sleeper.pick_ownership(
            rosters, traded, PICK_SEASONS, st.session_state["league_info"]["pick_rounds"])
    st.session_state["scoring"] = "My league"


def league():
    return st.session_state.get("league_info")


PICK_SEASONS = [season + 1, season + 2, season + 3]
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th"}


def pick_id(s_, rd, orig):
    return f"PICK_{s_}_{rd}_{orig}"


def parse_pick(pid):
    _, s_, rd, orig = pid.split("_")
    return int(s_), int(rd), int(orig)


def pick_label(pid):
    s_, rd, orig = parse_pick(pid)
    owner = league()["rosters"].get(orig, {}).get("owner", f"Team {orig}") if league() else f"Team {orig}"
    return f"{s_} {ORD.get(rd, f'{rd}th')} ({owner})"


def league_pick_ids():
    lg = league()
    if not lg or not lg.get("picks"):
        return []
    return [pick_id(*p_) for rid in lg["picks"] for p_ in lg["picks"][rid]]


def pick_tier(pid):
    """Next year's picks get an Early/Mid/Late slot from the original team's standing (worst teams pick early)."""
    s_, rd, orig = parse_pick(pid)
    lg = league()
    if s_ != PICK_SEASONS[0] or not lg:
        return None
    order = sorted(lg["rosters"], key=lambda r: (lg["rosters"][r]["wins"], lg["rosters"][r]["pf"]))
    n = len(order)
    pos_ = order.index(orig) if orig in order else n // 2
    return "Early" if pos_ < n / 3 else ("Mid" if pos_ < 2 * n / 3 else "Late")


# reconnect automatically when the page is opened from a saved link
if league() is None and st.query_params.get("sleeper") and st.query_params.get("league"):
    try:
        u = sl_user(st.query_params["sleeper"])
        lg = next((x for x in sl_leagues(u["user_id"], season) if x["league_id"] == st.query_params["league"]), None)
        if u and lg:
            connect_league(lg, u["user_id"])
    except Exception:
        pass


# draft picks become tradeable "assets" once a dynasty league is connected
asset_labels = list(labels)
for _pid in league_pick_ids():
    _lab = pick_label(_pid)
    id_by_label[_lab] = _pid
    label_by_id[_pid] = _lab
    asset_labels.append(_lab)

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


def tile_color(team):
    """Team color for a tile background, falling back to the secondary color when the main one is near-black."""
    c = team_attr(team, "team_color", "#3a4256")
    try:
        r_, g_, b_ = (int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        if 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ < 20:
            return team_attr(team, "team_color2", "#3a4256")
    except ValueError:
        pass
    return c


def usage_words(pos_):
    return "pass attempts and runs" if pos_ == "QB" else "targets and carries"


@st.cache_data
def breakout_usage(db_version):
    """Usage per game earlier this season vs. the last 3 games. QBs: attempts + runs. Others: targets + carries."""
    g = games_cur.sort_values(["player_id", "week"]).copy()
    g["use"] = np.where(g["position"] == "QB", g["attempts"] + g["carries"], g["targets"] + g["carries"])
    rows = {}
    for pid, x in g.groupby("player_id"):
        if len(x) >= 4:
            rows[pid] = (x["use"].iloc[:-3].mean(), x["use"].tail(3).mean())
    return rows


def render_home_summary():
    wk = proj[(proj["week"] == next_week) & ~proj["status"].isin(["Out", "Final"])]

    # 1) top projected player at each position: big tiles tinted in the team's color
    tiles = []
    for pos_ in ["QB", "RB", "WR", "TE"]:
        r = wk[wk["position"] == pos_].nlargest(1, "proj").iloc[0]
        tiles.append(f"""<div class="tp" style="--tc:{tile_color(r['team'])}">
          <img class="tp-logo" src="{r['logo']}">
          <div class="tp-top"><span>{pos_}1 this week</span><span>{r['team']} {r['matchup']}</span></div>
          <div class="tp-mid"><img class="tp-hs" src="{r['headshot_url']}"><div class="tp-name">{r['name']} {badge(r['status'])}</div></div>
          <div class="tp-pts">{r['proj']:.1f}<small>projected</small></div></div>""")

    # 2) the slate: every game this week with Vegas totals
    g = proj[proj["week"] == next_week].drop_duplicates(["game_id", "team"])
    games_wk = []
    for gid, x in g.groupby("game_id"):
        if len(x) < 2:
            continue
        home = x[x["home"] == 1].iloc[0] if (x["home"] == 1).any() else x.iloc[1]
        away = x[x["home"] != 1].iloc[0] if (x["home"] != 1).any() else x.iloc[0]
        done = (x["status"] == "Final").any()
        games_wk.append((home["implied_total"] + away["implied_total"], gid, away, home, done))
    kick = {gid: k for gid, k in zip(games_tbl["game_id"], games_tbl["kick"])}
    games_wk.sort(key=lambda t: (t[4], kick.get(t[1], ""), t[1]))
    hi_total = max((t[0] for t in games_wk if not t[4]), default=0)
    slate = []
    for total, gid, away, home, done in games_wk:
        day = pd.to_datetime(home["gameday"]).strftime("%a")
        fav = away if away["implied_total"] > home["implied_total"] else home
        spread = abs(away["implied_total"] - home["implied_total"])
        line = "Final" if done else (f"{fav['team']} −{spread:.1f}" if spread >= 0.5 else "Pick'em")
        hot = (not done) and total == hi_total
        slate.append(f"""<div class="gm {'done' if done else ''}"><span class="gm-day">{day}</span>
          <span class="gm-t"><img src="{away['logo']}">{away['team']}<b>{away['implied_total']:.1f}</b></span>
          <span class="gm-at">@</span>
          <span class="gm-t"><img src="{home['logo']}">{home['team']}<b>{home['implied_total']:.1f}</b></span>
          <span class="gm-line">{line}</span>
          <span class="gm-tot {'hot' if hot else ''}">{'' if done else f'O/U {total:.1f}'}</span></div>""")

    # 3) worth knowing, each with a face or logo
    picks = wk[wk["proj"] >= 12]
    smash = picks.nlargest(1, "def_factor").iloc[0]
    use = breakout_usage(DB_VERSION)
    starters = set(wk.loc[wk["proj"] >= 8, "player_id"])
    bo_pool = breakouts[breakouts["player_id"].isin(starters) & breakouts["player_id"].isin(use)]
    notes = [(smash["headshot_url"], "Best matchup",
              f"<b>{smash['name']}</b> {smash['matchup']}. That defense allows "
              f"{(smash['def_factor'] - 1) * 100:.0f}% more points than average to {smash['position']}s.")]
    if len(bo_pool):
        bo = bo_pool.iloc[0]
        u0, u1 = use[bo["player_id"]]
        notes.append((bo["headshot_url"], "Breakout to watch",
                      f"<b>{bo['name']}</b> ({bo['position']}) went from {u0:.1f} to {u1:.1f} "
                      f"{usage_words(bo['position'])} per game over his last 3."))
    hi = wk.drop_duplicates("team").nlargest(1, "implied_total").iloc[0]
    notes.append((hi["logo"], "Highest team total",
                  f"<b>{team_attr(hi['team'], 'team_name', hi['team'])}</b> {hi['matchup']}, "
                  f"{hi['implied_total']:.1f} points expected by Vegas."))
    if not injury_posted:
        notes.append((None, "Injuries", f"The Week {next_week} report isn't final. Statuses update Saturday and Sunday mornings."))
    windy = wk[wk["windy"]].drop_duplicates("game_id")
    if len(windy):
        games_txt = ", ".join(f"{r['team']} {r['matchup']} ({r['wind']:.0f} mph)" for _, r in windy.iterrows())
        notes.append((None, "Wind", f"{games_txt}. QBs in 15+ mph wind have scored about 3 points under projection."))
    note_rows = "".join(
        f'<div class="nt-row2">{f"<img src={img}>" if img else "<span class=nt-dot></span>"}'
        f'<div><div class="nt-k">{k}</div><div class="nt-v">{v}</div></div></div>' for img, k, v in notes)

    html(f"""<div class="tp-grid">{"".join(tiles)}</div>
    <div class="home2">
      <div class="panel"><div class="panel-h">Week {next_week} slate<span>Vegas implied points</span></div>{"".join(slate)}</div>
      <div class="panel"><div class="panel-h">Worth knowing</div>{note_rows}</div>
    </div>""")


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


def avatar_html(team, size=34):
    if team.get("avatar"):
        return f'<img class="av" src="{team["avatar"]}" style="width:{size}px;height:{size}px">'
    initials = "".join(w[0] for w in team.get("owner", "?").split()[:2]).upper()
    return f'<span class="av av-txt" style="width:{size}px;height:{size}px">{initials}</span>'


def record(team):
    return f"{team.get('wins', 0)}-{team.get('losses', 0)}" + (f"-{team['ties']}" if team.get("ties") else "")


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
    me_t = lg["rosters"].get(lg["my_roster_id"], {})
    op_t = lg["rosters"].get(opp["roster_id"], {})
    html(f"""<div class="sb">
      <div class="sb-team">{avatar_html(me_t, 52)}<div><div class="sb-name">{me_name}</div>
        <div class="sb-rec">{record(me_t)} · {me_t.get('pf', 0):.0f} PF</div></div>
        <div class="sb-pts">{my_tot:.1f}</div></div>
      <div class="sb-mid"><div class="sb-wk">Week {week}</div><div class="sb-vs">projected</div></div>
      <div class="sb-team right"><div class="sb-pts">{op_tot:.1f}</div><div><div class="sb-name">{op_name}</div>
        <div class="sb-rec">{record(op_t)} · {op_t.get('pf', 0):.0f} PF</div></div>{avatar_html(op_t, 52)}</div>
      <div class="sb-wp"><span>{p * 100:.0f}%</span><div class="wp-bar"><div style="width:{p * 100:.1f}%"></div></div>
        <span>{(1 - p) * 100:.0f}%</span></div>
    </div>""")

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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def market_rows(dynasty, superflex, teams, ppr):
    try:
        return fantasycalc.fetch_rows(dynasty, superflex, teams, ppr)
    except Exception:
        return []


def market_values(dynasty, superflex, teams, ppr):
    """FantasyCalc trade values keyed by our player IDs. Empty dict if the service can't be reached."""
    raw = fantasycalc.player_values(market_rows(dynasty, superflex, teams, ppr))
    return {g: v for g, v in ((to_gsis(sid), val) for sid, val in raw.items()) if g}


def league_market_values():
    """Player values plus every draft pick in the league (dynasty only), keyed by our IDs."""
    lg = league()
    if not lg:
        return {}
    args = (lg.get("dynasty", False), lg.get("superflex", False), lg["teams"], lg["ppr"])
    vals = dict(market_values(*args))
    if lg.get("dynasty") and lg.get("picks"):
        pv = fantasycalc.pick_values(market_rows(*args), lg["pick_seasons"], lg["pick_rounds"])
        for pid in league_pick_ids():
            s_, rd, _ = parse_pick(pid)
            v = pv.get((s_, rd, pick_tier(pid))) or pv.get((s_, rd, None))
            if v:
                vals[pid] = float(v)
    return vals


@st.cache_data(ttl=900, show_spinner=False)
def run_season_sim(league_id, scoring_key, db_version=None, n_sims=5000):
    lg = league()
    slots = my_slots()
    rids = list(lg["rosters"])
    idx = {r: i for i, r in enumerate(rids)}
    T = len(rids)

    def week_strength(week):
        mu, sd = np.zeros(T), np.zeros(T)
        wk_rows = proj[proj["week"] == week]
        for r, i in idx.items():
            avail = wk_rows[wk_rows["player_id"].isin(lg["rosters"][r]["ids"]) & (wk_rows["status"] != "Out")]
            starters = [x for _, x in optimize(avail, slots) if x is not None]
            mu[i] = sum(x["proj"] for x in starters)
            sd[i] = math.sqrt(sum(((x["ceiling"] - x["floor"]) / 2.07) ** 2 for x in starters)) or 1.0
        return mu, sd

    weeks, strength = [], []
    for w in range(next_week, lg["playoff_start"]):
        try:
            ms = sl_matchups(lg["id"], w)
        except sleeper.SleeperError:
            continue
        groups = {}
        for m in ms:
            if m.get("matchup_id") is not None and m.get("roster_id") in idx:
                groups.setdefault(m["matchup_id"], []).append(idx[m["roster_id"]])
        pairs = [tuple(g) for g in groups.values() if len(g) == 2]
        if not pairs:
            continue
        mu, sd = week_strength(w)
        weeks.append((mu, sd, pairs))
        strength.append(mu)
    pw_ = [week_strength(w) for w in range(lg["playoff_start"], min(lg["playoff_start"] + 3, 19))
           if w in set(proj["week"])]
    if pw_:
        p_mu, p_sd = np.mean([x[0] for x in pw_], axis=0), np.mean([x[1] for x in pw_], axis=0)
    else:
        p_mu, p_sd = week_strength(next_week)
    res = season_sim.simulate(rids, [lg["rosters"][r]["wins"] for r in rids], [lg["rosters"][r]["pf"] for r in rids],
                              weeks, lg["playoff_teams"], p_mu, p_sd, n_sims=n_sims)
    res["weekly"] = np.mean(strength, axis=0) if strength else p_mu
    res["rids"] = rids
    res["n_weeks"] = len(weeks)
    return res


def view_league():
    intro("Power rankings", "Each team's chance to make the playoffs and win the title, from 5,000 simulations of the "
          "rest of the season using every team's best projected lineup, your league's real schedule, and current records.",
          "Teams are ranked by title odds. Your team is highlighted.")
    lg = league()
    if not lg:
        st.info("Connect your Sleeper league in My Team to see power rankings and playoff odds.")
        return
    with st.spinner("Simulating the rest of the season..."):
        res = run_season_sim(lg["id"], scoring_name, DB_VERSION)
    rows = []
    for i, r in enumerate(res["rids"]):
        t = lg["rosters"][r]
        rows.append({"rid": r, "t": t, "weekly": res["weekly"][i], "wins": res["avg_wins"][i],
                     "playoff": res["playoff_pct"][i], "bye": res["bye_pct"][i], "title": res["title_pct"][i]})
    rows.sort(key=lambda x: (-x["title"], -x["playoff"], -x["weekly"]))
    games_left = res["n_weeks"]
    out = [f'<div class="pr-head"><span>#</span><span>Team</span><span>Record</span><span>Proj / wk</span>'
           f'<span>Proj wins</span><span>Playoffs</span><span>Title</span></div>']
    for k, x in enumerate(rows, 1):
        mine = x["rid"] == lg["my_roster_id"]
        bye = f'<small>bye {x["bye"] * 100:.0f}%</small>' if res["has_byes"] else ""
        out.append(f"""<div class="pr-row {'mine' if mine else ''}">
          <span class="pr-rank">{k}</span>
          <span class="pr-team">{avatar_html(x['t'], 32)}<span class="pr-name">{x['t']['owner']}</span></span>
          <span class="pr-num">{record(x['t'])}<small>{x['t'].get('pf', 0):.0f} PF</small></span>
          <span class="pr-num">{x['weekly']:.1f}</span>
          <span class="pr-num">{x['wins']:.1f}</span>
          <span class="pr-bar"><b>{x['playoff'] * 100:.0f}%</b><i><em style="width:{x['playoff'] * 100:.1f}%"></em></i>{bye}</span>
          <span class="pr-bar"><b>{x['title'] * 100:.0f}%</b><i><em class="t" style="width:{x['title'] * 100:.1f}%"></em></i></span>
        </div>""")
    html('<div class="pr">' + "".join(out) + "</div>")
    st.caption(f"{games_left} regular-season weeks left before playoffs start in Week {lg['playoff_start']} · "
               f"{lg['playoff_teams']} playoff teams · ties broken by points for. Refreshes with each projection update.")


def use_trade(give_labels, get_labels):
    st.session_state["give"] = list(give_labels)
    st.session_state["get"] = list(get_labels)
    st.session_state["tr_view"] = "Trade analyzer"


def package_value(vals):
    """What a group of players is worth in one trade. Extra pieces count for less than their sticker value,
    because one great player is worth more than two good ones (roster spots and lineup slots are limited)."""
    vals = sorted(vals, reverse=True)
    return vals[0] + 0.7 * sum(vals[1:]) if vals else 0.0


def view_trade_finder():
    intro("Trade finder", "Finds trades with each team in your league that are fair by market value and make both "
          "starting lineups better, so the other manager has a reason to accept.",
          "Connect your Sleeper league in My Team. Open any idea in the trade analyzer to look closer.")
    lg = league()
    if not lg:
        st.info("Connect your Sleeper league in My Team to find trades.")
        return
    weeks = sorted(proj["week"].unique())
    end_default = FANTASY_LAST_WEEK if FANTASY_LAST_WEEK in weeks else weeks[-1]
    c1, c2, c3 = st.columns([3, 1.3, 1.3])
    w0, w1 = c1.slider("Weeks to count for lineup gains", min_value=weeks[0], max_value=weeks[-1],
                       value=(weeks[0], end_default), key="tf_w")
    kinds = c2.selectbox("Trade types", ["1-for-1 and 2-for-1", "1-for-1 only"], key="tf_kinds")
    style = "Win now"
    if lg.get("dynasty") and lg.get("picks"):
        style = st.segmented_control("Your goal", ["Win now", "Build for the future"], key="tf_style",
                                     default="Win now", required=True,
                                     help="Win now: improve your lineup, paying with players or picks. "
                                          "Build for the future: trade lineup points to contenders for picks.")
    tol = c3.selectbox("Fairness", ["Strict (±8%)", "Normal (±15%)", "Loose (±25%)"], index=1, key="tf_tol")
    tol = {"Strict (±8%)": .08, "Normal (±15%)": .15, "Loose (±25%)": .25}[tol]

    ros = ros_values(w0, w1, lg["teams"], scoring_name, DB_VERSION).set_index("player_id")
    market = league_market_values()
    if market:
        value = lambda pid: market.get(pid, 0.0)
        src = "FantasyCalc " + ("dynasty" if lg.get("dynasty") else "redraft") + " market values" + \
              (" (superflex)" if lg.get("superflex") else "")
    else:
        value = lambda pid: float(ros.at[pid, "value"]) * 30 if pid in ros.index else 0.0
        src = "projected rest-of-season value (market values couldn't be loaded)"
    slots = my_slots()

    def plist(ids):
        return [(i, ros.at[i, "position"], float(ros.at[i, "ros"])) for i in ids if i in ros.index]

    def tradeable(ids):
        return [i for i in ids if i in ros.index and value(i) > 0 and ros.at[i, "position"] not in ("K", "DST")]

    def picks_of(rid):
        if not (market and lg.get("picks")):
            return []
        return [p_ for p_ in (pick_id(*x) for x in lg["picks"].get(rid, [])) if value(p_) >= 400]

    mine_ids = lg["rosters"].get(lg["my_roster_id"], {}).get("ids", [])
    mine, my_trade = plist(mine_ids), tradeable(mine_ids) + picks_of(lg["my_roster_id"])
    base_me = lineup_total(mine, slots)
    ideas = []

    def consider(give, get, theirs, base_them, owner):
        vg, vr = package_value([value(i) for i in give]), package_value([value(i) for i in get])
        if min(vg, vr) <= 0 or abs(vg - vr) / max(vg, vr) > tol:
            return
        me_after = [x for x in mine if x[0] not in give] + [x for x in theirs if x[0] in get]
        th_after = [x for x in theirs if x[0] not in get] + [x for x in mine if x[0] in give]
        gm = lineup_total(me_after, slots) - base_me
        gt = lineup_total(th_after, slots) - base_them
        gets_picks = any(p_.startswith("PICK_") for p_ in get)
        gives_picks = any(p_.startswith("PICK_") for p_ in give)
        if style == "Win now":
            pick_paid = sum(value(p_) for p_ in give if p_.startswith("PICK_"))
            # they may give up a little lineup for a real pick (2nd round or better), like a rebuilding team would
            ok = gm > 3 and (gt > 0 or (pick_paid >= 1000 and gt > -15))
        else:
            ok = gets_picks and gt > 3                        # you sell lineup points to a contender for picks
        if ok:
            ideas.append({"team": owner, "rid": cur_rid[0], "give": give, "get": get, "gm": gm, "gt": gt, "vg": vg, "vr": vr})

    two_for_one = kinds.startswith("1-for-1 and")
    cur_rid = [None]
    for rid, r in lg["rosters"].items():
        if rid == lg["my_roster_id"]:
            continue
        cur_rid[0] = rid
        theirs, their_trade = plist(r["ids"]), tradeable(r["ids"]) + picks_of(rid)
        base_them = lineup_total(theirs, slots)
        for a in my_trade:
            for b in their_trade:
                consider([a], [b], theirs, base_them, r["owner"])
        if two_for_one:
            for i, a1 in enumerate(my_trade):        # I give two, get one better player
                for a2 in my_trade[i + 1:]:
                    for b in their_trade:
                        consider([a1, a2], [b], theirs, base_them, r["owner"])
            for a in my_trade:                       # I give one, get two
                for j, b1 in enumerate(their_trade):
                    for b2 in their_trade[j + 1:]:
                        consider([a], [b1, b2], theirs, base_them, r["owner"])

    if not ideas:
        st.info("No fair trades found that help both lineups. Try a looser fairness setting or a different week range.")
        return
    # best combined lineup gain first; keep the list varied (each player at most twice)
    picked, seen = [], {}
    for t in sorted(ideas, key=lambda x: -(x["gm"] + x["gt"]) if style == "Win now"
                    else -(x["vr"] - x["vg"] + x["gt"] * 10)):
        if all(seen.get(p_, 0) < 2 for p_ in t["give"] + t["get"]):
            picked.append(t)
            for p_ in t["give"] + t["get"]:
                seen[p_] = seen.get(p_, 0) + 1
        if len(picked) == 12:
            break

    st.markdown(f'<div class="tf-count">{len(picked)} fair trade ideas · value from {src} · lineup gains are '
                f'rest-of-season starter points (Weeks {w0}-{w1})</div>', unsafe_allow_html=True)
    pinfo = players.set_index("player_id")

    def asset_html(p_):
        if p_.startswith("PICK_"):
            s_, rd, orig = parse_pick(p_)
            tier = pick_tier(p_)
            return (f'<div class="tf-pl"><span class="pk">{str(s_)[2:]}<br>R{rd}</span><div class="tf-txt">'
                    f'<div class="nm">{s_} {ORD.get(rd, "")}{" (" + tier + ")" if tier else ""}</div>'
                    f'<div class="mt">from {lg["rosters"].get(orig, {}).get("owner", "")} · value {value(p_):,.0f}'
                    f'</div></div></div>')
        return (f'<div class="tf-pl"><img class="hs" src="{pinfo.at[p_, "headshot_url"]}"><div class="tf-txt">'
                f'<div class="nm">{pinfo.at[p_, "name"]}</div><div class="mt">{pill(pinfo.at[p_, "position"])} '
                f'{pinfo.at[p_, "team"]} · value {value(p_):,.0f}</div></div></div>')

    def side_html(label, pids, val):
        rows_ = "".join(asset_html(p_) for p_ in pids)
        return (f'<div class="tf-side"><div class="tf-lbl">{label} <span>{val:,.0f}</span></div>{rows_}</div>')

    cols = st.columns(2, gap="medium")
    for k, t in enumerate(picked):
        diff = (t["vr"] - t["vg"]) / max(t["vg"], t["vr"]) * 100
        with cols[k % 2]:
            html(f"""<div class="tf-card">
              <div class="tf-head"><span class="tf-who">{avatar_html(lg["rosters"][t["rid"]], 26)} <b>{t['team']}</b>
                <span class="tf-rec">{record(lg["rosters"][t["rid"]])}</span></span>
                <span class="tf-fair">{'Even' if abs(diff) < 3 else (f'{diff:+.0f}% value to you')}</span></div>
              <div class="tf-sides">{side_html("You give", t["give"], t["vg"])}<div class="tf-arrow">for</div>
                {side_html("You get", t["get"], t["vr"])}</div>
              <div class="tf-foot"><span class="tf-g me">Your lineup {t['gm']:+.0f} pts</span>
                <span class="tf-g them">Their lineup {t['gt']:+.0f} pts</span></div>
            </div>""")
            st.button("Open in trade analyzer", key=f"tf_{k}", on_click=use_trade,
                      args=([label_by_id[p_] for p_ in t["give"]], [label_by_id[p_] for p_ in t["get"]]),
                      width="stretch")
            st.write("")
    st.caption("Fair = both sides within your fairness setting by market value. In 2-for-1 deals the second piece "
               "counts for 70% of its value. Picks under 400 value are left out to keep ideas realistic.")


# ---------- tabs ----------

def view_projections():
    intro("Weekly projections", "Projected points for every player. Floor is a bad day, ceiling is a good one.", "Pick a week and positions, or search a name. Bars show each player's floor-to-ceiling range.")
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
    rk = ros_values(rw0, rw1, rk_n, scoring_name, DB_VERSION)
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
    intro("My Team", "Your best lineup this week, close calls, and pickups worth making.", "Import your roster from Sleeper or add players by hand. Bookmark the page and your team is saved.")
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
    intro("Start / Sit", "Compare 2 to 4 players for a lineup decision.", "Pick the week and players. The card marked Start is the pick. Close calls get a tiebreaker tip.")
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
    intro("Breakouts", "Players getting more targets and carries lately. Usage tends to show up in the box score a week or two later.", "Look for the Usage spike and Due for points tags. Tap a row for the full profile.")
    c1, c2 = st.columns([3, 2])
    with c1:
        bpos = position_filter("bo_pos")
    b = breakouts[breakouts["position"].isin(bpos)]
    if league() and c2.toggle(f"Only available in {league()['name']}", key="bo_avail"):
        b = b[~b["player_id"].isin(league()["rostered"])]
    b = b.head(25).copy()
    use = breakout_usage(DB_VERSION)
    b["usage"] = [f"{use[p_][0]:.1f} → {use[p_][1]:.1f}" if p_ in use else "" for p_ in b["player_id"]]
    b["signal"] = b.apply(lambda r: " · ".join(
        s for s, ok in [("Usage spike", r["usage_change"] >= 3), ("Due for points", r["points_gap"] >= 3)] if ok), axis=1)
    sel = st.dataframe(
        b[["headshot_url", "name", "position", "team", "usage", "xfp_before", "xfp_last3", "ppr_last3",
           "signal", "breakout_score"]].style.map(lambda p: POS_STYLE.get(p, ""), subset=["position"]),
        hide_index=True, width="stretch", height=480, on_select="rerun", selection_mode="single-row", key="bo_table",
        column_config={
            "headshot_url": st.column_config.ImageColumn("", width="small"),
            "name": "Player", "position": "Pos", "team": "Team",
            "usage": st.column_config.TextColumn("Usage / game", help="Earlier this season → last 3 games. "
                                                 "QBs: pass attempts + runs. Others: targets + carries."),
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
    lg = league()
    dyn = bool(lg and lg.get("dynasty"))
    intro("Trade analyzer",
          "Judges a trade by FantasyCalc dynasty market value, including draft picks, plus what it does to your "
          "lineup this season." if dyn else
          "Judges a trade by value above replacement, so a 2-for-1 doesn't win just by adding bodies.",
          "Add players or picks to each side. Copy the page link to share the exact trade.")
    weeks = sorted(proj["week"].unique())
    end_default = FANTASY_LAST_WEEK if FANTASY_LAST_WEEK in weeks else weeks[-1]
    c0, c00 = st.columns([3, 1])
    w0, w1 = c0.slider("Weeks to count", min_value=weeks[0], max_value=weeks[-1], value=(weeks[0], end_default))
    sizes = [8, 10, 12, 14]
    default_n = lg["teams"] if lg and lg["teams"] in sizes else 12
    teams_n = c00.selectbox("League size", sizes, index=sizes.index(default_n), key="league_size")
    init_from_url("give", "give")
    init_from_url("get", "get")
    opts = asset_labels if dyn else labels
    for k in ("give", "get"):
        st.session_state[k] = [x for x in st.session_state.get(k, []) if x in opts]
    c1, c2 = st.columns(2)
    give = c1.multiselect("You give", opts, key="give", placeholder="Players or picks you're sending" if dyn else
                          "Players you're sending")
    get = c2.multiselect("You get", opts, key="get", placeholder="Players or picks you're receiving" if dyn else
                         "Players you're receiving")
    sync_url("give", give)
    sync_url("get", get)
    vals = ros_values(w0, w1, teams_n, scoring_name, DB_VERSION).set_index("player_id")
    mv = league_market_values()

    def side(labs):
        rows_ = []
        for lab in labs:
            pid = id_by_label[lab]
            if pid.startswith("PICK_"):
                rows_.append({"Asset": lab, "Market": mv.get(pid, 0.0), "ROS points": 0.0, "ROS value": 0.0})
            else:
                v = vals.loc[pid]
                rows_.append({"Asset": lab, "Market": mv.get(pid, 0.0), "ROS points": v["ros"], "ROS value": v["value"]})
        return pd.DataFrame(rows_)

    if not (give and get):
        st.info("Add at least one player" + (" or pick" if dyn else "") + " to each side.")
        return
    g, r = side(give), side(get)
    mg, mr = package_value(g["Market"].tolist()), package_value(r["Market"].tolist())
    gv, rv = g["ROS value"].sum(), r["ROS value"].sum()
    use_market = dyn and max(mg, mr) > 0
    if use_market:
        diff_pct = (mr - mg) / max(mg, mr) * 100
        verdict, fn = (("Fair trade", st.info) if abs(diff_pct) < 8 else
                       ("You win this trade", st.success) if diff_pct > 0 else ("You lose this trade", st.error))
        fn(f"**{verdict}** · you give **{mg:,.0f}** and get **{mr:,.0f}** in dynasty market value ({diff_pct:+.0f}%)")
        lineup = rv - gv
        if gv == 0 and rv == 0:
            raw = r["ROS points"].sum() - g["ROS points"].sum()
            st.caption(f"This season: you {'gain' if raw >= 0 else 'lose'} {abs(raw):.1f} projected points from Weeks "
                       f"{w0}-{w1} (none of these players project as a starter). Picks add nothing this season.")
        else:
            st.caption(f"This season: you {'gain' if lineup >= 0 else 'lose'} {abs(lineup):.1f} points of value above "
                       f"replacement from Weeks {w0}-{w1}. Picks add nothing this season.")
    elif gv == 0 and rv == 0:
        # nobody involved is a startable player, so value above replacement is 0 on both sides; compare raw points
        pg, pr_ = g["ROS points"].sum(), r["ROS points"].sum()
        diff = pr_ - pg
        verdict, fn = (("Fair trade", st.info) if abs(diff) < max(5, 0.08 * max(pg, pr_, 1)) else
                       ("You win this trade", st.success) if diff > 0 else ("You lose this trade", st.error))
        fn(f"**{verdict}** · none of these players project as a starter in a {teams_n}-team league, so this "
           f"compares raw points: you {'gain' if diff >= 0 else 'lose'} {abs(diff):.1f} projected points "
           f"(Weeks {w0}-{w1})")
    else:
        diff = rv - gv
        verdict, fn = (("Fair trade", st.info) if abs(diff) < max(5, 0.08 * max(gv, rv, 1)) else
                       ("You win this trade", st.success) if diff > 0 else ("You lose this trade", st.error))
        if max(mg, mr) > 0:
            st.markdown(f'<div class="tf-count">Trade market (FantasyCalc): you give <b>{mg:,.0f}</b> · you get '
                        f'<b>{mr:,.0f}</b> ({(mr - mg) / max(mg, mr) * 100:+.0f}%)</div>', unsafe_allow_html=True)
        fn(f"**{verdict}** · you {'gain' if diff >= 0 else 'give up'} {abs(diff):.1f} points of value above "
           f"replacement (Weeks {w0}-{w1}, {teams_n}-team league)")
    cfg = {"Market": st.column_config.NumberColumn("Market value", format="%,.0f"),
           "ROS points": st.column_config.NumberColumn("ROS points", format="%.1f"),
           "ROS value": st.column_config.NumberColumn("ROS value", format="%.1f",
                                                      help="Rest-of-season points above a replacement-level starter")}
    if not max(mg, mr):
        g, r = g.drop(columns="Market"), r.drop(columns="Market")
    a, bcol = st.columns(2)
    a.metric("You give", f"{mg:,.0f}" if use_market else f"{gv:.1f} value")
    a.dataframe(g, hide_index=True, width="stretch", column_config=cfg)
    bcol.metric("You get", f"{mr:,.0f}" if use_market else f"{rv:.1f} value")
    bcol.dataframe(r, hide_index=True, width="stretch", column_config=cfg)
    if any(id_by_label[x].startswith("DST_") or players.set_index("player_id").get("position", pd.Series()).get(
            id_by_label[x]) == "K" for x in give + get):
        st.caption("FantasyCalc doesn't put a market value on kickers or defenses, so they count as 0 there.")
    st.caption("In multi-asset deals the extra pieces count for 70% of their value, since one great player is worth "
               "more than two good ones. " + ("Next year's picks are valued as early, mid, or late from the original "
                                               "team's current record." if dyn else ""))


def view_accuracy():
    intro("Accuracy", "Projections compared with what actually happened, week by week.", "Pick a week to see the best calls and the biggest misses.")
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
    intro("How the model works", "Where the data comes from, how the model works, and how it was tested.", "Read top to bottom.")
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
names = ["This Week", "My Team", "League", "Rankings", "Players", "Trades", "How It Works"]
st.session_state.setdefault("nav", default_tab)
st.session_state["_nested"] = False
with nav_box:
    section = st.radio("Menu", names, key="nav", label_visibility="collapsed")
tabs = {n: st.container() if n == section else None for n in names}

if section == "This Week":
    live = (proj["week"] == next_week) & (proj["status"] == "Final")
    sub = (f"Week {next_week} in progress · {proj.loc[live, 'team'].nunique()} teams have played"
           if live.any() else f"Stats through Week {last_week}")
    section_header("This week", sub)
    render_home_summary()
    st.write("")
    st.session_state["_nested"] = True
    st.session_state.setdefault("tw_view", "Start / Sit" if "ss" in qp else "Projections")
    view = st.segmented_control("View", ["Projections", "Start / Sit"], required=True, key="tw_view",
                                label_visibility="collapsed")
    if view == "Projections":
        view_projections()
    else:
        view_startsit()
if section == "My Team":
    view_team()
if section == "League":
    section_header("League")
    st.session_state["_nested"] = True
    view_league()
if section == "Rankings":
    section_header("Rankings")
    st.session_state["_nested"] = True
    st.session_state.setdefault("rk_view", "Rest of season")
    view = st.segmented_control("View", ["Rest of season", "Breakouts"], required=True, key="rk_view",
                                label_visibility="collapsed")
    if view == "Rest of season":
        view_rankings()
    else:
        view_breakouts()
if section == "Players":
    view_players()
if section == "Trades":
    section_header("Trades")
    st.session_state["_nested"] = True
    st.session_state.setdefault("tr_view", "Trade analyzer")
    view = st.segmented_control("View", ["Trade analyzer", "Trade finder"], required=True, key="tr_view",
                                label_visibility="collapsed")
    if view == "Trade analyzer":
        view_trades()
    else:
        view_trade_finder()
if section == "How It Works":
    section_header("How it works")
    st.session_state["_nested"] = True
    st.session_state.setdefault("hw_view", "Accuracy")
    view = st.segmented_control("View", ["Accuracy", "The model"], required=True, key="hw_view",
                                label_visibility="collapsed")
    if view == "Accuracy":
        view_accuracy()
    else:
        view_model()

updated = metrics.get("updated_at", "")
html(f"""<div class="foot"><div>Red Zone Lab · projections update Tue, Sat and Sun mornings{' · last updated ' + updated if updated else ''}</div>
<div>Data: <a href="https://github.com/nflverse" target="_blank">nflverse</a> · <a href="https://sleeper.com" target="_blank">Sleeper</a> ·
<a href="https://open-meteo.com" target="_blank">Open-Meteo</a> · <a href="https://github.com/NicholasMatthews22/fantasy-football-analytics" target="_blank">Source code</a></div></div>""")
