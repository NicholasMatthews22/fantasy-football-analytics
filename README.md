# Red Zone Lab: Fantasy Football Analytics

A full-PPR fantasy football platform built on NFL play-by-play data: weekly projections with floor/ceiling ranges, a start/sit tool, a breakout detector for the waiver wire, and a rest-of-season trade analyzer. The data refreshes automatically every week.

**Live app:** _add your Streamlit Cloud link here_

## What it does

Seven sections, each with one job: **This Week** (projections and start/sit), **My Team** (lineup and matchup), **League** (power rankings and playoff odds), **Rankings** (rest of season and breakouts), **Players**, **Trades** (analyzer and finder), and **How It Works** (accuracy and the model). Covers QB, RB, WR, TE, K, and D/ST, in full PPR, half PPR, standard, or your Sleeper league's exact scoring.


| Tab | What you get |
|---|---|
| **At a glance** | Top projected player at each position, the best matchup, the top breakout candidate, and the highest-scoring offense for the upcoming week |
| **Rankings** | Rest-of-season big board ranked by value over a replacement-level starter, with league-size adjustment and a filter for players available in your Sleeper league |
| **Projections** | Every QB, RB, WR, and TE projected for each remaining week, with a floor, ceiling, matchup grade, Vegas implied team total, and injury status. Click a row to open the player's profile |
| **Players** | Profile for any player: season stats and positional rank, game log, usage share trend, and projections for every remaining week |
| **My Team** | Import your roster from Sleeper (or add players by hand) and save it in the page link. Shows get your optimal lineup (QB, 2 RB, 2 WR, TE, FLEX), close calls, bye/injury/wind alerts, and the best players actually available on your league's waiver wire |
| **Start / Sit** | Compare 2 to 4 players on side-by-side cards with team colors, matchup, injury status, and floor-to-ceiling range |
| **Breakouts** | Ranks players whose usage (targets + carries) is rising, or whose points haven't caught up to their usage yet |
| **Power rankings + playoff odds** | Simulates the rest of your Sleeper league's season 5,000 times (real schedule, current records, every team's best projected lineup, your playoff format) for each team's playoff, bye, and title odds |
| **Matchup preview** | Pulls this week's opponent from Sleeper, shows both starting lineups, and gives a win probability plus lineup tips that would raise it |
| **Trade finder** | Scans every roster in your Sleeper league for 1-for-1 and 2-for-1 trades (players and dynasty draft picks) that are fair by FantasyCalc market value (dynasty or redraft, superflex-aware) and improve both teams' starting lineups |
| **Trade Analyzer** | Values each side by rest-of-season points above a replacement-level starter (adjusts for league size), so 2-for-1 deals are judged fairly |
| **Accuracy** | Projections are saved before kickoff every week and scored against real results: weekly average miss vs. baseline, best calls, biggest misses |
| **Model** | Backtest results, what drives the projections, and feature experiments that didn't make the cut |

Comparisons, trades, and your roster are all saved in the page URL, so any view can be bookmarked or shared.

## Results

Backtested on games the model never trained on, measured on fantasy-relevant players:

| Test | Model avg miss | Baseline avg miss | Improvement | Floor-to-ceiling hit rate |
|---|---|---|---|---|
| 2025 season (3,129 player-games) | 5.78 pts | 5.98 pts | 3.3% | 69% |
| 2026 season through Week 6 (928 player-games) | 5.90 pts | 6.13 pts | 3.8% | 68% |

The baseline is each player's recent weighted scoring average. The floor and ceiling are the 15th and 85th percentile outcomes, so roughly 70% of results should land between them, which is what the backtest shows.

## How it works

```
nflverse (GitHub)  ->  SQLite database  ->  feature engineering  ->  model  ->  Streamlit app
   weekly stats         player_weeks          pre-game features      GBM        projections
   schedules + lines    games                                                    breakouts
```

**Data.** Weekly stats for every offensive skill player since 2021, the full schedule with betting lines, weekly injury reports, and team colors/logos, from [nflverse](https://github.com/nflverse). Stored in SQLite (`data/ff.db`) with tables for player games, games, injuries, teams, projections, and breakouts.

**Injuries.** When the upcoming week's injury report is out, players ruled Out are projected for 0 and Doubtful/Questionable players are tagged.

**Features.** Every feature uses only information available before kickoff:
- Player form: exponentially weighted averages of points, targets, carries, target share, carry share, air yards share, and WOPR, carried across seasons so early-season projections still have history
- Opponent: PPR points the defense has allowed to that position, relative to league average
- Game environment: Vegas implied team total, point spread, home/away

**Model.** Gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`). Floor and ceiling ranges are calibrated per position from how actual results spread around projections in a held-out season.

**Breakout detector.** Fits an expected fantasy points (xFP) model from opportunity alone (targets, carries, air yards, pass attempts), then scores each player on how much their xFP rose over the last 3 games and how far their actual points trail their xFP.

**Trade analyzer.** Projects every remaining game using each player's current form against that week's opponent and betting line, then sums across the selected weeks.

**Sleeper import.** Uses Sleeper's free public API (no login): username, then league, then roster. Sleeper player IDs are mapped to NFL IDs with the DynastyProcess ID database. Every rostered player in the league is marked taken, so waiver suggestions only show real free agents.

**Draft picks (dynasty).** Pick ownership comes from Sleeper, including traded picks. Values come from FantasyCalc's dynasty market; next year's picks are valued as early, mid, or late from the original team's current record, and rounds FantasyCalc doesn't list are extended from the drop-off between earlier rounds.

**Kickers and defenses.** Scored with Sleeper defaults. A small regression on recent form plus Vegas context (team implied total for kickers, opponent implied total for defenses) beat the recent-average baseline in a 2025 backtest: kickers 3.84 vs 3.98 average miss, defenses 4.34 vs 4.61.

**League scoring.** Projections are built in full PPR, then converted using each player's recent stat mix (receptions, yards, TDs, interceptions), so half PPR, TE premium, or 6-point passing TDs all re-rank players correctly.

**Season simulation.** For each remaining regular-season week, every team's optimal lineup gives a projected score and spread; matchups come from Sleeper's schedule. Each simulation plays out the weeks, seeds the playoffs by wins then points for, and runs the bracket (top seeds get byes when needed).

**Win probability.** Each starter's floor-to-ceiling range is treated as a bell curve; the lineup totals are compared to get the chance of outscoring the opponent.

**Weather.** Forecasts for upcoming outdoor games come from Open-Meteo using each stadium's coordinates. Weather isn't a model feature (it didn't improve the backtest overall), but QBs in 15+ mph wind scored about 3 points below projection, so windy games get flagged.

**Accuracy tracking.** Every pipeline run saves the upcoming week's projections. The last save before each game is the one that gets scored. Weeks before live tracking began are backfilled walk-forward: the model is retrained before each week using only earlier games.

**Experiments.** Snap share, red zone usage, usage trends, weather, stadium factors, and separate per-position models were all tested. None lowered the average miss in both backtests, so the simpler single model was kept. Results are listed in the app's Model tab.

## Tech stack

Python, pandas, SQLite, scikit-learn, Plotly, Streamlit, GitHub Actions, Sleeper API, FantasyCalc API, Open-Meteo API

## Project structure

```
app.py                     Streamlit app
ffa/
  config.py                seasons, paths, settings
  data.py                  download + SQLite loading
  features.py              feature engineering
  model.py                 training, backtesting, projections
  breakouts.py             xFP model and breakout scoring
  sleeper.py               Sleeper league import
  fantasycalc.py           trade market values
  season_sim.py            Monte Carlo season and playoff simulation
  kdst.py                  kicker and defense projections
  weather.py               game-day forecasts
scripts/update.py          runs the full pipeline
data/ff.db                 database the app reads
data/metrics.json          backtest results
.github/workflows/         weekly auto-update
```

## Run it locally

```bash
pip install -r requirements.txt
python scripts/update.py      # downloads the latest data, retrains, writes projections (~1 min)
streamlit run app.py
```

## Weekly updates

A GitHub Actions workflow runs `scripts/update.py` every Tuesday morning after Monday Night Football (new stats and projections), plus Saturday and Sunday mornings (final injury reports), and commits the refreshed database, so the deployed app always shows the next week. You can also trigger it by hand from the Actions tab.

## Limitations

- Injury statuses only reach the app on the scheduled refreshes, so a Sunday-morning scratch can be missed.
- Player-games where a player recorded no stats aren't in the source data.
- Trade values are raw points. They don't account for positional scarcity or your roster needs.
