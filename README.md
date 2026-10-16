# Fantasy Football Analytics

A full-PPR fantasy football platform built on NFL play-by-play data: weekly projections with floor/ceiling ranges, a start/sit tool, a breakout detector for the waiver wire, and a rest-of-season trade analyzer. The data refreshes automatically every week.

**Live app:** _add your Streamlit Cloud link here_

## What it does

| Tab | What you get |
|---|---|
| **At a glance** | Top projected player at each position, the best matchup, the top breakout candidate, and the highest-scoring offense for the upcoming week |
| **Projections** | Every QB, RB, WR, and TE projected for each remaining week, with a floor, ceiling, matchup grade, Vegas implied team total, and injury status. Click a row to open the player's profile |
| **Players** | Profile for any player: season stats and positional rank, game log, usage share trend, and projections for every remaining week |
| **Start / Sit** | Compare 2 to 4 players on side-by-side cards with team colors, matchup, injury status, and floor-to-ceiling range |
| **Breakouts** | Ranks players whose usage (targets + carries) is rising, or whose points haven't caught up to their usage yet |
| **Trade Analyzer** | Totals rest-of-season projected points for each side of a trade, accounting for remaining schedule and bye weeks |
| **Model** | Backtest results and what drives the projections |

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

## Tech stack

Python, pandas, SQLite, scikit-learn, Plotly, Streamlit, GitHub Actions

## Project structure

```
app.py                     Streamlit app
ffa/
  config.py                seasons, paths, settings
  data.py                  download + SQLite loading
  features.py              feature engineering
  model.py                 training, backtesting, projections
  breakouts.py             xFP model and breakout scoring
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
