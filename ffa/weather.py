"""Game-day weather forecasts for upcoming outdoor games (Open-Meteo, free, no API key).

The model doesn't use weather (it didn't improve the backtest overall), but strong wind
is a real red flag for passing games: in 2025-26 backtests, QBs playing in 15+ mph wind
scored about 3 points below their projection on average. So the app shows the forecast.
"""
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

# Home stadium coordinates (lat, lon). Shared stadiums use the same point.
STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LV": (36.0909, -115.1833), "LAC": (33.9535, -118.3392),
    "LA": (33.9535, -118.3392), "MIA": (25.9580, -80.2389), "MIN": (44.9737, -93.2577),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SF": (37.4030, -121.9700), "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
}
ET = ZoneInfo("America/New_York")
URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly=temperature_2m,wind_speed_10m,precipitation_probability"
       "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=UTC&forecast_days=10")


def _forecast(lat, lon, kickoff_utc):
    with urllib.request.urlopen(URL.format(lat=lat, lon=lon), timeout=15) as r:
        h = json.load(r)["hourly"]
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in h["time"]]
    i = min(range(len(times)), key=lambda k: abs(times[k] - kickoff_utc))
    return h["temperature_2m"][i], h["wind_speed_10m"][i], h["precipitation_probability"][i]


def game_weather(raw_games, season):
    """One row per upcoming game: roof type plus forecast for outdoor games in the next 9 days."""
    g = raw_games[(raw_games["season"] == season) & raw_games["home_score"].isna()].copy()
    now = datetime.now(timezone.utc)
    rows = []
    for _, r in g.iterrows():
        row = {"game_id": r["game_id"], "roof": r["roof"] if isinstance(r["roof"], str) else "unknown",
               "temp": None, "wind": None, "precip": None}
        try:
            kick = datetime.strptime(f"{r['gameday']} {r['gametime']}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        except (TypeError, ValueError):
            kick = None
        outdoor = row["roof"] in ("outdoors", "open")
        same_stadium = r["location"] == "Home" and r["home_team"] in STADIUMS
        if outdoor and same_stadium and kick and now < kick < now + timedelta(days=9):
            try:
                lat, lon = STADIUMS[r["home_team"]]
                row["temp"], row["wind"], row["precip"] = _forecast(lat, lon, kick.astimezone(timezone.utc))
            except Exception as e:  # forecast is a nice-to-have; never fail the pipeline over it
                print(f"  no forecast for {r['game_id']} ({e})")
        rows.append(row)
    return pd.DataFrame(rows)
