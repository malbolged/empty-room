"""Down-to-earth evidence the Skeptic can gather about a moment in February 2015."""

import math
import os
from datetime import datetime, timedelta, timezone

import httpx

# The office the dataset was recorded in (UMONS, Mons, Belgium).
LAT, LON = 50.4542, 3.9523
LOCAL = timezone(timedelta(hours=1))  # CET; the log is in local winter time


def sun_position(when: datetime) -> dict:
    """Solar elevation and azimuth in degrees (NOAA approximation, good to ~0.5°)."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=LOCAL)
    utc = when.astimezone(timezone.utc)
    day = utc.timetuple().tm_yday
    hour = utc.hour + utc.minute / 60 + utc.second / 3600
    g = 2 * math.pi / 365 * (day - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    )
    true_solar_min = hour * 60 + eqtime + 4 * LON
    ha = math.radians(true_solar_min / 4 - 180)
    lat = math.radians(LAT)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    elevation = 90 - math.degrees(zen)
    az = math.degrees(
        math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat))
    )
    return {
        "local_time": when.strftime("%Y-%m-%d %H:%M"),
        "sun_elevation_deg": round(elevation, 1),
        "sun_azimuth_deg": round((az + 180) % 360, 1),
        "sun_is_up": elevation > -0.833,
    }


def historical_weather(date: str, hours: tuple[int, int] = (6, 20)) -> dict:
    """Hourly cloud cover, sunshine and temperature for Mons from the Open-Meteo archive."""
    resp = httpx.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": date,
            "end_date": date,
            "hourly": "cloud_cover,sunshine_duration,temperature_2m,shortwave_radiation",
            "timezone": "Europe/Brussels",
        },
        timeout=30,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    rows = []
    for i, t in enumerate(h["time"]):
        hour = int(t[11:13])
        if hours[0] <= hour <= hours[1]:
            rows.append(
                {
                    "hour": t[11:16],
                    "cloud_cover_pct": h["cloud_cover"][i],
                    "sunshine_min": round((h["sunshine_duration"][i] or 0) / 60),
                    "radiation_w_m2": h["shortwave_radiation"][i],
                    "temp_c": h["temperature_2m"][i],
                }
            )
    return {"place": "Mons, Belgium", "date": date, "source": "Open-Meteo archive (ERA5)", "hourly": rows}


def web_search(query: str, num: int = 5) -> dict:
    """Live Google results through SerpApi. Needs SERPAPI_API_KEY."""
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        return {"error": "web_search is unavailable: SERPAPI_API_KEY is not set"}
    resp = httpx.get(
        "https://serpapi.com/search.json",
        params={"engine": "google", "q": query, "num": num, "api_key": key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    results = [
        {"title": r.get("title"), "snippet": r.get("snippet"), "link": r.get("link")}
        for r in data.get("organic_results", [])[:num]
    ]
    return {"query": query, "results": results}
