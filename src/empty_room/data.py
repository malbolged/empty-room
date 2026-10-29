"""Load the room's sensor log and derive per-minute features.

The data is the UCI Occupancy Detection dataset (Candanedo & Feldheim, 2016,
CC BY 4.0): one office in Mons, Belgium, 2–18 February 2015, one row per minute,
with a ground-truth label for whether anyone was in the room.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "occupancy"
FILES = ("datatest.txt", "datatraining.txt", "datatest2.txt")

SENSORS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

# What a body changes in a room: it breathes (CO2 climbs), it switches things on
# (Light jumps), it warms the air. Time of day is left out on purpose, so the
# model judges presence from physics rather than from office hours.
FEATURES = SENSORS + ["dCO2_5", "dCO2_15", "dLight_5", "dTemp_15"]


def load(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = [pd.read_csv(data_dir / name) for name in FILES]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dCO2_5"] = df["CO2"].diff(5)
    df["dCO2_15"] = df["CO2"].diff(15)
    df["dLight_5"] = df["Light"].diff(5)
    df["dTemp_15"] = df["Temperature"].diff(15)
    df[FEATURES] = df[FEATURES].fillna(0.0)
    df["day"] = df["date"].dt.date
    return df
