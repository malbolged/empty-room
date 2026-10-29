"""Teach TabPFN what a person looks like, then point it at the empty minutes.

Every minute gets an out-of-fold probability that someone is in the room: the
days are split into folds, and each fold is scored by a model whose context
only holds the *other* days. A minute the log calls empty but the model calls
occupied is a presence nobody signed in for. Runs of them are apparitions.
"""

import os
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import FEATURES

N_FOLDS = 4
# TabPFN learns in context: each fold model is "trained" by showing it this many
# labelled minutes. 1000 is TabPFN's default ceiling on CPU; raise it on a GPU.
CONTEXT_ROWS = int(os.environ.get("EMPTY_ROOM_CONTEXT", "1000"))
THRESHOLD = 0.5
MERGE_GAP_MIN = 3
MIN_MINUTES = 5


def model_version() -> str:
    """TabPFN-3.5 once its license is accepted (TABPFN_TOKEN); the open v2 weights otherwise."""
    return os.environ.get("EMPTY_ROOM_TABPFN") or ("3.5" if os.environ.get("TABPFN_TOKEN") else "2")


def default_model():
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # fail fast instead of waiting on a login page
    versions = {"2": ModelVersion.V2, "2.5": ModelVersion.V2_5, "3.5": ModelVersion.V3_5}
    return TabPFNClassifier.create_default_for_version(versions[model_version()])


@dataclass
class Apparition:
    id: str
    start: str
    end: str
    minutes: int
    peak_p: float
    mean_p: float
    fold: int
    light_mean: float
    co2_start: float
    co2_max: float
    co2_end: float
    temp_mean: float
    weekday: str

    def to_dict(self) -> dict:
        return asdict(self)


class Detector:
    def __init__(self, df: pd.DataFrame, model_factory: Callable = default_model, seed: int = 0):
        self.df = df
        self.model_factory = model_factory
        self.seed = seed
        days = sorted(df["day"].unique())
        self.fold_of_day = {day: i % N_FOLDS for i, day in enumerate(days)}
        self.df["fold"] = self.df["day"].map(self.fold_of_day)
        self.models: dict[int, object] = {}

    def _context(self, fold: int) -> pd.DataFrame:
        """A balanced sample of the other folds: half occupied, half empty."""
        rest = self.df[self.df["fold"] != fold]
        if rest.empty:  # a log shorter than N_FOLDS days has nothing to hold out
            rest = self.df
        half = CONTEXT_ROWS // 2
        parts = [
            g.sample(min(len(g), half), random_state=self.seed)
            for _, g in rest.groupby("Occupancy")
        ]
        return pd.concat(parts)

    def model_for(self, fold: int):
        if fold not in self.models:
            ctx = self._context(fold)
            model = self.model_factory()
            model.fit(ctx[FEATURES].to_numpy(), ctx["Occupancy"].to_numpy())
            self.models[fold] = model
        return self.models[fold]

    def score(self, progress: Callable[[int], None] | None = None) -> pd.DataFrame:
        self.df["p_occupied"] = np.nan
        for fold in range(N_FOLDS):
            mask = self.df["fold"] == fold
            model = self.model_for(fold)
            proba = model.predict_proba(self.df.loc[mask, FEATURES].to_numpy())
            self.df.loc[mask, "p_occupied"] = proba[:, 1]
            if progress:
                progress(fold)
        return self.df

    def presence_of(self, row: pd.Series, **overrides: float) -> float:
        """P(occupied) for one minute, optionally with some sensors replaced."""
        x = row[FEATURES].astype(float).copy()
        for name, value in overrides.items():
            x[name] = value
        model = self.model_for(int(row["fold"]))
        return float(model.predict_proba(x.to_numpy().reshape(1, -1))[0, 1])


def accuracy(df: pd.DataFrame) -> dict:
    pred = (df["p_occupied"] >= THRESHOLD).astype(int)
    truth = df["Occupancy"]
    tpr = ((pred == 1) & (truth == 1)).sum() / max((truth == 1).sum(), 1)
    tnr = ((pred == 0) & (truth == 0)).sum() / max((truth == 0).sum(), 1)
    return {
        "minutes": int(len(df)),
        "accuracy": round(float((pred == truth).mean()), 4),
        "balanced_accuracy": round(float((tpr + tnr) / 2), 4),
        "empty_minutes": int((truth == 0).sum()),
        "haunted_minutes": int(((pred == 1) & (truth == 0)).sum()),
    }


def find_apparitions(df: pd.DataFrame) -> list[Apparition]:
    """Group haunted minutes (empty by the log, occupied by the model) into events."""
    haunted = df[(df["Occupancy"] == 0) & (df["p_occupied"] >= THRESHOLD)]
    events: list[list[int]] = []
    last_time = None
    for idx, t in haunted["date"].items():
        if last_time is None or (t - last_time).total_seconds() > MERGE_GAP_MIN * 60:
            events.append([])
        events[-1].append(idx)
        last_time = t

    found = []
    for rows in events:
        run = df.loc[rows[0] : rows[-1]]
        if len(rows) < MIN_MINUTES:
            continue
        start = run["date"].iloc[0]
        found.append(
            Apparition(
                id=start.strftime("%Y%m%d-%H%M"),
                start=start.isoformat(sep=" "),
                end=run["date"].iloc[-1].isoformat(sep=" "),
                minutes=len(rows),
                peak_p=round(float(run["p_occupied"].max()), 3),
                mean_p=round(float(df.loc[rows, "p_occupied"].mean()), 3),
                fold=int(run["fold"].iloc[0]),
                light_mean=round(float(run["Light"].mean()), 1),
                co2_start=round(float(run["CO2"].iloc[0]), 1),
                co2_max=round(float(run["CO2"].max()), 1),
                co2_end=round(float(run["CO2"].iloc[-1]), 1),
                temp_mean=round(float(run["Temperature"].mean()), 2),
                weekday=start.strftime("%A"),
            )
        )
    found.sort(key=lambda a: a.minutes * a.mean_p, reverse=True)
    return found


def one_per_day(found: list[Apparition], n: int) -> list[Apparition]:
    """The strongest apparition of each day, so the cases are not one long afternoon."""
    seen, picked = set(), []
    for a in found:
        day = a.start[:10]
        if day not in seen:
            seen.add(day)
            picked.append(a)
    return picked[:n]
