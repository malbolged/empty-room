"""Fast tests: no TabPFN weights, no GPU, no network. A stub stands in for each model."""

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from empty_room import agents, evidence
from empty_room.data import FEATURES, add_features
from empty_room.detector import Detector, accuracy, find_apparitions


class LightModel:
    """Says 'occupied' whenever the light is on, like a very literal ghost hunter."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = (np.asarray(X)[:, FEATURES.index("Light")] > 200).astype(float)
        return np.column_stack([1 - p, p])


def room(minutes=240, lit=(60, 90), occupied=(150, 200)):
    t = pd.date_range("2015-02-07 09:00", periods=minutes, freq="min")
    df = pd.DataFrame(
        {
            "date": t,
            "Temperature": 21.0,
            "Humidity": 25.0,
            "Light": 0.0,
            "CO2": 450.0,
            "HumidityRatio": 0.004,
            "Occupancy": 0,
        }
    )
    df.loc[lit[0] : lit[1] - 1, "Light"] = 400.0
    df.loc[occupied[0] : occupied[1] - 1, ["Light", "Occupancy"]] = [450.0, 1]
    return add_features(df)


def test_features_are_deltas():
    df = room()
    assert df.loc[62, "dLight_5"] == 400.0
    assert df.loc[100, "dLight_5"] == 0.0
    assert not df[FEATURES].isna().any().any()


def test_light_with_nobody_logged_is_an_apparition():
    df = room()
    det = Detector(df, model_factory=LightModel)
    det.score()
    found = find_apparitions(df)
    assert len(found) == 1
    a = found[0]
    assert a.start == "2015-02-07 10:00:00"
    assert a.minutes == 30
    assert a.weekday == "Saturday"
    stats = accuracy(df)
    assert stats["haunted_minutes"] == 30


def test_short_flickers_are_ignored():
    df = room(lit=(60, 62))
    Detector(df, model_factory=LightModel).score()
    assert find_apparitions(df) == []


def test_counterfactual_without_light_clears_the_room():
    df = room()
    det = Detector(df, model_factory=LightModel)
    det.score()
    row = df.loc[70]
    assert det.presence_of(row) == 1.0
    assert det.presence_of(row, Light=0.0) == 0.0


def test_sun_over_mons():
    noon = evidence.sun_position(datetime(2015, 2, 7, 12, 50))
    assert 20 < noon["sun_elevation_deg"] < 28
    assert 170 < noon["sun_azimuth_deg"] < 190
    assert not evidence.sun_position(datetime(2015, 2, 7, 23, 0))["sun_is_up"]


def test_parse_verdict():
    text = "VERDICT: **DEBUNKED**\nCAUSE: Low winter sun.\nREPORT: The light rose with the sun.\nIt fell at dusk."
    assert agents.parse_verdict(text) == ("DEBUNKED", "Low winter sun.", "The light rose with the sun.\nIt fell at dusk.")
    assert agents.parse_verdict("VERDICT: Living\nCAUSE: A cleaner.\nREPORT: Lamp on before dawn.")[0] == "LIVING"
    assert agents.parse_verdict("I am not sure.")[0] == "UNEXPLAINED"


class ScriptedLLM:
    """Plays back chat completions: first a tool call, then a verdict."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


def message(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    msg.model_dump = lambda exclude_none=True: {"role": "assistant", "content": content or "", "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
        for c in tool_calls or []
    ]}
    return msg


def call(name, args):
    return SimpleNamespace(id=f"call_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def test_skeptic_uses_tools_then_rules(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    df = room()
    det = Detector(df, model_factory=LightModel)
    det.score()
    a = find_apparitions(df)[0]
    llm = ScriptedLLM(
        [
            message(tool_calls=[call("counterfactual", {"clue": "light"}), call("sun_position", {"time": "2015-02-07 10:15"})]),
            message("VERDICT: DEBUNKED\nCAUSE: Morning sun on the sensor.\nREPORT: Without light the model sees nobody."),
        ]
    )
    v = agents.skeptic(llm, df, det, a)
    assert v.verdict == "DEBUNKED"
    assert [c["tool"] for c in v.tool_calls] == ["counterfactual", "sun_position"]
    cf = v.tool_calls[0]["result"]
    assert cf["p_presence_as_logged"] == 1.0 and cf["p_presence_without_light"] == 0.0
    assert "p_presence_without_breath" not in cf
    assert v.tool_calls[1]["result"]["sun_is_up"]
    tool_names = {t["function"]["name"] for t in llm.seen[0]["tools"]}
    assert "web_search" not in tool_names
    assert "web_search" not in llm.seen[0]["messages"][0]["content"]


def test_search_hint_only_with_serpapi():
    assert "web_search" in agents.skeptic_system(True)
    assert agents.skeptic_system(True).index("web_search") < agents.skeptic_system(True).index("VERDICT")


def test_counterfactual_all_tries_every_clue():
    df = room()
    det = Detector(df, model_factory=LightModel)
    det.score()
    a = find_apparitions(df)[0]
    tools = {t.name: t for t in agents.skeptic_tools(df, det, a)}
    cf = agents.run_tool(tools["counterfactual"], {})
    assert cf["p_presence_without_light"] == 0.0
    assert cf["p_presence_without_breath"] == 1.0
    assert cf["p_presence_without_warmth"] == 1.0
    assert cf["clues_that_matter"] == ["light"]


def test_bad_tool_arguments_come_back_as_errors():
    df = room()
    det = Detector(df, model_factory=LightModel)
    det.score()
    a = find_apparitions(df)[0]
    tools = {t.name: t for t in agents.skeptic_tools(df, det, a)}
    assert "error" in agents.run_tool(tools["counterfactual"], {"clue": "ectoplasm"})
    assert "error" in agents.run_tool(tools["counterfactual"], {"sensor": "Light"})
    assert "error" in agents.run_tool(tools["sun_position"], {"time": "midnight-ish"})


def test_window_summary_sees_the_switch():
    df = room()
    Detector(df, model_factory=LightModel).score()
    summary = agents.window_summary(df, find_apparitions(df)[0])
    assert summary["log_in_10_min_before"] == "empty"
    assert summary["light_switched_on_at_start"] is True
    assert summary["light_switched_off_after"] is True
    assert summary["co2_trend_during"] == "flat"


def test_sun_position_tolerates_decorated_times():
    df = room()
    det = Detector(df, model_factory=LightModel)
    det.score()
    tools = {t.name: t for t in agents.skeptic_tools(df, det, find_apparitions(df)[0])}
    result = agents.run_tool(tools["sun_position"], {"time": "2015-02-07 12:00 at Mons"})
    assert result["local_time"] == "2015-02-07 12:00"


@pytest.mark.parametrize("before,after", [(0, 0), (500, 500)])
def test_sensor_window_is_clamped(before, after):
    df = room()
    det = Detector(df, model_factory=LightModel)
    det.score()
    a = find_apparitions(df)[0]
    tools = {t.name: t for t in agents.skeptic_tools(df, det, a)}
    rows = agents.run_tool(tools["sensor_window"], {"minutes_before": before, "minutes_after": after})["readings_every_5_min"]
    assert 1 <= len(rows) <= (30 + 240) // 5 + 1
