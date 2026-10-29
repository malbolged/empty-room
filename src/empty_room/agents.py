"""Two voices from one local Gemma: the Medium, who believes, and the Skeptic, who checks.

Gemma runs on this machine behind an OpenAI-compatible endpoint (Ollama or
LM Studio), so the same OpenAI client, and Sentry's OpenAI integration, work
unchanged. The agent and tool spans around it are added by hand, following
Sentry's gen_ai conventions.
"""

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd
import sentry_sdk
from openai import OpenAI

from . import evidence
from .detector import Apparition, Detector

MODEL = os.environ.get("EMPTY_ROOM_MODEL", "gemma-4-e4b")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
MAX_STEPS = 8

# Running totals across every model call, for the case files and the write-up.
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def _count(resp) -> None:
    USAGE["calls"] += 1
    usage = getattr(resp, "usage", None)
    if usage:
        USAGE["input_tokens"] += usage.prompt_tokens or 0
        USAGE["output_tokens"] += usage.completion_tokens or 0


def client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key="local")


@contextmanager
def _span(op: str, name: str, attributes: dict):
    with sentry_sdk.start_span(op=op, name=name) as span:
        for key, value in attributes.items():
            span.set_data(key, value)
        yield span


def _agent_span(name: str, prompt: str):
    return _span(
        "gen_ai.invoke_agent",
        f"invoke_agent {name}",
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": name,
            "gen_ai.request.model": MODEL,
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "parts": [{"type": "text", "content": prompt}]}]
            ),
        },
    )


def _finish(span, text: str) -> None:
    span.set_data(
        "gen_ai.output.messages",
        json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": text}]}]),
    )


def evidence_rows(df: pd.DataFrame, a: Apparition, before: int = 0, after: int = 0, every: int = 5) -> list[dict]:
    start = pd.Timestamp(a.start) - pd.Timedelta(minutes=before)
    end = pd.Timestamp(a.end) + pd.Timedelta(minutes=after)
    win = df[(df["date"] >= start) & (df["date"] <= end)].iloc[::every]
    return [
        {
            "time": r.date.strftime("%H:%M"),
            "logged_occupied": bool(r.Occupancy),
            "p_presence": round(float(r.p_occupied), 2),
            "light_lux": round(float(r.Light), 1),
            "co2_ppm": round(float(r.CO2), 1),
            "temp_c": round(float(r.Temperature), 2),
            "humidity_pct": round(float(r.Humidity), 1),
        }
        for r in win.itertuples()
    ]


def _logged(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "no data"
    share = rows["Occupancy"].mean()
    return "occupied" if share == 1 else "empty" if share == 0 else "mixed"


def window_summary(df: pd.DataFrame, a: Apparition, margin_min: int = 10) -> dict:
    """The facts a small model tends to misread off a table, computed once."""
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    margin = pd.Timedelta(minutes=margin_min)
    before = df[(df["date"] >= start - margin) & (df["date"] < start)]
    during = df[(df["date"] >= start) & (df["date"] <= end)]
    after = df[(df["date"] > end) & (df["date"] <= end + margin)]
    light_before = float(before["Light"].mean()) if not before.empty else float("nan")
    light_after = float(after["Light"].mean()) if not after.empty else float("nan")
    co2_change = float(during["CO2"].iloc[-1] - during["CO2"].iloc[0])
    return {
        f"log_in_{margin_min}_min_before": _logged(before),
        f"log_in_{margin_min}_min_after": _logged(after),
        "light_before_lux": round(light_before, 1),
        "light_during_lux": round(float(during["Light"].mean()), 1),
        "light_after_lux": round(light_after, 1),
        "light_switched_on_at_start": bool(light_before < 50 and during["Light"].max() > 200),
        "light_switched_off_after": bool(light_after < 50 and during["Light"].mean() > 200),
        "co2_change_during_ppm": round(co2_change, 1),
        "co2_trend_during": "rising" if co2_change > 10 else "falling" if co2_change < -10 else "flat",
    }


# --- The Medium ------------------------------------------------------------

MEDIUM_SYSTEM = """You are the Medium in a séance held over a sensor log.
An office in Mons, Belgium, recorded itself every minute in February 2015.
A camera took a picture every minute to record who was in the room. For the
minutes below, the pictures show it EMPTY, yet a model trained on what people
do to a room says someone was there.

Speak AS that presence, in the first person, to whoever reads the log today.
Rules:
- 110 to 160 words, plain prose, no headings, no lists, no emoji.
- Every detail you mention must come from the evidence: the time, the day,
  how long, the light, the CO2, the temperature. Quote at least three numbers.
- Eerie and tender, not gory. You may wonder what you are; do not explain yourself away."""


def medium(llm: OpenAI, df: pd.DataFrame, a: Apparition) -> str:
    payload = {"apparition": a.to_dict(), "minutes": evidence_rows(df, a)}
    prompt = "Evidence:\n" + json.dumps(payload, indent=1)
    with _agent_span("The Medium", prompt) as span:
        resp = llm.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": MEDIUM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
        )
        _count(resp)
        text = (resp.choices[0].message.content or "").strip()
        _finish(span, text)
    return text


# --- The Skeptic -----------------------------------------------------------

SKEPTIC_SYSTEM = """You are the Skeptic in a haunted-data investigation.
An office in Mons, Belgium (50.45 N, 3.95 E) logged light, CO2, temperature and
humidity every minute in February 2015. A presence model (TabPFN) flagged the
apparition below: the camera's record says the room was empty, the model says it was not.

Find the most likely explanation, using your tools. Work through these:
- Which clue convinced the model? Use counterfactual to take each one away.
- If it was the light: was the sun up? Check sun_position and historical_weather.
  A lamp does not switch itself on. If sensor_window says the light was
  switched on at the start and the sun was down, a person switched it on:
  that is LIVING, even if the camera missed them.
- Use sensor_window to see what the log said just before and after. If the
  room was logged occupied on either side and the readings carry on unchanged
  (CO2 still rising, the same light), the log probably forgot a person.
- If the light simply stayed on while CO2 fell steadily, the person had left
  and only their trace remained.
Call tools before you conclude. Never invent readings.

Then choose one verdict:
- DEBUNKED: nobody was there. Sunlight, a lamp left on, or a lingering breath fooled the model.
- LIVING: somebody really was there, and the log forgot them.
- UNEXPLAINED: the evidence fits neither.

Answer in exactly this format:
VERDICT: DEBUNKED, LIVING or UNEXPLAINED
CAUSE: one short sentence
REPORT: 80 to 140 words of plain prose citing the evidence you gathered,
naming the tool that gave you each fact. No markdown, no LaTeX."""

# Added only when SerpApi is configured, so the model is never told about a tool it lacks.
SEARCH_HINT = """
- Was it an ordinary working day? Use web_search once, e.g. for Belgian public or
  school holidays on that date, or for anything unusual in Mons that day."""


def skeptic_system(can_search: bool) -> str:
    if not can_search:
        return SKEPTIC_SYSTEM
    rules_end = "  and only their trace remained."
    return SKEPTIC_SYSTEM.replace(rules_end, rules_end + SEARCH_HINT)

# The three things a body does to a room, as groups of features.
CLUES = {
    "light": ["Light", "dLight_5"],
    "breath": ["CO2", "dCO2_5", "dCO2_15"],
    "warmth": ["Temperature", "dTemp_15"],
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., dict]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def skeptic_tools(df: pd.DataFrame, det: Detector, a: Apparition) -> list[Tool]:
    peak = df[(df["date"] >= pd.Timestamp(a.start)) & (df["date"] <= pd.Timestamp(a.end))]
    peak_row = peak.loc[peak["p_occupied"].idxmax()]
    # What the room looks like when it is truly empty: dark, unoccupied minutes.
    still = df[(df["Occupancy"] == 0) & (df["Light"] == 0)].median(numeric_only=True)

    def sensor_window(minutes_before: int = 30, minutes_after: int = 30) -> dict:
        before = max(0, min(int(minutes_before), 120))
        after = max(0, min(int(minutes_after), 120))
        return {
            "summary": window_summary(df, a),
            "readings_every_5_min": evidence_rows(df, a, before, after),
        }

    def sun_position(time: str) -> dict:
        # Small models decorate arguments ("2015-02-07 12:00 at Mons"); keep the timestamp.
        stamp = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", time)
        if not stamp:
            return {"error": "time must look like YYYY-MM-DD HH:MM"}
        return evidence.sun_position(datetime.fromisoformat(stamp.group(0)))

    def historical_weather(date: str) -> dict:
        day = re.search(r"\d{4}-\d{2}-\d{2}", date)
        if not day:
            return {"error": "date must look like YYYY-MM-DD"}
        return evidence.historical_weather(day.group(0))

    def counterfactual(clue: str = "all") -> dict:
        if clue != "all" and clue not in CLUES:
            return {"error": f"clue must be 'all' or one of {sorted(CLUES)}"}
        result = {
            "minute": peak_row["date"].strftime("%Y-%m-%d %H:%M"),
            "p_presence_as_logged": round(det.presence_of(peak_row), 3),
        }
        for name in CLUES if clue == "all" else [clue]:
            calm = {f: round(float(still[f]), 2) for f in CLUES[name]}
            result[f"p_presence_without_{name}"] = round(det.presence_of(peak_row, **calm), 3)
            result[f"{name}_set_to"] = calm
        drops = {
            name: result["p_presence_as_logged"] - result[f"p_presence_without_{name}"]
            for name in CLUES
            if f"p_presence_without_{name}" in result
        }
        decisive = [name for name, drop in drops.items() if drop >= 0.25]
        result["clues_that_matter"] = decisive or "none: no single clue changes the verdict"
        return result

    tools = [
        Tool(
            "sensor_window",
            "A summary of what changed around the apparition (what the log said before and after, "
            "whether the light was switched on or off, the CO2 trend), plus readings every 5 minutes.",
            {
                "type": "object",
                "properties": {
                    "minutes_before": {"type": "integer", "description": "0-120"},
                    "minutes_after": {"type": "integer", "description": "0-120"},
                },
            },
            sensor_window,
        ),
        Tool(
            "sun_position",
            "Sun elevation and azimuth over the office at a local time.",
            {
                "type": "object",
                "properties": {"time": {"type": "string", "description": "YYYY-MM-DD HH:MM local time"}},
                "required": ["time"],
            },
            sun_position,
        ),
        Tool(
            "historical_weather",
            "Hourly cloud cover, sunshine minutes and solar radiation in Mons for one day.",
            {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                "required": ["date"],
            },
            historical_weather,
        ),
        Tool(
            "counterfactual",
            "Ask the presence model again at the apparition's strongest minute with one clue "
            "(light, breath = CO2, warmth = temperature) reset to how the empty room looks at night. "
            "A big drop means that clue is what looked like a person. 'all' tries each clue in turn.",
            {
                "type": "object",
                "properties": {"clue": {"type": "string", "enum": ["all", *sorted(CLUES)]}},
            },
            counterfactual,
        ),
    ]
    if os.environ.get("SERPAPI_API_KEY"):
        tools.append(
            Tool(
                "web_search",
                "Search the live web, e.g. for events in Mons or Belgium on that date.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                lambda query: evidence.web_search(query),
            )
        )
    return tools


def run_tool(tool: Tool, args: dict) -> dict:
    with _span(
        "gen_ai.execute_tool",
        f"execute_tool {tool.name}",
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.agent.name": "The Skeptic",
            "gen_ai.tool.name": tool.name,
            "gen_ai.tool.description": tool.description,
            "gen_ai.tool.call.arguments": json.dumps(args),
        },
    ) as span:
        try:
            result = tool.run(**args)
        except Exception as exc:  # the agent sees the error and can try something else
            span.set_status("internal_error")
            span.set_data("error.type", type(exc).__name__)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        span.set_data("gen_ai.tool.call.result", json.dumps(result, default=str)[:4000])
    return result


@dataclass
class Verdict:
    verdict: str
    cause: str
    report: str
    tool_calls: list[dict] = field(default_factory=list)


def parse_verdict(text: str) -> tuple[str, str, str]:
    verdict = re.search(r"VERDICT:\s*\**\s*(DEBUNKED|LIVING|UNEXPLAINED)", text, re.I)
    cause = re.search(r"CAUSE:\s*(.+)", text)
    report = re.search(r"REPORT:\s*(.+)", text, re.S)
    return (
        verdict.group(1).upper() if verdict else "UNEXPLAINED",
        cause.group(1).strip().strip("*") if cause else "",
        report.group(1).strip() if report else text.strip(),
    )


def skeptic(llm: OpenAI, df: pd.DataFrame, det: Detector, a: Apparition) -> Verdict:
    tools = skeptic_tools(df, det, a)
    by_name = {t.name: t for t in tools}
    prompt = "Apparition:\n" + json.dumps(a.to_dict(), indent=1)
    messages: list[dict] = [
        {"role": "system", "content": skeptic_system("web_search" in by_name)},
        {"role": "user", "content": prompt},
    ]
    calls: list[dict] = []
    text = ""
    with _agent_span("The Skeptic", prompt) as span:
        for step in range(MAX_STEPS):
            last = step == MAX_STEPS - 1
            resp = llm.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=None if last else [t.schema() for t in tools],
                temperature=0,
            )
            _count(resp)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                text = msg.content or ""
                if "VERDICT" in text.upper() or last:
                    break
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "Now give your answer in the VERDICT/CAUSE/REPORT format."})
                continue
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool = by_name.get(name)
                result = run_tool(tool, args) if tool else {"error": f"no tool named {name}"}
                calls.append({"tool": name, "args": args, "result": result})
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)}
                )
        _finish(span, text)
    verdict, cause, report = parse_verdict(text)
    return Verdict(verdict, cause, report, calls)
