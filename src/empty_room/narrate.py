"""Read each case aloud with ElevenLabs: one voice for the presence, one for the Skeptic."""

import os
from pathlib import Path

import httpx
import sentry_sdk

TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
TTS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
MEDIUM_VOICE = os.environ.get("ELEVENLABS_MEDIUM_VOICE", "pFZP5JQG7iQjIQuC4Bku")
SKEPTIC_VOICE = os.environ.get("ELEVENLABS_SKEPTIC_VOICE", "JBFqnCBsd6RMkjVDRZzb")


def enabled() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def speak(text: str, voice: str, out: Path) -> Path:
    with sentry_sdk.start_span(op="http.client", name="ElevenLabs text-to-speech") as span:
        span.set_data("characters", len(text))
        resp = httpx.post(
            TTS_URL.format(voice=voice),
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            json={"text": text, "model_id": TTS_MODEL},
            timeout=120,
        )
        resp.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(resp.content)
    return out


def narrate_case(case_id: str, testimony: str, report: str, out_dir: Path) -> dict:
    return {
        "testimony": speak(testimony, MEDIUM_VOICE, out_dir / f"{case_id}-medium.mp3").name,
        "report": speak(report, SKEPTIC_VOICE, out_dir / f"{case_id}-skeptic.mp3").name,
    }
