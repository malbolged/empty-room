# The Empty Room

*A séance over a sensor log.*

**Case archive:** https://empty-room-6z2k.onrender.com/

In February 2015 an office in Mons, Belgium recorded its light, CO₂, temperature and humidity once a minute, while a camera took a picture each minute to record whether anyone was inside. That log ([UCI Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection)) has been a machine-learning benchmark for ten years. This project reads it as a ghost story.

1. **TabPFN learns what a person does to a room.** It sees thousands of labelled minutes and works out what a body looks like in the sensors: breath raising the CO₂, a hand on the light switch, warmth. Time of day is hidden from it, so it judges presence only from the physics.
2. **Then it listens to the empty minutes.** Each day is scored by a model that never saw that day. Minutes the log calls *empty* but the model calls *occupied* are presences nobody signed in for. Runs of them become **apparitions**.
3. **The Medium speaks for each one.** Gemma 4, running locally through LM Studio, writes the presence's testimony in the first person, using only the numbers in the evidence.
4. **The Skeptic tries to explain it away.** The same local Gemma becomes a tool-using agent. It checks the sun's position over Mons, pulls that day's cloud cover from the Open-Meteo archive, re-asks TabPFN with one clue taken away (*would it still look like a person without the light?*), searches the web through SerpApi, and returns a verdict: **DEBUNKED** (nobody was there), **LIVING** (somebody was, and the log forgot them) or **UNEXPLAINED**.
5. **Everything is read aloud and traced.** ElevenLabs voices the presence and the Skeptic. Sentry traces every agent run, model call and tool call, with latency and token counts.

The output is a single static page of case files: `site/index.html`.

## Run it

You need [uv](https://docs.astral.sh/uv/) and a local, OpenAI-compatible server running Gemma 4. It was built and tested with [LM Studio](https://lmstudio.ai) and Gemma 4 E4B on a laptop with a 4 GB GPU. It also runs on CPU alone, about three times slower:

```sh
# download Gemma 4 E4B (google/gemma-4-e4b) in LM Studio first
lms load google/gemma-4-e4b --gpu max --identifier gemma-4-e4b
lms server start

uv sync --extra model
cp .env.example .env        # add TABPFN_TOKEN and any optional keys
uv run empty-room investigate --top 7
```

[Ollama](https://ollama.com) should work too: set `LLM_BASE_URL=http://localhost:11434/v1` and `EMPTY_ROOM_MODEL` to your Gemma 4 tag.

Open `site/index.html` to read the cases.

| Flag | What it does |
| --- | --- |
| `--top N` | Investigate the strongest apparition of each of N days (default 5) |
| `--no-llm` | Only run TabPFN and list the apparitions |
| `--narrate` | Read each case aloud with ElevenLabs (needs `ELEVENLABS_API_KEY`) |
| `--rescore` | Ignore the cached TabPFN scores in `out/scores.csv` |

`uv run empty-room site` rebuilds the page from `out/cases.json` without calling any model, and `uv run empty-room narrate` voices the saved cases with ElevenLabs.

### Keys

| Variable | Needed for | |
| --- | --- | --- |
| `TABPFN_TOKEN` | TabPFN-3.5 weights | recommended, free for non-commercial use: accept the license at [platform.priorlabs.ai](https://platform.priorlabs.ai). Without it the open TabPFN v2 weights are used |
| `EMPTY_ROOM_MODEL` | Which local model plays both roles | default `gemma-4-e4b` |
| `LLM_BASE_URL` | Any OpenAI-compatible local server | default LM Studio, `http://localhost:1234/v1` |
| `SERPAPI_API_KEY` | The Skeptic's `web_search` tool | optional |
| `ELEVENLABS_API_KEY` | `--narrate` | optional |
| `SENTRY_DSN` | Agent tracing | optional |

Without the optional keys everything still runs offline, except the weather lookup.

## Tests

```sh
uv sync
uv run pytest
```

The tests replace TabPFN and Gemma with stubs, so they need no weights, GPU or network. GitHub Actions runs them on every push.

## Credits

- Data: Luis M. Candanedo and Véronique Feldheim, "Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models", *Energy and Buildings* 112 (2016). [UCI ML Repository](https://archive.ics.uci.edu/dataset/357/occupancy+detection), CC BY 4.0.
- Weather: [Open-Meteo](https://open-meteo.com/) historical archive (ERA5), CC BY 4.0.
- Presence model: [TabPFN](https://github.com/PriorLabs/TabPFN) by Prior Labs.
- Voices: [Gemma 4](https://ai.google.dev/gemma), run locally with [LM Studio](https://lmstudio.ai); narration by [ElevenLabs](https://elevenlabs.io).
- Sun position: NOAA's general solar position approximation.

Built for the [DEV Hacktoberfest Open-Source AI Challenge, Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26), 26–31 October 2026.

## License

MIT
