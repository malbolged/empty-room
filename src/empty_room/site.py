"""Render the case files as one static page: the haunted archive."""

import html
import json
from pathlib import Path

import pandas as pd

W, H, PAD = 640, 180, 28


def _path(xs: list[float], ys: list[float], lo: float, hi: float) -> str:
    span = (hi - lo) or 1.0
    pts = [f"{x:.1f},{H - PAD - (y - lo) / span * (H - 2 * PAD):.1f}" for x, y in zip(xs, ys)]
    return "M" + " L".join(pts)


def chart(df: pd.DataFrame, case: dict, margin_min: int = 90) -> str:
    start, end = pd.Timestamp(case["start"]), pd.Timestamp(case["end"])
    win = df[(df["date"] >= start - pd.Timedelta(minutes=margin_min)) & (df["date"] <= end + pd.Timedelta(minutes=margin_min))]
    if win.empty:
        return ""
    t0, t1 = win["date"].iloc[0], win["date"].iloc[-1]
    total = (t1 - t0).total_seconds() or 1.0

    def x(t):
        return PAD + (t - t0).total_seconds() / total * (W - 2 * PAD)

    xs = [x(t) for t in win["date"]]
    light = _path(xs, win["Light"].tolist(), 0, max(600.0, win["Light"].max()))
    co2 = _path(xs, win["CO2"].tolist(), 400, max(1200.0, win["CO2"].max()))
    pres = _path(xs, win["p_occupied"].tolist(), 0, 1)
    logged = _path(xs, win["Occupancy"].tolist(), 0, 1)
    x0, x1 = x(start), x(end)
    return f"""<svg viewBox="0 0 {W} {H}" role="img" aria-label="Sensors around the apparition">
  <rect x="{x0:.1f}" y="{PAD - 8}" width="{max(x1 - x0, 2):.1f}" height="{H - 2 * PAD + 16}" class="haunt"/>
  <path d="{logged}" class="logged"/>
  <path d="{light}" class="light"/>
  <path d="{co2}" class="co2"/>
  <path d="{pres}" class="pres"/>
  <text x="{PAD}" y="{H - 6}">{t0:%H:%M}</text>
  <text x="{W - PAD}" y="{H - 6}" text-anchor="end">{t1:%H:%M}</text>
</svg>"""


def _p(text: str) -> str:
    return "".join(f"<p>{html.escape(par.strip())}</p>" for par in text.split("\n") if par.strip())


def _audio(src: str | None, label: str) -> str:
    if not src:
        return ""
    return f'<figure class="audio"><figcaption>{label}</figcaption><audio controls preload="none" src="audio/{src}"></audio></figure>'


def _cost(case: dict) -> str:
    if "tokens" not in case:
        return ""
    return f"; {case['model_calls']} local model calls, {case['tokens']:,} tokens, {case['seconds'] // 60}m {case['seconds'] % 60}s, $0"


def render(df: pd.DataFrame, cases: list[dict], stats: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for c in cases:
        a = c["apparition"]
        audio = c.get("audio") or {}
        calls = "".join(
            f"<li><code>{html.escape(t['tool'])}({html.escape(json.dumps(t['args']))})</code></li>"
            for t in c.get("tool_calls", [])
        )
        items.append(f"""
<article class="case {c['verdict'].lower()}">
  <header>
    <h2>{pd.Timestamp(a['start']):%A %d %B %Y, %H:%M}</h2>
    <p class="meta">{a['minutes']} minutes marked empty · presence {a['mean_p']:.0%} (peak {a['peak_p']:.0%}) · light {a['light_mean']} lx · CO₂ {a['co2_start']}→{a['co2_end']} ppm</p>
  </header>
  {chart(df, a)}
  <section class="medium"><h3>The presence speaks</h3>{_p(c['testimony'])}{_audio(audio.get('testimony'), 'Hear it')}</section>
  <section class="skeptic"><h3>The Skeptic: <span class="stamp">{c['verdict']}</span></h3>
    <p class="cause">{html.escape(c['cause'])}</p>{_p(c['report'])}{_audio(audio.get('report'), 'Hear the verdict')}
    <details><summary>Evidence gathered ({len(c.get('tool_calls', []))} tool calls{_cost(c)})</summary><ul>{calls}</ul></details>
  </section>
</article>""")

    narrated = any(c.get("audio") for c in cases)
    spoken = ' Read aloud by <a href="https://elevenlabs.io/">ElevenLabs</a>.' if narrated else ""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Empty Room: case files</title>
<style>
:root {{ color-scheme: dark; --bg:#0d0b10; --fg:#e9e4d8; --dim:#9a93a6; --marigold:#f2a900; --ghost:#b9c7ff; --co2:#7fd1a6; --light:#f2a900; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:17px/1.6 Georgia, serif; }}
main {{ max-width:720px; margin:0 auto; padding:48px 20px 96px; }}
h1 {{ font-size:2.4rem; margin:0 0 .2em; letter-spacing:.02em; }}
.lede {{ color:var(--dim); }}
.stats {{ display:flex; gap:24px; flex-wrap:wrap; margin:28px 0 40px; font-family:ui-monospace,monospace; font-size:.85rem; color:var(--dim); }}
.stats b {{ display:block; color:var(--fg); font-size:1.4rem; }}
.case {{ border-top:1px solid #2a2530; padding:36px 0; }}
.case h2 {{ margin:0; font-size:1.35rem; }}
.meta {{ color:var(--dim); font-family:ui-monospace,monospace; font-size:.8rem; }}
svg {{ width:100%; height:auto; margin:12px 0; }}
svg path {{ fill:none; stroke-width:1.6; }}
svg text {{ fill:var(--dim); font:11px ui-monospace,monospace; }}
.haunt {{ fill:#b9c7ff22; }}
.light {{ stroke:var(--light); }} .co2 {{ stroke:var(--co2); }} .pres {{ stroke:var(--ghost); stroke-dasharray:4 3; }} .logged {{ stroke:#ffffff40; }}
.legend {{ font-family:ui-monospace,monospace; font-size:.75rem; color:var(--dim); }}
.legend span {{ margin-right:14px; }}
.medium p {{ font-style:italic; color:var(--ghost); }}
.stamp {{ font-family:ui-monospace,monospace; font-size:.8rem; padding:2px 8px; border:1px solid currentColor; border-radius:3px; }}
.debunked .stamp {{ color:var(--co2); }} .living .stamp {{ color:var(--ghost); }} .unexplained .stamp {{ color:var(--marigold); }}
.cause {{ font-weight:bold; }}
details {{ color:var(--dim); font-size:.85rem; }}
figure.audio {{ margin:8px 0; }} figcaption {{ font-size:.8rem; color:var(--dim); }}
footer {{ color:var(--dim); font-size:.8rem; margin-top:48px; }}
a {{ color:var(--marigold); }}
</style></head>
<body><main>
<h1>The Empty Room</h1>
<p class="lede">In February 2015 an office in Mons, Belgium wrote down its light, CO₂, temperature and humidity once a minute, while a camera took a picture each minute to record whether anyone was inside. A local TabPFN model learned what a person does to a room, then listened to the minutes the room was supposed to be empty. These are the times it heard someone anyway. A local Gemma speaks for each presence, then tries to explain it away.</p>
<div class="stats">
  <div><b>{stats['minutes']:,}</b>minutes logged</div>
  <div><b>{stats['empty_minutes']:,}</b>marked empty</div>
  <div><b>{stats['haunted_minutes']:,}</b>felt occupied anyway</div>
  <div><b>{stats['balanced_accuracy']:.1%}</b>presence model balanced accuracy</div>
</div>
<p class="legend"><span style="color:var(--light)">— light</span><span style="color:var(--co2)">— CO₂</span><span style="color:var(--ghost)">- - presence</span><span>— logged occupancy</span></p>
{''.join(items)}
<footer>Data: <a href="https://archive.ics.uci.edu/dataset/357/occupancy+detection">UCI Occupancy Detection</a> (Candanedo &amp; Feldheim, 2016, CC BY 4.0). Weather: <a href="https://open-meteo.com/">Open-Meteo</a> (CC BY 4.0). Presence model: TabPFN. Voices of the presence and the Skeptic: Gemma, running locally.{spoken}</footer>
</main></body></html>"""
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
