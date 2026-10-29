"""The Empty Room: a séance over a sensor log."""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import sentry_sdk
from dotenv import load_dotenv
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
SITE = ROOT / "site"

console = Console()


def init_tracing() -> None:
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return
    from sentry_sdk.integrations.openai import OpenAIIntegration

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=1.0,
        send_default_pii=True,  # lets the AI spans carry prompts and tool arguments
        integrations=[OpenAIIntegration(include_prompts=True)],
        environment=os.environ.get("SENTRY_ENVIRONMENT", "local"),
    )
    console.print("[dim]Sentry tracing on[/]")


def scores_file(version: str) -> Path:
    return OUT / f"scores-tabpfn{version}.csv"


def score(detector, rescore: bool) -> pd.DataFrame:
    from .detector import model_version

    cache = scores_file(model_version())
    df = detector.df
    if cache.exists() and not rescore:
        cached = pd.read_csv(cache, parse_dates=["date"])
        df["p_occupied"] = df["date"].map(cached.set_index("date")["p_occupied"])
        console.print(f"[dim]Presence scores loaded from {cache.relative_to(ROOT)}[/]")
        return df
    with sentry_sdk.start_span(op="ml.inference", name=f"TabPFN {model_version()} presence scoring"):
        with console.status("TabPFN is learning what a person looks like...") as status:
            detector.score(progress=lambda f: status.update(f"TabPFN: fold {f + 1} of 4 scored"))
    OUT.mkdir(exist_ok=True)
    df[["date", "p_occupied"]].to_csv(cache, index=False)
    return df


def investigate(args) -> None:
    from . import agents, narrate, site
    from .data import load
    from .detector import Detector, accuracy, find_apparitions, model_version, one_per_day

    df = load()
    det = Detector(df)
    with sentry_sdk.start_transaction(op="seance", name="The Empty Room séance"):
        score(det, args.rescore)
        stats = accuracy(df) | {"tabpfn": model_version(), "llm": agents.MODEL}
        found = find_apparitions(df)
        console.print(
            f"[bold]{stats['haunted_minutes']}[/] of {stats['empty_minutes']} empty minutes felt occupied "
            f"(TabPFN {stats['tabpfn']}, balanced accuracy {stats['balanced_accuracy']:.1%}); {len(found)} apparitions."
        )
        for a in found[: args.list]:
            console.print(f"  {a.id}  {a.weekday:<9} {a.minutes:>4} min  p={a.mean_p:.2f}  light={a.light_mean}  co2={a.co2_start}->{a.co2_end}")
        if args.no_llm:
            return

        llm = agents.client()
        cases = []
        for a in one_per_day(found, args.top):
            console.rule(f"{a.weekday} {a.start}")
            started, before = time.perf_counter(), dict(agents.USAGE)
            with console.status("The Medium is listening..."):
                testimony = agents.medium(llm, df, a)
            console.print(f"[italic cyan]{testimony}[/]\n")
            with console.status("The Skeptic is checking..."):
                v = agents.skeptic(llm, df, det, a)
            for c in v.tool_calls:
                console.print(f"[dim]  tool {c['tool']}({json.dumps(c['args'])})[/]")
            console.print(f"[bold]{v.verdict}[/]: {v.cause}\n{v.report}\n")
            case = {
                "apparition": a.to_dict(),
                "testimony": testimony,
                "verdict": v.verdict,
                "cause": v.cause,
                "report": v.report,
                "tool_calls": v.tool_calls,
                "seconds": round(time.perf_counter() - started),
                "model_calls": agents.USAGE["calls"] - before["calls"],
                "tokens": agents.USAGE["input_tokens"] + agents.USAGE["output_tokens"]
                - before["input_tokens"] - before["output_tokens"],
            }
            if args.narrate and narrate.enabled():
                with console.status("ElevenLabs is reading it aloud..."):
                    case["audio"] = narrate.narrate_case(a.id, testimony, v.report, SITE / "audio")
            cases.append(case)

    stats["usage"] = dict(agents.USAGE)
    OUT.mkdir(exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps({"stats": stats, "cases": cases}, indent=2, default=str), encoding="utf-8")
    page = site.render(df, cases, stats, SITE)
    console.print(f"Case files written to {page.relative_to(ROOT)}")


def narrate_saved(args) -> None:
    """Read the saved cases aloud with ElevenLabs, then rebuild the page."""
    from . import narrate

    if not narrate.enabled():
        raise SystemExit("Set ELEVENLABS_API_KEY in .env first.")
    path = OUT / "cases.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    for case in saved["cases"]:
        a = case["apparition"]
        with console.status(f"ElevenLabs is reading {a['id']} aloud..."):
            case["audio"] = narrate.narrate_case(a["id"], case["testimony"], case["report"], SITE / "audio")
        console.print(f"  {a['id']}: {case['audio']['testimony']}, {case['audio']['report']}")
    path.write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
    rebuild_site(args, saved)


def rebuild_site(args, saved: dict | None = None) -> None:
    """Re-render the page from out/cases.json without running any model."""
    from . import site
    from .data import load

    saved = saved or json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    df = load()
    scores = pd.read_csv(scores_file(saved["stats"]["tabpfn"]), parse_dates=["date"])
    df["p_occupied"] = df["date"].map(scores.set_index("date")["p_occupied"])
    page = site.render(df, saved["cases"], saved["stats"], SITE)
    console.print(f"Rebuilt {page.relative_to(ROOT)}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    warnings.filterwarnings("ignore", message="Running on CPU")
    parser = argparse.ArgumentParser(prog="empty-room", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("investigate", help="find apparitions and hold the séance")
    inv.add_argument("--top", type=int, default=5, help="how many apparitions to investigate")
    inv.add_argument("--list", type=int, default=12, help="how many apparitions to list")
    inv.add_argument("--rescore", action="store_true", help="ignore cached TabPFN scores")
    inv.add_argument("--no-llm", action="store_true", help="only run the presence model")
    inv.add_argument("--narrate", action="store_true", help="read cases aloud with ElevenLabs")
    inv.set_defaults(func=investigate)
    sub.add_parser("site", help="rebuild site/index.html from out/cases.json").set_defaults(func=rebuild_site)
    sub.add_parser("narrate", help="voice the saved cases with ElevenLabs").set_defaults(func=narrate_saved)
    args = parser.parse_args()
    init_tracing()
    try:
        args.func(args)
    finally:
        sentry_sdk.flush()
