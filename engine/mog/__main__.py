"""Command-line interface.

    python -m mog analyze TRACK [--json out.json] [--length 15]
    python -m mog debug   TRACK [--out DIR] [--length 15]
    python -m mog synth   OUT.wav [--bpm 145] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mog.errors import MogError


def _cmd_analyze(args: argparse.Namespace) -> None:
    from mog.audio import analyze_audio, pick_section

    t = time.time()
    an = analyze_audio(args.track, tracker=args.tracker, offbeat=args.offbeat)
    sec = pick_section(an, args.length) if args.length else None
    payload = {"analysis": an.model_dump(), "section": sec.model_dump() if sec else None}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json}")
    _summary(an, sec, time.time() - t)


def _cmd_debug(args: argparse.Namespace) -> None:
    from mog.audio import analyze_audio, pick_section
    from mog.audio.debug import write_click_track, write_debug_png
    from mog.ffmpeg.audio_io import decode_audio

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.track).stem
    t = time.time()
    an = analyze_audio(args.track, tracker=args.tracker, offbeat=args.offbeat)
    sec = pick_section(an, args.length)
    elapsed = time.time() - t

    (out / f"{stem}.analysis.json").write_text(json.dumps({"analysis": an.model_dump(), "section": sec.model_dump()},
                                                          indent=1))
    y_plot = decode_audio(args.track, sr=11025)
    png = write_debug_png(y_plot, 11025, an, out / f"{stem}.debug.png", section=sec)
    y_full = decode_audio(args.track, sr=44100)
    clicks = write_click_track(y_full, 44100, an, out / f"{stem}.clicks.wav")
    clicks_sec = write_click_track(y_full, 44100, an, out / f"{stem}.section-clicks.wav", section=sec)
    _summary(an, sec, elapsed)
    print(f"\n  plot:          {png}\n  click track:   {clicks}\n  section only:  {clicks_sec}")


def _cmd_synth(args: argparse.Namespace) -> None:
    import soundfile as sf

    from mog.testing.synth import make_track

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    y, truth = make_track(bpm=args.bpm, seed=args.seed, sr=44100)
    sf.write(args.out, y, 44100, subtype="PCM_16")
    print(f"wrote {args.out}: {args.bpm} BPM, true drop at {truth.drop:.3f}s, {len(truth.beats)} beats")


def _summary(an, sec, elapsed: float) -> None:
    kinds = {k: len(an.onsets_of(k)) for k in ("kick", "snare", "hihat", "cowbell")}
    print(f"\n{an.source}  ({an.duration:.1f}s, analyzed in {elapsed:.1f}s)")
    print(f"  BPM:        {an.bpm:.2f}   (tracker: {an.beat_tracker})")
    print(f"  beats:      {len(an.beats)}   downbeats: {len(an.downbeats)}")
    print("  onsets:     " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    if an.drop:
        print(f"  drop:       {an.drop.t:.3f}s   confidence {an.drop.confidence:.2f}")
    else:
        print("  drop:       none found")
    if sec:
        print(f"  section:    {sec.start:.3f}s -> {sec.end:.3f}s  ({sec.duration:.2f}s, drop after {sec.lead_bars} bars)")
    for w in an.warnings + (sec.warnings if sec else []):
        print(f"  ! {w}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mog", description="MOG auto phonk edit generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze a music track")
    a.add_argument("track")
    a.add_argument("--json", help="write the analysis to this JSON file")
    a.add_argument("--length", type=float, default=15, help="target edit length for the section picker (s)")
    a.add_argument("--tracker", choices=["auto", "beat_this", "librosa"], default="auto")
    a.add_argument("--offbeat", action="store_true", help="shift the beat grid by half a beat")
    a.set_defaults(func=_cmd_analyze)

    d = sub.add_parser("debug", help="analyze + write debug PNG and click tracks")
    d.add_argument("track")
    d.add_argument("--out", default="out/debug")
    d.add_argument("--length", type=float, default=15)
    d.add_argument("--tracker", choices=["auto", "beat_this", "librosa"], default="auto")
    d.add_argument("--offbeat", action="store_true", help="shift the beat grid by half a beat")
    d.set_defaults(func=_cmd_debug)

    s = sub.add_parser("synth", help="generate a synthetic phonk-style test track")
    s.add_argument("out")
    s.add_argument("--bpm", type=float, default=145)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_cmd_synth)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except MogError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
