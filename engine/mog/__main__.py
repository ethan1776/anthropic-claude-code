"""Command-line interface.

    python -m mog analyze TRACK [--json out.json] [--length 15]
    python -m mog debug   TRACK [--out DIR] [--length 15]
    python -m mog synth   OUT.wav [--bpm 145] [--seed 0]
    python -m mog render  TIMELINE.json --media DIR --out OUT.mp4 [--preview]
    python -m mog testclips DIR
    python -m mog schema  [--json OUT.schema.json] [--ts OUT.ts]
    python -m mog scout   CLIP [--debug DIR]
    python -m mog reframe CLIP --out OUT.mp4 [--max 15] [--preview]
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


def _cmd_render(args: argparse.Namespace) -> None:
    from pydantic import ValidationError

    from mog.render.renderer import render
    from mog.timeline.models import Timeline

    try:
        tl = Timeline.model_validate_json(Path(args.timeline).read_text())
    except FileNotFoundError:
        raise MogError(f"Timeline file not found: {args.timeline}") from None
    except ValidationError as e:
        lines = [f"  {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise MogError("The timeline is not valid:\n" + "\n".join(lines)) from None
    media = Path(args.media) if args.media else Path(args.timeline).parent
    last = [""]

    def progress(stage: str, frac: float) -> None:
        line = f"\r  {stage:<16} {frac * 100:5.1f}%"
        if line != last[0]:
            print(line, end="", flush=True)
            last[0] = line

    res = render(tl, args.out, media, preview=args.preview, progress=progress)
    print(f"\nwrote {res.path}  {res.width}x{res.height} @ {res.fps} fps, {res.frames} frames, "
          f"{res.frames / res.fps:.2f}s of video in {res.seconds:.1f}s")
    for w in res.warnings:
        print(f"  ! {w}")


def _cmd_testclips(args: argparse.Namespace) -> None:
    from mog.testing.clips import make_demo_clips

    for p in make_demo_clips(Path(args.dir)):
        print(f"  {p}")


def _cmd_schema(args: argparse.Namespace) -> None:
    from mog.timeline.export import write_json_schema, write_typescript

    if args.json:
        print(f"wrote {write_json_schema(Path(args.json))}")
    if args.ts:
        print(f"wrote {write_typescript(Path(args.ts))}")


def _cmd_scout(args: argparse.Namespace) -> None:
    from mog.video.analyze import analyze_video

    t = time.time()
    an = analyze_video(args.clip, use_cache=not args.fresh)
    print(f"\n{an.source}  {an.width}x{an.height} @ {an.fps:.2f} fps, {an.duration:.1f}s, "
          f"analysed in {time.time() - t:.1f}s")
    print(f"  content area: {an.content}")
    print("   #   start    end   face frontal motion  stare action quality  exposure")
    for s in an.shots:
        print(f"  {s.index:2d} {s.start:7.2f} {s.end:6.2f}   {s.face_score:.2f}   {s.frontal:.2f}   {s.motion_score:.2f}"
              f"   {s.stare_score:.2f}   {s.action_score:.2f}    {s.quality:.2f}    {s.exposure:+.2f}")
    for w in an.warnings:
        print(f"  ! {w}")
    if args.debug:
        from mog.video.reframe import debug_sheet

        d = Path(args.debug)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{Path(args.clip).stem}.analysis.json").write_text(an.model_dump_json(indent=1))
        print(f"  sheet: {debug_sheet(an, Path(args.clip), d / f'{Path(args.clip).stem}.shots.png')}")


def _cmd_reframe(args: argparse.Namespace) -> None:
    from mog.render.renderer import render
    from mog.video.analyze import analyze_video
    from mog.video.ingest import cache_dir
    from mog.video.reframe import reframe_timeline, silent_wav

    an = analyze_video(args.clip)
    clip = Path(args.clip)
    silent_wav(cache_dir().parent / "silence-600s.wav", 600)
    tl = reframe_timeline(an, "silence-600s.wav", max_duration=args.max)
    # Media dir must hold both the clip and the audio: link the silence next to the clip's cache
    media = cache_dir().parent / "reframe-media"
    media.mkdir(exist_ok=True)
    for src, name in ((clip.resolve(), clip.name), (cache_dir().parent / "silence-600s.wav", "silence-600s.wav")):
        link = media / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
    res = render(tl, args.out, media, preview=args.preview)
    print(f"wrote {res.path} ({len(tl.segments)} shots, {res.frames / res.fps:.1f}s) in {res.seconds:.1f}s")


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

    r = sub.add_parser("render", help="render a timeline JSON to MP4")
    r.add_argument("timeline")
    r.add_argument("--media", help="folder holding the clips and audio (default: the timeline's folder)")
    r.add_argument("--out", default="out/render.mp4")
    r.add_argument("--preview", action="store_true", help="fast half-resolution render")
    r.set_defaults(func=_cmd_render)

    c = sub.add_parser("testclips", help="generate synthetic demo clips")
    c.add_argument("dir")
    c.set_defaults(func=_cmd_testclips)

    sc = sub.add_parser("schema", help="export the timeline schema (JSON Schema and/or TypeScript)")
    sc.add_argument("--json", default="../web/lib/timeline.schema.json")
    sc.add_argument("--ts", default="../web/lib/timeline.ts")
    sc.set_defaults(func=_cmd_schema)

    sc2 = sub.add_parser("scout", help="analyse a video clip: shots, faces, scores")
    sc2.add_argument("clip")
    sc2.add_argument("--debug", help="write the analysis JSON and a debug image here")
    sc2.add_argument("--fresh", action="store_true", help="ignore the analysis cache")
    sc2.set_defaults(func=_cmd_scout)

    rf = sub.add_parser("reframe", help="auto-reframe a clip to 9:16 following faces")
    rf.add_argument("clip")
    rf.add_argument("--out", default="out/reframed.mp4")
    rf.add_argument("--max", type=float, default=15.0, help="max seconds to render")
    rf.add_argument("--preview", action="store_true")
    rf.set_defaults(func=_cmd_reframe)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except MogError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
