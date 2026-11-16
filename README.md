# MOG: auto phonk / velocity edit generator

Upload clips and a phonk track, pick a style, and get a beat-synced 9:16 edit.

> Only use clips and music you have the rights to post.

**Status:** Phases 1–2 of 8 done (audio analysis, basic renderer; CLI only). See [PROGRESS.md](PROGRESS.md).

## Setup

You need Python 3.11+ and FFmpeg.

```bash
cd engine
pip install -e ".[dev]"
# optional, more accurate beats + downbeats (downloads PyTorch, ~700 MB):
pip install -e ".[beats]"
```

## Try it

```bash
cd engine
# Analyze a track: BPM, beats, downbeats, drums, drop, best 15 s section
python -m mog analyze path/to/track.mp3 --length 15 --json analysis.json

# Same, plus a debug plot and click tracks to check accuracy by eye and ear
python -m mog debug path/to/track.mp3 --out out/debug --length 15

# No track handy? Generate a synthetic phonk-style one
python -m mog synth out/test.wav --bpm 145

# Run the tests
python -m pytest
```

`debug` writes:

| File | What it's for |
|---|---|
| `<name>.debug.png` | Waveform with beats (grey), downbeats (white), kicks / snares / cowbells / hi-hats, energy curves, the drop (red) and the chosen section (green). The bottom panel zooms in on the drop. |
| `<name>.clicks.wav` | The song with clicks on each detected beat: a high click on beat 1, a lower click on the others, and a ping at the drop. If the clicks sound locked to the drums, detection is right. |
| `<name>.section-clicks.wav` | The same, cut to just the section the edit would use. |
| `<name>.analysis.json` | Everything the planner will use. |

See `samples/phase1/` for example output.

## Render a timeline (Phase 2)

A timeline is a JSON file describing the edit: which clip plays when, at what speed and
crop, plus the colour grade and the music section. The format is defined in
`engine/mog/timeline/models.py`; TypeScript types for the web app are in `web/lib/timeline.ts`.

```bash
cd engine
# Demo media: 4 synthetic clips + a synthetic phonk track
python -m mog testclips out/media
python -m mog synth out/media/phonk145.wav --bpm 145

# Render the hand-written sample timeline
python -m mog render ../samples/phase2/sample_timeline.json --media out/media --out out/sample.mp4
# Quick half-resolution preview
python -m mog render ../samples/phase2/sample_timeline.json --media out/media --out out/preview.mp4 --preview

# After changing the timeline models, regenerate the JSON Schema + TypeScript types
python -m mog schema
```

Converted clips and generated LUTs are cached in `~/.cache/mog` (set `MOG_CACHE` to move it).
