# MOG: auto phonk / velocity edit generator

Upload clips and a phonk track, pick a style, and get a beat-synced 9:16 edit.

> Only use clips and music you have the rights to post.

**Status:** Phase 1 of 8 (audio analysis, CLI only). See [PROGRESS.md](PROGRESS.md).

## Setup (Phase 1)

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
