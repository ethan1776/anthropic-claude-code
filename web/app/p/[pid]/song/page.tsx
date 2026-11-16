"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { ErrorNote, Header, Warnings, fmtTime } from "@/components/ui";
import Waveform from "@/components/Waveform";
import { api, type SongAnalysis } from "@/lib/api";

const LENGTHS = [10, 15, 20, 30];

export default function SongPage({ params }: { params: { pid: string } }) {
  const { pid } = params;
  const router = useRouter();
  const [song, setSong] = useState<SongAnalysis | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [length, setLength] = useState(15);
  const [start, setStart] = useState<number | null>(null);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const audio = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    api.song(pid).then(setSong).catch((e) => setErr(e.message));
  }, [pid]);

  const auto = song?.sections[String(length)];
  const s = start ?? auto?.start ?? 0;
  // The section end snaps to the downbeat nearest start + length (like the engine does)
  const end = useMemo(() => {
    if (!song) return s + length;
    const after = song.downbeats.filter((d) => d > s + length * 0.5);
    if (!after.length) return Math.min(song.duration, s + length);
    return Math.min(song.duration, after.reduce((a, b) => (Math.abs(b - s - length) < Math.abs(a - s - length) ? b : a)));
  }, [song, s, length]);
  const containsDrop = song?.drop ? song.drop.t >= s && song.drop.t < end : false;

  function seek(t: number) {
    if (!song) return;
    const maxStart = Math.max(0, song.duration - length);
    const target = Math.min(Math.max(t, 0), maxStart);
    const snapped = song.downbeats.reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a), song.downbeats[0] ?? 0);
    setStart(snapped);
    stop();
  }

  function play() {
    const a = audio.current;
    if (!a) return;
    a.currentTime = s;
    a.play().catch(() => setErr("Tap play again: the browser blocked audio."));
  }
  function stop() {
    audio.current?.pause();
    setPlayhead(null);
  }

  return (
    <>
      <Header pid={pid} step="song" />
      <main className="mx-auto max-w-3xl space-y-6 px-4 pb-28 pt-6">
        <div>
          <h1 className="font-display text-3xl uppercase tracking-wide">Pick the part</h1>
          <p className="mt-1 text-sm text-bone/60">
            We picked the section around the drop. Drag to move it; it snaps to the bar lines.
          </p>
        </div>
        {!song && !err && <p className="text-sm text-ink-400">Loading the song…</p>}
        <ErrorNote msg={err} />
        {song && (
          <>
            <Waveform
              peaks={song.waveform}
              duration={song.duration}
              beats={song.beats}
              downbeats={song.downbeats}
              drop={song.drop?.t ?? null}
              start={s}
              length={end - s}
              playhead={playhead}
              onSeek={seek}
            />
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-ink-400">
              <span>
                {fmtTime(s)} → {fmtTime(end)} · {(end - s).toFixed(1)}s · {Math.round(song.bpm)} BPM
              </span>
              <span className={containsDrop ? "text-blood" : ""}>{containsDrop ? "includes the drop" : "no drop in this part"}</span>
            </div>
            <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Edit length">
              {LENGTHS.map((L) => (
                <button
                  key={L}
                  role="radio"
                  aria-checked={length === L}
                  className={length === L ? "chip-on" : "chip"}
                  onClick={() => {
                    setLength(L);
                    setStart(null);
                    stop();
                  }}
                >
                  {L}s
                </button>
              ))}
              {start !== null && (
                <button className="chip" onClick={() => setStart(null)}>
                  Reset to auto
                </button>
              )}
            </div>
            <div className="flex gap-3">
              <button className="btn-ghost" onClick={playhead === null ? play : stop}>
                {playhead === null ? "▶ Play section" : "■ Stop"}
              </button>
            </div>
            <audio
              ref={audio}
              src={song.playback}
              preload="auto"
              onTimeUpdate={(e) => {
                const t = e.currentTarget.currentTime;
                if (t >= end) stop();
                else setPlayhead(t);
              }}
              onPause={() => setPlayhead(null)}
            />
            <Warnings
              items={song.warnings
                .filter((w) => !w.startsWith("Using the librosa"))
                .map((w) => (w.includes("--offbeat") ? "This track's drums hit both on and between the beats. If the cuts feel off, try another section." : w))}
            />
          </>
        )}
      </main>
      <footer className="fixed inset-x-0 bottom-0 z-30 border-t border-ink-800 bg-ink-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-3xl justify-end px-4 py-3">
          <button
            className="btn-primary"
            disabled={!song}
            onClick={() => {
              stop();
              const q = new URLSearchParams({ len: String(length) });
              if (start !== null) q.set("start", String(start));
              router.push(`/p/${pid}/style?${q}`);
            }}
          >
            Continue
          </button>
        </div>
      </footer>
    </>
  );
}
