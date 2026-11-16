"use client";

import { useEffect, useRef } from "react";

type Props = {
  peaks: number[];
  duration: number;
  beats: number[];
  downbeats: number[];
  drop: number | null;
  start: number;
  length: number;
  playhead: number | null;
  onSeek: (t: number) => void; // drag/tap: requested section start (unsnapped)
};

/** Waveform with beat ticks, the drop, and a draggable selection window. Canvas: fast for
 * thousands of beat lines, crisp on phones (devicePixelRatio aware). */
export default function Waveform(p: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ dx: number } | null>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const x = (t: number) => (t / p.duration) * w;
    const mid = h / 2;

    // selection background
    g.fillStyle = "rgba(236,235,230,0.08)";
    g.fillRect(x(p.start), 0, x(p.start + p.length) - x(p.start), h);

    // beats + downbeats
    for (const t of p.beats) {
      g.fillStyle = "rgba(122,122,136,0.18)";
      g.fillRect(Math.round(x(t)), 0, 1, h);
    }
    for (const t of p.downbeats) {
      g.fillStyle = "rgba(122,122,136,0.45)";
      g.fillRect(Math.round(x(t)), 0, 1, h);
    }

    // waveform bars
    const n = p.peaks.length;
    const bw = w / n;
    for (let i = 0; i < n; i++) {
      const t = (i / n) * p.duration;
      const inSel = t >= p.start && t < p.start + p.length;
      const a = Math.max(1, p.peaks[i] * (h * 0.46));
      g.fillStyle = inSel ? "#ecebe6" : "#4a4a55";
      g.fillRect(i * bw, mid - a, Math.max(bw - 0.4, 0.6), a * 2);
    }

    // drop
    if (p.drop !== null) {
      g.fillStyle = "#ff2d55";
      g.fillRect(x(p.drop) - 1, 0, 2, h);
      g.font = "600 10px var(--font-inter), sans-serif";
      g.fillText("DROP", Math.min(x(p.drop) + 4, w - 30), 12);
    }
    // selection edges
    g.fillStyle = "#ecebe6";
    g.fillRect(x(p.start), 0, 2, h);
    g.fillRect(x(p.start + p.length) - 2, 0, 2, h);
    // playhead
    if (p.playhead !== null) {
      g.fillStyle = "#7fd6e0";
      g.fillRect(x(p.playhead), 0, 2, h);
    }
  }, [p.peaks, p.duration, p.beats, p.downbeats, p.drop, p.start, p.length, p.playhead]);

  function timeAt(clientX: number) {
    const r = ref.current!.getBoundingClientRect();
    return ((clientX - r.left) / r.width) * p.duration;
  }

  return (
    <canvas
      ref={ref}
      data-testid="waveform"
      className="h-28 w-full touch-none select-none rounded-xl bg-ink-900"
      onPointerDown={(e) => {
        (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
        const t = timeAt(e.clientX);
        const inside = t >= p.start && t <= p.start + p.length;
        drag.current = { dx: inside ? t - p.start : p.length / 2 };
        p.onSeek(t - drag.current.dx);
      }}
      onPointerMove={(e) => drag.current && p.onSeek(timeAt(e.clientX) - drag.current.dx)}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
    />
  );
}
