"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ErrorNote, Header } from "@/components/ui";
import { api, type Look } from "@/lib/api";

export default function StylePage({ params, searchParams }: {
  params: { pid: string };
  searchParams: { len?: string; start?: string };
}) {
  const { pid } = params;
  const router = useRouter();
  const [looks, setLooks] = useState<Look[] | null>(null);
  const [preset, setPreset] = useState("mog");
  const [intensity, setIntensity] = useState(60);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const length = Number(searchParams.len ?? 15);
  const start = searchParams.start !== undefined ? Number(searchParams.start) : null;

  useEffect(() => {
    api.looks(pid).then(setLooks).catch((e) => setErr(e.message));
  }, [pid]);

  async function generate() {
    setBusy(true);
    setErr(null);
    try {
      const e = await api.createEdit(pid, {
        preset, length, section_start: start, intensity: intensity / 100, text: text.trim() || null, seed: 1, fps: 30,
      });
      router.push(`/p/${pid}/generate/${e.id}`);
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <>
      <Header pid={pid} step="style" />
      <main className="mx-auto max-w-3xl space-y-6 px-4 pb-28 pt-6">
        <h1 className="font-display text-3xl uppercase tracking-wide">Pick a style</h1>
        {!looks && !err && <p className="text-sm text-ink-400">Loading styles…</p>}
        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3" role="radiogroup" aria-label="Style">
          {looks?.map((l) => (
            <li key={l.id}>
              <button
                role="radio"
                aria-checked={preset === l.id}
                data-testid={`preset-${l.id}`}
                onClick={() => setPreset(l.id)}
                className={`card block w-full overflow-hidden text-left transition ${
                  preset === l.id ? "border-bone ring-2 ring-bone" : "hover:border-ink-400"
                }`}
              >
                <div className="relative aspect-[9/16] bg-ink-800">
                  {l.image ? (
                    // your own footage in this look
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={l.image} alt={`Your footage in the ${l.name} look`} className="h-full w-full object-cover" />
                  ) : (
                    <video src={l.preview} autoPlay muted loop playsInline className="h-full w-full object-cover" />
                  )}
                  {l.image && (
                    // the style's motion (cuts, punches, flashes) as a small loop
                    <video
                      src={l.preview}
                      autoPlay
                      muted
                      loop
                      playsInline
                      aria-hidden
                      className="absolute bottom-2 right-2 h-24 w-[54px] rounded-md border border-bone/30 object-cover shadow-lg"
                    />
                  )}
                </div>
                <div className="p-2.5">
                  <div className="font-display text-lg uppercase tracking-wide">{l.name}</div>
                  <div className="line-clamp-3 text-[11px] leading-snug text-bone/60">{l.description}</div>
                </div>
              </button>
            </li>
          ))}
        </ul>

        <div className="card space-y-3 p-4">
          <label className="flex items-center justify-between text-sm" htmlFor="intensity">
            <span className="label">Intensity</span>
            <span className="tabular-nums text-ink-400">{intensity}%</span>
          </label>
          <input
            id="intensity"
            type="range"
            min={0}
            max={100}
            value={intensity}
            onChange={(e) => setIntensity(Number(e.target.value))}
            className="w-full accent-[#ff2d55]"
          />
          <p className="text-xs text-ink-400">More effects, stronger punches and shakes.</p>
        </div>

        <div className="card space-y-2 p-4">
          <label className="label" htmlFor="caption">Caption (optional)</label>
          <input
            id="caption"
            maxLength={120}
            value={text}
            placeholder="he doesn't chase."
            onChange={(e) => setText(e.target.value)}
            className="w-full rounded-lg border border-ink-600 bg-ink-950 px-3 py-2.5 text-sm outline-none focus:border-bone"
          />
          <p className="text-xs text-ink-400">One line that pops in after the drop.</p>
        </div>
        <ErrorNote msg={err} />
      </main>
      <footer className="fixed inset-x-0 bottom-0 z-30 border-t border-ink-800 bg-ink-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-3 px-4 py-3">
          <span className="text-xs text-ink-400">{length}s edit</span>
          <button className="btn-blood" onClick={generate} disabled={busy || !looks} data-testid="generate">
            {busy ? "Starting…" : "Generate"}
          </button>
        </div>
      </footer>
    </>
  );
}
