"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Header, ErrorNote } from "@/components/ui";
import { api, type ProjectSummary } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.projects().then(setProjects).catch(() => setProjects([]));
  }, []);

  async function create() {
    setBusy(true);
    setErr(null);
    try {
      const p = await api.createProject();
      router.push(`/p/${p.id}/upload`);
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <>
      <Header />
      <main className="mx-auto max-w-3xl px-4 pb-16">
        <section className="flex flex-col items-center pb-12 pt-16 text-center sm:pt-24">
          <h1 className="font-display text-6xl uppercase leading-[0.95] tracking-wide sm:text-7xl">
            Phonk edits
            <br />
            <span className="text-blood">on the beat</span>
          </h1>
          <p className="mt-5 max-w-sm text-sm leading-relaxed text-bone/60">
            Drop in your clips and a track. MOG finds the beat and the drop, picks the shots and cuts a vertical edit
            ready to post.
          </p>
          <button className="btn-blood mt-8 w-full max-w-xs py-4 text-base" onClick={create} disabled={busy}>
            {busy ? "Starting…" : "Create edit"}
          </button>
          <div className="mt-4 w-full max-w-xs">
            <ErrorNote msg={err} />
          </div>
        </section>

        <section>
          <h2 className="label mb-3">Recent</h2>
          {projects === null ? (
            <p className="text-sm text-ink-400">Loading…</p>
          ) : projects.length === 0 ? (
            <p className="text-sm text-ink-400">No edits yet. Your projects show up here.</p>
          ) : (
            <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {projects.map((p) => (
                <li key={p.id}>
                  <Link href={`/p/${p.id}/${p.edits ? "style" : "upload"}`} className="card block overflow-hidden">
                    <div className="relative aspect-[9/16] bg-ink-800">
                      {p.latest_preview ? (
                        <video src={p.latest_preview} muted playsInline preload="metadata" className="h-full w-full object-cover" />
                      ) : p.thumb ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={p.thumb} alt="" className="h-full w-full object-cover" />
                      ) : null}
                      {p.saved && <span className="absolute right-2 top-2 rounded bg-ink-950/70 px-1.5 py-0.5 text-[10px]">Saved</span>}
                    </div>
                    <div className="p-2.5">
                      <div className="truncate text-sm font-semibold">{p.name}</div>
                      <div className="text-xs text-ink-400">
                        {p.edits} edit{p.edits === 1 ? "" : "s"} · {new Date(p.created_at * 1000).toLocaleDateString()}
                      </div>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>
    </>
  );
}
