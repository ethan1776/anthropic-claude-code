"use client";

import Link from "next/link";

const STEPS = [
  { key: "upload", label: "Upload" },
  { key: "song", label: "Song" },
  { key: "style", label: "Style" },
  { key: "edit", label: "Edit" },
] as const;

export function Header({ pid, step }: { pid?: string; step?: (typeof STEPS)[number]["key"] }) {
  const idx = STEPS.findIndex((s) => s.key === step);
  return (
    <header className="sticky top-0 z-40 border-b border-ink-800 bg-ink-950/85 backdrop-blur">
      <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-3">
        <Link href="/" className="font-display text-2xl tracking-wide">
          MOG<span className="text-blood">.</span>
        </Link>
        {pid && step && (
          <ol className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-widest">
            {STEPS.map((s, i) => {
              const done = i < idx;
              const on = i === idx;
              const href = s.key === "edit" ? undefined : `/p/${pid}/${s.key}`;
              const cls = on ? "text-bone" : done ? "text-bone/60 hover:text-bone" : "text-ink-600";
              return (
                <li key={s.key} className="flex items-center gap-1.5">
                  {i > 0 && <span className="h-px w-3 bg-ink-600" />}
                  {href && done ? (
                    <Link href={href} className={cls}>
                      {s.label}
                    </Link>
                  ) : (
                    <span className={cls}>{s.label}</span>
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </header>
  );
}

export function Bar({ value, className = "" }: { value: number; className?: string }) {
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full bg-ink-700 ${className}`}>
      <div
        className="h-full rounded-full bg-bone transition-[width] duration-500 ease-out"
        style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }}
      />
    </div>
  );
}

export function ErrorNote({ msg, onRetry }: { msg: string | null; onRetry?: () => void }) {
  if (!msg) return null;
  return (
    <div role="alert" className="flex items-start justify-between gap-3 rounded-xl border border-blood/40 bg-blood/10 p-3 text-sm">
      <span>{msg}</span>
      {onRetry && (
        <button className="shrink-0 font-semibold underline" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Warnings({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <ul className="space-y-1 text-xs text-amber-200/80">
      {items.map((w) => (
        <li key={w}>! {w}</li>
      ))}
    </ul>
  );
}

export function fmtTime(s: number) {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
}

export function fmtBytes(b: number) {
  if (b > 1024 ** 3) return `${(b / 1024 ** 3).toFixed(1)} GB`;
  if (b > 1024 ** 2) return `${(b / 1024 ** 2).toFixed(0)} MB`;
  return `${Math.ceil(b / 1024)} KB`;
}
