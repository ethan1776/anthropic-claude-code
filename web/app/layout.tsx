import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import "./globals.css";

const inter = localFont({ src: "./fonts/Inter.ttf", variable: "--font-inter", display: "swap" });
const anton = localFont({ src: "./fonts/Anton.ttf", variable: "--font-anton", display: "swap" });

export const metadata: Metadata = {
  title: "MOG: auto phonk edits",
  description: "Upload clips and a phonk track, pick a style, get a beat-synced vertical edit.",
};

export const viewport: Viewport = { themeColor: "#070708", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${anton.variable}`}>
      <body className="min-h-dvh bg-ink-950 font-sans text-bone antialiased">{children}</body>
    </html>
  );
}
