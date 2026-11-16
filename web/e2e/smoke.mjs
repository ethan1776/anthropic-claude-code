// End-to-end smoke test of the whole app in a phone-sized browser.
//   BASE=http://localhost:3000 CLIPS=a.mp4,b.mov AUDIO=song.mp3 node e2e/smoke.mjs
// Needs the API + worker running. Screenshots go to e2e/out/.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
// Playwright is not a project dependency: use your global install (npm i -g playwright).
const { chromium } = require("playwright");
import fs from "node:fs";

const BASE = process.env.BASE || "http://localhost:3000";
const CLIPS = (process.env.CLIPS || "").split(",").filter(Boolean);
const AUDIO = process.env.AUDIO;
const OUT = new URL("./out/", import.meta.url).pathname;
fs.mkdirSync(OUT, { recursive: true });
if (!CLIPS.length || !AUDIO) throw new Error("Set CLIPS (comma-separated) and AUDIO");

const t0 = Date.now();
const log = (m) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0).padStart(4)}s] ${m}`);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 1 });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
const shot = (n) => page.screenshot({ path: `${OUT}${n}.png`, fullPage: true });

try {
  await page.goto(BASE);
  await page.getByText("on the beat").waitFor();
  await shot("1-home");
  await page.getByRole("button", { name: "Create edit" }).click();
  await page.waitForURL(/\/upload$/);
  log("upload page");

  await page.getByTestId("drop-clips").setInputFiles(CLIPS);
  await page.getByTestId("drop-audio").setInputFiles(AUDIO);
  await page.waitForTimeout(1500);
  await shot("2-uploading");
  await page.getByText("✓ clips · ✓ song").waitFor({ timeout: 300_000 });
  await page.getByRole("button", { name: "Continue" }).waitFor({ state: "visible" });
  await page.waitForFunction(() => !document.querySelector("footer button")?.hasAttribute("disabled"), null, { timeout: 120_000 });
  await shot("3-uploaded");
  log("uploads analysed");
  await page.getByRole("button", { name: "Continue" }).click();

  await page.waitForURL(/\/song$/);
  await page.getByTestId("waveform").waitFor();
  await page.getByRole("radio", { name: "15s" }).click();
  await page.waitForTimeout(500);
  await shot("4-song");
  log("song page");
  await page.getByRole("button", { name: "Continue" }).click();

  await page.waitForURL(/\/style/);
  await page.getByTestId("preset-dark_triad").waitFor({ timeout: 60_000 });
  await page.getByTestId("preset-dark_triad").click();
  await page.getByLabel("Caption (optional)").fill("he doesn't chase.");
  await page.waitForTimeout(1500);
  await shot("5-style");
  log("style page");
  await page.getByTestId("generate").click();

  await page.waitForURL(/\/generate\//);
  await page.waitForTimeout(4000);
  await shot("6-generating");
  await page.waitForURL(/\/edit\//, { timeout: 400_000 });
  await page.getByTestId("preview").waitFor();
  log("edit generated");
  await page.waitForTimeout(1500);
  await shot("7-editor");

  await page.getByTestId("seg-s4").click();
  await page.getByRole("button", { name: "slow", exact: true }).click();
  await page.getByText("Changes not in the preview yet").waitFor();
  await page.getByRole("button", { name: /Shake/ }).click();
  await page.waitForTimeout(800);
  await shot("8-segment-panel");
  const before = await page.getByTestId("preview").getAttribute("src");
  await page.getByTestId("update-preview").click();
  await page.getByText("Rendering preview…").waitFor();
  await page.waitForFunction((b) => document.querySelector("[data-testid=preview]")?.getAttribute("src") !== b,
    before, { timeout: 300_000 });
  await page.getByText("Changes not in the preview yet").waitFor({ state: "detached" });
  log("preview re-rendered after edits");

  await page.getByTestId("open-export").click();
  const dialog = page.getByRole("dialog", { name: "Export" });
  await dialog.getByRole("button", { name: "Fast", exact: true }).click();
  await page.getByTestId("export").click();
  await page.getByTestId("download").waitFor({ timeout: 400_000 });
  await shot("9-exported");
  const href = await page.getByTestId("download").getAttribute("href");
  const res = await page.request.get(BASE + href);
  fs.writeFileSync(`${OUT}export.mp4`, await res.body());
  log(`exported ${(await res.body()).length} bytes`);
  if (errors.length) console.log("browser errors:", errors);
  console.log("SMOKE OK");
} catch (e) {
  await shot("error");
  console.log("browser errors:", errors);
  throw e;
} finally {
  await browser.close();
}
