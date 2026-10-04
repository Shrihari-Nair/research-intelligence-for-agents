// Records a real run in the real UI (docker compose up first). Output: ../docs/demo/demo.webm + frames/ (used to build demo.gif).
import { chromium } from "@playwright/test";
import fs from "node:fs";

const BASE = process.env.UI_URL ?? "http://localhost:3201";
const QUESTION = process.env.QUESTION ?? "What license does LangGraph use?";
const OUT = "../docs/demo";
fs.rmSync(`${OUT}/frames`, { recursive: true, force: true });
fs.mkdirSync(`${OUT}/frames`, { recursive: true });

const browser = await chromium.launch({ args: ["--no-sandbox"] });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, recordVideo: { dir: `${OUT}/tmp`, size: { width: 1280, height: 800 } } });
const page = await ctx.newPage();
let n = 0;
const caption = (t) => page.evaluate((text) => {
  let el = document.getElementById("demo-caption");
  if (!el) {
    el = document.createElement("div");
    el.id = "demo-caption";
    el.style.cssText = "position:fixed;left:50%;bottom:18px;transform:translateX(-50%);max-width:900px;z-index:99999;background:rgba(15,23,42,.92);color:#fff;padding:10px 18px;border-radius:10px;font:600 17px system-ui;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,.35)";
    document.body.appendChild(el);
  }
  el.textContent = text;
}, t);
const shot = async (label) => { await page.screenshot({ path: `${OUT}/frames/${String(++n).padStart(2, "0")}-${label}.png` }); };
const beat = async (ms) => page.waitForTimeout(ms);

await page.goto(BASE, { waitUntil: "networkidle" });
await caption("Research Intelligence OS: ask a research question");
await beat(2200); await shot("home");

const box = page.getByTestId("request-input");
await box.fill("");
await box.pressSequentially(QUESTION, { delay: 55 });
await caption("A fine-tuned MiniLM router will classify it in about 30 ms, with no LLM call");
await beat(1800); await shot("typed");
await page.getByTestId("start-button").click();
await page.getByTestId("router-panel").waitFor({ timeout: 30000 }).catch(() => {});
await caption("Plan runs under a budget: search, read, extract claims, verify, synthesize");
await beat(2500); await shot("running");
await page.getByTestId("run-status").filter({ hasText: /SUCCESS|PARTIAL|FAILED/ }).waitFor({ timeout: 240000 });
await caption("Done. Router decision, plan and budget used are all visible");
await beat(2500); await shot("done");

await page.getByTestId("tab-evidence").click();
await caption("Every claim is tied to a verbatim quote from a stored source");
await page.getByTestId("claim-row").first().click();
await beat(2800); await shot("evidence");

await page.getByTestId("tab-report").click();
await caption("A cited report: findings, evidence, limitations. Export as Markdown or PDF");
await beat(3000); await shot("report");
await page.evaluate(() => window.scrollTo({ top: 600, behavior: "smooth" }));
await beat(2200); await shot("report-evidence");

await page.getByTestId("nav-history").click();
await caption("Every run is stored and can be reopened or replayed");
await beat(2500); await shot("history");

await ctx.close();   // flushes the video
await browser.close();
const vid = fs.readdirSync(`${OUT}/tmp`).find((f) => f.endsWith(".webm"));
fs.renameSync(`${OUT}/tmp/${vid}`, `${OUT}/demo.webm`);
fs.rmSync(`${OUT}/tmp`, { recursive: true, force: true });
console.log("wrote", `${OUT}/demo.webm`, "and", n, "frames");
