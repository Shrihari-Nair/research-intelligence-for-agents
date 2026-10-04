import { defineConfig } from "@playwright/test";

// Boots the real API (offline demo LLM + fixture corpus) and the production build of the UI.
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: "http://localhost:3101", launchOptions: { args: ["--no-sandbox"] }, trace: "retain-on-failure" },
  webServer: [
    {
      command: "cd .. && RESEARCH_OS_LLM=demo RESEARCH_OS_CORS_ORIGINS=http://localhost:3101 DATABASE_URL=sqlite:///runs/e2e.db .venv/bin/uvicorn research_os.api.app:app --port 8765",
      url: "http://localhost:8765/health",
      timeout: 180_000,
      reuseExistingServer: false,  // other projects run on this machine's usual ports; never attach to someone else's server
    },
    { command: "npm run start:e2e", url: "http://localhost:3101", timeout: 60_000, reuseExistingServer: false },
  ],
});
