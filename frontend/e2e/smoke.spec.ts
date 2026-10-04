import { expect, test } from "@playwright/test";

// A request that mentions many technologies produces an expensive plan, which must wait for approval.
const EXPENSIVE = "Compare Redis, Memcached, Kafka, Pulsar, RabbitMQ and Nomad for event handling";

test("submit, watch live execution, open a claim's evidence, read the report and download it", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("request-input").fill("Compare LangGraph and CrewAI");
  await page.getByTestId("start-button").click();
  await page.waitForURL(/\/runs\/[a-f0-9]+/);

  // live execution: graph + router panel appear, then the run finishes
  await expect(page.getByTestId("run-graph")).toBeVisible();
  await expect(page.getByTestId("router-panel")).toContainText("TECHNOLOGY_COMPARISON");
  await expect(page.getByTestId("run-status")).toContainText("SUCCESS");
  await expect(page.getByTestId("node-verify")).toHaveAttribute("data-status", "succeeded");
  await expect(page.getByTestId("budget-meters")).toContainText("Steps");
  await expect(page.getByTestId("event-log")).toContainText("planner");

  // evidence explorer: click a claim -> its source opens with the verbatim quote highlighted
  await page.getByTestId("tab-evidence").click();
  await expect(page.getByTestId("group-supported")).toBeVisible();
  await page.getByTestId("claim-row").first().click();
  await expect(page.getByTestId("source-viewer")).toBeVisible();
  await expect(page.getByTestId("quote-highlight").first()).toBeVisible();

  // report: sections, clickable citation, downloads
  await page.getByTestId("tab-report").click();
  await expect(page.getByTestId("report")).toContainText("Executive summary");
  await expect(page.getByTestId("report")).toContainText("Limitations");
  await page.getByTestId("citation").first().click();
  await expect(page.getByTestId("citation-dialog")).toBeVisible();
  await expect(page.getByTestId("citation-dialog").getByTestId("quote-highlight").first()).toBeVisible();
  await page.keyboard.press("Escape");
  const [md] = await Promise.all([page.waitForEvent("download"), page.getByTestId("download-md").click()]);
  expect(md.suggestedFilename()).toMatch(/report-.*\.md$/);
  const [json] = await Promise.all([page.waitForEvent("download"), page.getByTestId("download-json").click()]);
  expect(json.suggestedFilename()).toMatch(/report-.*\.json$/);
});

test("an expensive plan asks for approval before anything runs", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("request-input").fill(EXPENSIVE);
  await page.getByTestId("start-button").click();
  await page.waitForURL(/\/runs\/[a-f0-9]+/);
  const dialog = page.getByTestId("approval-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("Approval required");
  await expect(dialog).toContainText("searches");
  await expect(page.getByTestId("run-status")).toContainText("AWAITING");
  await dialog.getByRole("button", { name: "Approve" }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByTestId("run-status")).toContainText(/SUCCESS|PARTIAL/, { timeout: 90_000 });
});

test("rejecting an approval cancels the run", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("request-input").fill(EXPENSIVE);
  await page.getByTestId("start-button").click();
  await page.waitForURL(/\/runs\/[a-f0-9]+/);
  await page.getByTestId("approval-dialog").getByRole("button", { name: "Cancel run" }).click();
  await expect(page.getByTestId("run-status")).toContainText("CANCELLED");
  await expect(page.getByTestId("cancelled-banner")).toBeVisible();
});

test("history lists runs and can be filtered", async ({ page }) => {
  await page.goto("/history");
  await expect(page.getByTestId("history-table")).toBeVisible();
  await expect(page.getByTestId("history-row").first()).toBeVisible();
  await page.getByTestId("history-status").selectOption("CANCELLED");
  await expect(page.getByTestId("history-row").first()).toContainText("CANCELLED");
  await page.getByTestId("history-search").fill("zzz-no-such-run");
  await expect(page.getByTestId("history-table")).toContainText("No runs match");
});
