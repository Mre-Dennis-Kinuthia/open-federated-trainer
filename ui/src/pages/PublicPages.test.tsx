import { render, screen } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";
import { describe, expect, it, vi } from "vitest";
import { PrivacyPage } from "./PrivacyPage";
import { StatusPage } from "./StatusPage";

expect.extend(toHaveNoViolations);

vi.mock("../api", () => ({
  fetchOverview: vi.fn(async () => ({
    operator_auth_required: true,
    registered_clients: [],
    global: { total_rounds: 0 },
    job_stats: { total: 0, completed: 0, counts: { QUEUED: 0 } },
  })),
  fetchActivity: vi.fn(async () => ({
    online_count: 0,
    nodes: [],
  })),
}));

describe("authoritative public pages", () => {
  it("status page states this instance scope", async () => {
    render(<StatusPage />);
    const page = await screen.findByTestId("status-page");
    expect(page.textContent).toMatch(/this coordinator instance/i);
    expect(
      screen.getByRole("heading", { name: /coordinator status/i })
    ).toBeInTheDocument();
  });

  it("privacy page rejects complete-privacy marketing", () => {
    render(<PrivacyPage />);
    const page = screen.getByTestId("privacy-page");
    expect(page.textContent).toMatch(/does\s+not\s+mean complete privacy/i);
    expect(page.textContent).toMatch(/GEO_LOOKUP_DISABLED/i);
  });

  it("privacy page has no critical axe violations", async () => {
    const { container } = render(<PrivacyPage />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
