import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PrivacyDisclosure } from "./PrivacyDisclosure";

describe("PrivacyDisclosure", () => {
  it("states that FL alone is not complete privacy", () => {
    render(<PrivacyDisclosure workload="general" />);
    expect(
      screen.getByText(/does not guarantee privacy/i)
    ).toBeInTheDocument();
  });

  it("discloses that inference inputs leave the node", () => {
    render(<PrivacyDisclosure workload="inference" />);
    expect(screen.getByText(/payload\.inputs/i)).toBeInTheDocument();
    expect(screen.getByTestId("privacy-disclosure")).toBeInTheDocument();
  });

  it("does not claim secure aggregation is enabled", () => {
    render(<PrivacyDisclosure showFlags />);
    const row = screen
      .getAllByRole("row")
      .find((r) => r.textContent?.includes("Secure aggregation"));
    expect(row).toBeTruthy();
    expect(row).toHaveTextContent("Not available");
  });
});

