import { PrivacyDisclosure } from "../components/PrivacyDisclosure";

export function PrivacyPage() {
  return (
    <div className="public-page" data-testid="privacy-page">
      <a className="skip-link" href="#privacy-main">
        Skip to privacy
      </a>
      <header className="public-page-header">
        <a href="#/" className="public-brand">
          fed-compute
        </a>
        <nav aria-label="Public">
          <a href="#/status">Status</a>
          <a href="#/overview">Console</a>
        </nav>
      </header>
      <main id="privacy-main" className="public-page-main">
        <h1>Privacy model</h1>
        <p className="public-lede">
          Accurate summary of what this software does today. Federated learning
          alone does <strong>not</strong> mean complete privacy.
        </p>

        <section aria-labelledby="geo-heading">
          <h2 id="geo-heading">Location / activity globe</h2>
          <ul>
            <li>
              Optional coarse presence for the public landing map (city-level with
              jitter).
            </li>
            <li>
              Disable lookups with <code>GEO_LOOKUP_DISABLED=true</code> on the
              coordinator.
            </li>
            <li>
              Public activity API does not expose client IDs or IP addresses.
            </li>
            <li>
              Globe arcs between points are decorative illustrations — not measured
              network paths.
            </li>
          </ul>
        </section>

        <section aria-labelledby="workloads-heading">
          <h2 id="workloads-heading">By workload</h2>
          <PrivacyDisclosure workload="train" />
          <PrivacyDisclosure workload="lora" />
          <PrivacyDisclosure workload="inference" />
          <PrivacyDisclosure workload="compute" />
        </section>

        <section aria-labelledby="flags-heading">
          <h2 id="flags-heading">Capability flags</h2>
          <PrivacyDisclosure workload="general" showFlags compact />
        </section>
      </main>
    </div>
  );
}
