import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import {
  fetchActivity,
  fetchOverview,
  type Activity,
  type Overview,
} from "../api";
import { buildActivityArcs } from "./activityArcs";
import type { GlobeConfig } from "../components/Globe";
import "./landing.css";

const World = lazy(() =>
  import("../components/Globe").then((m) => ({ default: m.World }))
);

const GLOBE_CONFIG: GlobeConfig = {
  pointSize: 3.5,
  globeColor: "#0a1628",
  showAtmosphere: true,
  atmosphereColor: "#c5d4e8",
  atmosphereAltitude: 0.14,
  emissive: "#0a1628",
  emissiveIntensity: 0.2,
  shininess: 0.9,
  polygonColor: "rgba(220, 230, 240, 0.65)",
  ambientLight: "#b8c9dc",
  directionalLeftLight: "#ffffff",
  directionalTopLight: "#ffffff",
  pointLight: "#dbe7f3",
  arcTime: 1400,
  arcLength: 0.88,
  rings: 1,
  maxRings: 2,
  initialPosition: { lat: 12, lng: 20 },
  autoRotate: true,
  autoRotateSpeed: 0.4,
};

const ACTIVITY_POLL_MS = 8000;

const NAV_LINKS = [
  { href: "#why", label: "Why" },
  { href: "#product", label: "Product" },
  { href: "#proof", label: "Console" },
  { href: "#/privacy", label: "Privacy" },
  { href: "#/status", label: "Status" },
] as const;

export function Landing({ onEnter }: { onEnter: () => void }) {
  const reduceMotion = useReducedMotion();
  const [overview, setOverview] = useState<Overview | null>(null);
  const [activity, setActivity] = useState<Activity | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchOverview(25)
      .then((data) => {
        if (!cancelled) setOverview(data);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const data = await fetchActivity();
        if (!cancelled) setActivity(data);
      } catch {
        /* keep last */
      }
    }
    void poll();
    const id = window.setInterval(() => {
      if (!document.hidden) void poll();
    }, ACTIVITY_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const arcs = useMemo(
    () => buildActivityArcs(activity?.nodes ?? []),
    [activity?.nodes]
  );

  const onlineCount = activity?.online_count ?? 0;
  const locatedCount = activity?.nodes.length ?? 0;
  const rounds = overview?.global?.total_rounds;
  const jobsQueued = overview?.job_stats?.counts?.QUEUED;
  const jobsCompleted = overview?.job_stats?.completed;
  const emptyNetwork = locatedCount === 0;

  const enter = reduceMotion
    ? undefined
    : { initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 } };

  function closeMenu() {
    setMenuOpen(false);
  }

  return (
    <div className="landing">
      <a className="skip-link" href="#lp-main">
        Skip to content
      </a>

      <header className="lp-nav">
        <a className="lp-nav-brand" href="#/">
          fed-compute
        </a>
        <nav className="lp-nav-links" aria-label="Primary">
          {NAV_LINKS.map((link) => (
            <a key={link.href} href={link.href}>
              {link.label}
            </a>
          ))}
        </nav>
        <div className="lp-nav-actions">
          <button
            type="button"
            className="lp-menu-toggle"
            aria-expanded={menuOpen}
            aria-controls="lp-mobile-nav"
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? "Close" : "Menu"}
          </button>
          <button type="button" className="lp-btn solid" onClick={onEnter}>
            Console
          </button>
        </div>
      </header>

      <nav
        id="lp-mobile-nav"
        className={`lp-mobile-nav${menuOpen ? " is-open" : ""}`}
        aria-label="Mobile"
      >
        {NAV_LINKS.map((link) => (
          <a key={link.href} href={link.href} onClick={closeMenu}>
            {link.label}
          </a>
        ))}
      </nav>

      <main id="lp-main">
        <section className="lp-hero">
          <motion.div
            className="lp-hero-text"
            {...enter}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          >
            <p className="lp-wordmark">fed-compute</p>
            <h1>
              <span className="lp-promise">Private data.</span>
              <span className="lp-promise">Shared progress.</span>
            </h1>
            <p className="lp-sub">
              The open coordination layer for federated training, LoRA, and
              distributed jobs — data stays on the node unless you send it.
            </p>
            <div className="lp-actions">
              <button type="button" className="lp-btn solid" onClick={onEnter}>
                Open console
              </button>
              <a className="lp-btn ghost" href="#/privacy">
                Privacy model
              </a>
            </div>
          </motion.div>

          <motion.div
            className="lp-hero-stage"
            aria-label="Live activity globe for this coordinator"
            initial={reduceMotion ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1.1, delay: 0.15 }}
          >
            <Suspense
              fallback={<div className="lp-stage-loading">Loading network…</div>}
            >
              <World data={arcs} globeConfig={GLOBE_CONFIG} />
            </Suspense>
          </motion.div>
        </section>

        <p className="lp-stage-caption">
          {emptyNetwork
            ? "No located nodes on this coordinator yet — the globe stays quiet until clients report city-level presence. Arcs are decorative, not measured routes."
            : "Globe arcs are decorative — not measured routes. Points are city-level with jitter from this coordinator’s activity feed."}
        </p>

        <section id="why" className="lp-band">
          <h2 className="lp-band-title">
            Most AI asks you to bring the world to a single cluster.
          </h2>
          <p className="lp-band-lead">
            fed-compute coordinates the model toward the data — so collaboration
            compounds without a warehouse of everyone else’s rows.
          </p>

          <div className="lp-split">
            <div>
              <p className="lp-kicker">Centralized default</p>
              <p>
                Copy datasets. Hope the contract holds. Retrain slowly. Contort
                prompts when the average model misses your edge.
              </p>
            </div>
            <div className="lp-split-accent">
              <p className="lp-kicker">Federated coordination</p>
              <p>
                Keep training corpora local. Exchange deltas, adapters, and job
                results. Disclose what actually leaves the node — no absolute
                privacy theater.
              </p>
            </div>
          </div>
        </section>

        <section id="product" className="lp-band lp-band-tight">
          <h2 className="lp-band-title">What it runs</h2>
          <ul className="lp-rows">
            <li>
              <h3>Federated training</h3>
              <p>
                Shared PyTorch models. Clients train locally; the coordinator
                averages weight deltas — never the raw training rows.
              </p>
            </li>
            <li>
              <h3>LoRA fine-tuning</h3>
              <p>
                Collaborative PEFT on registry LLMs. Lightweight adapters are
                aggregated, evaluated, and versioned on the coordinator.
              </p>
            </li>
            <li>
              <h3>Distributed jobs</h3>
              <p>
                Inference, labeling, allowlisted science compute. Prefer dataset
                aliases; text in <code>payload.inputs</code> leaves the node.
              </p>
            </li>
          </ul>
        </section>

        <section id="proof" className="lp-band">
          <h2 className="lp-band-title">The operator console</h2>
          <p className="lp-band-lead">
            Same product you open from this page — rounds, clients, LoRA, and
            jobs on the coordinator you are talking to.
          </p>
          <div className="lp-proof" aria-hidden="true">
            <div className="lp-proof-bar">
              <span className="lp-proof-dot" />
              fed-compute · Operator console
            </div>
            <div className="lp-proof-body">
              <aside className="lp-proof-side">
                <strong>Navigate</strong>
                Overview · Launch · Rounds · Clients · LoRA · Jobs
              </aside>
              <div className="lp-proof-main">
                <div className="lp-proof-statrow">
                  <div className="lp-proof-stat">
                    <span>Rounds</span>
                    <strong>{rounds ?? "—"}</strong>
                  </div>
                  <div className="lp-proof-stat">
                    <span>Online</span>
                    <strong>{onlineCount}</strong>
                  </div>
                  <div className="lp-proof-stat">
                    <span>Jobs done</span>
                    <strong>{jobsCompleted ?? "—"}</strong>
                  </div>
                </div>
                <div className="lp-proof-table">
                  Classic FL · <em>simple_mlp</em> · updates accepted when
                  clients train and submit · aggregate from Rounds
                </div>
              </div>
            </div>
          </div>
          <p className="lp-proof-note">
            Live numbers above mirror this API when reachable. Open the console
            for confirms, launch, and operator auth.
          </p>
          <div className="lp-actions">
            <button type="button" className="lp-btn solid" onClick={onEnter}>
              Open console
            </button>
          </div>
        </section>

        <section id="how" className="lp-band">
          <h2 className="lp-band-title">How a round moves</h2>
          <ol className="lp-flow">
            <li>
              <strong>Nodes hold data</strong>
              <span>Corpora and local paths stay on participants.</span>
            </li>
            <li>
              <strong>Coordinator orchestrates</strong>
              <span>Rounds, leases, auth, aggregation on this instance.</span>
            </li>
            <li>
              <strong>Artifacts return</strong>
              <span>Models, adapters, and results — versioned, not datasets.</span>
            </li>
          </ol>
        </section>

        <section id="trust" className="lp-band lp-trust">
          <h2 className="lp-band-title">Privacy without the fairy tale</h2>
          <p className="lp-band-lead">
            Federated learning alone does not guarantee privacy. Secure
            aggregation and differential privacy are not enabled in this build.
            Boundaries are listed — not implied.
          </p>
          <a className="lp-inline" href="#/privacy">
            Read the privacy model
          </a>
        </section>

        <section id="network" className="lp-band lp-live">
          <h2 className="lp-band-title">This coordinator</h2>
          <p className="lp-band-lead">
            Live figures for the API you are talking to. Presence on the globe
            is city-level with jitter when geo lookup is enabled.
          </p>
          <dl className="lp-metrics">
            <div>
              <dt>Online</dt>
              <dd>{onlineCount}</dd>
            </div>
            <div>
              <dt>Located (24h)</dt>
              <dd>{locatedCount}</dd>
            </div>
            <div>
              <dt>Rounds</dt>
              <dd>{rounds ?? "—"}</dd>
            </div>
            <div>
              <dt>Jobs queued</dt>
              <dd>{jobsQueued ?? "—"}</dd>
            </div>
            <div>
              <dt>Jobs done</dt>
              <dd>{jobsCompleted ?? "—"}</dd>
            </div>
          </dl>
          <p className="lp-fine">
            Disable geo with <code>GEO_LOOKUP_DISABLED</code>.{" "}
            <a href="#/status">Status page</a>
            {onlineCount === 0 && locatedCount === 0 && !rounds
              ? " — zeros are live empties, not placeholders."
              : "."}
          </p>
        </section>

        <section className="lp-band lp-end">
          <h2 className="lp-band-title">Start coordinating</h2>
          <p className="lp-band-lead">
            Launch the console, bring up nodes, run a round. Training rows stay
            home unless you put them in a job payload.
          </p>
          <button type="button" className="lp-btn solid lg" onClick={onEnter}>
            Open console
          </button>
        </section>
      </main>

      <footer className="lp-foot">
        <div>
          <strong>fed-compute</strong>
          <span>Private data. Shared progress.</span>
        </div>
        <nav aria-label="Footer">
          <a href="#/status">Status</a>
          <a href="#/privacy">Privacy</a>
          <button type="button" onClick={onEnter}>
            Console
          </button>
        </nav>
      </footer>
    </div>
  );
}
