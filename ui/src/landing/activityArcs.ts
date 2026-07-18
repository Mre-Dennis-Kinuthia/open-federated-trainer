import type { ActivityNode } from "../api";
import type { Position } from "../components/Globe";

/**
 * Arcs are built from real coordinator presence data: every node is a client
 * that recently contacted this coordinator. Locations arrive already
 * anonymized (city-level + per-client jitter) — see /dashboard/activity.
 */

const ONLINE_COLOR = "#0070f3";
const RECENT_COLOR = "#6366f1";

/** Build globe arcs connecting recently active nodes to each other. */
export function buildActivityArcs(nodes: ActivityNode[]): Position[] {
  if (nodes.length === 0) return [];

  if (nodes.length === 1) {
    // A single node still deserves visible activity: short local arcs.
    const n = nodes[0];
    return [0, 1, 2].map((i) => ({
      order: i + 1,
      startLat: n.lat,
      startLng: n.lng,
      endLat: n.lat + [6, -5, 3][i],
      endLng: n.lng + [8, 6, -9][i],
      arcAlt: 0.15,
      color: n.online ? ONLINE_COLOR : RECENT_COLOR,
    }));
  }

  const arcs: Position[] = [];
  const limited = nodes.slice(0, 60);
  for (let i = 0; i < limited.length; i++) {
    const from = limited[i];
    // Two hops per node keeps density reasonable and the pairing stable.
    for (const hop of [1, 2]) {
      const to = limited[(i + hop) % limited.length];
      if (to === from) continue;
      const distance =
        Math.abs(from.lat - to.lat) + Math.abs(from.lng - to.lng);
      arcs.push({
        order: (i % 14) + 1,
        startLat: from.lat,
        startLng: from.lng,
        endLat: to.lat,
        endLng: to.lng,
        // Altitude scales with distance so nearby nodes get shallow arcs.
        arcAlt: Math.min(0.6, 0.05 + distance / 250),
        color: from.online && to.online ? ONLINE_COLOR : RECENT_COLOR,
      });
    }
  }
  return arcs;
}
