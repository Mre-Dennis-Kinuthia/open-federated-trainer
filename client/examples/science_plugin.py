"""Example real scientific compute plugin: Lennard-Jones particle dynamics.

Enable it on a worker with:
  COMPUTE_PLUGIN_ALLOWLIST=examples.science_plugin
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def lennard_jones(work_unit: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate a small particle system using velocity Verlet.

    Work unit fields:
      positions: N x 3 coordinates (required)
      velocities: N x 3 coordinates (optional, defaults to zero)
      steps, dt, epsilon, sigma, mass: simulation parameters
    """
    positions = np.asarray(work_unit.get("positions"), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must be an N x 3 array with N >= 2")
    if not np.isfinite(positions).all():
        raise ValueError("positions contain non-finite values")

    velocities = np.asarray(
        work_unit.get("velocities", np.zeros_like(positions)),
        dtype=np.float64,
    )
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions")

    steps = max(1, min(int(work_unit.get("steps", 100)), 1_000_000))
    dt = float(work_unit.get("dt", 0.001))
    epsilon = float(work_unit.get("epsilon", 1.0))
    sigma = float(work_unit.get("sigma", 1.0))
    mass = float(work_unit.get("mass", 1.0))
    if dt <= 0 or epsilon <= 0 or sigma <= 0 or mass <= 0:
        raise ValueError("dt, epsilon, sigma, and mass must be positive")

    def forces_and_energy(points: np.ndarray):
        forces = np.zeros_like(points)
        energy = 0.0
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                delta = points[i] - points[j]
                radius_squared = float(np.dot(delta, delta))
                if radius_squared < 1e-16:
                    raise ValueError("particles overlap")
                inverse_r2 = (sigma * sigma) / radius_squared
                inverse_r6 = inverse_r2**3
                inverse_r12 = inverse_r6**2
                energy += 4.0 * epsilon * (inverse_r12 - inverse_r6)
                scalar = (
                    24.0
                    * epsilon
                    * (2.0 * inverse_r12 - inverse_r6)
                    / radius_squared
                )
                pair_force = scalar * delta
                forces[i] += pair_force
                forces[j] -= pair_force
        return forces, energy

    forces, potential = forces_and_energy(positions)
    for _ in range(steps):
        velocities += 0.5 * dt * forces / mass
        positions += dt * velocities
        forces, potential = forces_and_energy(positions)
        velocities += 0.5 * dt * forces / mass

    kinetic = 0.5 * mass * float(np.sum(velocities**2))
    return {
        "steps": steps,
        "particle_count": len(positions),
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "potential_energy": potential,
        "kinetic_energy": kinetic,
        "total_energy": potential + kinetic,
    }
