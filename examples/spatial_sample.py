"""Spatial/depth composition model for the simulated 10 mm NMC sample.

Coordinates ``x`` and ``y`` are in mm. Depth ``z`` uses the areal-density
units used by the generation examples and spans 0 to ``MAX_DEPTH``.
"""

from __future__ import annotations

import numpy as np


ELEMENTS = ("Li", "Ni", "Mn", "Co", "O", "Fe", "Cu", "Al", "Si")
MAX_DEPTH = 500_000.0
SAMPLE_SIZE_MM = 10.0
SUBLAYER_COUNT = 10
SUBLAYER_GROWTH_RATIO = 1.6

# Atomic fractions of stoichiometric LiNi0.8Mn0.1Co0.1O2.
NMC811 = np.array([0.25, 0.20, 0.025, 0.025, 0.50, 0.0, 0.0, 0.0, 0.0])

# (element, x, y, diameter, surface strength, penetration depth)
IMPURITY_SPOTS = (
    ("Fe", 0.95, 2.15, 1.0, 0.30, 50_000.0),
    ("Cu", 2.05, 1.85, 0.6, 0.25, 35_000.0),
    ("Al", 2.85, 1.68, 0.4, 0.20, 25_000.0),
    ("Si", 3.50, 1.55, 0.2, 0.15, 15_000.0),
)


def _surface_matrix_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return unnormalised surface weights for Li/Ni/Mn/Co/O."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))

    # Background NMC811 with a smooth, modest Li increase from left to right.
    li = 0.90 + 0.20 * np.clip(x / SAMPLE_SIZE_MM, 0.0, 1.0)
    ni = np.full_like(li, 0.8)
    mn = np.full_like(li, 0.1)
    co = np.full_like(li, 0.1)
    oxygen = np.full_like(li, 2.0)

    radius = np.hypot(x - 5.0, y - 5.0)
    central = (np.abs(x - 5.0) <= 1.5) & (np.abs(y - 5.0) <= 1.5)
    ring1 = (radius <= 2.10) & ~central
    ring2 = (radius > 2.10) & (radius <= 2.85)
    ring3 = (radius > 2.85) & (radius <= 3.55)

    li = np.where(ring3, 1.18, li)             # Li recovery/enrichment
    li = np.where(ring2, 1.00, li)
    ni, mn, co = (np.where(ring2, value, array) for value, array in
                  ((0.6, ni), (0.2, mn), (0.2, co)))  # NMC622-like ring
    li = np.where(ring1, 0.72, li)             # weak Li depletion
    li = np.where(central, 0.35, li)           # strong Li depletion
    oxygen = np.where(central, 1.90, oxygen)   # mild coupled oxygen loss

    zeros = np.zeros_like(li)
    return np.stack((li, ni, mn, co, oxygen, zeros, zeros, zeros, zeros), axis=-1)


def surface_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the unnormalised composition weights at the exposed surface."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    weights = _surface_matrix_weights(x, y)
    for element, cx, cy, diameter, strength, _ in IMPURITY_SPOTS:
        inside = np.hypot(x - cx, y - cy) <= diameter / 2.0
        weights[..., ELEMENTS.index(element)] = np.where(inside, strength, 0.0)
    return weights


def composition_weights(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate w_i(x, y, z) before concentration normalisation.

    Matrix elements follow ``d_i + (a_i-d_i)(1-z/Z)^2``, so the measured
    surface map is recovered at z=0 and deviations arrive smoothly at NMC811
    with zero slope at the maximum depth. Impurities use shallower compact
    quadratic profiles and are exactly zero below their penetration depths.
    """
    x, y, z = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    )
    if np.any((z < 0.0) | (z > MAX_DEPTH)):
        raise ValueError(f"z must lie between 0 and {MAX_DEPTH:g}")

    surface = surface_weights(x, y)
    # Convert the bulk fractions to the same arbitrary scale as the surface
    # formula-unit weights (Li + transition metals + two oxygen atoms = 4).
    bulk = NMC811 * 4.0
    u = z / MAX_DEPTH
    weights = bulk + (surface - bulk) * (1.0 - u[..., None]) ** 2

    for element, _, _, _, _, depth in IMPURITY_SPOTS:
        index = ELEMENTS.index(element)
        decay = np.maximum(1.0 - z / depth, 0.0) ** 2
        weights[..., index] = surface[..., index] * decay
    return weights


def composition(x: np.ndarray, y: np.ndarray, z: np.ndarray = 0.0) -> np.ndarray:
    """Return normalized atomic fractions n_i(x, y, z), ordered by ELEMENTS."""
    weights = np.clip(composition_weights(x, y, z), 0.0, None)
    return weights / weights.sum(axis=-1, keepdims=True)


def scan_grid(points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return an edge-inclusive square measurement grid in millimetres."""
    axis = np.linspace(0.0, SAMPLE_SIZE_MM, points)
    return np.meshgrid(axis, axis, indexing="xy")


def sublayer_thicknesses(
    count: int = SUBLAYER_COUNT,
    total: float = MAX_DEPTH,
    growth_ratio: float = SUBLAYER_GROWTH_RATIO,
) -> np.ndarray:
    """Return geometrically increasing layer thicknesses summing to ``total``."""
    if count < 1 or total <= 0.0 or growth_ratio <= 1.0:
        raise ValueError("count and total must be positive and growth_ratio must exceed one")
    thicknesses = growth_ratio ** np.arange(count, dtype=float)
    thicknesses *= total / thicknesses.sum()
    # Make the stated total exact despite floating-point summation.
    thicknesses[-1] += total - float(thicknesses.sum())
    return thicknesses


def sublayer_compositions(
    x: float,
    y: float,
    thicknesses: np.ndarray | None = None,
) -> np.ndarray:
    """Return one representative composition per sublayer.

    The continuous depth function is sampled at each layer's areal-density
    midpoint. With the geometric grid this gives the finest resolution where
    the surface impurity profiles change most rapidly.
    """
    values = sublayer_thicknesses() if thicknesses is None else np.asarray(thicknesses, float)
    if values.ndim != 1 or np.any(values <= 0.0):
        raise ValueError("thicknesses must be a one-dimensional positive array")
    edges = np.concatenate(([0.0], np.cumsum(values)))
    if not np.isclose(edges[-1], MAX_DEPTH):
        raise ValueError(f"thicknesses must sum to {MAX_DEPTH:g}")
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    return composition(x, y, midpoints)
