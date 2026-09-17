"""Transparent Python-side sampling helpers.

Sampling is deliberately implemented in Python.  The native generator accepts
the finished parameter matrix and is responsible only for validation,
simulation, and dataset storage.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .generation import GenerationConfig, SpeciesConcentration


def _bounded_simplex(
    raw: np.ndarray, lower: np.ndarray, upper: np.ndarray, total: float
) -> np.ndarray:
    """Map positive weights onto a bounded simplex with the requested sum."""

    tolerance = 1e-12
    if total < float(lower.sum()) - tolerance or total > float(upper.sum()) + tolerance:
        raise ValueError("concentration bounds cannot represent the remaining layer fraction")

    values = lower.copy()
    weights = np.maximum(raw - lower, 0.0)
    remaining = total - float(values.sum())
    while remaining > tolerance:
        active = values < upper - tolerance
        if not np.any(active):
            raise ValueError("concentration upper bounds leave an unallocated fraction")
        active_weights = weights[active]
        if float(active_weights.sum()) <= tolerance:
            shares = np.full(active.sum(), remaining / int(active.sum()))
        else:
            shares = remaining * active_weights / float(active_weights.sum())
        additions = np.minimum(upper[active] - values[active], shares)
        allocated = float(additions.sum())
        if allocated <= tolerance:
            raise ValueError("failed to normalize layer concentrations")
        values[active] += additions
        remaining -= allocated
    return values


def normalize_layer_concentrations(
    config: GenerationConfig, parameter_rows: Sequence[Sequence[float]]
) -> np.ndarray:
    """Normalize open concentration columns independently for every layer."""

    rows = np.asarray(parameter_rows, dtype=np.float64).copy()
    if rows.ndim != 2:
        raise ValueError("parameter_rows must be a two-dimensional matrix")

    open_parameters = [parameter for parameter in config.parameters if parameter.fixed_value is None]
    if rows.shape[1] != len(open_parameters):
        raise ValueError("parameter row width does not match the open parameter count")

    groups: dict[int, list[tuple[int, object]]] = defaultdict(list)
    all_concentrations: dict[tuple[int, str], object] = {}
    for parameter in config.parameters:
        if isinstance(parameter.target, SpeciesConcentration):
            key = (parameter.target.layer, parameter.target.element)
            all_concentrations[key] = parameter
    for column, parameter in enumerate(open_parameters):
        if isinstance(parameter.target, SpeciesConcentration):
            groups[parameter.target.layer].append((column, parameter))

    for layer_index, entries in groups.items():
        fixed_sum = 0.0
        for species in config.sample.layers[layer_index].species:
            parameter = all_concentrations.get((layer_index, species.element))
            if parameter is None:
                fixed_sum += species.concentration
            elif parameter.fixed_value is not None:
                fixed_sum += parameter.fixed_value
        total = max(0.0, 1.0 - fixed_sum)
        columns = [column for column, _ in entries]
        lower = np.asarray([parameter.lower_bound for _, parameter in entries])
        upper = np.asarray([parameter.upper_bound for _, parameter in entries])
        for row in rows:
            row[columns] = _bounded_simplex(row[columns], lower, upper, total)
    return rows


def sample_uniform_parameters(
    config: GenerationConfig,
    sample_count: int,
    seed: int,
    *,
    normalize_concentrations: bool = True,
) -> np.ndarray:
    """Sample every open parameter uniformly within its declared bounds."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    config.validate()
    parameters = [parameter for parameter in config.parameters if parameter.fixed_value is None]
    rng = np.random.default_rng(seed)
    rows = (
        np.column_stack(
            [
                rng.uniform(parameter.lower_bound, parameter.upper_bound, sample_count)
                for parameter in parameters
            ]
        )
        if parameters
        else np.empty((sample_count, 0), dtype=np.float64)
    )
    if normalize_concentrations:
        rows = normalize_layer_concentrations(config, rows)
    return rows


def sampling_config_toml(
    *, strategy: str, seed: int, sample_count: int, concentration_normalization: bool
) -> str:
    """Serialize the common sampling provenance fields without a TOML dependency."""

    escaped = strategy.replace("\\", "\\\\").replace('"', '\\"')
    normalized = "true" if concentration_normalization else "false"
    return (
        f'strategy = "{escaped}"\n'
        f"seed = {seed}\n"
        f"sample_count = {sample_count}\n"
        f"concentration_normalization = {normalized}\n"
    )


__all__ = [
    "normalize_layer_concentrations",
    "sample_uniform_parameters",
    "sampling_config_toml",
]
