"""Reproducible sampling policies for dataset generation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True, slots=True)
class ConcentrationRegion:
    """Half-open raw-weight interval; the final region may include its upper bound."""
    name: str
    lower: float
    upper: float

@dataclass(frozen=True, slots=True)
class ElementDistribution:
    """Activation probability and raw-weight region probabilities for one element."""
    selection_probability: float
    region_probabilities: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class ThicknessEnvelope:
    """The legacy ibamlkit exponential layer-envelope policy."""
    minimum_total: float = 50_000.0
    maximum_total: float = 700_000.0
    growth_ratio: float = 1.6

    def maximum_for(self, layer_count: int, maximum_layers: int) -> float:
        if maximum_layers <= 1:
            return self.maximum_total
        fraction = (layer_count - 1) / (maximum_layers - 1)
        return self.minimum_total + fraction * (self.maximum_total - self.minimum_total)

@dataclass(frozen=True, slots=True)
class SamplingBatch:
    """Sampled thicknesses/compositions and an audit report."""
    thicknesses: NDArray[np.float64]
    concentrations: NDArray[np.float64]
    components: tuple[str, ...]
    report: dict[str, object]

def _rng(master_seed: int, layer_count: int, stream: int) -> np.random.Generator:
    """Create a stable stream unaffected by call ordering in other components."""
    return np.random.default_rng(np.random.SeedSequence([master_seed, layer_count, stream]))

def sample_thicknesses(count: int, layer_count: int, maximum_layers: int,
                       policy: ThicknessEnvelope, rng: np.random.Generator) -> NDArray[np.float64]:
    """Mirror ibamlkit: sample a total envelope, then values inside exponential sub-envelopes."""
    maximum = policy.maximum_for(layer_count, maximum_layers)
    totals = rng.uniform(policy.minimum_total, maximum, count) if maximum > policy.minimum_total else np.full(count, maximum)
    weights = policy.growth_ratio ** np.arange(layer_count, dtype=np.float64)
    weights /= weights.sum()
    envelopes = totals[:, None] * weights[None, :]
    return rng.uniform(0.0, envelopes)

def _validate(regions: Sequence[ConcentrationRegion], elements: Mapping[str, ElementDistribution]) -> None:
    if not regions:
        raise ValueError("at least one concentration region is required")
    previous = 0.0
    for region in regions:
        if not region.name or not 0 <= region.lower < region.upper <= 1:
            raise ValueError(f"invalid concentration region {region!r}")
        if region.lower < previous:
            raise ValueError("concentration regions must be ordered and non-overlapping")
        previous = region.upper
    if not elements:
        raise ValueError("at least one element distribution is required")
    for symbol, distribution in elements.items():
        if not symbol or not 0 <= distribution.selection_probability <= 1:
            raise ValueError(f"invalid selection probability for {symbol}")
        probabilities = np.asarray(distribution.region_probabilities, dtype=float)
        if len(probabilities) != len(regions) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
            raise ValueError(f"region probabilities for {symbol} must be non-negative and sum to one")
    if not any(x.selection_probability > 0 for x in elements.values()):
        raise ValueError("at least one element must be selectable")

def sample_regional_compositions(count: int, layer_count: int,
        regions: Sequence[ConcentrationRegion], elements: Mapping[str, ElementDistribution],
        rng: np.random.Generator) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Sample guided raw weights, set inactive elements to zero, and normalize each layer."""
    _validate(regions, elements)
    symbols = tuple(elements)
    n_elements = len(symbols)
    concentrations = np.zeros((count, layer_count, n_elements), dtype=np.float64)
    selected_counts = np.zeros(n_elements, dtype=np.int64)
    proposed_regions = np.zeros((n_elements, len(regions)), dtype=np.int64)
    final_regions = np.zeros_like(proposed_regions)
    for layer in range(layer_count):
        active = np.column_stack([
            rng.random(count) < elements[symbol].selection_probability for symbol in symbols
        ])
        empty = ~active.any(axis=1)
        while np.any(empty):
            for column, symbol in enumerate(symbols):
                active[empty, column] = rng.random(int(empty.sum())) < elements[symbol].selection_probability
            empty = ~active.any(axis=1)
        weights = np.zeros((count, n_elements), dtype=np.float64)
        for column, symbol in enumerate(symbols):
            rows = np.flatnonzero(active[:, column])
            selected_counts[column] += len(rows)
            if not len(rows):
                continue
            distribution = elements[symbol]
            choices = rng.choice(len(regions), size=len(rows), p=distribution.region_probabilities)
            for region_index, region in enumerate(regions):
                selected = rows[choices == region_index]
                proposed_regions[column, region_index] += len(selected)
                if len(selected):
                    weights[selected, column] = rng.uniform(region.lower, region.upper, len(selected))
        zero_weight = weights.sum(axis=1) == 0
        # This is only possible when all active draws hit an exact zero endpoint.
        while np.any(zero_weight):
            weights[zero_weight, np.argmax(active[zero_weight], axis=1)] = np.nextafter(0.0, 1.0)
            zero_weight = weights.sum(axis=1) == 0
        values = weights / weights.sum(axis=1, keepdims=True)
        concentrations[:, layer, :] = values
        for column in range(n_elements):
            for region_index, region in enumerate(regions):
                if region_index == len(regions) - 1:
                    mask = active[:, column] & (values[:, column] >= region.lower) & (values[:, column] <= region.upper)
                else:
                    mask = active[:, column] & (values[:, column] >= region.lower) & (values[:, column] < region.upper)
                final_regions[column, region_index] += int(mask.sum())
    report = {
        "selection_opportunities": count * layer_count,
        "selected": {symbol: int(selected_counts[i]) for i, symbol in enumerate(symbols)},
        "proposed_regions": {symbol: {r.name: int(proposed_regions[i, j]) for j, r in enumerate(regions)} for i, symbol in enumerate(symbols)},
        "final_regions": {symbol: {r.name: int(final_regions[i, j]) for j, r in enumerate(regions)} for i, symbol in enumerate(symbols)},
    }
    return concentrations, report

def sample_layer_system(*, layer_count: int, maximum_layers: int, mixed_count: int,
        pure_per_element: int, master_seed: int, thickness: ThicknessEnvelope,
        regions: Sequence[ConcentrationRegion], elements: Mapping[str, ElementDistribution]) -> SamplingBatch:
    """Build mixed rows plus the special pure-element subset for a one-layer system."""
    symbols = tuple(elements)
    mixed_thicknesses = sample_thicknesses(mixed_count, layer_count, maximum_layers, thickness,
                                           _rng(master_seed, layer_count, 1))
    mixed_compositions, composition_report = sample_regional_compositions(
        mixed_count, layer_count, regions, elements, _rng(master_seed, layer_count, 2))
    components = ["regional_mixture"] * mixed_count
    thickness_parts = [mixed_thicknesses]
    composition_parts = [mixed_compositions]
    pure_counts: dict[str, int] = {}
    if layer_count == 1 and pure_per_element:
        maximum = thickness.maximum_for(1, maximum_layers)
        for element_index, symbol in enumerate(symbols):
            rng = _rng(master_seed, 1, 100 + element_index)
            edges = np.linspace(0.0, maximum, pure_per_element + 1)
            values = rng.uniform(edges[:-1], edges[1:])
            rng.shuffle(values)
            pure_thickness = values[:, None]
            pure_composition = np.zeros((pure_per_element, 1, len(symbols)), dtype=np.float64)
            pure_composition[:, 0, element_index] = 1.0
            thickness_parts.append(pure_thickness)
            composition_parts.append(pure_composition)
            components.extend([f"pure_{symbol}"] * pure_per_element)
            pure_counts[symbol] = pure_per_element
    all_thicknesses = np.concatenate(thickness_parts)
    all_compositions = np.concatenate(composition_parts)
    order_rng = _rng(master_seed, layer_count, 999)
    order = order_rng.permutation(len(all_thicknesses))
    report: dict[str, object] = {
        "layer_count": layer_count,
        "mixed_samples": mixed_count,
        "pure_samples": pure_counts,
        "total_samples": int(len(order)),
        "composition": composition_report,
        "thickness": {
            "minimum": float(all_thicknesses.min()),
            "maximum": float(all_thicknesses.max()),
            "mean_total": float(all_thicknesses.sum(axis=1).mean()),
        },
    }
    return SamplingBatch(all_thicknesses[order], all_compositions[order],
                         tuple(np.asarray(components, dtype=object)[order]), report)

__all__ = ["ConcentrationRegion", "ElementDistribution", "SamplingBatch", "ThicknessEnvelope", "sample_layer_system", "sample_regional_compositions", "sample_thicknesses"]
