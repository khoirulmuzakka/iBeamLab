"""Reproducible compound, shared-set and independent layerwise composition sampling."""
from __future__ import annotations
from dataclasses import dataclass
from math import fsum
import numpy as np
from numpy.typing import NDArray

METHODS = ("compound", "shared_random_elements", "random_elements")


@dataclass(frozen=True, slots=True)
class ConcentrationRegion:
    """Raw positive-weight interval, before within-layer normalization."""
    name: str
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ElementDistribution:
    """Region probabilities conditional on this element having been selected."""
    region_probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CompoundSampling:
    """Predefined sets shared across layers, with independent layer concentrations."""
    fraction: float
    subsets: tuple[tuple[str, tuple[str, ...]], ...]
    subset_allocation: str = "proportional_to_element_count"


@dataclass(frozen=True, slots=True)
class RandomElementSampling:
    """Inclusive uniform element-count range and selection without replacement."""
    fraction: float
    minimum_elements: int
    maximum_elements: int
    count_distribution: str = "uniform"
    selection: str = "uniform_without_replacement"


@dataclass(frozen=True, slots=True)
class ThicknessEnvelope:
    """Exponential layer envelopes, with a total limit depending on layer count."""
    minimum_total: float = 50_000.0
    maximum_total: float = 700_000.0
    growth_ratio: float = 1.6

    def maximum_for(self, layer_count, maximum_layers):
        if maximum_layers <= 1:
            return self.maximum_total
        return self.minimum_total + (layer_count - 1) / (maximum_layers - 1) * (self.maximum_total - self.minimum_total)


@dataclass(frozen=True, slots=True)
class SamplingBatch:
    """Physical arrays, shuffled component labels and method-specific diagnostics."""
    thicknesses: NDArray[np.float64]
    concentrations: NDArray[np.float64]
    components: tuple[str, ...]
    report: dict[str, object]


def _rng(master_seed, layer_count, stream):
    return np.random.default_rng(np.random.SeedSequence([master_seed, layer_count, stream]))


def _validate(regions, elements):
    if not regions or not elements:
        raise ValueError("at least one concentration region and element are required")
    previous, names = 0., set()
    for region in regions:
        if (not region.name or region.name in names or not 0 <= region.lower < region.upper <= 1
                or region.lower < previous):
            raise ValueError("regions require unique names and ordered non-overlapping intervals in [0,1]")
        previous = region.upper
        names.add(region.name)
    for symbol, distribution in elements.items():
        p = np.asarray(distribution.region_probabilities, dtype=float)
        if (not symbol or p.shape != (len(regions),) or not np.all(np.isfinite(p))
                or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError(f"region probabilities for {symbol} must be finite, nonnegative and sum to one")


def validate_methods(compound, shared_random_elements, random_elements, elements):
    methods = (compound, shared_random_elements, random_elements)
    fractions = [m.fraction for m in methods]
    if (any(not np.isfinite(f) or not 0 <= f <= 1 for f in fractions)
            or not np.isclose(fsum(fractions), 1, rtol=0, atol=1e-12)):
        raise ValueError("composition method fractions must be in [0,1] and sum to one")
    if compound.subset_allocation != "proportional_to_element_count":
        raise ValueError("compound.subset_allocation must be proportional_to_element_count")
    if compound.fraction and not compound.subsets:
        raise ValueError("compound.subsets are required for a positive compound fraction")
    names = set()
    for name, members in compound.subsets:
        if (not name or name in names or not members or len(set(members)) != len(members)
                or any(s not in elements for s in members)):
            raise ValueError("compound subsets require unique names and distinct configured elements")
        names.add(name)
    for name, method in zip(METHODS[1:], methods[1:]):
        if (type(method.minimum_elements) is not int or type(method.maximum_elements) is not int
                or not 1 <= method.minimum_elements <= method.maximum_elements <= len(elements)):
            raise ValueError(f"{name} element-count bounds must satisfy 1 <= minimum <= maximum <= {len(elements)}")
        if method.count_distribution != "uniform" or method.selection != "uniform_without_replacement":
            raise ValueError(f"{name} requires uniform counts and uniform_without_replacement selection")


def allocate_counts(total, weights):
    """Largest-remainder allocation preserves the exact budget; ties follow declaration order."""
    weights = np.asarray(weights, dtype=float)
    quotas = total * (weights / weights.sum())
    counts = np.floor(quotas).astype(np.int64)
    remainder = total - int(counts.sum())
    counts[np.argsort(-(quotas - counts), kind="stable")[:remainder]] += 1
    return counts


def method_counts(total, compound, shared_random_elements, random_elements):
    return dict(zip(METHODS, map(int, allocate_counts(total, [
        compound.fraction, shared_random_elements.fraction, random_elements.fraction]))))


def sample_thicknesses(count, layer_count, maximum_layers, policy, rng):
    maximum = policy.maximum_for(layer_count, maximum_layers)
    totals = rng.uniform(policy.minimum_total, maximum, count) if maximum > policy.minimum_total else np.full(count, maximum)
    weights = policy.growth_ratio ** np.arange(layer_count, dtype=np.float64)
    weights /= weights.sum()
    return rng.uniform(0., totals[:, None] * weights[None, :])


def _random_membership(count, n_elements, method, rng):
    """Choose a count first, then a uniform subset of that size for each row."""
    counts = rng.integers(method.minimum_elements, method.maximum_elements + 1, size=count)
    order = np.argsort(rng.random((count, n_elements)), axis=1)
    active = np.zeros((count, n_elements), dtype=bool)
    active[np.arange(count)[:, None], order] = np.arange(n_elements)[None, :] < counts[:, None]
    return active


def _fill_weights(output, active, regions, elements, rng):
    """Fill a preallocated layer slice. Selected entries stay positive; others stay zero."""
    for column, distribution in enumerate(elements.values()):
        rows = np.flatnonzero(active[:, column])
        choices = rng.choice(len(regions), size=len(rows), p=distribution.region_probabilities)
        for index, region in enumerate(regions):
            selected = rows[choices == index]
            # Prevent an exact zero endpoint from removing a selected element.
            output[selected, column] = np.maximum(
                rng.uniform(region.lower, region.upper, len(selected)), np.finfo(float).tiny)
    output /= output.sum(axis=1, keepdims=True)


def _composition_report(values, symbols):
    histogram = np.zeros(len(symbols) + 1, dtype=np.int64)
    selected = np.zeros(len(symbols), dtype=np.int64)
    for layer in range(values.shape[1]):
        active = values[:, layer, :] > 0
        histogram += np.bincount(active.sum(axis=1), minlength=len(histogram))
        selected += active.sum(axis=0)
    return {
        "samples": len(values),
        "element_count_histogram": {str(i): int(n) for i, n in enumerate(histogram) if n},
        "element_layer_occurrences": dict(zip(symbols, map(int, selected))),
    }


def sampling_method(component):
    if component.startswith("compound_"):
        return "compound"
    if component.startswith("pure_"):
        return "pure_element"
    return component


def sample_layer_system(*, layer_count, maximum_layers, mixed_count, pure_per_element,
        master_seed, thickness, regions, elements, compound, shared_random_elements, random_elements):
    """Mix three budgeted families and additional one-layer pure samples.

    Allocate physical arrays once; each method fills its own slice. Selection
    and concentration streams are separate. Only compounds and the shared-set
    method reuse element membership across layers.
    """
    _validate(regions, elements)
    validate_methods(compound, shared_random_elements, random_elements, elements)
    symbols = tuple(elements)
    counts = method_counts(mixed_count, compound, shared_random_elements, random_elements)
    pure_total = len(symbols) * pure_per_element if layer_count == 1 else 0
    total = mixed_count + pure_total
    thicknesses = np.empty((total, layer_count), dtype=np.float64)
    thicknesses[:mixed_count] = sample_thicknesses(mixed_count, layer_count, maximum_layers,
                                                  thickness, _rng(master_seed, layer_count, 1))
    values = np.zeros((total, layer_count, len(symbols)), dtype=np.float64)
    components, compound_counts, reports = [], {}, {}
    offset = 0
    if compound.subsets:
        subset_counts = allocate_counts(counts["compound"], [len(m) for _, m in compound.subsets])
        for index, ((name, members), count) in enumerate(zip(compound.subsets, subset_counts)):
            count = int(count)
            compound_counts[name] = count
            active = np.broadcast_to(np.isin(symbols, members), (count, len(symbols)))
            rng = _rng(master_seed, layer_count, 1000 + index)
            for layer in range(layer_count):
                _fill_weights(values[offset:offset + count, layer, :], active, regions, elements, rng)
            components.extend([f"compound_{name}"] * count)
            offset += count
    reports["compound"] = _composition_report(values[:offset], symbols)
    for method_index, (name, method) in enumerate(zip(METHODS[1:], (shared_random_elements, random_elements))):
        count = counts[name]
        selection_rng = _rng(master_seed, layer_count, 20 + method_index * 2)
        weight_rng = _rng(master_seed, layer_count, 21 + method_index * 2)
        if name == "shared_random_elements":
            active = _random_membership(count, len(symbols), method, selection_rng)
        for layer in range(layer_count):
            if name == "random_elements":
                active = _random_membership(count, len(symbols), method, selection_rng)
            _fill_weights(values[offset:offset + count, layer, :], active, regions, elements, weight_rng)
        reports[name] = _composition_report(values[offset:offset + count], symbols)
        components.extend([name] * count)
        offset += count
    pure_counts = {}
    if pure_total:
        maximum = thickness.maximum_for(1, maximum_layers)
        for index, symbol in enumerate(symbols):
            rng = _rng(master_seed, 1, 100 + index)
            edges = np.linspace(0., maximum, pure_per_element + 1)
            samples = rng.uniform(edges[:-1], edges[1:])
            rng.shuffle(samples)
            thicknesses[offset:offset + pure_per_element, 0] = samples
            values[offset:offset + pure_per_element, 0, index] = 1.
            components.extend([f"pure_{symbol}"] * pure_per_element)
            pure_counts[symbol] = pure_per_element
            offset += pure_per_element
    report = {
        "layer_count": layer_count, "mixed_samples": mixed_count,
        "method_samples": counts, "compound_samples": compound_counts,
        "pure_samples": pure_counts, "total_samples": total, "composition": reports,
        "thickness": {"minimum": float(thicknesses.min()), "maximum": float(thicknesses.max()),
                      "mean_total": float(thicknesses.sum(axis=1).mean())},
    }
    order = _rng(master_seed, layer_count, 999).permutation(total)
    return SamplingBatch(thicknesses[order], values[order],
                         tuple(np.asarray(components, dtype=object)[order]), report)
