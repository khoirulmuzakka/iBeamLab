"""Declarative parameter studies and native dataset generation."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .exceptions import DatasetError, ValidationError
from .sample import Experiment, _to_native_sample, _to_native_setup
from .simulator import SimnraMethod, Simulator

@dataclass(frozen=True, slots=True)
class Parameter:
    """One varied physical quantity and its inclusive sampling bounds."""
    kind: str
    name: str
    lower: float
    upper: float
    layer: int | None = None
    element: str | None = None
    detector: str | None = None
    unit: str = ""
    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("parameter name cannot be empty")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper) or self.lower > self.upper:
            raise ValidationError(f"invalid bounds for parameter {self.name!r}")

class _Vary:
    @staticmethod
    def concentration(element: str, *, layer: int, bounds: tuple[float, float], name: str | None = None) -> Parameter:
        """Vary an element's atomic fraction in a zero-based layer."""
        return Parameter("concentration", name or f"layer_{layer}_{element}_concentration",
                         *bounds, layer=layer, element=element, unit="fraction")
    @staticmethod
    def layer_thickness(*, layer: int, bounds: tuple[float, float], name: str | None = None) -> Parameter:
        """Vary layer thickness in ``1e15 atoms/cm2``."""
        return Parameter("layer_thickness", name or f"layer_{layer}_thickness", *bounds,
                         layer=layer, unit="1e15 atoms/cm2")
    @staticmethod
    def beam_energy(detector: str, *, bounds: tuple[float, float], name: str | None = None) -> Parameter:
        return Parameter("beam_energy", name or f"{detector}_beam_energy", *bounds, detector=detector, unit="keV")
    @staticmethod
    def beam_spread(detector: str, *, bounds: tuple[float, float], name: str | None = None) -> Parameter:
        return Parameter("beam_spread", name or f"{detector}_beam_spread", *bounds, detector=detector, unit="keV")
    @staticmethod
    def detector_resolution(detector: str, *, bounds: tuple[float, float], name: str | None = None) -> Parameter:
        return Parameter("detector_resolution", name or f"{detector}_resolution", *bounds, detector=detector, unit="keV")
    @staticmethod
    def particles_sr(detector: str, *, bounds: tuple[float, float], name: str | None = None) -> Parameter:
        return Parameter("particles_sr", name or f"{detector}_particles_sr", *bounds, detector=detector)

vary = _Vary()

class FailurePolicy(Enum):
    STOP = "stop"
    RECORD = "record"
    DISCARD = "discard"

@dataclass(frozen=True, slots=True)
class GenerationProgress:
    attempted: int
    accepted: int
    invalid: int
    failed: int
    total: int

@dataclass(frozen=True, slots=True)
class GenerationSummary:
    requested: int
    accepted: int
    invalid: int
    failed: int
    cancelled: bool
    path: Path

def _native_parameter(parameter: Parameter):
    from ._native import native
    targets = {
        "concentration": native.generation.SpeciesConcentration,
        "layer_thickness": native.generation.LayerThickness,
        "beam_energy": native.generation.BeamEnergy,
        "beam_spread": native.generation.BeamSpread,
        "detector_resolution": native.generation.DetectorResolution,
        "particles_sr": native.generation.ParticlesSr,
    }
    try:
        target = targets[parameter.kind]()
    except KeyError as error:
        raise ValidationError(f"unsupported parameter kind: {parameter.kind}") from error
    if parameter.layer is not None:
        target.layer = parameter.layer
    if parameter.element is not None:
        target.element = parameter.element
    if parameter.detector is not None:
        target.detector = parameter.detector
    value = native.generation.ParameterSpec()
    value.name, value.target = parameter.name, target
    value.lower_bound, value.upper_bound = parameter.lower, parameter.upper
    value.unit = parameter.unit
    return value

class GenerationStudy:
    """A reusable experiment schema and the quantities varied within it."""
    def __init__(self, experiment: Experiment, parameters: Sequence[Parameter], *,
                 methods: Sequence[SimnraMethod] = ()):
        self.experiment = experiment
        self.parameters = tuple(parameters)
        self.methods = tuple(methods)
        names = [x.name for x in self.parameters]
        if len(names) != len(set(names)):
            raise ValidationError("parameter names must be unique")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self.parameters)

    def sample(self, count: int, *, seed: int | None = None,
               normalize_concentrations: bool = True) -> NDArray[np.float64]:
        """Uniformly sample all bounds, optionally normalizing concentration groups."""
        if count <= 0:
            raise ValueError("count must be positive")
        rng = np.random.default_rng(seed)
        rows = (np.column_stack([rng.uniform(x.lower, x.upper, count) for x in self.parameters])
                if self.parameters else np.empty((count, 0), dtype=np.float64))
        if normalize_concentrations:
            groups: dict[int, list[int]] = {}
            for index, parameter in enumerate(self.parameters):
                if parameter.kind == "concentration":
                    groups.setdefault(parameter.layer or 0, []).append(index)
            for layer, columns in groups.items():
                varied = {self.parameters[i].element for i in columns}
                fixed = sum(x.concentration for x in self.experiment.sample.layers[layer].species
                            if x.element not in varied)
                target = 1.0 - fixed
                totals = rows[:, columns].sum(axis=1)
                if target < 0 or np.any(totals <= 0):
                    raise ValidationError(f"concentrations in layer {layer} cannot be normalized")
                rows[:, columns] *= (target / totals)[:, None]
                for i in columns:
                    p = self.parameters[i]
                    if np.any(rows[:, i] < p.lower - 1e-12) or np.any(rows[:, i] > p.upper + 1e-12):
                        raise ValidationError("normalization violates concentration bounds; adjust the bounds")
        return rows

    def _config(self):
        from ._native import native
        config = native.generation.GenerationConfig()
        config.sample = _to_native_sample(self.experiment.sample)
        config.setup = _to_native_setup(self.experiment.detectors)
        methods = []
        configured_methods = self.methods or tuple(
            SimnraMethod(detector.label, Path("not-applicable.xnra"))
            for detector in self.experiment.detectors
        )
        for method in configured_methods:
            value = native.generation.MethodConfig()
            value.label, value.iba_method = method.label, method.label
            value.reference_file = method.reference_file
            methods.append(value)
        config.methods = methods
        config.parameters = [_native_parameter(x) for x in self.parameters]
        config.validate()
        return config

    def generate(self, path: str | Path, *, simulator: Simulator,
                 samples: int | ArrayLike, seed: int = 0, batch_size: int = 1024,
                 shards: int = 1, failure_policy: FailurePolicy = FailurePolicy.STOP,
                 progress: Callable[[GenerationProgress], None] | None = None) -> GenerationSummary:
        """Generate a native dataset from a count or explicit parameter matrix."""
        from ._native import native
        output = Path(path)
        rows = self.sample(samples, seed=seed) if isinstance(samples, int) else np.asarray(samples, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.parameters):
            raise ValidationError(f"samples must have shape (n, {len(self.parameters)})")
        options = native.generation.GenerationOptions()
        options.batch_size, options.shard_count, options.seed = batch_size, shards, seed
        options.sampler, options.sampler_version = "python-uniform", 1
        policies = {FailurePolicy.STOP: native.generation.FailurePolicy.STOP,
                    FailurePolicy.RECORD: native.generation.FailurePolicy.RECORD,
                    FailurePolicy.DISCARD: native.generation.FailurePolicy.DISCARD}
        options.failure_policy = policies[failure_policy]
        callback = None
        if progress:
            callback = lambda x: progress(GenerationProgress(x.attempted, x.accepted, x.invalid, x.failed, x.total))
        try:
            raw = native.generation.DataGenerator(self._config(), simulator._native).generate(
                output, rows.tolist(), options, callback)
        except Exception as error:
            raise DatasetError(str(error)) from error
        return GenerationSummary(raw.requested, raw.accepted, raw.invalid, raw.failed, raw.cancelled, output)

__all__ = ["FailurePolicy", "GenerationProgress", "GenerationStudy", "GenerationSummary", "Parameter", "vary"]
