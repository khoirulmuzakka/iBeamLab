"""Versioned TOML configuration for reproducible dataset generation."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import csv
import platform
import sys
from typing import Any, Mapping
import numpy as np

from .exceptions import ConfigurationError
from .generation import FailurePolicy, GenerationStudy
from .sample import Beam, Detector, Experiment, Layer, LinearCalibration, Sample
from .sampling import ConcentrationRegion, ElementDistribution, ThicknessEnvelope, sample_layer_system
from .simulator import SimnraMethod, SimnraSimulator

def _toml_reader():
    try:
        import tomllib
        return tomllib
    except ImportError as error:
        try:
            import tomli
            return tomli
        except ImportError:
            raise ConfigurationError("TOML support requires tomli on Python 3.10") from error

def _toml_writer():
    try:
        import tomli_w
        return tomli_w
    except ImportError as error:
        raise ConfigurationError("Writing generation metadata requires tomli-w") from error

def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{location}: expected a mapping")
    return value

def _number(value: Any, location: str) -> float:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ConfigurationError(f"{location}: expected a finite number")
    return float(value)

def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigurationError(f"{location}: expected an integer >= {minimum}")
    return value

def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

@dataclass(frozen=True, slots=True)
class SetupVariation:
    target: str
    detector: str
    distribution: str
    bounds: tuple[float, float]
    unit: str = ""

@dataclass(frozen=True, slots=True)
class GenerationConfiguration:
    """Validated, resolved generation configuration loaded from TOML."""
    source: Path
    raw: dict[str, Any]
    name: str
    output: Path
    maximum_layers: int
    mixed_samples: int
    pure_samples_per_element: int
    seed: int
    elements: dict[str, ElementDistribution]
    regions: tuple[ConcentrationRegion, ...]
    thickness: ThicknessEnvelope
    detectors: tuple[Detector, ...]
    methods: tuple[SimnraMethod, ...]
    variations: tuple[SetupVariation, ...]
    workers: int
    batch_size: int
    shards: int
    failure_policy: FailurePolicy

    def experiment(self, layer_count: int) -> Experiment:
        composition = {symbol: 1.0 / len(self.elements) for symbol in self.elements}
        total = self.thickness.maximum_for(layer_count, self.maximum_layers)
        return Experiment(Sample([Layer(total / layer_count, composition) for _ in range(layer_count)]), self.detectors)

    def study(self, layer_count: int) -> GenerationStudy:
        from .generation import vary
        parameters = []
        for layer in range(layer_count):
            parameters.append(vary.layer_thickness(layer=layer, bounds=(0, self.thickness.maximum_total)))
            parameters.extend(vary.concentration(symbol, layer=layer, bounds=(0, 1)) for symbol in self.elements)
        factories = {
            "beam_energy": vary.beam_energy,
            "beam_spread": vary.beam_spread,
            "detector_resolution": vary.detector_resolution,
            "particles_sr": vary.particles_sr,
        }
        for item in self.variations:
            parameters.append(factories[item.target](item.detector, bounds=item.bounds))
        return GenerationStudy(self.experiment(layer_count), parameters, methods=self.methods)

    def rows(self, layer_count: int):
        batch = sample_layer_system(layer_count=layer_count, maximum_layers=self.maximum_layers,
            mixed_count=self.mixed_samples, pure_per_element=self.pure_samples_per_element,
            master_seed=self.seed, thickness=self.thickness, regions=self.regions, elements=self.elements)
        pieces = []
        for layer in range(layer_count):
            pieces.append(batch.thicknesses[:, layer:layer + 1])
            pieces.append(batch.concentrations[:, layer, :])
        rows = np.concatenate(pieces, axis=1)
        if self.variations:
            setup_rng = np.random.default_rng(np.random.SeedSequence([self.seed, layer_count, 3]))
            setup_columns = []
            for item in self.variations:
                low, high = item.bounds
                if item.distribution == "uniform":
                    values = setup_rng.uniform(low, high, len(rows))
                elif item.distribution == "log_uniform":
                    if low <= 0:
                        raise ConfigurationError("log_uniform setup bounds must be positive")
                    values = np.exp(setup_rng.uniform(np.log(low), np.log(high), len(rows)))
                else:
                    raise ConfigurationError(f"unsupported setup distribution: {item.distribution}")
                setup_columns.append(values)
            rows = np.column_stack([rows, *setup_columns])
        return rows, batch

    def resolved(self) -> dict[str, Any]:
        try:
            package_version = version("ibeamlab")
        except PackageNotFoundError:
            package_version = "0.1.0.dev0"
        methods = [{"label": x.label, "reference_file": str(x.reference_file),
                    "reference_file_sha256": _hash(x.reference_file)} for x in self.methods]
        return {
            **self.raw,
            "format": "ibeamlab.generation-resolved",
            "source_configuration": str(self.source),
            "source_configuration_sha256": _hash(self.source),
            "software": {"ibeamlab_version": package_version, "python_version": platform.python_version(),
                         "numpy_version": np.__version__, "platform": platform.platform()},
            "resolved": {"output_directory": str(self.output), "methods": methods,
                         "seed_streams": "SeedSequence([master_seed, layer_count, stream_id])"},
        }

    def validate_files(self) -> None:
        for method in self.methods:
            if not method.reference_file.is_file():
                raise ConfigurationError(f"simulation.methods[{method.label}].reference_file does not exist: {method.reference_file}")

    def run(self, *, progress=None) -> dict[str, Any]:
        """Generate every configured layer-count dataset and write TOML audit artifacts."""
        tomli_w = _toml_writer()
        self.validate_files()
        if self.output.exists():
            raise ConfigurationError(f"output directory already exists: {self.output}")
        self.output.mkdir(parents=True)
        (self.output / "generation.toml").write_text(self.source.read_text(encoding="utf-8"), encoding="utf-8")
        (self.output / "generation.resolved.toml").write_text(
            tomli_w.dumps(self.resolved()), encoding="utf-8")
        summaries: dict[str, Any] = {}
        total_requested = total_accepted = total_invalid = total_failed = 0
        complete = False
        try:
            with SimnraSimulator(self.methods, workers=self.workers) as simulator:
                for layer_count in range(1, self.maximum_layers + 1):
                    rows, batch = self.rows(layer_count)
                    label = f"layers_{layer_count:02d}"
                    summary = self.study(layer_count).generate(self.output / label, simulator=simulator,
                        samples=rows, seed=self.seed, batch_size=self.batch_size, shards=self.shards,
                        failure_policy=self.failure_policy, progress=progress)
                    with (summary.path / "sampling-components.csv").open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("sample_index", "sampling_component"))
                        writer.writerows(enumerate(batch.components))
                    audit = {**batch.report, "components": {
                        component: batch.components.count(component) for component in sorted(set(batch.components))}}
                    (summary.path / "sampling-summary.toml").write_text(
                        tomli_w.dumps(audit), encoding="utf-8")
                    summaries[label] = {"requested": summary.requested, "accepted": summary.accepted,
                        "invalid": summary.invalid, "failed": summary.failed, "cancelled": summary.cancelled}
                    total_requested += summary.requested
                    total_accepted += summary.accepted
                    total_invalid += summary.invalid
                    total_failed += summary.failed
            complete = True
            return summaries
        finally:
            result = {"complete": complete, "requested": total_requested, "accepted": total_accepted,
                      "invalid": total_invalid, "failed": total_failed, "datasets": summaries}
            (self.output / "generation-summary.toml").write_text(
                tomli_w.dumps(result), encoding="utf-8")

def load_generation_configuration(path: str | Path) -> GenerationConfiguration:
    """Load, validate, and resolve a version-1 generation TOML file."""
    tomllib = _toml_reader()
    source = Path(path).resolve()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
    except Exception as error:
        raise ConfigurationError(f"cannot read configuration {source}: {error}") from error
    root = _mapping(raw, "configuration")
    if root.get("format") != "ibeamlab.generation" or root.get("format_version") != 1:
        raise ConfigurationError("configuration requires format=ibeamlab.generation and format_version=1")
    dataset = _mapping(root.get("dataset"), "dataset")
    random = _mapping(root.get("random"), "random")
    output_section = _mapping(root.get("output"), "output")
    composition = _mapping(root.get("composition_sampling"), "composition_sampling")
    thickness_data = _mapping(root.get("thickness_sampling"), "thickness_sampling")
    simulation = _mapping(root.get("simulation"), "simulation")
    if composition.get("strategy") != "regional_weights" or composition.get("normalization") != "sum_to_one":
        raise ConfigurationError("composition_sampling requires strategy=regional_weights and normalization=sum_to_one")
    if thickness_data.get("strategy") != "exponential_envelope":
        raise ConfigurationError("thickness_sampling.strategy must be exponential_envelope")
    if simulation.get("backend") != "simnra":
        raise ConfigurationError("simulation.backend must currently be simnra")
    if random.get("algorithm", "PCG64") != "PCG64":
        raise ConfigurationError("random.algorithm must currently be PCG64")
    regions = tuple(ConcentrationRegion(str(x["name"]), _number(x["lower"], "regions.lower"),
                    _number(x["upper"], "regions.upper")) for x in composition.get("regions", []))
    elements: dict[str, ElementDistribution] = {}
    for index, value in enumerate(root.get("elements", [])):
        item = _mapping(value, f"elements[{index}]")
        symbol = str(item.get("symbol", ""))
        if symbol in elements:
            raise ConfigurationError(f"elements[{index}].symbol: duplicate {symbol}")
        probabilities = tuple(float(x) for x in item.get("region_probabilities", []))
        elements[symbol] = ElementDistribution(_number(item.get("selection_probability"),
            f"elements[{index}].selection_probability"), probabilities)
    thickness = ThicknessEnvelope(_number(thickness_data.get("minimum_total_envelope"), "thickness_sampling.minimum_total_envelope"),
        _number(thickness_data.get("maximum_total_envelope"), "thickness_sampling.maximum_total_envelope"),
        _number(thickness_data.get("growth_ratio"), "thickness_sampling.growth_ratio"))
    if thickness.minimum_total < 0 or thickness.maximum_total < thickness.minimum_total or thickness.growth_ratio <= 0:
        raise ConfigurationError("thickness_sampling: invalid envelope bounds or growth ratio")
    experiment = _mapping(root.get("experiment"), "experiment")
    detectors = []
    for index, value in enumerate(experiment.get("detectors", [])):
        item = _mapping(value, f"experiment.detectors[{index}]")
        beam_data = _mapping(item.get("beam"), f"experiment.detectors[{index}].beam")
        calibration = _mapping(item.get("calibration", {}), f"experiment.detectors[{index}].calibration")
        detectors.append(Detector(str(item["label"]), Beam(str(beam_data["particle"]),
            _number(beam_data["energy"], "beam.energy"), _number(beam_data.get("spread", 0), "beam.spread")),
            LinearCalibration(_number(calibration.get("linear", 1), "calibration.linear"),
                _number(calibration.get("offset", 0), "calibration.offset"),
                _number(calibration.get("quadratic", 0), "calibration.quadratic")),
            resolution=_number(item.get("resolution", 0), "detector.resolution"),
            particles_sr=_number(item.get("particles_sr", 0), "detector.particles_sr"),
            real_time=_number(item.get("real_time", 0), "detector.real_time"),
            live_time=_number(item.get("live_time", 0), "detector.live_time")))
    methods = []
    for index, value in enumerate(simulation.get("methods", [])):
        item = _mapping(value, f"simulation.methods[{index}]")
        reference = Path(item["reference_file"])
        if not reference.is_absolute():
            reference = (source.parent / reference).resolve()
        methods.append(SimnraMethod(str(item["label"]), reference))
    variations = []
    vary = _mapping(root.get("vary", {}), "vary")
    for index, value in enumerate(vary.get("setup_parameters", [])):
        item = _mapping(value, f"vary.setup_parameters[{index}]")
        bounds = item.get("bounds", [])
        if len(bounds) != 2:
            raise ConfigurationError(f"vary.setup_parameters[{index}].bounds: expected two values")
        variations.append(SetupVariation(str(item["target"]), str(item["detector"]),
            str(item.get("distribution", "uniform")), (float(bounds[0]), float(bounds[1])), str(item.get("unit", ""))))
    policies = {"stop": FailurePolicy.STOP, "record": FailurePolicy.RECORD, "discard": FailurePolicy.DISCARD}
    try:
        failure_policy = policies[str(simulation.get("failure_policy", "stop"))]
    except KeyError as error:
        raise ConfigurationError("simulation.failure_policy must be stop, record, or discard") from error
    one_layer = _mapping(root.get("one_layer_sampling", {}), "one_layer_sampling")
    pure = _mapping(one_layer.get("pure_element_samples", {}), "one_layer_sampling.pure_element_samples")
    maximum_layers = _integer(dataset.get("maximum_layers"), "dataset.maximum_layers", 1)
    mixed_samples = _integer(dataset.get("mixed_samples_per_layer_count"), "dataset.mixed_samples_per_layer_count", 1)
    pure_count = _integer(pure.get("samples_per_element", 0) if pure.get("enabled", True) else 0,
                          "one_layer_sampling.pure_element_samples.samples_per_element")
    output = Path(output_section.get("directory", ""))
    if not output.is_absolute():
        output = (source.parent / output).resolve()
    allowed_targets = {"beam_energy", "beam_spread", "detector_resolution", "particles_sr"}
    labels = {x.label for x in detectors}
    for item in variations:
        if item.target not in allowed_targets or item.detector not in labels or item.bounds[0] > item.bounds[1]:
            raise ConfigurationError(f"invalid setup variation: {item}")
    config = GenerationConfiguration(source, dict(root), str(root.get("name", source.stem)), output,
        maximum_layers, mixed_samples, pure_count, _integer(random.get("seed", 0), "random.seed"),
        elements, regions, thickness, tuple(detectors), tuple(methods), tuple(variations),
        _integer(simulation.get("workers", 1), "simulation.workers", 1),
        _integer(simulation.get("batch_size", 1), "simulation.batch_size", 1),
        _integer(output_section.get("shards", 1), "output.shards", 1), failure_policy)
    # Reuse the sampler's detailed region/element validation without generating rows.
    from .sampling import _validate
    try:
        _validate(config.regions, config.elements)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    if not detectors or {x.label for x in methods} != labels:
        raise ConfigurationError("simulation method labels must exactly match detector labels")
    return config

__all__ = ["GenerationConfiguration", "SetupVariation", "load_generation_configuration"]
