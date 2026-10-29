"""Example-specific TOML configuration and multilayer generation orchestration."""
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import csv
import json
import platform
import numpy as np
import ibeamlab as ibl
from .sampling import (
    ConcentrationRegion, ElementDistribution, ThicknessEnvelope,
    CompoundSampling, RandomElementSampling, sample_layer_system, _validate,
    validate_methods, method_counts, sampling_method,
)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'unbounded'}]")
    return value


def _hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Configuration:
    """Resolved sampling policy; relative TOML paths are based on the TOML directory."""
    source: Path
    name: str
    output: Path
    maximum_layers: int
    mixed_samples: int
    pure_samples: int
    seed: int
    elements: dict
    regions: tuple
    thickness: ThicknessEnvelope
    compound: CompoundSampling
    shared_random_elements: RandomElementSampling
    random_elements: RandomElementSampling
    setup: ibl.ExperimentalSetup
    methods: tuple
    backend: str
    workers: int
    channels: int
    records_per_shard: int
    failure_policy: ibl.FailurePolicy
    multithreaded_apartment: bool = True
    batch_size: int = 512
    fast_calculation: bool = True

    @property
    def total_samples(self):
        return self.maximum_layers * self.mixed_samples + len(self.elements) * self.pure_samples

    def validate_files(self):
        if self.backend == "simnra":
            particles = {m.label: m.beam.particle for m in self.setup.measurements}
            if len(self.methods) != len(particles) or {m.measurement: m.particle for m in self.methods} != particles:
                raise ValueError("SIMNRA methods must match each measurement label and particle exactly once")
            for method in self.methods:
                if not method.reference.is_file():
                    raise ValueError(f"missing SIMNRA reference for {method.measurement}: {method.reference}")

    def plan(self, layer_count):
        """Layer-major columns: thickness, then each element's atomic fraction."""
        composition = dict.fromkeys(self.elements, 1 / len(self.elements))
        thickness = self.thickness.maximum_for(layer_count, self.maximum_layers) / layer_count
        reference = ibl.Experiment(ibl.Sample([
            ibl.Layer(thickness, composition) for _ in range(layer_count)
        ]), self.setup)
        parameters = []
        for layer in range(layer_count):
            parameters.append(ibl.vary.layer_thickness(layer=layer, bounds=(0, self.thickness.maximum_total)))
            parameters.extend(ibl.vary.concentration(symbol, layer=layer, bounds=(0, 1)) for symbol in self.elements)
        return ibl.GenerationPlan(reference, parameters)

    def rows(self, layer_count):
        """Materialize one layer-count system at a time, preserving named RNG streams."""
        batch = sample_layer_system(
            layer_count=layer_count, maximum_layers=self.maximum_layers,
            mixed_count=self.mixed_samples, pure_per_element=self.pure_samples,
            master_seed=self.seed, thickness=self.thickness, regions=self.regions,
            elements=self.elements, compound=self.compound,
            shared_random_elements=self.shared_random_elements, random_elements=self.random_elements,
        )
        columns = len(self.elements) + 1
        rows = np.empty((len(batch.thicknesses), layer_count * columns), dtype=np.float64)
        for layer in range(layer_count):
            rows[:, layer * columns] = batch.thicknesses[:, layer]
            rows[:, layer * columns + 1:(layer + 1) * columns] = batch.concentrations[:, layer, :]
        return rows, batch

    def resolved(self):
        return {
            "format": "ibeamlab.generation-resolved", "format_version": 1,
            "configuration": str(self.source), "configuration_sha256": _hash(self.source),
            "output": str(self.output), "maximum_layers": self.maximum_layers,
            "mixed_samples_per_layer_count": self.mixed_samples,
            "pure_samples_per_element": self.pure_samples, "total_samples": self.total_samples,
            "elements": list(self.elements), "backend": self.backend, "workers": self.workers,
            "dummy_channels": self.channels, "records_per_shard": self.records_per_shard,
            "multithreaded_apartment": self.multithreaded_apartment,
            "batch_size": self.batch_size, "fast_calculation": self.fast_calculation,
            "failure_policy": self.failure_policy.value, "seed": self.seed,
            "seed_streams": "SeedSequence([master_seed, layer_count, stream_id])",
            "composition_sampling": {
                "compound": asdict(self.compound),
                "shared_random_elements": asdict(self.shared_random_elements),
                "random_elements": asdict(self.random_elements),
            },
            "method_samples_per_layer_count": method_counts(self.mixed_samples, self.compound,
                self.shared_random_elements, self.random_elements),
            "methods": [{"measurement": m.measurement, "particle": m.particle,
                "reference": str(m.reference),
                "sha256": _hash(m.reference) if m.reference.is_file() else None} for m in self.methods],
            "software": {"ibeamlab": ibl.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
        }

    def run(self, progress=None, layer_progress=None):
        """Write one core dataset per layer count, plus sampling and provenance artifacts.

        progress receives cumulative completed and total sample counts.
        layer_progress(layer_count, completed, total) starts at zero per system.
        Existing
        output is refused. Exceptions and cancellation leave complete=false in
        generation-summary.json and preserve the core's incomplete dataset marker.
        """
        self.validate_files()
        self.output.mkdir(parents=True, exist_ok=False)
        summaries = {}
        status = {"complete": False, "requested": self.total_samples, "datasets": summaries}
        try:
            (self.output / "generation.toml").write_bytes(self.source.read_bytes())
            _json(self.output / "generation.resolved.json", self.resolved())
            with ExitStack() as stack:
                simulator = (ibl.DummySimulator(self.channels) if self.backend == "dummy" else
                    stack.enter_context(ibl.SimnraSimulator(self.methods, workers=self.workers,
                        multithreaded_apartment=self.multithreaded_apartment,
                        fast_calculation=self.fast_calculation)))
                completed = 0
                for layer_count in range(1, self.maximum_layers + 1):
                    label = f"layers_{layer_count:02d}"
                    rows, batch = self.rows(layer_count)
                    if layer_progress:
                        layer_progress(layer_count, 0, len(rows))
                    destination = self.output / label
                    status["active_dataset"] = label
                    def update(done, total):
                        if progress:
                            progress(completed + done, self.total_samples)
                        if layer_progress:
                            layer_progress(layer_count, done, total)
                    summary = self.plan(layer_count).generate(
                        rows, simulator, destination, failure_policy=self.failure_policy,
                        progress=update, records_per_shard=self.records_per_shard,
                        batch_size=self.batch_size,
                        provenance=f"{self.name}; seed={self.seed}; layers={layer_count}; config_sha256={_hash(self.source)}",
                    )
                    summaries[label] = {**asdict(summary), "path": str(summary.path)}
                    status.pop("active_dataset")
                    with (destination / "sampling-components.csv").open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("sample_index", "sampling_method", "sampling_component",
                                         "layer_element_counts", "layer_elements"))
                        symbols = tuple(self.elements)
                        for index, component in enumerate(batch.components):
                            members = [[symbols[j] for j in np.flatnonzero(layer)]
                                       for layer in batch.concentrations[index]]
                            writer.writerow((index, sampling_method(component), component,
                                json.dumps([len(m) for m in members]), json.dumps(members)))
                    _json(destination / "sampling-summary.json", {
                        **batch.report, "components": dict(Counter(batch.components)),
                    })
                    completed += len(rows)
                    del rows, batch
                    if summary.cancelled:
                        break
                else:
                    status["complete"] = True
        except BaseException as error:
            status["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            for key in ("accepted", "failed", "discarded"):
                status[key] = sum(s[key] for s in summaries.values())
            _json(self.output / "generation-summary.json", status)
        return status


def load_configuration(path):
    """Validate TOML settings without loading the compiled extension or starting SIMNRA."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as error:
            raise ValueError("Python 3.10 TOML support requires: python -m pip install '.[examples]'") from error
    source = Path(path).resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    if raw.get("format") != "ibeamlab.generation" or type(raw.get("format_version")) is not int or raw["format_version"] != 1:
        raise ValueError("expected ibeamlab.generation format_version=1")
    composition, thickness = raw["composition_sampling"], raw["thickness_sampling"]
    if composition["normalization"] != "sum_to_one":
        raise ValueError("expected sum_to_one composition normalization")
    if (thickness["strategy"] != "exponential_envelope" or thickness["layer_value_distribution"] != "uniform"
            or thickness["unit"] != "1e15_atoms_per_cm2"):
        raise ValueError("unsupported thickness strategy, distribution or units")
    random = raw["random"]
    if random.get("algorithm") != "PCG64" or random.get("seed_strategy") != "named_streams_per_layer_count":
        raise ValueError("expected PCG64 named_streams_per_layer_count")
    regions = tuple(ConcentrationRegion(r["name"], _number(r["lower"], "region.lower"),
        _number(r["upper"], "region.upper")) for r in composition["regions"])
    elements = {}
    for element in raw["elements"]:
        symbol = element["symbol"]
        if not isinstance(symbol, str) or not symbol.strip() or symbol in elements:
            raise ValueError("element symbols must be nonempty and unique")
        if "selection_probability" in element:
            raise ValueError("selection_probability is incompatible with explicit element counts")
        elements[symbol] = ElementDistribution(
            tuple(_number(p, "region_probability") for p in element["region_probabilities"]))
    _validate(regions, elements)
    envelope = ThicknessEnvelope(*(_number(thickness[name], name) for name in (
        "minimum_total_envelope", "maximum_total_envelope", "growth_ratio")))
    if envelope.minimum_total < 0 or envelope.maximum_total < envelope.minimum_total or envelope.growth_ratio <= 0:
        raise ValueError("invalid thickness envelope bounds or growth ratio")
    compound_data = composition["compound"]
    compound = CompoundSampling(
        _number(compound_data["fraction"], "compound.fraction"),
        tuple((s["name"], tuple(s["elements"])) for s in compound_data.get("subsets", [])),
        compound_data.get("subset_allocation", "proportional_to_element_count"))
    def random_method(name):
        data = composition[name]
        return RandomElementSampling(_number(data["fraction"], f"{name}.fraction"),
            _integer(data["minimum_elements"], f"{name}.minimum_elements", 1),
            _integer(data["maximum_elements"], f"{name}.maximum_elements", 1),
            data.get("count_distribution", "uniform"),
            data.get("selection", "uniform_without_replacement"))
    shared = random_method("shared_random_elements")
    layerwise = random_method("random_elements")
    validate_methods(compound, shared, layerwise, elements)
    measurements = []
    for m in raw["experiment"]["measurements"]:
        measurements.append(ibl.Measurement(m["label"], ibl.Beam(**m["beam"]),
            ibl.Detector(**m.get("detector", {})), ibl.Acquisition(**m.get("acquisition", {}))))
    setup = ibl.ExperimentalSetup(measurements)
    simulation = raw["simulation"]
    mta = simulation.get("multithreaded_apartment", True)
    if not isinstance(mta, bool):
        raise ValueError("simulation.multithreaded_apartment must be boolean")
    fast = simulation.get("fast_calculation", True)
    if not isinstance(fast, bool):
        raise ValueError("simulation.fast_calculation must be boolean")
    backend = simulation["backend"]
    if backend not in ("simnra", "dummy"):
        raise ValueError("simulation.backend must be simnra or dummy")
    methods = tuple(ibl.SimnraMethod(m["measurement"],
        (source.parent / m["reference_file"]).resolve(), m["particle"])
        for m in simulation.get("methods", []))
    if methods or backend == "simnra":
        particles = {m.label: m.beam.particle for m in setup.measurements}
        if len(methods) != len(particles) or {m.measurement: m.particle for m in methods} != particles:
            raise ValueError("SIMNRA methods must match each measurement label and particle exactly once")
    pure = raw.get("one_layer_sampling", {}).get("pure_element_samples", {})
    if not isinstance(pure.get("enabled", True), bool):
        raise ValueError("pure_element_samples.enabled must be boolean")
    if pure.get("thickness_sampling", "stratified") != "stratified":
        raise ValueError("pure-element thickness sampling must be stratified")
    dataset, output = raw["dataset"], raw["output"]
    if not isinstance(output["directory"], str) or not output["directory"].strip():
        raise ValueError("output.directory must be a nonempty path")
    return Configuration(source, raw.get("name", source.stem),
        (source.parent / output["directory"]).resolve(),
        _integer(dataset["maximum_layers"], "maximum_layers", 1),
        _integer(dataset["mixed_samples_per_layer_count"], "mixed_samples", 1),
        _integer(pure.get("samples_per_element", 0) if pure.get("enabled", True) else 0, "pure_samples"),
        _integer(random["seed"], "seed"), elements, regions, envelope, compound, shared, layerwise, setup,
        methods, backend, _integer(simulation.get("workers", 1), "workers", 1, 256),
        _integer(simulation.get("channels", 5000), "channels", 1),
        _integer(output.get("records_per_shard", 64), "records_per_shard", 1, 100000),
        ibl.FailurePolicy(simulation.get("failure_policy", "stop").upper()), mta,
        _integer(simulation.get("batch_size", 512), "batch_size", 1, 4096), fast)
