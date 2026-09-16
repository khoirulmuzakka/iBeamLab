"""Generate 1,000 RBS/NRA spectra with the native SIMNRA backend.

The two reference ``.xnra`` files are resolved from this repository's ``xnra``
directory. SIMNRA must be installed and registered as a Windows COM server.
Run this example from any working directory with::

    python examples/generate.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SIMNRA_WORKERS = 8
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibeamlab.generation import (
    DataGenerator,
    FailurePolicy,
    GenerationConfig,
    GenerationOptions,
    GenerationProgress,
    MethodConfig,
    ParameterSpec,
    SpeciesConcentration,
)
from ibeamlab.sample import Beam, Detector, ExperimentalSetup, Layer, SampleModel, Species
from ibeamlab.simulator import SimnraMethod, SimnraSimulator, SimnraSimulatorConfig


def species(element: str, concentration: float) -> Species:
    value = Species()
    value.element = element
    value.concentration = concentration
    return value


def layer(thickness: float, composition: list[tuple[str, float]]) -> Layer:
    value = Layer()
    value.thickness = thickness
    value.species = [species(element, concentration) for element, concentration in composition]
    return value


def detector(
    label: str,
    beam_particle: str,
    beam_energy: float,
    beam_spread: float,
    calibration_linear: float,
    calibration_offset: float,
    calibration_quadratic: float,
    resolution: float,
    particles_sr: float,
    real_time: float = 0.001,
    live_time: float = 0.001,
    info: str = "",
) -> Detector:
    beam = Beam()
    beam.particle = beam_particle
    beam.energy = beam_energy
    beam.spread = beam_spread

    value = Detector()
    value.label = label
    value.beam = beam
    value.calibration_linear = calibration_linear
    value.calibration_offset = calibration_offset
    value.calibration_quadratic = calibration_quadratic
    value.resolution = resolution
    value.particles_sr = particles_sr
    value.real_time = real_time
    value.live_time = live_time
    value.info = info
    return value


def method(label: str, reference_file: Path) -> MethodConfig:
    value = MethodConfig()
    value.label = label
    value.iba_method = label
    value.reference_file = reference_file
    return value


def parameter(
    name: str,
    target: object,
    lower: float,
    upper: float,
    unit: str = "",
    fixed_value: float | None = None,
) -> ParameterSpec:
    value = ParameterSpec()
    value.name = name
    value.target = target
    value.lower_bound = lower
    value.upper_bound = upper
    value.fixed_value = fixed_value
    value.unit = unit
    return value


def concentration_target(layer_index: int, element: str) -> SpeciesConcentration:
    target = SpeciesConcentration()
    target.layer = layer_index
    target.element = element
    return target


def build_generation_config() -> GenerationConfig:
    sample = SampleModel()
    sample.layers = [
        layer(
            5.0e5,
            [("Li", 0.2), ("Ni", 0.2), ("Mn", 0.2), ("Co", 0.2), ("O", 0.2)],
        )
    ]

    setup = ExperimentalSetup()
    setup.detectors = [
        detector(
            label="RBS",
            beam_particle="H",
            beam_energy=2974.0,
            beam_spread=0.0,
            calibration_linear=2.63714,
            calibration_offset=0.0,
            calibration_quadratic=0.0,
            resolution=20.0,
            particles_sr=1.0e12,
        ),
        detector(
            label="NRA",
            beam_particle="H",
            beam_energy=2974.0,
            beam_spread=0.0,
            calibration_linear=7.55,
            calibration_offset=0.0,
            calibration_quadratic=0.0,
            resolution=20.0,
            particles_sr=1.0e13,
        ),
    ]

    config = GenerationConfig()
    config.sample = sample
    config.setup = setup
    config.methods = [
        method("RBS", ROOT / "xnra" / "Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra"),
        method("NRA", ROOT / "xnra" / "Ref_NRA_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra"),
    ]
    config.parameters = [
        # Thickness is fixed in SampleModel. These five open concentrations are
        # sampled jointly and normalized so that the layer always sums to one.
        parameter("Conc_1_Li", concentration_target(0, "Li"), 0.0, 1.0, "fraction"),
        parameter("Conc_1_Ni", concentration_target(0, "Ni"), 0.0, 1.0, "fraction"),
        parameter("Conc_1_Mn", concentration_target(0, "Mn"), 0.0, 1.0, "fraction"),
        parameter("Conc_1_Co", concentration_target(0, "Co"), 0.0, 1.0, "fraction"),
        parameter("Conc_1_O", concentration_target(0, "O"), 0.0, 1.0, "fraction"),
    ]
    config.validate()
    return config


def build_simulator(config: GenerationConfig) -> SimnraSimulator:
    methods: list[SimnraMethod] = []
    for configured_method in config.methods:
        value = SimnraMethod()
        value.label = configured_method.label
        value.reference_file = configured_method.reference_file
        methods.append(value)

    simulator_config = SimnraSimulatorConfig()
    simulator_config.methods = methods
    simulator_config.workers = SIMNRA_WORKERS
    simulator_config.multithreaded_apartment = True
    simulator_config.fast_calculation = True
    return SimnraSimulator(simulator_config)


def main() -> None:
    config = build_generation_config()

    options = GenerationOptions()
    options.samples = 1000
    # Progress is reported after each native batch. Match the batch size to the
    # SIMNRA worker count so all workers stay busy and tqdm advances every wave.
    options.batch_size = SIMNRA_WORKERS
    options.shard_count = 1
    options.seed = 1
    # A SIMNRA/COM failure is usually systemic, so stop on the first failed
    # sample instead of repeating the same error for the entire dataset.
    options.failure_policy = FailurePolicy.STOP

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "examples" / "datasets" / f"linimncoo_rbs_nra_1000_{run_id}"

    # The context manager closes all SIMNRA workers and releases their COM
    # interfaces before propagating any Python or native exception.
    with build_simulator(config) as simulator:
        generator = DataGenerator(config, simulator)
        with tqdm(total=options.samples, unit="sample", desc="SIMNRA") as progress_bar:
            def show_progress(progress: GenerationProgress) -> None:
                progress_bar.update(progress.attempted - progress_bar.n)
                progress_bar.set_postfix(
                    accepted=progress.accepted,
                    invalid=progress.invalid,
                    failed=progress.failed,
                )

            summary = generator.generate(output, options, show_progress)

    print(
        f"Finished: {summary.accepted} accepted, {summary.invalid} invalid, "
        f"{summary.failed} failed; output={output}"
    )


if __name__ == "__main__":
    main()
