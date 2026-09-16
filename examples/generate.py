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
    CalibrationLinear,
    CalibrationOffset,
    DataGenerator,
    FailurePolicy,
    GenerationConfig,
    GenerationOptions,
    GenerationProgress,
    LayerThickness,
    MethodConfig,
    ParameterSpec,
    ParticlesSr,
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
    calibration_linear: float,
    calibration_quadratic: float,
    resolution: float,
    particles_sr: float,
) -> Detector:
    beam = Beam()
    beam.particle = "He"
    beam.energy = 2950.0

    value = Detector()
    value.label = label
    value.beam = beam
    value.calibration_linear = calibration_linear
    value.calibration_quadratic = calibration_quadratic
    value.resolution = resolution
    value.particles_sr = particles_sr
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


def detector_target(target_type: type, label: str) -> object:
    target = target_type()
    target.detector = label
    return target


def concentration_target(layer_index: int, element: str) -> SpeciesConcentration:
    target = SpeciesConcentration()
    target.layer = layer_index
    target.element = element
    return target


def build_generation_config() -> GenerationConfig:
    sample = SampleModel()
    sample.layers = [
        layer(100.0, [("C", 0.5), ("O", 0.5)]),
        layer(
            300_000.0,
            [
                ("C", 0.10),
                ("O", 0.20),
                ("Y", 0.02),
                ("Zr", 0.13),
                ("Ba", 0.19),
                ("Ce", 0.04),
                ("D", 0.32),
            ],
        ),
    ]

    setup = ExperimentalSetup()
    setup.detectors = [
        detector("RBS", 2.63714, -1.47615e-5, 22.0, 8.40491e10),
        detector("NRA", 7.55, 0.0, 20.0, 9.08e11),
    ]

    thickness = LayerThickness()
    thickness.layer = 0  # Native indices are zero-based.

    config = GenerationConfig()
    config.sample = sample
    config.setup = setup
    config.methods = [
        method("RBS", ROOT / "xnra" / "Ref_all_rbs_nopu.xnra"),
        method("NRA", ROOT / "xnra" / "Ref_all_nra_nopu.xnra"),
    ]
    config.parameters = [
        parameter("Thickness_1", thickness, 10.0, 2000.0, "1e15 at/cm2"),
        # Open concentrations in each layer are sampled jointly and normalized
        # to the fraction remaining after fixed concentrations.
        parameter("Conc_1_C", concentration_target(0, "C"), 0.0, 1.0, "fraction"),
        parameter("Conc_1_O", concentration_target(0, "O"), 0.0, 1.0, "fraction"),
        parameter("Conc_2_C", concentration_target(1, "C"), 0.0, 1.0, "fraction"),
        parameter("Conc_2_O", concentration_target(1, "O"), 0.0, 1.0, "fraction"),
        parameter("Conc_2_D", concentration_target(1, "D"), 0.0, 1.0, "fraction"),
        parameter(
            "Conc_2_Y", concentration_target(1, "Y"), 0.0, 1.0, "fraction", 0.02
        ),
        parameter(
            "Conc_2_Zr", concentration_target(1, "Zr"), 0.0, 1.0, "fraction", 0.13
        ),
        parameter(
            "Conc_2_Ba", concentration_target(1, "Ba"), 0.0, 1.0, "fraction", 0.19
        ),
        parameter(
            "Conc_2_Ce", concentration_target(1, "Ce"), 0.0, 1.0, "fraction", 0.04
        ),
        parameter(
            "Calib_Linear_RBS",
            detector_target(CalibrationLinear, "RBS"),
            2.50528,
            2.769,
            "keV/channel",
        ),
        parameter(
            "Calib_Linear_NRA",
            detector_target(CalibrationLinear, "NRA"),
            7.1725,
            7.9275,
            "keV/channel",
        ),
        parameter(
            "Calib_Offset_RBS",
            detector_target(CalibrationOffset, "RBS"),
            -20.0,
            20.0,
            "keV",
        ),
        parameter(
            "ParticlesSr_RBS",
            detector_target(ParticlesSr, "RBS"),
            1.0e9,
            8.40491e10,
            "particles/sr",
        ),
        parameter(
            "ParticlesSr_NRA",
            detector_target(ParticlesSr, "NRA"),
            1.0e9,
            9.08e11,
            "particles/sr",
        ),
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
    options.samples = 100
    # Progress is reported after each native batch. Match the batch size to the
    # SIMNRA worker count so all workers stay busy and tqdm advances every wave.
    options.batch_size = SIMNRA_WORKERS
    options.shard_count = 10
    options.seed = 1
    # A SIMNRA/COM failure is usually systemic, so stop on the first failed
    # sample instead of repeating the same error for the entire dataset.
    options.failure_policy = FailurePolicy.STOP

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "examples" / "datasets" / f"bzcy_rbs_nra_1000_{run_id}"

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
