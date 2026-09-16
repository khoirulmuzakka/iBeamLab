"""Round-trip a Python SampleModel and detector setup through SIMNRA COM."""

from __future__ import annotations

import math
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[2]))

from ibeamlab.sample import (
    Beam,
    Detector,
    ExperimentalSetup,
    Isotope,
    Layer,
    SampleModel,
    Species,
)
from ibeamlab.simulator import (
    SimnraMethod,
    SimnraSimulator,
    SimnraSimulatorConfig,
    SimulationInput,
)


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (
        actual,
        expected,
    )


def make_species(element: str, concentration: float) -> Species:
    value = Species()
    value.element = element
    value.concentration = concentration
    return value


def make_isotope(mass_number: int, exact_mass: float, fraction: float) -> Isotope:
    value = Isotope()
    value.mass_number = mass_number
    value.exact_mass = exact_mass
    value.fraction = fraction
    return value


def main(reference_file: Path) -> None:
    surface = Layer()
    surface.thickness = 123.5
    surface.roughness = 4.25
    surface.porosity_fraction = 0.12
    surface.pore_diameter = 7.5
    surface.species = [make_species("C", 0.35), make_species("O", 0.65)]
    surface.species[0].isotopes = [
        make_isotope(12, 12.0, 0.9),
        make_isotope(13, 13.0, 0.1),
    ]

    substrate = Layer()
    substrate.thickness = 4567.0
    substrate.species = [make_species("Si", 0.8), make_species("O", 0.2)]

    sample = SampleModel()
    sample.layers = [surface, substrate]

    beam = Beam()
    beam.particle = "He"
    beam.energy = 2345.0
    beam.spread = 8.5

    detector = Detector()
    detector.label = "RBS"
    detector.beam = beam
    detector.calibration_linear = 2.75
    detector.calibration_offset = -3.5
    detector.calibration_quadratic = -1.25e-5
    detector.resolution = 17.5
    detector.particles_sr = 7.25e10
    detector.real_time = 91.0
    detector.live_time = 87.0

    setup = ExperimentalSetup()
    setup.detectors = [detector]

    simulation_input = SimulationInput()
    simulation_input.sample = sample
    simulation_input.setup = setup

    method = SimnraMethod()
    method.label = "RBS"
    method.reference_file = reference_file.resolve()

    config = SimnraSimulatorConfig()
    config.methods = [method]
    config.workers = 1
    config.multithreaded_apartment = True

    with SimnraSimulator(config) as simulator:
        actual = simulator.inspect_configuration(simulation_input, "RBS")

    assert len(actual.sample.layers) == 2
    for actual_layer, expected_layer in zip(actual.sample.layers, sample.layers, strict=True):
        close(actual_layer.thickness, expected_layer.thickness)
        close(actual_layer.roughness, expected_layer.roughness)
        close(actual_layer.porosity_fraction, expected_layer.porosity_fraction)
        close(actual_layer.pore_diameter, expected_layer.pore_diameter)
        actual_elements = [item.element for item in actual_layer.species]
        expected_elements = [item.element for item in expected_layer.species]
        assert actual_elements == expected_elements, (actual_elements, expected_elements)
        for actual_species, expected_species in zip(
            actual_layer.species, expected_layer.species, strict=True
        ):
            close(actual_species.concentration, expected_species.concentration)
            if expected_species.isotopes:
                assert len(actual_species.isotopes) == len(expected_species.isotopes)
                for actual_isotope, expected_isotope in zip(
                    actual_species.isotopes, expected_species.isotopes, strict=True
                ):
                    close(actual_isotope.exact_mass, expected_isotope.exact_mass)
                    close(actual_isotope.fraction, expected_isotope.fraction)
            else:
                # An empty list requests natural abundance; SIMNRA expands it
                # into its explicit natural-isotope table.
                close(sum(item.fraction for item in actual_species.isotopes), 1.0)

    actual_detector = actual.setup.detectors[0]
    close(actual_detector.beam.energy, detector.beam.energy)
    close(actual_detector.beam.spread, detector.beam.spread)
    close(actual_detector.calibration_linear, detector.calibration_linear)
    close(actual_detector.calibration_offset, detector.calibration_offset)
    close(actual_detector.calibration_quadratic, detector.calibration_quadratic)
    close(actual_detector.resolution, detector.resolution)
    close(actual_detector.particles_sr, detector.particles_sr)
    close(actual_detector.real_time, detector.real_time)
    close(actual_detector.live_time, detector.live_time)

    print("SIMNRA target and setup round-trip passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: simnra_python_integration.py REFERENCE.xnra")
    main(Path(sys.argv[1]))
