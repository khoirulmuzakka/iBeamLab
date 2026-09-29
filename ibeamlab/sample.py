"""Python-first descriptions of samples and experimental setups.

Units are keV for beam values and resolution, seconds for times, and
``1e15 atoms/cm2`` for layer thickness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence

from .exceptions import ValidationError


def _nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValidationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Isotope:
    """One isotope; ``mass_number=0`` selects natural abundance."""
    mass_number: int = 0
    exact_mass: float = 0.0
    fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.mass_number < 0:
            raise ValidationError("mass_number cannot be negative")
        _nonnegative("exact_mass", self.exact_mass)
        _nonnegative("fraction", self.fraction)


@dataclass(frozen=True, slots=True)
class Species:
    """An element and its atomic fraction in a layer."""
    element: str
    concentration: float
    isotopes: tuple[Isotope, ...] = ()

    def __post_init__(self) -> None:
        if not self.element.strip():
            raise ValidationError("element cannot be empty")
        _nonnegative("concentration", self.concentration)
        object.__setattr__(self, "isotopes", tuple(self.isotopes))
        if self.isotopes and abs(sum(x.fraction for x in self.isotopes) - 1.0) > 1e-6:
            raise ValidationError(f"isotope fractions for {self.element} must sum to 1")


@dataclass(frozen=True, slots=True, init=False)
class Layer:
    """A layer ordered from surface to depth; composition must sum to one."""
    thickness: float
    species: tuple[Species, ...]
    roughness: float
    porosity_fraction: float
    pore_diameter: float

    def __init__(self, thickness: float, composition: Mapping[str, float] | Sequence[Species],
                 *, roughness: float = 0.0, porosity_fraction: float = 0.0,
                 pore_diameter: float = 0.0) -> None:
        values = (tuple(Species(k, v) for k, v in composition.items())
                  if isinstance(composition, Mapping) else tuple(composition))
        object.__setattr__(self, "thickness", float(thickness))
        object.__setattr__(self, "species", values)
        object.__setattr__(self, "roughness", float(roughness))
        object.__setattr__(self, "porosity_fraction", float(porosity_fraction))
        object.__setattr__(self, "pore_diameter", float(pore_diameter))
        _nonnegative("thickness", self.thickness)
        if not values:
            raise ValidationError("a layer must contain at least one species")
        elements = [x.element for x in values]
        if len(elements) != len(set(elements)):
            raise ValidationError("a layer cannot contain duplicate elements")
        total = sum(x.concentration for x in values)
        if abs(total - 1.0) > 1e-6:
            raise ValidationError(f"layer concentrations sum to {total:g}; expected 1")
        _nonnegative("roughness", self.roughness)
        _nonnegative("pore_diameter", self.pore_diameter)
        if not 0 <= self.porosity_fraction <= 1:
            raise ValidationError("porosity_fraction must be between 0 and 1")

    @property
    def composition(self) -> dict[str, float]:
        """Atomic fractions keyed by element symbol."""
        return {x.element: x.concentration for x in self.species}


@dataclass(frozen=True, slots=True)
class Sample:
    """A surface-to-depth ordered collection of layers."""
    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        if not self.layers:
            raise ValidationError("a sample must contain at least one layer")


@dataclass(frozen=True, slots=True)
class Beam:
    """Incident particle, energy (keV), and FWHM spread (keV)."""
    particle: str
    energy: float
    spread: float = 0.0

    def __post_init__(self) -> None:
        if not self.particle.strip():
            raise ValidationError("beam particle cannot be empty")
        _nonnegative("beam energy", self.energy)
        _nonnegative("beam spread", self.spread)


@dataclass(frozen=True, slots=True)
class LinearCalibration:
    """Calibration ``offset + linear*channel + quadratic*channel²``."""
    linear: float = 1.0
    offset: float = 0.0
    quadratic: float = 0.0


@dataclass(frozen=True, slots=True)
class Detector:
    """Configuration for one uniquely labelled measurement method."""
    label: str
    beam: Beam
    calibration: LinearCalibration = field(default_factory=LinearCalibration)
    resolution: float = 0.0
    particles_sr: float = 0.0
    real_time: float = 0.0
    live_time: float = 0.0
    info: str = ""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValidationError("detector label cannot be empty")
        for name in ("resolution", "particles_sr", "real_time", "live_time"):
            _nonnegative(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Experiment:
    """A sample and the detector configurations used to measure it."""
    sample: Sample
    detectors: tuple[Detector, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "detectors", tuple(self.detectors))
        if not self.detectors:
            raise ValidationError("an experiment must contain at least one detector")
        labels = [x.label for x in self.detectors]
        if len(labels) != len(set(labels)):
            raise ValidationError("detector labels must be unique")


def _to_native_sample(sample: Sample):
    from ._native import native
    result = native.sample.SampleModel()
    layers = []
    for layer in sample.layers:
        item = native.sample.Layer()
        item.thickness, item.roughness = layer.thickness, layer.roughness
        item.porosity_fraction, item.pore_diameter = layer.porosity_fraction, layer.pore_diameter
        species_values = []
        for species in layer.species:
            value = native.sample.Species()
            value.element, value.concentration = species.element, species.concentration
            isotopes = []
            for isotope in species.isotopes:
                native_isotope = native.sample.Isotope()
                native_isotope.mass_number = isotope.mass_number
                native_isotope.exact_mass = isotope.exact_mass
                native_isotope.fraction = isotope.fraction
                isotopes.append(native_isotope)
            value.isotopes = isotopes
            species_values.append(value)
        item.species = species_values
        layers.append(item)
    result.layers = layers
    result.validate()
    return result


def _to_native_setup(detectors: Sequence[Detector]):
    from ._native import native
    result = native.sample.ExperimentalSetup()
    values = []
    for detector in detectors:
        beam = native.sample.Beam()
        beam.particle, beam.energy, beam.spread = detector.beam.particle, detector.beam.energy, detector.beam.spread
        value = native.sample.Detector()
        value.label, value.beam = detector.label, beam
        value.calibration_linear = detector.calibration.linear
        value.calibration_offset = detector.calibration.offset
        value.calibration_quadratic = detector.calibration.quadratic
        value.resolution, value.particles_sr = detector.resolution, detector.particles_sr
        value.real_time, value.live_time, value.info = detector.real_time, detector.live_time, detector.info
        values.append(value)
    result.detectors = values
    result.validate()
    return result


__all__ = ["Beam", "Detector", "Experiment", "Isotope", "Layer", "LinearCalibration", "Sample", "Species"]
