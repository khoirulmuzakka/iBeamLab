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

    def to_core(self):
        """Materialize an owning C++ sample."""
        from ._engine import engine
        core = engine()
        result = core.Sample()
        layers = []
        for layer in self.layers:
            item = core.Layer()
            item.thickness, item.roughness = layer.thickness, layer.roughness
            item.porosityFraction, item.poreDiameter = layer.porosity_fraction, layer.pore_diameter
            species = []
            for entry in layer.species:
                value = core.Species()
                value.element, value.concentration = entry.element, entry.concentration
                isotopes = []
                for isotope in entry.isotopes:
                    raw = core.Isotope()
                    raw.massNumber, raw.exactMass, raw.fraction = isotope.mass_number, isotope.exact_mass, isotope.fraction
                    isotopes.append(raw)
                value.isotopes = isotopes
                species.append(value)
            item.species = species
            layers.append(item)
        result.layers = layers
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class Beam:
    """Incident particle, positive energy in keV, and spread in keV FWHM."""
    particle: str
    energy: float
    spread: float = 0.0

    def __post_init__(self):
        if not self.particle.strip() or not isfinite(self.energy) or self.energy <= 0:
            raise ValidationError("beam requires a particle and positive finite energy")
        _nonnegative("beam spread", self.spread)


@dataclass(frozen=True, slots=True)
class Detector:
    """Response and channel-center energy calibration, in keV.

    E(i) = calibration_offset + calibration_linear*i + calibration_quadratic*i*i.
    Resolution is FWHM. Measurement identity and beam belong to Measurement.
    """
    calibration_linear: float = 1.0
    calibration_offset: float = 0.0
    calibration_quadratic: float = 0.0
    resolution: float = 0.0
    info: str = ""

    def __post_init__(self):
        for value in (self.calibration_linear, self.calibration_offset, self.calibration_quadratic):
            if not isfinite(value):
                raise ValidationError("calibration coefficients must be finite")
        _nonnegative("resolution", self.resolution)


@dataclass(frozen=True, slots=True)
class Acquisition:
    """Incident particle count times solid angle, and real/live acquisition times in seconds."""
    particles_sr: float = 0.0
    real_time: float = 0.0
    live_time: float = 0.0

    def __post_init__(self):
        for name in ("particles_sr", "real_time", "live_time"):
            _nonnegative(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Measurement:
    """One labelled measurement; separate measurements can have different beams."""
    label: str
    beam: Beam
    detector: Detector = field(default_factory=Detector)
    acquisition: Acquisition = field(default_factory=Acquisition)

    def __post_init__(self):
        if not self.label.strip():
            raise ValidationError("measurement label cannot be empty")


@dataclass(frozen=True, slots=True)
class ExperimentalSetup:
    """Independent, uniquely labelled measurements of the same sample."""
    measurements: tuple[Measurement, ...]

    def __post_init__(self):
        object.__setattr__(self, "measurements", tuple(self.measurements))
        labels = [m.label for m in self.measurements]
        if not labels or len(labels) != len(set(labels)):
            raise ValidationError("setup requires nonempty, unique measurement labels")

    def to_core(self):
        """Materialize an owning C++ setup."""
        from ._engine import engine
        core = engine()
        result = core.ExperimentalSetup()
        measurements = []
        for m in self.measurements:
            value = core.Measurement()
            value.label = m.label
            beam = core.Beam()
            beam.particle, beam.energy, beam.spread = m.beam.particle, m.beam.energy, m.beam.spread
            value.beam = beam
            detector = core.Detector()
            detector.calibrationLinear = m.detector.calibration_linear
            detector.calibrationOffset = m.detector.calibration_offset
            detector.calibrationQuadratic = m.detector.calibration_quadratic
            detector.resolution, detector.info = m.detector.resolution, m.detector.info
            value.detector = detector
            acquisition = core.Acquisition()
            acquisition.particlesSr = m.acquisition.particles_sr
            acquisition.realTime, acquisition.liveTime = m.acquisition.real_time, m.acquisition.live_time
            value.acquisition = acquisition
            measurements.append(value)
        result.measurements = measurements
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class Experiment:
    """Immutable sample and setup; its C++ representation is cached on first use.

    Use dataclasses.replace to create variations. Treat to_core() as read-only;
    use ibeamlab.native directly when mutable C++ objects are needed.
    """
    sample: Sample
    setup: ExperimentalSetup
    _core: object = field(default=None, init=False, repr=False, compare=False)

    def to_core(self):
        from ._engine import engine
        if self._core is None:
            value = engine().Experiment()
            value.sample, value.setup = self.sample.to_core(), self.setup.to_core()
            value.validate()
            object.__setattr__(self, "_core", value)
        return self._core


def setup_from_core(value):
    """Convert C++ setup metadata to Python value objects."""
    return ExperimentalSetup(tuple(Measurement(
        m.label, Beam(m.beam.particle, m.beam.energy, m.beam.spread),
        Detector(m.detector.calibrationLinear, m.detector.calibrationOffset,
                 m.detector.calibrationQuadratic, m.detector.resolution, m.detector.info),
        Acquisition(m.acquisition.particlesSr, m.acquisition.realTime, m.acquisition.liveTime),
    ) for m in value.measurements))


def sample_from_core(value):
    """Convert C++ sample metadata to Python value objects."""
    return Sample(tuple(Layer(l.thickness, tuple(Species(s.element, s.concentration,
        tuple(Isotope(i.massNumber, i.exactMass, i.fraction) for i in s.isotopes))
        for s in l.species), roughness=l.roughness, porosity_fraction=l.porosityFraction,
        pore_diameter=l.poreDiameter) for l in value.layers))


def experiment_from_core(value):
    """Convert C++ experiment metadata to Python value objects."""
    return Experiment(sample_from_core(value.sample), setup_from_core(value.setup))


