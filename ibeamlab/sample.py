"""Python-first descriptions of samples and experimental setups.

Units are keV for beam values and resolution, seconds for times, and
``1e15 atoms/cm2`` for layer thickness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sin, cos, radians
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from .native import IsotopeData

from .exceptions import ValidationError


def _nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValidationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Isotope:
    """Specify one isotope fraction within an element.

    Parameters
    ----------
    mass_number : int
        Positive isotope mass number, such as 28 for Si-28. The containing Species
        provides the element identity; exact masses come from the framework database.
    fraction : float, default=1.0
        Atomic fraction within this element, finite and nonnegative. Explicit isotope
        fractions must sum to one per Species.

    Raises
    ------
    ValidationError
        The mass number or fraction is invalid.

    Notes
    -----
    Represent natural composition with an empty Species.isotopes sequence.
    Isotope availability and duplicate identities are checked at native validation.
    """
    mass_number: int
    fraction: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.mass_number, bool) or not isinstance(self.mass_number, int) or self.mass_number <= 0:
            raise ValidationError("mass_number must be a positive integer")
        _nonnegative("fraction", self.fraction)


@dataclass(frozen=True, slots=True)
class Species:
    """Describe an element and its atomic concentration in a layer.

    Parameters
    ----------
    element : str
        Canonical element symbol, for example "Si".
    concentration : float
        Finite, nonnegative atomic fraction. Species concentrations sum to one per layer.
    isotopes : sequence of Isotope, optional
        Explicit isotope mixture. Empty uses the framework natural composition;
        elements without a natural default require an explicit mixture.

    Raises
    ------
    ValidationError
        The symbol is empty, a concentration is invalid, or fractions are not normalized.

    Notes
    -----
    Element concentration and isotope fractions are normalized independently.
    A valid enriched sample can still be incompatible with a particular surrogate.
    """
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
    """Describe one homogeneous layer in surface-to-depth order.

    Parameters
    ----------
    thickness : float
        Finite, nonnegative areal thickness in 1e15 atoms/cm2.
    composition : mapping of str to float or sequence of Species
        Element fractions, either {symbol: fraction} for natural elements or explicit
        Species values for isotope mixtures. Symbols must be unique; fractions sum to one.
    roughness : float, default=0.0
        Nonnegative backend roughness width. Its interpretation follows the backend
        and reference roughness model; it is not an RMS height in nanometres.
    porosity_fraction : float, default=0.0
        Pore volume fraction in [0, 1].
    pore_diameter : float, default=0.0
        Nonnegative pore diameter in the backend areal-length convention.
        The SIMNRA backend uses 1e15 atoms/cm2.

    Raises
    ------
    ValidationError
        Layer dimensions or composition violate physical constraints.

    Notes
    -----
    Values are immutable. Use dataclasses.replace to construct a changed layer.
    A backend or model may reject roughness or porosity even when the sample is valid.

    Examples
    --------
    >>> layer = Layer(100.0, {"Si": 0.5, "O": 0.5})
    >>> layer.composition
    {'Si': 0.5, 'O': 0.5}
    """
    thickness: float
    species: tuple[Species, ...]
    roughness: float
    porosity_fraction: float
    pore_diameter: float

    def __init__(self, thickness: float, composition: Mapping[str, float] | Sequence[Species],
                 *, roughness: float = 0.0, porosity_fraction: float = 0.0,
                 pore_diameter: float = 0.0) -> None:
        """Initialize a layer; see Layer for parameters, units and constraints.
        """
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
        """Return element concentrations as a newly allocated dictionary.

        Returns
        -------
        dict[str, float]
            Atomic fractions keyed by symbol; isotope mixtures remain in species.
        """
        return {x.element: x.concentration for x in self.species}


@dataclass(frozen=True, slots=True)
class Sample:
    """Describe a nonempty stack of layers from the surface into the sample.

    Parameters
    ----------
    layers : sequence of Layer
        Layers in physical depth order; stored as an immutable tuple.

    Raises
    ------
    ValidationError
        No layers are supplied.

    Notes
    -----
    Sample validity does not imply model compatibility. Model metadata determines
    supported elements, isotope mixtures and layer-count behavior.
    """
    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        if not self.layers:
            raise ValidationError("a sample must contain at least one layer")

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.Sample
            Native value for lower-level workflows.

        Notes
        -----
        Copies layers, species and isotope lists, then performs full native validation.
        An empty isotope list remains the compact natural-composition representation.
        """
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
                    raw.massNumber, raw.fraction = isotope.mass_number, isotope.fraction
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
    """Describe the incident particle and its energy distribution.

    Parameters
    ----------
    particle : str
        Nonempty backend particle identifier. For SIMNRA it must agree with the
        particle declared by the corresponding reference method.
    energy : float
        Positive finite beam energy in keV.
    spread : float, default=0.0
        Nonnegative beam energy spread in keV FWHM.

    Raises
    ------
    ValidationError
        Particle, energy or spread is invalid.
    """
    particle: str
    energy: float
    spread: float = 0.0

    def __post_init__(self):
        if not self.particle.strip() or not isfinite(self.energy) or self.energy <= 0:
            raise ValidationError("beam requires a particle and positive finite energy")
        _nonnegative("beam spread", self.spread)


@dataclass(frozen=True, slots=True)
class Geometry:
    """Describe the complete geometry for one detector measurement.

    Parameters
    ----------
    incidence_angle : float, default=0.0
        SIMNRA alpha in degrees, measured from the inward surface normal; [0, 90).
    exit_angle : float, default=0.0
        SIMNRA beta in degrees, measured from the outward surface normal; [0, 180].
        Values above 90 describe transmission directions.
    scattering_angle : float, default=180.0
        SIMNRA theta in degrees, between incident and outgoing trajectories; [0, 180].

    Raises
    ------
    ValidationError
        An angle is nonfinite, outside its range, or the directions cannot coexist.

    Notes
    -----
    All three angles are stored explicitly under Detector.geometry. Non-coplanar
    arrangements are supported; alpha + beta + theta need not equal 180 degrees.
    The default describes normal incidence and exact backscattering. Supply actual
    angles for physical measurements. Fixed-geometry surrogates reject changes.

    Examples
    --------
    >>> geometry = Geometry(incidence_angle=0, exit_angle=15, scattering_angle=165)
    """
    incidence_angle: float = 0.0
    exit_angle: float = 0.0
    scattering_angle: float = 180.0

    def __post_init__(self):
        a, b, t = self.incidence_angle, self.exit_angle, self.scattering_angle
        if not (isfinite(a) and 0 <= a < 90 and isfinite(b) and 0 <= b <= 180
                and isfinite(t) and 0 <= t <= 180):
            raise ValidationError("geometry requires incidence in [0,90), exit/scattering in [0,180] degrees")
        a, b, t = radians(a), radians(b), radians(t)
        if abs(cos(t) + cos(a) * cos(b)) > sin(a) * sin(b) + 1e-10:
            raise ValidationError("inconsistent incidence, exit and scattering angles")

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.Geometry
            Native value for lower-level workflows.

        Notes
        -----
        Copies three scalar angles. Construction of the Python Geometry already
        checks physical consistency.
        """
        from ._engine import engine
        value = engine().Geometry()
        value.incidenceAngle, value.exitAngle = self.incidence_angle, self.exit_angle
        value.scatteringAngle = self.scattering_angle
        return value


@dataclass(frozen=True, slots=True)
class Detector:
    """Describe detector geometry, energy response and channel calibration.

    Parameters
    ----------
    calibration_linear : float, default=1.0
        Linear coefficient in keV/channel.
    calibration_offset : float, default=0.0
        Calibration offset in keV.
    calibration_quadratic : float, default=0.0
        Quadratic coefficient in keV/channel squared.
    resolution : float, default=0.0
        Nonnegative energy resolution in keV FWHM.
    info : str, default=""
        Descriptive detector information.
    geometry : Geometry or mapping, optional
        Complete alpha/beta/theta geometry for this detector measurement. A mapping
        uses the Geometry keyword names. Defaults to Geometry().

    Raises
    ------
    ValidationError
        Calibration, resolution or geometry is invalid.

    Notes
    -----
    For zero-based channel i, the channel-center energy is
    E(i) = offset + linear*i + quadratic*i*i. Edges use half-integer indices.
    Spectrum processing requires the applicable energy grid to increase monotonically.
    """
    calibration_linear: float = 1.0
    calibration_offset: float = 0.0
    calibration_quadratic: float = 0.0
    resolution: float = 0.0
    info: str = ""
    geometry: Geometry = field(default_factory=Geometry)

    def __post_init__(self):
        if isinstance(self.geometry, Mapping):
            object.__setattr__(self, "geometry", Geometry(**self.geometry))
        if not isinstance(self.geometry, Geometry):
            raise ValidationError("detector.geometry must be Geometry")
        for value in (self.calibration_linear, self.calibration_offset, self.calibration_quadratic):
            if not isfinite(value):
                raise ValidationError("calibration coefficients must be finite")
        _nonnegative("resolution", self.resolution)


@dataclass(frozen=True, slots=True)
class Acquisition:
    """Describe exposure and acquisition times for one measurement.

    Parameters
    ----------
    particles_sr : float, default=0.0
        Incident particle count multiplied by detector solid angle in steradians.
    real_time : float, default=0.0
        Real acquisition time in seconds.
    live_time : float, default=0.0
        Live acquisition time in seconds.

    Raises
    ------
    ValidationError
        A value is nonfinite or negative.

    Notes
    -----
    Exposure scaling requires positive source and destination exposures. Pileup
    requires appropriate positive times. SIMNRA uses reference times when requested
    real_time or live_time is zero; other consumers apply their own declared policy.
    """
    particles_sr: float = 0.0
    real_time: float = 0.0
    live_time: float = 0.0

    def __post_init__(self):
        for name in ("particles_sr", "real_time", "live_time"):
            _nonnegative(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Measurement:
    """Describe one labelled observation of a sample.

    Parameters
    ----------
    label : str
        Unique nonempty measurement identifier, used to match spectra and metadata.
    beam : Beam
        Beam for this measurement; measurements may use different particles or energies.
    detector : Detector, optional
        Detector response, calibration and geometry. Defaults to Detector().
    acquisition : Acquisition, optional
        Exposure and real/live times. Defaults to Acquisition().

    Raises
    ------
    ValidationError
        The label is empty.
    """
    label: str
    beam: Beam
    detector: Detector = field(default_factory=Detector)
    acquisition: Acquisition = field(default_factory=Acquisition)

    def __post_init__(self):
        if not self.label.strip():
            raise ValidationError("measurement label cannot be empty")


@dataclass(frozen=True, slots=True)
class ExperimentalSetup:
    """Group independently configured measurements of the same sample.

    Parameters
    ----------
    measurements : sequence of Measurement
        Nonempty ordered measurements with unique labels; stored as a tuple.

    Raises
    ------
    ValidationError
        Labels are empty or duplicated, or no measurements are supplied.

    Notes
    -----
    Labels establish identity independently of ordering in inference requests.
    Each measurement owns its beam, detector geometry and acquisition settings.
    """
    measurements: tuple[Measurement, ...]

    def __post_init__(self):
        object.__setattr__(self, "measurements", tuple(self.measurements))
        labels = [m.label for m in self.measurements]
        if not labels or len(labels) != len(set(labels)):
            raise ValidationError("setup requires nonempty, unique measurement labels")

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.ExperimentalSetup
            Native value for lower-level workflows.

        Notes
        -----
        Allocates owning measurement values and performs native setup validation,
        including each detector geometry.
        """
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
            detector.geometry = m.detector.geometry.to_core()
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
    """Combine one sample with the measurements made on it.

    Parameters
    ----------
    sample : Sample
        Shared physical sample.
    setup : ExperimentalSetup
        Independently configured measurements.

    Notes
    -----
    Physical value objects are immutable. Use dataclasses.replace to create
    variations. The native representation is cached on first to_core() call; treat
    it as read-only so it remains consistent with the Python description.
    """
    sample: Sample
    setup: ExperimentalSetup
    _core: object = field(default=None, init=False, repr=False, compare=False)

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.Experiment
            Native value for lower-level workflows.

        Notes
        -----
        Materializes and validates the native experiment once, then returns the same
        cached object. Mutating that object bypasses Python immutability and invalidates
        the description/cache relationship; use native workflows explicitly for mutation.
        """
        from ._engine import engine
        if self._core is None:
            value = engine().Experiment()
            value.sample, value.setup = self.sample.to_core(), self.setup.to_core()
            value.validate()
            object.__setattr__(self, "_core", value)
        return self._core


def setup_from_core(value):
    """Construct Python-facing values from a native result.

    Parameters
    ----------
    value : ibeamlab.native.ExperimentalSetup
        Source value; it is not modified.

    Returns
    -------
    ExperimentalSetup
        Python representation of the supplied value.

    Notes
    -----
    Builds Python value objects; ordinary simulation and dataset APIs do this automatically.
    """
    return ExperimentalSetup(tuple(Measurement(
        m.label, Beam(m.beam.particle, m.beam.energy, m.beam.spread),
        Detector(m.detector.calibrationLinear, m.detector.calibrationOffset,
                 m.detector.calibrationQuadratic, m.detector.resolution, m.detector.info,
                 Geometry(m.detector.geometry.incidenceAngle, m.detector.geometry.exitAngle,
                          m.detector.geometry.scatteringAngle)),
        Acquisition(m.acquisition.particlesSr, m.acquisition.realTime, m.acquisition.liveTime),
    ) for m in value.measurements))


def sample_from_core(value):
    """Construct Python-facing values from a native result.

    Parameters
    ----------
    value : ibeamlab.native.Sample
        Source value; it is not modified.

    Returns
    -------
    Sample
        Python representation of the supplied value.

    Notes
    -----
    Builds Python value objects; ordinary simulation and dataset APIs do this automatically.
    """
    return Sample(tuple(Layer(l.thickness, tuple(Species(s.element, s.concentration,
        tuple(Isotope(i.massNumber, i.fraction) for i in s.isotopes))
        for s in l.species), roughness=l.roughness, porosity_fraction=l.porosityFraction,
        pore_diameter=l.poreDiameter) for l in value.layers))


def experiment_from_core(value):
    """Construct Python-facing values from a native result.

    Parameters
    ----------
    value : ibeamlab.native.Experiment
        Source value; it is not modified.

    Returns
    -------
    Experiment
        Python representation of the supplied value.

    Notes
    -----
    Builds Python value objects; ordinary simulation and dataset APIs do this automatically.
    """
    return Experiment(sample_from_core(value.sample), setup_from_core(value.setup))


def natural_isotopes(element: str) -> tuple[Isotope, ...]:
    """Return the framework natural isotope fractions for an element.

    Parameters
    ----------
    element : str
        Canonical element symbol.

    Returns
    -------
    tuple[Isotope, ...]
        Owned Python isotope values. An empty tuple means no natural default exists.

    Raises
    ------
    ValueError
        The element is unknown.
    NativeExtensionError
        The compiled isotope database cannot be loaded.

    Notes
    -----
    Fractions follow the framework database snapshot. C++ consumers can obtain
    immutable static views without allocation; Python receives owned values.
    Exported models store their assumed fractions explicitly.
    """
    from ._engine import engine
    return tuple(Isotope(i.massNumber, i.fraction) for i in engine().natural_isotopes(element))


def isotope_data(element: str) -> tuple[IsotopeData, ...]:
    """Return supported isotope identities and exact atomic masses.

    Parameters
    ----------
    element : str
        Canonical element symbol.

    Returns
    -------
    tuple[ibeamlab.native.IsotopeData, ...]
        Read-only records exposing massNumber and exactMass, in atomic mass units (u).

    Raises
    ------
    ValueError
        The element is unknown.

    Notes
    -----
    The list may include isotopes with zero natural abundance. Availability in the
    database does not guarantee support by a particular simulator or trained model.
    """
    from ._engine import engine
    return tuple(engine().isotope_data(element))
