"""Typed metadata descriptors and domain values exposed by the lazy native facade.

Other low-level bindings remain dynamically accessible. This file describes
types only; runtime imports still defer loading the compiled extension.
"""
from collections.abc import Sequence
from enum import Enum
from typing import Any, TypeAlias

def __getattr__(name: str) -> Any: ...

class ExecutionProvider(Enum):
    """Choose the native ONNX provider.

    Notes
    -----
    CPU selects CPU execution. CUDA requires a GPU-enabled native build and
    working compatible device/runtime libraries.
    """
    CPU = 0
    CUDA = 1

class InferenceOptions:
    """Native per-session execution settings.

    Attributes
    ----------
    execution_provider : ExecutionProvider
        CPU or CUDA.
    device_id : int
        Nonnegative CUDA device index.
    intra_op_threads : int
        Positive operator-level thread count.
    inter_op_threads : int
        Positive graph-level thread count.
    enable_graph_optimizations : bool
        Enable graph optimizations.
    allow_tf32 : bool
        Allow TensorFloat-32 CUDA operations.
    apply_pileup : bool
        Enable declared forward pileup operations.

    Notes
    -----
    Use the Python InferenceOptions facade for validated keyword construction.
    These settings do not change the model scientific contract.
    """
    execution_provider: ExecutionProvider
    device_id: int
    intra_op_threads: int
    inter_op_threads: int
    enable_graph_optimizations: bool
    allow_tf32: bool
    apply_pileup: bool
    def __init__(self) -> None:
        """Initialize InferenceOptions; see the class documentation for parameters and units.
        """
        ...
    def validate(self) -> None:
        """Validate provider enumeration, nonnegative device and positive thread counts.
        Does not initialize a device; raises ValueError for invalid scalar options.
        """
        ...

def available_execution_providers() -> list[str]:
    """Return provider names exposed by this native runtime.

    Returns a list containing supported cpu/cuda names, or an empty list when
    ONNX support is disabled. Enumeration does not guarantee device initialization.
    """
    ...

class Isotope:
    """Native isotope identity and atomic fraction.

    Attributes
    ----------
    massNumber : int
        Positive mass number; element identity is supplied by the containing Species.
    fraction : float
        Finite nonnegative fraction within the element. Explicit mixtures sum to one.

    Notes
    -----
    Use the Python Isotope facade for immutable descriptions. An empty Species
    isotope list requests natural composition; massNumber=0 is not a natural marker.
    """
    massNumber: int
    fraction: float
    def __init__(self) -> None:
        """Initialize Isotope; see the class documentation for parameters and units.
        """
        ...

class Species:
    """Native element composition record.

    Attributes
    ----------
    element : str
        Canonical element symbol.
    concentration : float
        Atomic fraction within a layer.
    isotopes : list[Isotope]
        Explicit mixture; empty selects the framework natural default.
    """
    element: str
    concentration: float
    isotopes: list[Isotope]
    def __init__(self) -> None:
        """Initialize Species; see the class documentation for parameters and units.
        """
        ...

class Layer:
    """Native layer in surface-to-depth order.

    Attributes
    ----------
    thickness : float
        Areal thickness in 1e15 atoms/cm2.
    roughness : float
        Backend roughness width.
    porosityFraction : float
        Pore volume fraction in [0,1].
    poreDiameter : float
        Pore diameter in backend areal-length units.
    species : list[Species]
        Unique elements with atomic concentrations summing to one.
    """
    thickness: float
    roughness: float
    porosityFraction: float
    poreDiameter: float
    species: list[Species]
    def __init__(self) -> None:
        """Initialize Layer; see the class documentation for parameters and units.
        """
        ...

class Sample:
    """Native owning stack of physical layers.

    Attributes
    ----------
    layers : list[Layer]
        Surface-to-depth layer sequence.

    Notes
    -----
    validate() checks dimensions, element/isotope identities and normalized
    compositions. Use ibeamlab.Sample for immutable Python descriptions.
    """
    layers: list[Layer]
    def __init__(self) -> None:
        """Initialize Sample; see the class documentation for parameters and units.
        """
        ...
    def validate(self) -> None:
        """Validate dimensions, element/isotope identities and normalized compositions.
        Raises ValueError when the physical sample contract is invalid.
        """
        ...

class Beam:
    """Native beam description.

    Attributes
    ----------
    particle : str
        Backend particle identifier.
    energy : float
        Positive beam energy in keV.
    spread : float
        Beam energy spread in keV FWHM.
    """
    particle: str
    energy: float
    spread: float
    def __init__(self) -> None:
        """Initialize Beam; see the class documentation for parameters and units.
        """
        ...

class Geometry:
    """Native detector measurement geometry in degrees.

    Attributes
    ----------
    incidenceAngle : float
        Alpha from the inward surface normal, in [0,90).
    exitAngle : float
        Beta from the outward normal, in [0,180].
    scatteringAngle : float
        Theta between incident and outgoing trajectories, in [0,180].

    Notes
    -----
    validate() checks directional consistency, including non-coplanar geometry.
    Default (0,0,180) means normal incidence and exact backscattering.
    """
    incidenceAngle: float
    exitAngle: float
    scatteringAngle: float
    def __init__(self) -> None:
        """Initialize Geometry; see the class documentation for parameters and units.
        """
        ...
    def validate(self) -> None:
        """Validate angle ranges and coexistence of the three directions.
        Angles use degrees; non-coplanar and transmission geometries are supported.
        Raises ValueError for inconsistent geometry.
        """
        ...

class Detector:
    """Native detector response, calibration and geometry.

    Attributes
    ----------
    calibrationLinear : float
        Linear calibration in keV/channel.
    calibrationOffset : float
        Channel-center offset in keV.
    calibrationQuadratic : float
        Quadratic term in keV/channel squared.
    resolution : float
        Energy resolution in keV FWHM.
    info : str
        Descriptive information.
    geometry : Geometry
        Complete detector-specific measurement geometry.
    """
    geometry: Geometry
    calibrationLinear: float
    calibrationOffset: float
    calibrationQuadratic: float
    resolution: float
    info: str
    def __init__(self) -> None:
        """Initialize Detector; see the class documentation for parameters and units.
        """
        ...

class Acquisition:
    """Native measurement exposure and acquisition times.

    Attributes
    ----------
    particlesSr : float
        Incident particle count times solid angle in sr.
    realTime : float
        Real time in seconds.
    liveTime : float
        Live time in seconds.
    """
    particlesSr: float
    realTime: float
    liveTime: float
    def __init__(self) -> None:
        """Initialize Acquisition; see the class documentation for parameters and units.
        """
        ...

class Measurement:
    """Native labelled measurement of the shared sample.

    Attributes
    ----------
    label : str
        Measurement identity.
    beam : Beam
        Particle and energy settings.
    detector : Detector
        Response, calibration and geometry.
    acquisition : Acquisition
        Exposure and times.
    """
    label: str
    beam: Beam
    detector: Detector
    acquisition: Acquisition
    def __init__(self) -> None:
        """Initialize Measurement; see the class documentation for parameters and units.
        """
        ...

class ExperimentalSetup:
    """Native owning collection of independently configured measurements.

    Attributes
    ----------
    measurements : list[Measurement]
        Nonempty measurements with unique labels.

    Notes
    -----
    validate() includes geometry and physical setup checks.
    """
    measurements: list[Measurement]
    def __init__(self) -> None:
        """Initialize ExperimentalSetup; see the class documentation for parameters and units.
        """
        ...
    def validate(self) -> None:
        """Validate measurement identities, beam values and detector/acquisition settings.
        Includes per-detector geometry validation. Raises ValueError on invalid input.
        """
        ...

class LayerThickness:
    """Select the areal thickness of a zero-based sample layer.

    Parameters
    ----------
    layer : int
        Zero-based layer index; values use 1e15 atoms/cm2.
    """
    layer: int
    def __init__(self, layer: int, /) -> None:
        """Initialize LayerThickness; see the class documentation for parameters and units.
        """
        ...

class SpeciesConcentration:
    """Select an elemental concentration in a zero-based layer.

    Parameters
    ----------
    layer : int
        Zero-based layer index.
    element : str
        Element symbol; values are atomic fractions.
    """
    layer: int
    element: str
    def __init__(self, layer: int, element: str, /) -> None:
        """Initialize SpeciesConcentration; see the class documentation for parameters and units.
        """
        ...

class BeamEnergy:
    """Select beam energy by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use keV.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize BeamEnergy; see the class documentation for parameters and units.
        """
        ...

class BeamSpread:
    """Select beam spread by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use keV FWHM.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize BeamSpread; see the class documentation for parameters and units.
        """
        ...

class IncidenceAngle:
    """Select detector incidence alpha by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use degree.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize IncidenceAngle; see the class documentation for parameters and units.
        """
        ...

class ExitAngle:
    """Select detector exit beta by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use degree.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize ExitAngle; see the class documentation for parameters and units.
        """
        ...

class ScatteringAngle:
    """Select detector scattering theta by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use degree.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize ScatteringAngle; see the class documentation for parameters and units.
        """
        ...

class CalibrationLinear:
    """Select linear calibration by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use keV/channel.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize CalibrationLinear; see the class documentation for parameters and units.
        """
        ...

class CalibrationOffset:
    """Select calibration offset by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use keV.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize CalibrationOffset; see the class documentation for parameters and units.
        """
        ...

class CalibrationQuadratic:
    """Select quadratic calibration by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use keV/channel squared.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize CalibrationQuadratic; see the class documentation for parameters and units.
        """
        ...

class DetectorResolution:
    """Select detector resolution by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use keV FWHM.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize DetectorResolution; see the class documentation for parameters and units.
        """
        ...

class ParticlesSr:
    """Select exposure by measurement label.

    Parameters
    ----------
    measurement : str
        Exact measurement label. Physical values use particles*sr.
    """
    measurement: str
    def __init__(self, measurement: str, /) -> None:
        """Initialize ParticlesSr; see the class documentation for parameters and units.
        """
        ...

ParameterTarget: TypeAlias = LayerThickness | SpeciesConcentration | BeamEnergy | BeamSpread | CalibrationLinear | CalibrationOffset | CalibrationQuadratic | DetectorResolution | ParticlesSr

class Parameter:
    """Declare one named native physical parameter column.

    Parameters
    ----------
    name : str
        Unique column name.
    target : ParameterTarget
        Physical target location.
    unit : str
        Descriptive unit label; values use the target physical convention.
    """
    name: str
    target: ParameterTarget
    unit: str
    def __init__(self, name: str, target: ParameterTarget, unit: str = ...) -> None:
        """Initialize Parameter; see the class documentation for parameters and units.
        """
        ...

class SpectrumState(Enum):
    """Declare the physical pileup state of a reference spectrum.

    Notes
    -----
    BARE denotes pileup-free counts. PILED_UP denotes counts with pileup.
    Metadata validation uses this state to reject inconsistent preparation.
    """
    BARE = 0
    PILED_UP = 1

class TransformDirection(Enum):
    """Choose the direction of stored numerical training operations.

    Notes
    -----
    FORWARD applies steps in stored order. INVERSE applies inverse operations
    in reverse order. Non-invertible descriptors are rejected in inverse pipelines.
    """
    FORWARD = 0
    INVERSE = 1

class TransformLocation(Enum):
    """Declare where a numerical transform executes.

    Notes
    -----
    RUNTIME executes outside ONNX. GRAPH declares operations already embedded
    in ONNX; the runtime skips them to avoid double processing.
    """
    RUNTIME = 0
    GRAPH = 1

class PredictionDomain(Enum):
    """Choose requested-measurement or reference-condition inference.

    Notes
    -----
    MEASUREMENT executes physical preparation. REFERENCE skips physical
    preparation while retaining numerical training transforms and output decoding.
    """
    MEASUREMENT = 0
    REFERENCE = 1

class SpectrumSpec:
    """Declare one reference-condition graph spectrum.

    Parameters
    ----------
    measurement : str
        Label matching a reference measurement.
    channels : int
        Positive reference channel count.
    state : SpectrumState, default=SpectrumState.BARE
        Physical pileup state.
    unit : str, default="counts"
        Integrated-count unit.

    Notes
    -----
    Energy calibration and exposure come from the matching reference setup.
    Physical output preparation may change the final grid, length and pileup state.
    """
    measurement: str
    channels: int
    state: SpectrumState
    unit: str
    def __init__(self, measurement: str, channels: int, state: SpectrumState = ..., unit: str = ...) -> None:
        """Initialize SpectrumSpec; see the class documentation for parameters and units.
        """
        ...

class SpectraSchema:
    """Pack labelled spectra consecutively along a tensor feature axis.

    Parameters
    ----------
    spectra : sequence of SpectrumSpec, optional
        Ordered unique measurement descriptors.

    Notes
    -----
    width() returns total channels. offset(label) returns the first feature index
    for a measurement, independently of caller spectrum ordering.
    """
    spectra: list[SpectrumSpec]
    def __init__(self, spectra: Sequence[SpectrumSpec] = ..., /) -> None:
        """Initialize SpectraSchema; see the class documentation for parameters and units.
        """
        ...
    def width(self) -> int:
        """Return the summed channel count of all packed reference spectra.
        """
        ...
    def offset(self, measurement: str) -> int:
        """Return the first packed feature index for a measurement label.
        Raises ValueError for an unknown label.
        """
        ...

class Log1p:
    """Describe the fitted numerical transform log(1 + x).

    Notes
    -----
    Requires nonnegative inputs. Inverse direction uses expm1. This is a tensor
    transform descriptor; it does not perform physical detector corrections.
    """
    def __init__(self) -> None:
        """Initialize Log1p; see the class documentation for parameters and units.
        """
        ...

class MinMaxScale:
    """Describe fitted per-feature min-max scaling.

    Parameters
    ----------
    minimum : sequence of float
        Fitted input minima; one scalar or one value per feature.
    maximum : sequence of float
        Fitted maxima, larger than corresponding minima.
    low : float, default=0.0
        Destination lower bound.
    high : float, default=1.0
        Destination upper bound.
    clip : bool, default=False
        Clamp transformed values; clipping prevents exact inversion.

    Notes
    -----
    Maps x to low + (x-minimum)*(high-low)/(maximum-minimum). Coefficients are
    recorded from training and never refitted during inference.
    """
    minimum: list[float]
    maximum: list[float]
    low: float
    high: float
    clip: bool
    def __init__(self, minimum: Sequence[float], maximum: Sequence[float], low: float = ..., high: float = ..., clip: bool = ...) -> None:
        """Initialize MinMaxScale; see the class documentation for parameters and units.
        """
        ...

class StandardScale:
    """Describe fitted standardization (x - mean) / deviation.

    Parameters
    ----------
    mean : sequence of float
        Fitted means; scalar or one per feature.
    deviation : sequence of float
        Positive finite fitted deviations with compatible broadcasting.

    Notes
    -----
    Inverse direction restores x*deviation + mean. Layer schemas broadcast
    coefficients independently over each nonempty layer.
    """
    mean: list[float]
    deviation: list[float]
    def __init__(self, mean: Sequence[float], deviation: Sequence[float], /) -> None:
        """Initialize StandardScale; see the class documentation for parameters and units.
        """
        ...

class MultiplyScale:
    """Describe numerical multiplication by fitted factors.

    Parameters
    ----------
    factors : sequence of float
        One broadcast factor or one per feature.

    Notes
    -----
    Inverse direction divides by factors and requires nonzero factors. This is
    a numerical transform, distinct from physical particlesSr exposure scaling.
    """
    factors: list[float]
    def __init__(self, factors: Sequence[float], /) -> None:
        """Initialize MultiplyScale; see the class documentation for parameters and units.
        """
        ...

class CustomTransform:
    """Describe an application-registered numerical transform.

    Attributes
    ----------
    identifier : str
        Kernel registration identifier.
    version : int, default=1
        Kernel contract version.
    parameters : dict[str, list[float]]
        Named numeric coefficient arrays.

    Notes
    -----
    Packages contain descriptors, not executable code. A runtime transform
    requires a matching C++ kernel registered by the consuming application.
    """
    identifier: str
    version: int
    parameters: dict[str, list[float]]
    def __init__(self) -> None:
        """Initialize CustomTransform; see the class documentation for parameters and units.
        """
        ...

TensorTransform: TypeAlias = Log1p | MinMaxScale | StandardScale | MultiplyScale | CustomTransform

class TransformPipeline:
    """Describe fitted numerical operations and their execution boundary.

    Parameters
    ----------
    steps : sequence of tensor transforms, optional
        Ordered fitted operations.
    direction : TransformDirection, default=FORWARD
        Apply or invert the stored operations.
    location : TransformLocation, default=RUNTIME
        Execute in native processing or declare graph-embedded operations.

    Notes
    -----
    Numerical transforms preserve feature width. Physical spectrum operations
    are declared separately through SpectrumProcessing.
    """
    steps: list[TensorTransform]
    direction: TransformDirection
    location: TransformLocation
    def __init__(self, steps: Sequence[TensorTransform] = ..., direction: TransformDirection = ..., location: TransformLocation = ...) -> None:
        """Initialize TransformPipeline; see the class documentation for parameters and units.
        """
        ...

class SpectrumTransform:
    """Associate one numerical transform pipeline with a detector label.

    Parameters
    ----------
    measurement : str
        Measurement identifier.
    transform : TransformPipeline
        Fitted numerical operations for this spectrum.
    """
    measurement: str
    transform: TransformPipeline
    def __init__(self, measurement: str, transform: TransformPipeline, /) -> None:
        """Initialize SpectrumTransform; see the class documentation for parameters and units.
        """
        ...

class ExposureScale:
    """Describe physical scaling by the ratio of particlesSr exposures.

    Notes
    -----
    Requires positive reference and requested exposures. Forward preparation maps
    reference to requested exposure; inverse preparation maps measured to reference.
    Ordering relative to nonlinear pileup matters.
    """
    def __init__(self) -> None:
        """Initialize ExposureScale; see the class documentation for parameters and units.
        """
        ...

class Rebin:
    """Describe physical integrated-count rebinning between detector grids.

    Parameters
    ----------
    channels : int, default=0
        Destination channels; zero selects the measurement schema channel count.
    require_full_coverage : bool, default=True
        Require the source energy range to cover the destination grid.

    Notes
    -----
    Forward preparation targets the requested calibration; inverse preparation
    targets reference calibration. No angle change is represented by this operation.
    """
    channels: int
    require_full_coverage: bool
    def __init__(self, channels: int = ..., require_full_coverage: bool = ...) -> None:
        """Initialize Rebin; see the class documentation for parameters and units.
        """
        ...

class AcquisitionSource(Enum):
    """Select the times used by a physical pileup operation.

    Notes
    -----
    REQUESTED uses inference-request real/live times. REFERENCE uses package
    reference times. This does not change the exposure-scaling policy.
    """
    REQUESTED = 0
    REFERENCE = 1

class Pileup:
    """Describe pair-sum pileup applied to a bare reference spectrum.

    Parameters
    ----------
    tau_seconds : float
        Nonnegative resolving time in seconds.
    acquisition : AcquisitionSource, default=REQUESTED
        Source of positive real/live times.

    Notes
    -----
    Expands N channels to the complete 2*N-1 support and changes physical state
    to piled-up. Execution follows the declared per-detector operation order.
    """
    tau_seconds: float
    acquisition: AcquisitionSource
    def __init__(self, tau_seconds: float, acquisition: AcquisitionSource = ...) -> None:
        """Initialize Pileup; see the class documentation for parameters and units.
        """
        ...

class RemovePileup:
    """Describe inversion of a complete pair-sum piled-up spectrum.

    Parameters
    ----------
    tau_seconds : float
        Resolving time in seconds, matching the forward pileup model.
    relative_tolerance : float, default=1e-7
        Forward-reconstruction consistency tolerance.
    acquisition : AcquisitionSource, default=REQUESTED
        Source of real/live acquisition times.

    Notes
    -----
    Requires complete 2*N-1 support and reconstructs N bare bins. Truncated or
    incompatible spectra are rejected; arbitrary pileup deconvolution is unsupported.
    """
    tau_seconds: float
    relative_tolerance: float
    acquisition: AcquisitionSource
    def __init__(self, tau_seconds: float, relative_tolerance: float = ..., acquisition: AcquisitionSource = ...) -> None:
        """Initialize RemovePileup; see the class documentation for parameters and units.
        """
        ...

SpectrumOperation: TypeAlias = ExposureScale | Rebin | Pileup | RemovePileup

class SpectrumProcessing:
    """Associate ordered physical spectrum operations with one measurement.

    Parameters
    ----------
    measurement : str
        Detector measurement label.
    steps : sequence of spectrum operations
        Ordered exposure, rebin and pileup operations.

    Notes
    -----
    Different detectors can use different pipelines. Operations are not
    automatically reordered; nonlinear corrections make order scientifically relevant.
    """
    measurement: str
    steps: list[SpectrumOperation]
    def __init__(self, measurement: str, steps: Sequence[SpectrumOperation], /) -> None:
        """Initialize SpectrumProcessing; see the class documentation for parameters and units.
        """
        ...

class EdpSchema:
    """Declare a fixed-maximum-layer elemental depth-profile output.

    Parameters
    ----------
    elements : sequence of str
        Ordered element symbols.
    layers : int
        Positive fixed maximum number of output layers.
    uncertainty : bool, default=False
        Pack means, presence probabilities and posterior standard deviations.
    unit : str, default="1e15 atoms/cm2"
        Physical elemental areal-density unit.

    Notes
    -----
    Unlike a variable-layer forward schema, this declares a finite output layer
    count. Values are ordered surface to depth, then by element.
    """
    elements: list[str]
    layers: int
    uncertainty: bool
    unit: str
    def __init__(self, elements: Sequence[str], layers: int, uncertainty: bool = ..., unit: str = ...) -> None:
        """Initialize EdpSchema; see the class documentation for parameters and units.
        """
        ...

class ParameterInputSchema:
    """Native fixed sample and physical parameter packing.

    Attributes
    ----------
    reference_sample : Sample
        Reference sample for unvaried properties.
    parameters : list[Parameter]
        Ordered physical graph columns.

    Notes
    -----
    The public Python metadata facade offers dataclass construction and field
    context for conversion errors. Complete validation occurs at metadata/package boundaries.
    """
    reference_sample: Sample
    parameters: list[Parameter]
    def __init__(self) -> None:
        """Initialize ParameterInputSchema; see the class documentation for parameters and units.
        """
        ...

class IsotopeData:
    """Read-only isotope database record.

    Attributes
    ----------
    massNumber : int
        Isotope mass number.
    exactMass : float
        Atomic mass in atomic mass units (u).
    """
    massNumber: int
    exactMass: float

def natural_isotopes(element: str) -> list[Isotope]:
    """Return native natural-isotope fraction records for an element.

    Parameters
    ----------
    element : str
        Canonical symbol.

    Returns
    -------
    list[Isotope]
        Owned records; empty when no natural default exists. Unknown symbols raise ValueError.
    """
    ...
def isotope_data(element: str) -> list[IsotopeData]:
    """Return read-only native isotope identity/mass records.

    Parameters
    ----------
    element : str
        Canonical element symbol.

    Returns
    -------
    list[IsotopeData]
        Supported masses in atomic mass units. Unknown elements raise ValueError.
    """
    ...
def validate_isotopes(element: str, isotopes: list[Isotope]) -> None:
    """Validate an explicit or default isotope composition.

    Parameters
    ----------
    element : str
        Canonical symbol.
    values : sequence of Isotope
        Unique supported mass numbers and normalized finite nonnegative fractions.
        Empty requests natural composition.

    Raises
    ------
    ValueError
        Composition is invalid or no requested default is available.
    """
    ...

class ElementComposition:
    """Native fixed isotope assumption for one element.

    Attributes
    ----------
    element : str
        Supported element symbol.
    isotopes : list[Isotope]
        Fixed mixture; empty resolves to natural composition.

    Notes
    -----
    The public Python metadata facade offers dataclass construction and field
    context for conversion errors. Complete validation occurs at metadata/package boundaries.
    """
    element: str
    isotopes: list[Isotope]
    def __init__(self) -> None:
        """Initialize ElementComposition; see the class documentation for parameters and units.
        """
        ...

class LayerInputSchema:
    """Native unbounded layer-sequence input packing.

    Attributes
    ----------
    elements : list[str]
        Ordered elemental AD features in 1e15 atoms/cm2.
    isotope_compositions : list[ElementComposition]
        Fixed per-element mixtures; omitted entries resolve to natural composition.

    Notes
    -----
    The public Python metadata facade offers dataclass construction and field
    context for conversion errors. Complete validation occurs at metadata/package boundaries.
    """
    elements: list[str]
    isotope_compositions: list[ElementComposition]
    def __init__(self) -> None:
        """Initialize LayerInputSchema; see the class documentation for parameters and units.
        """
        ...

class ForwardModelMetadata:
    """Native forward scientific model contract.

    Attributes
    ----------
    reference_setup : ExperimentalSetup
        Reference measurements including geometry.
    input_preparation : ParameterInputSchema or LayerInputSchema
        Physical sample packing.
    output_spectra : SpectraSchema
        Packed reference spectra.
    input_transform : TransformPipeline
        Fitted graph input transforms.
    output_transform : list[SpectrumTransform]
        Per-detector numerical output transforms.
    output_preparation : list[SpectrumProcessing]
        Per-detector physical output corrections.

    Notes
    -----
    The public Python metadata facade offers dataclass construction and field
    context for conversion errors. Complete validation occurs at metadata/package boundaries.
    """
    reference_setup: ExperimentalSetup
    input_preparation: ParameterInputSchema | LayerInputSchema
    output_spectra: SpectraSchema
    input_transform: TransformPipeline
    output_transform: list[SpectrumTransform]
    output_preparation: list[SpectrumProcessing]
    def __init__(self) -> None:
        """Initialize ForwardModelMetadata; see the class documentation for parameters and units.
        """
        ...

class InverseModelMetadata:
    """Native inverse scientific model contract.

    Attributes
    ----------
    reference_setup : ExperimentalSetup
        Reference setup and geometry.
    input_spectra : SpectraSchema
        Packed reference input spectra.
    input_preparation : list[SpectrumProcessing]
        Measured-to-reference physical corrections.
    input_transform : list[SpectrumTransform]
        Per-detector numerical input transforms.
    output_transform : TransformPipeline
        Numerical EDP output processing.
    output_preparation : EdpSchema
        Fixed-layer EDP layout and units.

    Notes
    -----
    The public Python metadata facade offers dataclass construction and field
    context for conversion errors. Complete validation occurs at metadata/package boundaries.
    """
    reference_setup: ExperimentalSetup
    input_spectra: SpectraSchema
    input_preparation: list[SpectrumProcessing]
    input_transform: list[SpectrumTransform]
    output_transform: TransformPipeline
    output_preparation: EdpSchema
    def __init__(self) -> None:
        """Initialize InverseModelMetadata; see the class documentation for parameters and units.
        """
        ...

class Direction(Enum):
    """Identify a model scientific mapping.

    Notes
    -----
    FORWARD maps sample parameters to spectra; INVERSE maps spectra to an
    elemental depth profile.
    """
    FORWARD = 0
    INVERSE = 1

class ModelMetadata:
    """Native directional scientific contract and graph identifiers.

    Attributes
    ----------
    model : ForwardModelMetadata or InverseModelMetadata
        Directional contract.
    input_name : str, default="inputs"
        Single graph input name.
    output_name : str, default="outputs"
        Single graph output name.
    provenance : str
        User/model provenance.

    Notes
    -----
    The public Python metadata facade offers dataclass construction and field
    context for conversion errors. Complete validation occurs at metadata/package boundaries.
    """
    model: ForwardModelMetadata | InverseModelMetadata
    input_name: str
    output_name: str
    provenance: str
    @property
    def direction(self) -> Direction:
        """Return FORWARD or INVERSE from the active directional metadata.
        """
        ...
    def __init__(self, model: ForwardModelMetadata | InverseModelMetadata = ...) -> None:
        """Initialize ModelMetadata; see the class documentation for parameters and units.
        """
        ...
    def validate(self) -> None:
        """Validate the directional scientific contract without executing ONNX.
        Checks schemas, mixtures, geometry, units and processing; raises ValueError.
        """
        ...
    def input_width(self) -> int:
        """Return physical-parameter, per-layer element or packed spectral width.
        """
        ...
    def output_width(self) -> int:
        """Return packed graph output width before physical spectrum preparation.
        """
        ...
    def input_rank(self) -> int:
        """Return 3 for a layer sequence and 2 for other graph inputs.
        """
        ...
    def layer_input(self) -> bool:
        """Return True when this is a forward variable-layer input contract.
        """
        ...
