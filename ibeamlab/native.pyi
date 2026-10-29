"""Typed metadata descriptors and domain values exposed by the lazy native facade.

Other low-level bindings remain dynamically accessible. This file describes
types only; runtime imports still defer loading the compiled extension.
"""
from collections.abc import Sequence
from enum import Enum
from typing import Any, TypeAlias

def __getattr__(name: str) -> Any: ...

class ExecutionProvider(Enum):
    CPU = 0
    CUDA = 1

class InferenceOptions:
    execution_provider: ExecutionProvider
    device_id: int
    intra_op_threads: int
    inter_op_threads: int
    enable_graph_optimizations: bool
    allow_tf32: bool
    def __init__(self) -> None: ...
    def validate(self) -> None: ...

def available_execution_providers() -> list[str]: ...

class Isotope:
    massNumber: int
    exactMass: float
    fraction: float
    def __init__(self) -> None: ...

class Species:
    element: str
    concentration: float
    isotopes: list[Isotope]
    def __init__(self) -> None: ...

class Layer:
    thickness: float
    roughness: float
    porosityFraction: float
    poreDiameter: float
    species: list[Species]
    def __init__(self) -> None: ...

class Sample:
    layers: list[Layer]
    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class Beam:
    particle: str
    energy: float
    spread: float
    def __init__(self) -> None: ...

class Detector:
    calibrationLinear: float
    calibrationOffset: float
    calibrationQuadratic: float
    resolution: float
    info: str
    def __init__(self) -> None: ...

class Acquisition:
    particlesSr: float
    realTime: float
    liveTime: float
    def __init__(self) -> None: ...

class Measurement:
    label: str
    beam: Beam
    detector: Detector
    acquisition: Acquisition
    def __init__(self) -> None: ...

class ExperimentalSetup:
    measurements: list[Measurement]
    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class LayerThickness:
    layer: int
    def __init__(self, layer: int, /) -> None: ...

class SpeciesConcentration:
    layer: int
    element: str
    def __init__(self, layer: int, element: str, /) -> None: ...

class BeamEnergy:
    measurement: str
    def __init__(self, measurement: str, /) -> None: ...

class BeamSpread:
    measurement: str
    def __init__(self, measurement: str, /) -> None: ...

class CalibrationLinear:
    measurement: str
    def __init__(self, measurement: str, /) -> None: ...

class CalibrationOffset:
    measurement: str
    def __init__(self, measurement: str, /) -> None: ...

class CalibrationQuadratic:
    measurement: str
    def __init__(self, measurement: str, /) -> None: ...

class DetectorResolution:
    measurement: str
    def __init__(self, measurement: str, /) -> None: ...

class ParticlesSr:
    measurement: str
    def __init__(self, measurement: str, /) -> None: ...

ParameterTarget: TypeAlias = LayerThickness | SpeciesConcentration | BeamEnergy | BeamSpread | CalibrationLinear | CalibrationOffset | CalibrationQuadratic | DetectorResolution | ParticlesSr

class Parameter:
    name: str
    target: ParameterTarget
    unit: str
    def __init__(self, name: str, target: ParameterTarget, unit: str = ...) -> None: ...

class SpectrumSpec:
    measurement: str
    channels: int
    def __init__(self, measurement: str, channels: int, /) -> None: ...

class SpectraSchema:
    spectra: list[SpectrumSpec]
    def __init__(self, spectra: Sequence[SpectrumSpec] = ..., /) -> None: ...
    def width(self) -> int: ...
    def offset(self, measurement: str) -> int: ...

class Log1p:
    def __init__(self) -> None: ...

class MinMaxScale:
    minimum: list[float]
    maximum: list[float]
    low: float
    high: float
    clip: bool
    def __init__(self, minimum: Sequence[float], maximum: Sequence[float], low: float = ..., high: float = ..., clip: bool = ...) -> None: ...

class StandardScale:
    mean: list[float]
    deviation: list[float]
    def __init__(self, mean: Sequence[float], deviation: Sequence[float], /) -> None: ...

class MultiplyScale:
    factors: list[float]
    def __init__(self, factors: Sequence[float], /) -> None: ...

class CustomTransform:
    identifier: str
    version: int
    parameters: dict[str, list[float]]
    def __init__(self) -> None: ...

TensorTransform: TypeAlias = Log1p | MinMaxScale | StandardScale | MultiplyScale | CustomTransform

class MultiplyCounts:
    factor: float
    def __init__(self, factor: float, /) -> None: ...

class ExposureScale:
    def __init__(self) -> None: ...

class Rebin:
    channels: int
    require_full_coverage: bool
    def __init__(self, channels: int = ..., require_full_coverage: bool = ...) -> None: ...

class AcquisitionSource(Enum):
    REQUESTED = 0
    REFERENCE = 1

class Pileup:
    tau_seconds: float
    acquisition: AcquisitionSource
    def __init__(self, tau_seconds: float, acquisition: AcquisitionSource = ...) -> None: ...

class RemovePileup:
    tau_seconds: float
    relative_tolerance: float
    acquisition: AcquisitionSource
    def __init__(self, tau_seconds: float, relative_tolerance: float = ..., acquisition: AcquisitionSource = ...) -> None: ...

SpectrumOperation: TypeAlias = MultiplyCounts | ExposureScale | Rebin | Pileup | RemovePileup

class SpectrumProcessing:
    measurement: str
    steps: list[SpectrumOperation]
    def __init__(self, measurement: str, steps: Sequence[SpectrumOperation], /) -> None: ...

class ScaleEdp:
    factor: float
    def __init__(self, factor: float, /) -> None: ...

class ParameterInputSchema:
    reference_sample: Sample
    parameters: list[Parameter]
    preprocessing: list[TensorTransform]
    def __init__(self) -> None: ...

class LayerInputSchema:
    elements: list[str]
    preprocessing: list[TensorTransform]
    def __init__(self) -> None: ...

class ForwardModelMetadata:
    reference_setup: ExperimentalSetup
    input: ParameterInputSchema | LayerInputSchema
    output_spectra: SpectraSchema
    postprocessing: list[SpectrumProcessing]
    def __init__(self) -> None: ...

class InverseModelMetadata:
    reference_setup: ExperimentalSetup
    input_spectra: SpectraSchema
    elements: list[str]
    layers: int
    uncertainty: bool
    input_preparation: list[SpectrumProcessing]
    preprocessing: list[TensorTransform]
    postprocessing: list[ScaleEdp]
    def __init__(self) -> None: ...

class Direction(Enum):
    FORWARD = 0
    INVERSE = 1

class ModelMetadata:
    model: ForwardModelMetadata | InverseModelMetadata
    input_name: str
    output_name: str
    provenance: str
    @property
    def direction(self) -> Direction: ...
    def __init__(self, model: ForwardModelMetadata | InverseModelMetadata = ...) -> None: ...
    def validate(self) -> None: ...
    def input_width(self) -> int: ...
    def output_width(self) -> int: ...
    def input_rank(self) -> int: ...
    def layer_input(self) -> bool: ...
