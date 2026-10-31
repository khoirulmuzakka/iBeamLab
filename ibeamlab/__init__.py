"""Python API for ion-beam simulation, dataset generation and model inference.

Physical descriptions use keV, degrees, seconds and 1e15 atoms/cm2. Element
concentrations and isotope fractions have separate normalization constraints.
The main workflow uses Sample, Experiment, Simulator, GenerationPlan, Package
and ForwardModel/InverseModel. Native descriptors are loaded lazily on access;
optional PyTorch and ONNX dependencies are imported when their features are used.

Numerical training transforms and physical detector preparation are separate
model stages. Device/thread settings belong to InferenceOptions. See each public
class for array shapes, ownership and fixed scientific assumptions.
"""
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING
from .exceptions import (
    IBeamLabError, ValidationError, ConfigurationError, NativeExtensionError,
    SimulationError, SimnraUnavailableError, DatasetError, ModelPackageError, InferenceError,
)
from .sample import natural_isotopes, isotope_data, Geometry, Isotope, Species, Layer, Sample, Beam, Detector, Acquisition, Measurement, ExperimentalSetup, Experiment
from .simulator import Spectrum, SimulationFailure, SimulationResult, Simulator, DummySimulator, SimnraMethod, SimnraSimulator
from .parameters import ParameterSpec, Parameter, ParameterTarget, vary
from .generation import FailurePolicy, GenerationPlan, GenerationSummary, generate
from .datasets import Dataset, DatasetMetadata, DatasetRecord, open_dataset
from .metadata import ElementComposition, ForwardModelMetadata, InverseModelMetadata, ModelMetadata, ParameterInputSchema, LayerInputSchema
from .metadata import TensorTransform, TensorPipeline, SpectrumOperation, SpectrumPipeline, SpectrumTransforms
from .package import Package
from .models import Model, ForwardModel, InverseInput, InverseModel, InversePrediction, load_model
from .inference import InferenceOptions, available_execution_providers
from .export import export_package
from .processing import rebin, apply_pileup, remove_pileup

if TYPE_CHECKING:
    from .native import (
        LayerThickness as LayerThickness, SpeciesConcentration as SpeciesConcentration,
        BeamEnergy as BeamEnergy, BeamSpread as BeamSpread, CalibrationLinear as CalibrationLinear,
        CalibrationOffset as CalibrationOffset, CalibrationQuadratic as CalibrationQuadratic,
        DetectorResolution as DetectorResolution, ParticlesSr as ParticlesSr,
        IncidenceAngle as IncidenceAngle, ExitAngle as ExitAngle, ScatteringAngle as ScatteringAngle,
        SpectrumSpec as SpectrumSpec, SpectraSchema as SpectraSchema, Log1p as Log1p,
        MinMaxScale as MinMaxScale, StandardScale as StandardScale, MultiplyScale as MultiplyScale,
        CustomTransform as CustomTransform, ExposureScale as ExposureScale,
        Rebin as Rebin, Pileup as Pileup, RemovePileup as RemovePileup,
        AcquisitionSource as AcquisitionSource, SpectrumProcessing as SpectrumProcessing,
        Direction as Direction, TransformPipeline as TransformPipeline, SpectrumTransform as SpectrumTransform,
        TransformDirection as TransformDirection, TransformLocation as TransformLocation,
        SpectrumState as SpectrumState, EdpSchema as EdpSchema, PredictionDomain as PredictionDomain,
    )

try:
    __version__ = version("ibeamlab")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

_DESCRIPTORS = {
    "LayerThickness", "SpeciesConcentration", "BeamEnergy", "BeamSpread", "CalibrationLinear",
    "CalibrationOffset", "CalibrationQuadratic", "DetectorResolution", "ParticlesSr", "IncidenceAngle", "ExitAngle", "ScatteringAngle",
    "ParameterBounds", "SpectrumSpec", "SpectraSchema", "Log1p", "MinMaxScale", "StandardScale",
    "MultiplyScale", "CustomTransform", "ExposureScale", "Rebin", "Pileup",
    "RemovePileup", "AcquisitionSource", "SpectrumProcessing", "TransformPipeline", "SpectrumTransform", "TransformDirection",
    "TransformLocation", "SpectrumState", "EdpSchema", "PredictionDomain", "Direction", "PileupWorkspace",
}


def __getattr__(name):
    if name in _DESCRIPTORS:
        from ._engine import engine
        value = getattr(engine(), name)
        globals()[name] = value
        return value
    if name in {"fit", "TrainingHistory"}:
        from . import training
        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Geometry", "ElementComposition", "natural_isotopes", "isotope_data", "Isotope", "Species", "Layer", "Sample", "Beam", "Detector", "Acquisition", "Measurement",
    "ExperimentalSetup", "Experiment", "Spectrum", "SimulationFailure", "SimulationResult", "Simulator",
    "DummySimulator", "SimnraMethod", "SimnraSimulator", "ParameterSpec", "Parameter", "vary",
    "FailurePolicy", "GenerationPlan", "GenerationSummary", "generate", "Dataset", "DatasetMetadata",
    "DatasetRecord", "open_dataset", "ForwardModelMetadata", "InverseModelMetadata", "ModelMetadata",
    "ParameterInputSchema", "LayerInputSchema",
    "ParameterTarget", "TensorTransform", "TensorPipeline", "SpectrumOperation", "SpectrumPipeline", "SpectrumTransforms",
    "Package", "Model", "ForwardModel", "InverseInput", "InverseModel", "InversePrediction", "load_model",
    "InferenceOptions", "available_execution_providers",
    "export_package", "rebin", "apply_pileup", "remove_pileup", "fit", "TrainingHistory",
    "IBeamLabError", "ValidationError", "ConfigurationError", "NativeExtensionError", "SimulationError",
    "SimnraUnavailableError", "DatasetError", "ModelPackageError", "InferenceError",
] + sorted(_DESCRIPTORS)
