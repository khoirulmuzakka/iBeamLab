"""Ion-beam simulation, generation, model packaging and inference.

Physical values use keV, seconds and 1e15 atoms/cm2. The compiled extension and
optional PyTorch dependency are loaded only when their capabilities are used.
"""
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING
from .exceptions import (
    IBeamLabError, ValidationError, ConfigurationError, NativeExtensionError,
    SimulationError, SimnraUnavailableError, DatasetError, ModelPackageError, InferenceError,
)
from .sample import Isotope, Species, Layer, Sample, Beam, Detector, Acquisition, Measurement, ExperimentalSetup, Experiment
from .simulator import Spectrum, SimulationFailure, SimulationResult, Simulator, DummySimulator, SimnraMethod, SimnraSimulator
from .parameters import ParameterSpec, Parameter, ParameterTarget, vary
from .generation import FailurePolicy, GenerationPlan, GenerationSummary, generate
from .datasets import Dataset, DatasetMetadata, DatasetRecord, open_dataset
from .metadata import ForwardModelMetadata, InverseModelMetadata, ModelMetadata, ParameterInputSchema, LayerInputSchema
from .metadata import TensorTransform, TensorPipeline, SpectrumOperation, SpectrumPipeline, EdpPipeline
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
        SpectrumSpec as SpectrumSpec, SpectraSchema as SpectraSchema, Log1p as Log1p,
        MinMaxScale as MinMaxScale, StandardScale as StandardScale, MultiplyScale as MultiplyScale,
        CustomTransform as CustomTransform, MultiplyCounts as MultiplyCounts, ExposureScale as ExposureScale,
        Rebin as Rebin, Pileup as Pileup, RemovePileup as RemovePileup,
        AcquisitionSource as AcquisitionSource, SpectrumProcessing as SpectrumProcessing,
        ScaleEdp as ScaleEdp, Direction as Direction,
    )

try:
    __version__ = version("ibeamlab")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

_DESCRIPTORS = {
    "LayerThickness", "SpeciesConcentration", "BeamEnergy", "BeamSpread", "CalibrationLinear",
    "CalibrationOffset", "CalibrationQuadratic", "DetectorResolution", "ParticlesSr",
    "ParameterBounds", "SpectrumSpec", "SpectraSchema", "Log1p", "MinMaxScale", "StandardScale",
    "MultiplyScale", "CustomTransform", "MultiplyCounts", "ExposureScale", "Rebin", "Pileup",
    "RemovePileup", "AcquisitionSource", "SpectrumProcessing", "ScaleEdp", "Direction", "PileupWorkspace",
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
    "Isotope", "Species", "Layer", "Sample", "Beam", "Detector", "Acquisition", "Measurement",
    "ExperimentalSetup", "Experiment", "Spectrum", "SimulationFailure", "SimulationResult", "Simulator",
    "DummySimulator", "SimnraMethod", "SimnraSimulator", "ParameterSpec", "Parameter", "vary",
    "FailurePolicy", "GenerationPlan", "GenerationSummary", "generate", "Dataset", "DatasetMetadata",
    "DatasetRecord", "open_dataset", "ForwardModelMetadata", "InverseModelMetadata", "ModelMetadata",
    "ParameterInputSchema", "LayerInputSchema",
    "ParameterTarget", "TensorTransform", "TensorPipeline", "SpectrumOperation", "SpectrumPipeline", "EdpPipeline",
    "Package", "Model", "ForwardModel", "InverseInput", "InverseModel", "InversePrediction", "load_model",
    "InferenceOptions", "available_execution_providers",
    "export_package", "rebin", "apply_pileup", "remove_pileup", "fit", "TrainingHistory",
    "IBeamLabError", "ValidationError", "ConfigurationError", "NativeExtensionError", "SimulationError",
    "SimnraUnavailableError", "DatasetError", "ModelPackageError", "InferenceError",
] + sorted(_DESCRIPTORS)
