"""Python interface for ion-beam simulation, datasets, and inference."""

from importlib.metadata import PackageNotFoundError, version
from .configuration import GenerationConfiguration, SetupVariation, load_generation_configuration
from .datasets import Dataset, DatasetMetadata, DatasetRecord, FailureRecord, open_dataset
from .exceptions import (
    ConfigurationError, DatasetError, IBeamLabError, InferenceError,
    ModelPackageError, NativeExtensionError, SimnraUnavailableError,
    SimulationError, ValidationError,
)
from .generation import FailurePolicy, GenerationProgress, GenerationStudy, GenerationSummary, Parameter, vary
from .models import ForwardModel, InverseModel, InversePrediction, Model, NamedValue, load_model
from .sample import Beam, Detector, Experiment, Isotope, Layer, LinearCalibration, Sample, Species
from .simulator import DummySimulator, SimnraMethod, SimnraSimulator, SimulationFailure, SimulationResult, Simulator, Spectrum

try:
    __version__ = version("ibeamlab")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = [
    "Beam", "ConfigurationError", "Dataset", "DatasetError", "DatasetMetadata",
    "DatasetRecord", "Detector", "DummySimulator", "Experiment", "FailurePolicy",
    "FailureRecord", "ForwardModel", "GenerationConfiguration", "GenerationProgress", "GenerationStudy",
    "GenerationSummary", "IBeamLabError", "InferenceError", "InverseModel",
    "InversePrediction", "Isotope", "Layer", "LinearCalibration", "LRNModel", "Model",
    "ModelPackageError", "NamedValue", "NativeExtensionError", "Parameter", "Sample",
    "SimnraMethod", "SimnraSimulator", "SimnraUnavailableError", "SimulationError",
    "SimulationFailure", "SimulationResult", "Simulator", "Species", "Spectrum",
    "SetupVariation", "ValidationError", "load_generation_configuration", "load_model", "open_dataset", "vary",
]


def __getattr__(name: str):
    """Load the optional PyTorch LRN model only when explicitly requested."""
    if name == "LRNModel":
        from .lrn import LRNModel
        return LRNModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
