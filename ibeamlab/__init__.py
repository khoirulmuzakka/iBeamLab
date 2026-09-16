"""Python interface to the native iBeamLab simulation and ML pipeline."""

try:
    from . import _ibeamlab_cpp as native
except ImportError as error:
    raise ImportError(
        "The native iBeamLab extension is not built. Run `compile.bat --release` first."
    ) from error

# Import the readable Python facades instead of publishing pybind11 submodules.
from . import datasets, generation, inference, model, preprocessing, sample, simulator

__all__ = [
    "datasets", "generation", "inference", "model", "native",
    "preprocessing", "sample", "simulator",
]
