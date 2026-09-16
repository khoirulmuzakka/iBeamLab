"""Python interface to the native iBeamLab libraries."""

try:
    from . import _ibeamlab_cpp as native
except ImportError as error:
    raise ImportError(
        "The native iBeamLab extension is not built. Run `compile.bat Release` first."
    ) from error

sample = native.sample
simulator = native.simulator
generation = native.generation
datasets = native.datasets
preprocessing = native.preprocessing
model = native.model
inference = native.inference

__all__ = [
    "native", "sample", "simulator", "generation", "datasets",
    "preprocessing", "model", "inference",
]
