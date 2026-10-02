"""High-level loading and execution of packaged ONNX models."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from .exceptions import InferenceError, ModelPackageError
from .sample import Experiment
from .simulator import SimulationResult, Spectrum, _input

@dataclass(frozen=True, slots=True)
class NamedValue:
    name: str
    value: float
    unit: str = ""

@dataclass(frozen=True, slots=True)
class InversePrediction:
    parameters: tuple[NamedValue, ...]
    @property
    def values(self) -> dict[str, float]:
        return {x.name: x.value for x in self.parameters}

class Model:
    """Base type returned by :func:`load_model`."""
    def __init__(self, path: str | Path, *, threads: int = 1):
        self.path = Path(path)
        self.threads = threads

class InverseModel(Model):
    """A spectra-to-physical-parameters model."""
    def __init__(self, path: str | Path, *, threads: int = 1):
        super().__init__(path, threads=threads)
        from ._native import native
        options = native.inference.InferenceOptions()
        options.intra_op_threads = threads
        self._native = native.inference.InverseModel(self.path, options)
    def predict(self, spectra: Mapping[str, Spectrum] | Sequence[Mapping[str, Spectrum]]) -> InversePrediction | list[InversePrediction]:
        single = isinstance(spectra, Mapping)
        batches = [spectra] if single else spectra
        from ._native import native
        native_batch = []
        for item in batches:
            row = []
            for spectrum in item.values():
                value = native.simulator.Spectrum()
                value.label, value.counts = spectrum.label, spectrum.counts
                row.append(value)
            native_batch.append(row)
        try:
            results = [InversePrediction(tuple(NamedValue(x.name, x.value, x.unit) for x in result.parameters))
                       for result in self._native.predict(native_batch)]
        except Exception as error:
            raise InferenceError(str(error)) from error
        return results[0] if single else results

class ForwardModel(Model):
    """An experiment-to-spectra model."""
    def __init__(self, path: str | Path, *, threads: int = 1, correction_threads: int = 1, apply_pileup_on_inference: bool | None = None):
        super().__init__(path, threads=threads)
        from ._native import native
        options = native.inference.InferenceOptions()
        options.intra_op_threads = threads
        options.correction_threads = correction_threads
        options.apply_pileup_on_inference = apply_pileup_on_inference
        self._native = native.inference.ForwardModel(self.path, options)
    def predict(self, experiments: Experiment | Sequence[Experiment]) -> SimulationResult | list[SimulationResult]:
        single = isinstance(experiments, Experiment)
        values = [experiments] if single else experiments
        try:
            results = [SimulationResult([Spectrum(x.label, x.counts) for x in result.spectra])
                       for result in self._native.predict([_input(x) for x in values])]
        except Exception as error:
            raise InferenceError(str(error)) from error
        return results[0] if single else results

def load_model(path: str | Path, *, threads: int = 1, correction_threads: int = 1, apply_pileup_on_inference: bool | None = None) -> Model:
    """Load an inverse or forward model by inspecting its package metadata."""
    from ._native import native
    try:
        metadata = native.model.ModelPackage.open(Path(path)).metadata
        cls = InverseModel if metadata.model_type == native.model.ModelType.INVERSE else ForwardModel
        return cls(path, threads=threads, **({"apply_pileup_on_inference": apply_pileup_on_inference, "correction_threads": correction_threads} if cls is ForwardModel else {}))
    except (InferenceError, ModelPackageError):
        raise
    except Exception as error:
        raise ModelPackageError(f"cannot load model package {path}: {error}") from error

def __getattr__(name: str):
    if name == "LRNModel":
        try:
            from .lrn import LRNModel
        except ImportError as error:
            raise ImportError("LRNModel requires the optional dependencies; install ibeamlab[training].") from error
        return LRNModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ForwardModel", "InverseModel", "InversePrediction", "LRNModel", "Model", "NamedValue", "load_model"]
