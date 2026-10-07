"""High-level loading and execution of packaged ONNX models."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from .exceptions import InferenceError, ModelPackageError
from .sample import Detector, Experiment, _to_native_setup
from .simulator import SimulationResult, Spectrum, _input

@dataclass(frozen=True, slots=True)
class NamedValue:
    name: str
    value: float
    unit: str = ""

@dataclass(frozen=True, slots=True)
class InverseInput:
    """Measured spectra and their detector settings.

    pileup_already_removed applies to every detector in this input and means
    counts are pileup-free with live/real-time scaling already undone.
    """
    spectra: Mapping[str, Spectrum]
    detectors: Sequence[Detector]
    pileup_already_removed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "detectors", tuple(self.detectors))
        if not isinstance(self.pileup_already_removed, bool):
            raise ValueError("pileup_already_removed must be a boolean")


@dataclass(frozen=True, slots=True)
class InversePrediction:
    """Mean densities, presence probabilities, and posterior std from surface to depth."""
    edp: object
    @property
    def values(self):
        import numpy as np
        return np.asarray(self.edp.values, dtype=np.float32)
    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(self.edp.elements)
    @property
    def presence_probability(self):
        import numpy as np
        return np.asarray(self.edp.presence_probability, dtype=np.float32)
    @property
    def posterior_std(self):
        import numpy as np
        return np.asarray(self.edp.posterior_std, dtype=np.float32)
    @property
    def uncertainty_predicted(self) -> bool:
        return self.edp.uncertainty_predicted
    @property
    def unit(self) -> str:
        return self.edp.unit

class Model:
    """Base type returned by :func:`load_model`."""
    def __init__(self, path: str | Path, *, threads: int = 1):
        self.path = Path(path)
        self.threads = threads

class InverseModel(Model):
    """A spectra-to-elemental-depth-profile model."""
    def __init__(self, path: str | Path, *, threads: int = 1):
        super().__init__(path, threads=threads)
        from ._native import native
        options = native.inference.InferenceOptions()
        options.intra_op_threads = threads
        self._native = native.inference.InverseModel(self.path, options)
    @property
    def metadata(self):
        return self._native.metadata

    @staticmethod
    def _spectra(spectra):
        from ._native import native
        row = []
        for label, spectrum in spectra.items():
            if label != spectrum.label:
                raise InferenceError("spectrum mapping key must match its label")
            value = native.simulator.Spectrum()
            value.label, value.counts = spectrum.label, spectrum.counts
            row.append(value)
        return row

    def predict(self, inputs: InverseInput | Sequence[InverseInput]) -> InversePrediction | list[InversePrediction]:
        """Prepare experimental spectra and return physical EDPs."""
        from ._native import native
        single = isinstance(inputs, InverseInput)
        batches = [inputs] if single else inputs
        if isinstance(inputs, Mapping):
            raise InferenceError("predict requires InverseInput with detector settings; use predict_prepared for training-reference spectra")
        try:
            native_batch = []
            for item in batches:
                if not isinstance(item, InverseInput):
                    raise ValueError("predict requires InverseInput values")
                value = native.inference.InverseInput()
                value.spectra = self._spectra(item.spectra)
                value.setup = _to_native_setup(item.detectors)
                value.pileup_already_removed = item.pileup_already_removed
                native_batch.append(value)
            results = [InversePrediction(result.edp) for result in self._native.predict(native_batch)]
        except Exception as error:
            raise InferenceError(str(error)) from error
        return results[0] if single else results

    def predict_prepared(self, spectra: Mapping[str, Spectrum] | Sequence[Mapping[str, Spectrum]]) -> InversePrediction | list[InversePrediction]:
        """Skip physical corrections; packaged input transforms still run once."""
        single = isinstance(spectra, Mapping)
        batches = [spectra] if single else spectra
        try:
            results = [InversePrediction(result.edp) for result in
                       self._native.predict_prepared([self._spectra(item) for item in batches])]
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
    if name == "LRN_AD":
        from .lrn_ad import LRN_AD
        return LRN_AD
    if name == "LRNModel":
        try:
            from .lrn import LRNModel
        except ImportError as error:
            raise ImportError("LRNModel requires the optional dependencies; install ibeamlab[training].") from error
        return LRNModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ForwardModel", "InverseInput", "InverseModel", "InversePrediction", "LRNModel", "LRN_AD", "Model", "NamedValue", "load_model"]
