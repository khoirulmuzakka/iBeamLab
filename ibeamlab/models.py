"""Forward and inverse execution with labelled spectra and NumPy predictions."""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ._engine import engine, to_core
from .exceptions import InferenceError
from .metadata import ModelMetadata
from .package import Package
from .simulator import Spectrum, result_from_core


def _spectra(values):
    if isinstance(values, Mapping):
        result = []
        for label, spectrum in values.items():
            if not isinstance(spectrum, Spectrum):
                spectrum = Spectrum(label, spectrum)
            if label != spectrum.measurement:
                raise ValueError("spectrum mapping key must match its measurement label")
            result.append(to_core(spectrum))
        return result
    return [to_core(s) for s in values]


@dataclass(frozen=True, slots=True)
class InverseInput:
    """Measured setup and labelled spectra, supplied as a mapping or sequence.

    pileup_already_removed skips the first requested-acquisition RemovePileup
    step for every measurement. Caller spectrum order does not affect packing.
    """
    setup: object
    spectra: object
    pileup_already_removed: bool = False

    def to_core(self):
        if not isinstance(self.pileup_already_removed, bool):
            raise ValueError("pileup_already_removed must be boolean")
        value = engine().InverseInput()
        value.setup, value.spectra = to_core(self.setup), _spectra(self.spectra)
        value.pileup_already_removed = self.pileup_already_removed
        return value


@dataclass(frozen=True, slots=True)
class InversePrediction:
    """Physical float64 arrays shaped [layer,element], surface to depth."""
    elements: tuple[str, ...]
    values: np.ndarray
    presence_probability: np.ndarray
    posterior_std: np.ndarray
    uncertainty_predicted: bool
    unit: str

    @classmethod
    def from_core(cls, value):
        return cls(tuple(value.elements), value.values_array,
                   value.presence_probability_array,
                   value.posterior_std_array,
                   value.uncertainty_predicted, value.unit)


class Model:
    """Loaded model; create one instance per worker rather than calling concurrently."""
    def __init__(self, package, *, threads=1):
        self.package = Package.open(package) if isinstance(package, (str, Path)) else package
        self._package = to_core(self.package)
        self.metadata = ModelMetadata.from_core(self._package.metadata)
        self.threads = threads


class ForwardModel(Model):
    """Physical experiment to one or several independently processed spectra."""
    def __init__(self, package, *, threads=1):
        super().__init__(package, threads=threads)
        try:
            self._core = engine().ForwardModel(self._package, threads)
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict(self, experiment):
        try:
            return result_from_core(self._core.predict(to_core(experiment)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_batch(self, experiments):
        try:
            return [result_from_core(r) for r in self._core.predict_batch([to_core(x) for x in experiments])]
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error


class InverseModel(Model):
    """Measured multi-spectrum input to physical elemental depth profiles."""
    def __init__(self, package, *, threads=1):
        super().__init__(package, threads=threads)
        try:
            self._core = engine().InverseModel(self._package, threads)
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict(self, inputs):
        try:
            return InversePrediction.from_core(self._core.predict(to_core(inputs)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_batch(self, inputs):
        try:
            return [InversePrediction.from_core(r) for r in self._core.predict_batch([to_core(x) for x in inputs])]
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_prepared(self, spectra):
        """Skip physical preparation; retain packed-tensor preprocessing and EDP processing."""
        try:
            return InversePrediction.from_core(self._core.predict_prepared(_spectra(spectra)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error


def load_model(path, *, threads=1):
    """Read the package direction and construct the matching inference model."""
    package = Package.open(path)
    cls = ForwardModel if package.to_core().metadata.direction == engine().Direction.FORWARD else InverseModel
    return cls(package, threads=threads)
