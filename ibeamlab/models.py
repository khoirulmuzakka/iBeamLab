"""Forward and inverse execution with labelled spectra and NumPy predictions."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ._engine import engine, to_core
from .exceptions import InferenceError, ValidationError
from . import native as core
from .metadata import ModelMetadata
from .package import Package
from .simulator import Spectrum, result_from_core
from .inference import InferenceOptions


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
    def __init__(self, package, *, options: InferenceOptions | core.InferenceOptions | None = None) -> None:
        self.package = Package.open(package) if isinstance(package, (str, Path)) else package
        self._package = to_core(self.package)
        self.metadata = ModelMetadata.from_core(self._package.metadata)
        requested = InferenceOptions() if options is None else options
        if not isinstance(requested, (InferenceOptions, engine().InferenceOptions)):
            raise ValidationError("options must be an InferenceOptions value")
        self._options = to_core(requested)
        self.options = InferenceOptions.from_core(self._options)


class ForwardModel(Model):
    """Physical experiment to one or several independently processed spectra."""
    def __init__(self, package, *, options: InferenceOptions | core.InferenceOptions | None = None) -> None:
        super().__init__(package, options=options)
        try:
            self._core = engine().ForwardModel(self._package, self._options)
            self.options = InferenceOptions.from_core(self._core.options)
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

    def predict_areal_density(self, densities, *, setup=None):
        """Physical AD [layers,elements] in metadata order; no thickness column.

        Contiguous float32 arrays are passed through without a Python copy.
        Other array types/layouts are converted once. Caller storage stays unchanged.
        """
        values = np.ascontiguousarray(densities, dtype=np.float32)
        try:
            return result_from_core(self._core.predict_areal_density(values, to_core(setup)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error


class InverseModel(Model):
    """Measured multi-spectrum input to physical elemental depth profiles."""
    def __init__(self, package, *, options: InferenceOptions | core.InferenceOptions | None = None) -> None:
        super().__init__(package, options=options)
        try:
            self._core = engine().InverseModel(self._package, self._options)
            self.options = InferenceOptions.from_core(self._core.options)
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


def load_model(path, *, options: InferenceOptions | core.InferenceOptions | None = None):
    """Read the package direction and construct the matching inference model."""
    package = Package.open(path)
    cls = ForwardModel if package.to_core().metadata.direction == engine().Direction.FORWARD else InverseModel
    return cls(package, options=options)
