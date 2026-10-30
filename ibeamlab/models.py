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
        """Execute one ONNX batch; differing layer counts are zero-padded."""
        try:
            return [result_from_core(r) for r in self._core.predict_batch([to_core(x) for x in experiments])]
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference(self, experiment):
        """Physical input to reference-condition spectra; numerical transforms still run."""
        try:
            return result_from_core(self._core.predict_reference(to_core(experiment)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference_batch(self, experiments):
        """One ONNX population call, returning physical reference spectra."""
        try:
            return [result_from_core(r) for r in self._core.predict_batch(
                [to_core(x) for x in experiments], engine().PredictionDomain.REFERENCE)]
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference_areal_density(self, densities):
        """Physical AD [layers,elements] to spectra before measurement corrections."""
        values = np.ascontiguousarray(densities, dtype=np.float32)
        try:
            return result_from_core(self._core.predict_areal_density(values, None, engine().PredictionDomain.REFERENCE))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference_areal_density_batch(self, densities, *, out=None):
        """One graph call for AD [population,layers,elements], with reusable output.

        out is writable contiguous float64 [population,sum(reference channels)].
        Returned per-detector arrays view out; all training transforms execute
        exactly once at their declared graph/runtime location. No physical output
        preparation (pileup, exposure or rebinning) is executed.
        """
        values = np.ascontiguousarray(densities, dtype=np.float32)
        try:
            return dict(self._core.predict_areal_density_batch(values, None, out, engine().PredictionDomain.REFERENCE))
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

    def predict_areal_density_batch(self, densities, *, setup=None, out=None):
        """One ONNX call for physical AD [population,layers,elements].

        Returns {measurement: float64 [population,processed_channels]} arrays.
        Element order follows metadata. setup applies to the whole population.
        Pass writable contiguous float64 out [population,total_processed_channels]
        to reuse output storage across DE iterations. Returned arrays view that
        storage and change when it is reused. Without out, results own new storage.
        Contiguous float32 input is borrowed; transforms use retained scratch.
        One model instance per fitting worker; calls on an instance must serialize.
        """
        values = np.ascontiguousarray(densities, dtype=np.float32)
        try:
            return dict(self._core.predict_areal_density_batch(values, to_core(setup), out))
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

    def predict_reference(self, spectra):
        """Reference-condition spectra to physical EDP; training transforms still run."""
        try:
            return InversePrediction.from_core(self._core.predict_reference(_spectra(spectra)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference_batch(self, spectra):
        """One ONNX call for a sequence of reference spectrum mappings."""
        try:
            inputs = []
            for item in spectra:
                value = engine().InverseInput()
                value.setup = self._package.metadata.model.reference_setup
                value.spectra = _spectra(item)
                inputs.append(value)
            return [InversePrediction.from_core(r) for r in self._core.predict_batch(
                inputs, engine().PredictionDomain.REFERENCE)]
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error


def load_model(path, *, options: InferenceOptions | core.InferenceOptions | None = None):
    """Read the package direction and construct the matching inference model."""
    package = Package.open(path)
    cls = ForwardModel if package.to_core().metadata.direction == engine().Direction.FORWARD else InverseModel
    return cls(package, options=options)
