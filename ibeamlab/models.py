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
    """Describe measured spectra and setup for inverse inference.

    Parameters
    ----------
    setup : ExperimentalSetup or native ExperimentalSetup
        Measurement conditions at which the spectra were acquired.
    spectra : mapping of str to array_like or sequence of Spectrum
        Finite nonnegative count spectra, identified by measurement label.
    pileup_already_removed : bool, default=False
        Skip the first requested-acquisition RemovePileup operation for each
        measurement. Other preparation and numerical transforms remain active.

    Notes
    -----
    Spectrum input order is independent of metadata packing order. Avoid marking
    pileup as removed unless the supplied data has actually undergone that correction.
    """
    setup: object
    spectra: object
    pileup_already_removed: bool = False

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.InverseInput
            Native value for lower-level workflows.

        Notes
        -----
        Converts setup and spectra into owning native inputs. Count arrays are copied
        at this boundary; ordinary inverse inference performs conversion automatically.
        """
        if not isinstance(self.pileup_already_removed, bool):
            raise ValueError("pileup_already_removed must be boolean")
        value = engine().InverseInput()
        value.setup, value.spectra = to_core(self.setup), _spectra(self.spectra)
        value.pileup_already_removed = self.pileup_already_removed
        return value


@dataclass(frozen=True, slots=True)
class InversePrediction:
    """Expose physical elemental depth profiles and optional graph uncertainty.

    Parameters
    ----------
    elements : tuple[str, ...]
        Ordered element symbols.
    values : numpy.ndarray
        Read-only float64 elemental AD, shaped (layers, elements), surface to depth.
    presence_probability : numpy.ndarray
        Read-only matrix with the same shape, containing layer/element presence
        probabilities. Without graph uncertainty this is derived from positive AD.
    posterior_std : numpy.ndarray
        Read-only physical posterior standard deviations with the same shape.
        Zero when uncertainty was not predicted.
    uncertainty_predicted : bool
        Whether the graph supplied uncertainty channels.
    unit : str
        Physical AD unit, normally "1e15 atoms/cm2".

    Notes
    -----
    layers is the fixed maximum declared by the inverse schema. Numerical
    processing restores physical values and propagates affine standard deviations.
    Arrays retain their native owner. Predicted uncertainty is a model output,
    not a guarantee of statistical calibration.
    """
    elements: tuple[str, ...]
    values: np.ndarray
    presence_probability: np.ndarray
    posterior_std: np.ndarray
    uncertainty_predicted: bool
    unit: str

    @classmethod
    def from_core(cls, value):
        """Construct Python-facing values from a native result.

        Parameters
        ----------
        value : ibeamlab.native.Prediction
            Source value; it is not modified.

        Returns
        -------
        InversePrediction
            Python representation of the supplied value.

        Notes
        -----
        Returns read-only array views retaining the native prediction owner.
        """
        return cls(tuple(value.elements), value.values_array,
                   value.presence_probability_array,
                   value.posterior_std_array,
                   value.uncertainty_predicted, value.unit)


class Model:
    """Hold package metadata and runtime options for an inference session.

    Parameters
    ----------
    package : Package, native Package, str or pathlib.Path
        Owning package or path to a package file.
    options : InferenceOptions or native InferenceOptions, optional
        Execution provider, device, threading and runtime controls.

    Raises
    ------
    ValidationError
        options has an unsupported type.
    ModelPackageError
        A supplied package path cannot be opened.

    Notes
    -----
    Use ForwardModel, InverseModel or load_model() for executable inference.
    Loaded metadata should be treated as read-only. One model instance belongs
    to each concurrent worker; calls on a shared instance must be serialized.
    """
    def __init__(self, package, *, options: InferenceOptions | core.InferenceOptions | None = None) -> None:
        """Load package descriptions and convert session options; see Model.
        """
        self.package = Package.open(package) if isinstance(package, (str, Path)) else package
        self._package = to_core(self.package)
        self.metadata = ModelMetadata.from_core(self._package.metadata)
        requested = InferenceOptions() if options is None else options
        if not isinstance(requested, (InferenceOptions, engine().InferenceOptions)):
            raise ValidationError("options must be an InferenceOptions value")
        self._options = to_core(requested)
        self.options = InferenceOptions.from_core(self._options)


class ForwardModel(Model):
    """Predict labelled detector spectra from a physical sample description.

    Parameters
    ----------
    package : Package, native Package, str or pathlib.Path
        Forward package containing an ONNX graph and scientific metadata.
    options : InferenceOptions or native InferenceOptions, optional
        Session provider/device options; defaults to CPU with one thread per setting.

    Raises
    ------
    InferenceError
        The graph, directional contract or requested runtime cannot initialize.

    Notes
    -----
    Ordinary use is predict(experiment) or predict_batch(experiments). Advanced
    methods accept direct elemental areal-density arrays or return reference spectra.
    Runtime scratch and native capacity are retained, but convenience methods that
    return owning results still allocate output objects. Use out on direct-AD batch
    methods to reuse caller storage.

    Each instance owns an ONNX session and must not be called concurrently.
    Independent CPU and CUDA instances can be used by separate fitting workers.
    """
    def __init__(self, package, *, options: InferenceOptions | core.InferenceOptions | None = None) -> None:
        """Initialize a forward ONNX session; see ForwardModel for arguments and errors.
        """
        super().__init__(package, options=options)
        try:
            self._core = engine().ForwardModel(self._package, self._options)
            self.options = InferenceOptions.from_core(self._core.options)
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict(self, experiment):
        """Predict spectra at the requested measurement conditions.

        Parameters
        ----------
        experiment : Experiment
            Physical sample and setup. Unsupported changes to geometry, beam, isotope
            mixtures or unvaried fields are rejected by the package contract.

        Returns
        -------
        SimulationResult
            Labelled float64 count spectra; arrays retain their owning native result
            and remain valid after subsequent predictions.

        Raises
        ------
        InferenceError
            The request is incompatible, graph execution fails, or processing is invalid.

        Notes
        -----
        Applies numerical transforms and the ordered per-detector physical output
        preparation declared in the package.
        """
        try:
            return result_from_core(self._core.predict(to_core(experiment)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_batch(self, experiments):
        """Predict spectra at the requested measurement conditions in one graph batch.

        Parameters
        ----------
        experiments : sequence of Experiment
            Physical sample and setup. Unsupported changes to geometry, beam, isotope
            mixtures or unvaried fields are rejected by the package contract.

        Returns
        -------
        list[SimulationResult]
            Labelled float64 count spectra; arrays retain their owning native result
            and remain valid after subsequent predictions.

        Raises
        ------
        InferenceError
            The request is incompatible, graph execution fails, or processing is invalid.

        Notes
        -----
        Applies numerical transforms and the ordered per-detector physical output
        preparation declared in the package.
        Different layer counts are zero-padded for layer-input graphs. An empty input
        sequence returns an empty list. Each result retains its own output storage.
        """
        try:
            return [result_from_core(r) for r in self._core.predict_batch([to_core(x) for x in experiments])]
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference(self, experiment):
        """Predict reference-condition spectra.

        Parameters
        ----------
        experiment : Experiment
            Physical sample and setup. Unsupported changes to geometry, beam, isotope
            mixtures or unvaried fields are rejected by the package contract.

        Returns
        -------
        SimulationResult
            Labelled float64 count spectra; arrays retain their owning native result
            and remain valid after subsequent predictions.

        Raises
        ------
        InferenceError
            The request is incompatible, graph execution fails, or processing is invalid.

        Notes
        -----
        Skips physical output_preparation only. Input packing and numerical training
        transforms still run. Returned counts use the reference spectrum grid/state.
        """
        try:
            return result_from_core(self._core.predict_reference(to_core(experiment)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference_batch(self, experiments):
        """Predict reference-condition spectra in one graph batch.

        Parameters
        ----------
        experiments : sequence of Experiment
            Physical sample and setup. Unsupported changes to geometry, beam, isotope
            mixtures or unvaried fields are rejected by the package contract.

        Returns
        -------
        list[SimulationResult]
            Labelled float64 count spectra; arrays retain their owning native result
            and remain valid after subsequent predictions.

        Raises
        ------
        InferenceError
            The request is incompatible, graph execution fails, or processing is invalid.

        Notes
        -----
        Skips physical output_preparation only. Input packing and numerical training
        transforms still run. Returned counts use the reference spectrum grid/state.
        Different layer counts are zero-padded for layer-input graphs. An empty input
        sequence returns an empty list. Each result retains its own output storage.
        """
        try:
            return [result_from_core(r) for r in self._core.predict_batch(
                [to_core(x) for x in experiments], engine().PredictionDomain.REFERENCE)]
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference_areal_density(self, densities):
        """Predict reference spectra directly from elemental areal-density arrays.

        Parameters
        ----------
        densities : array_like, shape (layers, elements)
            Nonnegative physical elemental AD in 1e15 atoms/cm2, ordered as metadata
            elements. There is no total-thickness column. Isotope mixtures are the
            fixed assumptions of the layer schema.

        Returns
        -------
        SimulationResult
            Labelled physical float64 spectra retaining their native owning result.

        Raises
        ------
        InferenceError
            The model is not layer-input, dimensions/values are invalid, setup is
            incompatible, an output buffer is invalid, or graph execution fails.

        Notes
        -----
        Contiguous float32 inputs are borrowed; other types/layouts are converted.
        Numerical transforms use retained native scratch and never modify the input.
        Physical output preparation is skipped; numerical transforms still run.
        This avoids constructing layers/species for repeated optimization evaluations.
        """
        values = np.ascontiguousarray(densities, dtype=np.float32)
        try:
            return result_from_core(self._core.predict_areal_density(values, None, engine().PredictionDomain.REFERENCE))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference_areal_density_batch(self, densities, *, out=None):
        """Predict reference spectra directly from elemental areal-density arrays.

        Parameters
        ----------
        densities : array_like, shape (batch, layers, elements)
            Nonnegative physical elemental AD in 1e15 atoms/cm2, ordered as metadata
            elements. There is no total-thickness column. Isotope mixtures are the
            fixed assumptions of the layer schema.
        out : numpy.ndarray, optional
            Writable contiguous float64 array shaped (batch, sum(output channels)).
            Uses reference channels for reference inference and processed channels
            otherwise. Must not overlap input storage.

        Returns
        -------
        dict[str, numpy.ndarray]
            Per-detector float64 arrays shaped (batch, detector_channels). When out is
            supplied these are views of caller storage; reusing out changes previous views.
            Without out, newly allocated output storage is retained by the returned arrays.

        Raises
        ------
        InferenceError
            The model is not layer-input, dimensions/values are invalid, setup is
            incompatible, an output buffer is invalid, or graph execution fails.

        Notes
        -----
        Contiguous float32 inputs are borrowed; other types/layouts are converted.
        Numerical transforms use retained native scratch and never modify the input.
        Physical output preparation is skipped; numerical transforms still run.
        This is one ONNX call for the whole population, including reduced populations.
        Per-detector output views may be strided when multiple detectors share out.
        """
        values = np.ascontiguousarray(densities, dtype=np.float32)
        try:
            return dict(self._core.predict_areal_density_batch(values, None, out, engine().PredictionDomain.REFERENCE))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_areal_density(self, densities, *, setup=None):
        """Predict physical spectra directly from elemental areal-density arrays.

        Parameters
        ----------
        densities : array_like, shape (layers, elements)
            Nonnegative physical elemental AD in 1e15 atoms/cm2, ordered as metadata
            elements. There is no total-thickness column. Isotope mixtures are the
            fixed assumptions of the layer schema.
        setup : ExperimentalSetup, optional
            Requested detector calibration/exposure/times within the model contract.
            Defaults to the package reference; fixed geometry must still match.

        Returns
        -------
        SimulationResult
            Labelled physical float64 spectra retaining their native owning result.

        Raises
        ------
        InferenceError
            The model is not layer-input, dimensions/values are invalid, setup is
            incompatible, an output buffer is invalid, or graph execution fails.

        Notes
        -----
        Contiguous float32 inputs are borrowed; other types/layouts are converted.
        Numerical transforms use retained native scratch and never modify the input.
        Per-detector physical preparation is applied in declared order.
        This avoids constructing layers/species for repeated optimization evaluations.
        """
        values = np.ascontiguousarray(densities, dtype=np.float32)
        try:
            return result_from_core(self._core.predict_areal_density(values, to_core(setup)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_areal_density_batch(self, densities, *, setup=None, out=None):
        """Predict physical spectra directly from elemental areal-density arrays.

        Parameters
        ----------
        densities : array_like, shape (batch, layers, elements)
            Nonnegative physical elemental AD in 1e15 atoms/cm2, ordered as metadata
            elements. There is no total-thickness column. Isotope mixtures are the
            fixed assumptions of the layer schema.
        setup : ExperimentalSetup, optional
            Requested detector calibration/exposure/times within the model contract.
            Defaults to the package reference; fixed geometry must still match.
        out : numpy.ndarray, optional
            Writable contiguous float64 array shaped (batch, sum(output channels)).
            Uses reference channels for reference inference and processed channels
            otherwise. Must not overlap input storage.

        Returns
        -------
        dict[str, numpy.ndarray]
            Per-detector float64 arrays shaped (batch, detector_channels). When out is
            supplied these are views of caller storage; reusing out changes previous views.
            Without out, newly allocated output storage is retained by the returned arrays.

        Raises
        ------
        InferenceError
            The model is not layer-input, dimensions/values are invalid, setup is
            incompatible, an output buffer is invalid, or graph execution fails.

        Notes
        -----
        Contiguous float32 inputs are borrowed; other types/layouts are converted.
        Numerical transforms use retained native scratch and never modify the input.
        Per-detector physical preparation is applied in declared order.
        This is one ONNX call for the whole population, including reduced populations.
        Per-detector output views may be strided when multiple detectors share out.
        """
        values = np.ascontiguousarray(densities, dtype=np.float32)
        try:
            return dict(self._core.predict_areal_density_batch(values, to_core(setup), out))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error


class InverseModel(Model):
    """Infer a fixed-maximum-layer elemental profile from measured spectra.

    Parameters
    ----------
    package : Package, native Package, str or pathlib.Path
        Inverse model package.
    options : InferenceOptions or native InferenceOptions, optional
        Execution provider and session controls.

    Raises
    ------
    InferenceError
        The directional contract, graph or runtime cannot initialize.

    Notes
    -----
    Physical input preparation is followed by fitted numerical transforms, ONNX
    execution and physical EDP processing. Use one instance per concurrent worker.
    Model metadata defines a fixed maximum layer count and reference geometry.
    """
    def __init__(self, package, *, options: InferenceOptions | core.InferenceOptions | None = None) -> None:
        """Initialize an inverse ONNX session; see InverseModel for its contract.
        """
        super().__init__(package, options=options)
        try:
            self._core = engine().InverseModel(self._package, self._options)
            self.options = InferenceOptions.from_core(self._core.options)
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict(self, inputs):
        """Prepare measured spectra and infer elemental depth profiles.

        Parameters
        ----------
        inputs : InverseInput
            Labelled counts and their acquisition setup, consistent with fixed model assumptions.

        Returns
        -------
        InversePrediction
            Physical float64 EDP and uncertainty arrays retaining native owners.

        Raises
        ------
        InferenceError
            Labels, dimensions, geometry, state or processing are incompatible, or
            graph execution fails.

        Notes
        -----
        Each input follows its per-detector physical and numerical preparation.
        """
        try:
            return InversePrediction.from_core(self._core.predict(to_core(inputs)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_batch(self, inputs):
        """Prepare measured spectra and infer elemental depth profiles in one graph batch.

        Parameters
        ----------
        inputs : sequence of InverseInput
            Labelled counts and their acquisition setup, consistent with fixed model assumptions.

        Returns
        -------
        list[InversePrediction]
            Physical float64 EDP and uncertainty arrays retaining native owners.

        Raises
        ------
        InferenceError
            Labels, dimensions, geometry, state or processing are incompatible, or
            graph execution fails.

        Notes
        -----
        Each input follows its per-detector physical and numerical preparation.
        An empty input sequence returns an empty list.
        """
        try:
            return [InversePrediction.from_core(r) for r in self._core.predict_batch([to_core(x) for x in inputs])]
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference(self, spectra):
        """Infer from reference-condition spectra.

        Parameters
        ----------
        spectra : mapping or sequence of Spectrum
            Labelled counts already on the declared reference grid, exposure and pileup state.

        Returns
        -------
        InversePrediction
            Physical float64 EDP and uncertainty arrays retaining native owners.

        Raises
        ------
        InferenceError
            Labels, dimensions, geometry, state or processing are incompatible, or
            graph execution fails.

        Notes
        -----
        Skips physical input_preparation only. Fitted numerical preprocessing and
        physical EDP output processing still run.
        """
        try:
            return InversePrediction.from_core(self._core.predict_reference(_spectra(spectra)))
        except (ValueError, RuntimeError) as error:
            raise InferenceError(str(error)) from error

    def predict_reference_batch(self, spectra):
        """Infer from reference-condition spectra in one graph batch.

        Parameters
        ----------
        spectra : sequence of spectrum mappings
            Labelled counts already on the declared reference grid, exposure and pileup state.

        Returns
        -------
        list[InversePrediction]
            Physical float64 EDP and uncertainty arrays retaining native owners.

        Raises
        ------
        InferenceError
            Labels, dimensions, geometry, state or processing are incompatible, or
            graph execution fails.

        Notes
        -----
        Skips physical input_preparation only. Fitted numerical preprocessing and
        physical EDP output processing still run.
        An empty input sequence returns an empty list.
        """
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
    """Load a package and construct its directional inference session.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing format-1 .ibm package.
    options : InferenceOptions or native InferenceOptions, optional
        Execution provider, device and thread settings.

    Returns
    -------
    ForwardModel or InverseModel
        Initialized model selected from the package direction.

    Raises
    ------
    ModelPackageError
        The package cannot be read or validated.
    InferenceError
        The directional graph or runtime cannot initialize.

    Notes
    -----
    Opening a package alone does not require ONNX support, but constructing this
    executable model does. Create one returned model per concurrent worker.
    """
    package = Package.open(path)
    cls = ForwardModel if package.to_core().metadata.direction == engine().Direction.FORWARD else InverseModel
    return cls(package, options=options)
