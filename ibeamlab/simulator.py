"""Simulation backends and labelled, double-precision NumPy spectra."""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import weakref
import xml.etree.ElementTree as ET
from .sample import Geometry, experiment_from_core
from ._engine import engine, to_core
from .exceptions import SimulationError, SimnraUnavailableError, ValidationError


@dataclass(frozen=True, slots=True)
class Spectrum:
    """Hold one labelled, integrated-count spectrum.

    Parameters
    ----------
    measurement : str
        Nonempty measurement label.
    counts : array_like, shape (channels,)
        Finite, nonnegative integrated bin counts. Converted to contiguous float64;
        compatible arrays are retained without copying.

    Raises
    ------
    ValidationError
        The label, dimensionality or counts are invalid.

    Notes
    -----
    Native results expose read-only NumPy views that retain their native owner.
    User-provided arrays retain their own writeability; this value type does not
    make an independent copy when the input already has a compatible layout.
    """
    measurement: str
    counts: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.counts, dtype=np.float64)
        if not self.measurement or values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValidationError("spectrum requires a label and finite nonnegative 1D counts")
        object.__setattr__(self, "counts", np.ascontiguousarray(values))

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.Spectrum
            Native value for lower-level workflows.

        Notes
        -----
        Copies count values into an owning native spectrum. Ordinary result adapters
        handle this boundary automatically.
        """
        return engine().Spectrum(self.measurement, self.counts)


@dataclass(frozen=True, slots=True)
class SimulationFailure:
    """Describe a backend failure associated with an experiment.

    Parameters
    ----------
    measurement : str
        Affected measurement label, or an empty string for an experiment-wide failure.
    message : str
        Human-readable diagnostic describing the failure.
    """
    measurement: str
    message: str


class SimulationResult(Mapping):
    """Expose labelled spectra and an optional backend diagnostic.

    Parameters
    ----------
    spectra : sequence of Spectrum or mapping, optional
        Spectra with unique measurement labels. A mapping may contain count arrays
        or Spectrum values whose labels agree with their keys.
    failure : SimulationFailure, optional
        Failure diagnostic. None denotes a successful outcome.

    Raises
    ------
    ValidationError
        Labels are duplicated, mapping keys disagree, or spectrum values are invalid.

    Notes
    -----
    Implements the read-only mapping interface: result[label] returns Spectrum.
    Check failed before accessing spectra when a backend reports failures.
    A failed result may have no spectra. Native-backed arrays keep their storage
    alive independently of the original native result wrapper.
    """
    def __init__(self, spectra=(), failure=None):
        """Initialize labelled spectra and failure state; see SimulationResult.
        """
        if isinstance(spectra, Mapping):
            values = []
            for label, value in spectra.items():
                if isinstance(value, Spectrum) and value.measurement != label:
                    raise ValidationError("spectrum mapping key must match its measurement label")
                values.append(value if isinstance(value, Spectrum) else Spectrum(label, value))
        else:
            values = tuple(spectra)
        self._spectra = {s.measurement: s for s in values}
        if len(values) != len(self._spectra):
            raise ValidationError("duplicate spectrum label")
        self.failure = failure

    def __getitem__(self, measurement):
        """Return the Spectrum identified by a measurement label.

        Parameters
        ----------
        measurement : str
            Exact measurement label.

        Returns
        -------
        Spectrum
            The stored spectrum.

        Raises
        ------
        KeyError
            The result contains no spectrum for this label.
        """
        return self._spectra[measurement]

    def __iter__(self):
        """Iterate stored measurement labels in insertion order.

        Returns
        -------
        iterator of str
            Labels available for mapping lookup.
        """
        return iter(self._spectra)

    def __len__(self):
        """Return the number of stored measurement spectra.

        Returns
        -------
        int
            Number of spectra; failure state is independent of this count.
        """
        return len(self._spectra)

    @property
    def spectra(self):
        """Return stored spectra in insertion order.

        Returns
        -------
        tuple[Spectrum, ...]
            Spectrum objects retaining their associated array storage.
        """
        return tuple(self._spectra.values())

    @property
    def failed(self):
        """Report whether a failure diagnostic is present.

        Returns
        -------
        bool
            True when failure is not None.
        """
        return self.failure is not None


def result_from_core(value):
    """Construct Python-facing values from a native result.

    Parameters
    ----------
    value : ibeamlab.native.SimulationResult
        Source value; it is not modified.

    Returns
    -------
    SimulationResult
        Python representation of the supplied value.

    Notes
    -----
    Count arrays are read-only views retaining their native owners; no additional
    full-spectrum Python copy is required for compatible contiguous arrays.
    """
    failure = None if value.failure is None else SimulationFailure(
        value.failure.measurement, value.failure.message)
    return SimulationResult((Spectrum(s.measurement, s.counts_array) for s in value.spectra), failure)


def _result(value, experiment):
    if isinstance(value, Mapping) and not isinstance(value, SimulationResult):
        value = SimulationResult(value)
    if not isinstance(value, SimulationResult):
        raise ValidationError("simulate must return SimulationResult or a mapping of labels to spectra")
    if value.failure is None and set(value) != {m.label for m in experiment.setup.measurements}:
        raise ValidationError("simulator result labels must match the experiment measurements")
    return value


def _result_to_core(value, experiment):
    result = _result(value, experiment)
    return engine()._simulation_result(list(result.spectra), result.failure)


class Simulator:
    """Define a Python simulation backend or wrap a built-in native backend.

    Notes
    -----
    Subclass this class and implement simulate(experiment). Return SimulationResult
    or a mapping from measurement labels to count arrays. No base constructor or
    native binding objects are required. Successful results used by generation must
    include exactly the requested measurement labels.

    The default Python batch path calls simulate per input and records ordinary
    exceptions as per-input failures. Override simulate_batch for vectorized execution.
    A batch override returns one result per input in order and should use explicit
    failed results for per-input errors. An exception from a batch override aborts
    the batch. KeyboardInterrupt propagates.

    Calls on an instance must be serialized. Native backends release the GIL around
    execution; the adapter acquires it for Python callbacks. Independent workers
    should use independent backend instances.

    Examples
    --------
    >>> class FlatSimulator(Simulator):
    ...     def simulate(self, experiment):
    ...         return {m.label: [1.0, 2.0, 1.0]
    ...                 for m in experiment.setup.measurements}
    """
    def simulate(self, experiment):
        """Simulate all measurements of one experiment.

        Parameters
        ----------
        experiment : Experiment
            Physical sample and setup, including detector geometry.

        Returns
        -------
        SimulationResult or mapping
            Built-in backends return SimulationResult. Python implementations may
            return a mapping from measurement labels to integrated count arrays.

        Raises
        ------
        SimulationError
            Native execution or conversion raises an error.
        NotImplementedError
            A Python subclass has not implemented the scalar simulation method.

        Notes
        -----
        Built-in backend calculation failures can be returned through result.failure
        rather than raised. Inspect failed before indexing a SimulationResult.
        """
        if not hasattr(self, "_core"):
            raise NotImplementedError("implement simulate() or override simulate_batch()")
        try:
            return result_from_core(self._core.simulate(to_core(experiment)))
        except (ValueError, RuntimeError) as error:
            raise SimulationError(str(error)) from error

    def simulate_batch(self, experiments):
        """Simulate a sequence of experiments in input order.

        Parameters
        ----------
        experiments : sequence of Experiment
            Inputs for one bounded batch.

        Returns
        -------
        list of SimulationResult or backend result mappings
            One result per input when execution completes. Cooperative cancellation may
            return fewer results; generation recognizes the associated stop flag.

        Raises
        ------
        SimulationError
            Native execution or conversion raises an error.

        Notes
        -----
        The inherited Python implementation catches Exception for each scalar call
        and creates a failed result. Batch overrides should return explicit per-input
        failures when using RECORD or DISCARD. A batch-level exception aborts generation.
        Generation reports Python callback progress after the callback returns.
        """
        if hasattr(self, "_core"):
            try:
                return [result_from_core(r) for r in self._core.simulate_batch([to_core(x) for x in experiments])]
            except (ValueError, RuntimeError) as error:
                raise SimulationError(str(error)) from error
        results = []
        for experiment in experiments:
            if self.stopped:
                break
            try:
                results.append(_result(self.simulate(experiment), experiment))
            except Exception as error:
                results.append(SimulationResult(failure=SimulationFailure("", str(error))))
        return results

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.Simulator
            Native value for lower-level workflows.

        Notes
        -----
        Built-in wrappers return their native backend. Python subclasses lazily create
        and cache a callback adapter that holds the Python owner weakly. Keep the Python
        backend alive when using this adapter directly. Ordinary generation does so.
        """
        if hasattr(self, "_core"):
            return self._core
        if not hasattr(self, "_native_adapter"):
            # Callbacks borrow the owner weakly: owner -> adapter -> callback must
            # not form an uncollectable Python/C++ reference cycle.
            reference = weakref.ref(self)
            def owner():
                result = reference()
                if result is None:
                    raise SimulationError("Python simulator owner no longer exists")
                return result
            def single(raw):
                experiment = experiment_from_core(raw)
                return _result_to_core(owner().simulate(experiment), experiment)
            def batch(raw):
                experiments = [experiment_from_core(x) for x in raw]
                result = list(owner().simulate_batch(experiments))
                if len(result) != len(experiments) and not owner().stopped:
                    raise ValidationError("simulate_batch must return one result per experiment")
                if len(result) > len(experiments):
                    raise ValidationError("simulate_batch returned too many results")
                return [_result_to_core(value, experiment) for value, experiment in zip(result, experiments)]
            def provenance():
                return str(owner().provenance)
            self._native_adapter = engine()._PythonSimulator(single, batch, provenance)
        return self._native_adapter

    def request_stop(self):
        """Request cooperative cancellation of current or subsequent work.

        Notes
        -----
        This sets an atomic native stop flag. It does not forcibly interrupt an
        external process or an executing Python callback. Batch overrides may poll
        stopped. Call reset_stop() before starting subsequent work.
        """
        self.to_core().request_stop()

    def reset_stop(self):
        """Clear the cancellation flag before reusing an active backend.

        Notes
        -----
        Does not reopen or guarantee reuse of a backend whose resources were closed.
        """
        self.to_core().reset_stop()

    @property
    def stopped(self):
        """Read the cooperative cancellation flag.

        Returns
        -------
        bool
            True after request_stop(), until reset_stop() is called.
        """
        return self.to_core().stopped()

    @property
    def provenance(self):
        """Return the backend identity recorded during generation.

        Returns
        -------
        str
            Native backend provenance or the Python subclass module and qualified name.

        Notes
        -----
        Python subclasses may override this property to include a simulator version,
        physics-data version and other reproducibility information.
        """
        if hasattr(self, "_core"):
            return self._core.provenance()
        return f"python:{type(self).__module__}.{type(self).__qualname__}"


class DummySimulator(Simulator):
    """Generate deterministic synthetic spectra for workflow development.

    Parameters
    ----------
    channels : int, default=5000
        Positive number of channels for each measurement.

    Raises
    ------
    ValidationError
        channels is not a positive integer.

    Notes
    -----
    This backend is synthetic and does not implement ion-beam interaction physics.
    Its interface and dataset behavior match the built-in simulation workflow.
    """
    def __init__(self, channels=5000):
        """Create the synthetic backend; see DummySimulator for the channels parameter.
        """
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValidationError("channels must be a positive integer")
        self._core = engine().DummySimulator(channels)


@dataclass(frozen=True, slots=True)
class SimnraMethod:
    """Associate one measurement with a SIMNRA reference project.

    Parameters
    ----------
    measurement : str
        Measurement label to simulate.
    reference : str or pathlib.Path
        Reference project supplying geometry defaults and backend physics choices.
    particle : str
        Expected particle identifier for this method; must match the request.

    Notes
    -----
    A measurement may explicitly override reference geometry through its Detector.
    For XNRA files, reference_geometry() reads the actual saved angles without
    starting SIMNRA. Other project formats require explicit geometry.
    """
    measurement: str
    reference: Path
    particle: str

    def reference_geometry(self):
        """Read the detector geometry from a single-spectrum XNRA reference.

        Returns
        -------
        Geometry
            Saved incidence, exit and scattering angles in degrees.

        Raises
        ------
        ValidationError
            The file is unreadable, is not supported XML, has missing or ambiguous
            spectrum geometry, or contains invalid angles.

        Notes
        -----
        No COM worker is started. References containing multiple spectrum geometries
        require an explicit caller-selected Geometry instead of an inferred choice.
        """
        try:
            root = ET.parse(self.reference).getroot()
            for node in root.iter():
                node.tag = node.tag.rsplit("}", 1)[-1]
            geometries = root.findall("./sample/spectra/spectrum/geometry")
            if len(geometries) != 1:
                raise ValueError("expected one spectrum geometry; specify geometry explicitly for ambiguous references")
            geometry = geometries[0]
            return Geometry(float(geometry.findtext("incidenceangle")),
                            float(geometry.findtext("exitangle")),
                            float(geometry.findtext("scatteringangle")))
        except (OSError, ET.ParseError, TypeError, ValueError) as error:
            raise ValidationError(f"Cannot read detector geometry from {self.reference}: {error}") from error

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.SimnraMethod
            Native value for lower-level workflows.

        Notes
        -----
        Copies the measurement, path and particle mapping; it does not start SIMNRA.
        """
        value = engine().SimnraMethod()
        value.measurement, value.reference, value.particle = self.measurement, Path(self.reference), self.particle
        return value


class SimnraSimulator(Simulator):
    """Simulate spectra with persistent Windows SIMNRA COM workers.

    Parameters
    ----------
    methods : sequence of SimnraMethod
        Exactly one reference mapping per requested measurement.
    workers : int, default=1
        Number of native COM workers.
    multithreaded_apartment : bool, default=True
        Initialize each worker using the multithreaded COM apartment model.
    reuse_target_topology : bool, default=True
        Reuse compatible layers and isotope identities between calls. Replacements
        append new layers before removing old layers to retain cross-section choices.
    fast_calculation : bool, default=False
        Use SIMNRA CalculateSpectrumFast. This is a distinct approximation and can
        differ numerically from the standard calculation.

    Raises
    ------
    SimnraUnavailableError
        The native backend cannot be created with the supplied build or configuration.

    Notes
    -----
    Requires a SIMNRA-enabled Windows build and an installed SIMNRA runtime.
    COM activation may occur when simulation starts. Inspect returned failure
    diagnostics as well as constructor errors.

    Use as a context manager to release workers promptly. Workers claim unowned
    applications using OLEUser; interactive and already-owned applications are skipped.
    Calls on one wrapper must be serialized. A completed batch is ordered by input,
    regardless of worker completion order. Geometry maps directly to Alpha/Beta/Theta.
    """
    def __init__(self, methods, *, workers=1, multithreaded_apartment=True,
                 reuse_target_topology=True, fast_calculation=False):
        """Configure persistent COM workers; see SimnraSimulator for all options.
        """
        config = engine().SimnraConfig()
        config.methods = [to_core(m) for m in methods]
        config.workers = workers
        config.multithreaded_apartment = multithreaded_apartment
        config.fast_calculation = fast_calculation
        config.reuse_target_topology = reuse_target_topology
        try:
            self._core = engine().SimnraSimulator(config)
        except (ValueError, RuntimeError) as error:
            raise SimnraUnavailableError(str(error)) from error

    def close(self):
        """Request cancellation and release the native worker sessions.

        Notes
        -----
        Safe to call repeatedly. Prefer constructing a new backend after closing;
        reset_stop() alone does not guarantee that closed resources can be reused.
        """
        self._core.close()

    def __enter__(self):
        """Return this simulator for context-managed execution.

        Returns
        -------
        SimnraSimulator
            This instance.
        """
        return self

    def __exit__(self, *args):
        """Release worker sessions when leaving the context.

        Parameters
        ----------
        args : tuple
            Exception context supplied by the context-manager protocol.

        Notes
        -----
        Exceptions from the context body are not suppressed.
        """
        self.close()
