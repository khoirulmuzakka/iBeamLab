"""Simulation backends and labelled, double-precision NumPy spectra."""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ._engine import engine, to_core
from .exceptions import SimulationError, SimnraUnavailableError, ValidationError


@dataclass(frozen=True, slots=True)
class Spectrum:
    """Integrated counts for one measurement; contiguous float64 with no copy if compatible."""
    measurement: str
    counts: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.counts, dtype=np.float64)
        if not self.measurement or values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValidationError("spectrum requires a label and finite nonnegative 1D counts")
        object.__setattr__(self, "counts", np.ascontiguousarray(values))

    def to_core(self):
        return engine().Spectrum(self.measurement, self.counts)


@dataclass(frozen=True, slots=True)
class SimulationFailure:
    """Backend diagnostic identifying a measurement when available."""
    measurement: str
    message: str


class SimulationResult(Mapping):
    """Read-only mapping from measurement labels to spectra, with an optional failure."""
    def __init__(self, spectra=(), failure=None):
        values = tuple(spectra)
        self._spectra = {s.measurement: s for s in values}
        if len(values) != len(self._spectra):
            raise ValidationError("duplicate spectrum label")
        self.failure = failure

    def __getitem__(self, measurement):
        return self._spectra[measurement]

    def __iter__(self):
        return iter(self._spectra)

    def __len__(self):
        return len(self._spectra)

    @property
    def spectra(self):
        return tuple(self._spectra.values())

    @property
    def failed(self):
        return self.failure is not None


def result_from_core(value):
    failure = None if value.failure is None else SimulationFailure(
        value.failure.measurement, value.failure.message)
    return SimulationResult((Spectrum(s.measurement, s.counts_array) for s in value.spectra), failure)


class Simulator:
    """Common backend interface; calls on one instance must be serialized."""
    def simulate(self, experiment):
        try:
            return result_from_core(self._core.simulate(to_core(experiment)))
        except (ValueError, RuntimeError) as error:
            raise SimulationError(str(error)) from error

    def simulate_batch(self, experiments):
        try:
            return [result_from_core(r) for r in self._core.simulate_batch([to_core(x) for x in experiments])]
        except (ValueError, RuntimeError) as error:
            raise SimulationError(str(error)) from error

    def to_core(self):
        return self._core

    def request_stop(self):
        self._core.request_stop()

    def reset_stop(self):
        self._core.reset_stop()

    @property
    def provenance(self):
        return self._core.provenance()


class DummySimulator(Simulator):
    """Deterministic synthetic backend for workflow development, without physical simulation."""
    def __init__(self, channels=5000):
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValidationError("channels must be a positive integer")
        self._core = engine().DummySimulator(channels)


@dataclass(frozen=True, slots=True)
class SimnraMethod:
    """Map a measurement label and particle identity to a SIMNRA reference file."""
    measurement: str
    reference: Path
    particle: str

    def to_core(self):
        value = engine().SimnraMethod()
        value.measurement, value.reference, value.particle = self.measurement, Path(self.reference), self.particle
        return value


class SimnraSimulator(Simulator):
    """Persistent Windows COM workers requiring a locally installed SIMNRA.

    Use as a context manager to release sessions promptly. References preserve
    geometry and cross-section settings independently for each measurement.
    fast_calculation explicitly opts into CalculateSpectrumFast, whose numerical
    results can differ from the standard algorithm used by default.
    """
    def __init__(self, methods, *, workers=1, multithreaded_apartment=True,
                 reuse_target_topology=True, fast_calculation=False):
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
        self._core.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
