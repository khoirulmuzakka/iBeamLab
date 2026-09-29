"""Simulation backends and Python-friendly spectrum results."""

from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from .exceptions import SimulationError, SimnraUnavailableError
from .sample import Experiment, _to_native_sample, _to_native_setup

@dataclass(frozen=True, slots=True)
class Spectrum:
    """One labelled, one-dimensional float32 count spectrum."""
    label: str
    counts: NDArray[np.float32]
    def __post_init__(self) -> None:
        values = np.asarray(self.counts, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("spectrum counts must be one-dimensional")
        object.__setattr__(self, "counts", values)
    @property
    def channels(self) -> NDArray[np.int64]:
        return np.arange(self.counts.size)
    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.counts, dtype=dtype)

@dataclass(frozen=True, slots=True)
class SimulationFailure:
    sample_index: int
    method_label: str
    type: str
    message: str

class SimulationResult(Mapping[str, Spectrum]):
    """Mapping from detector label to spectrum for one experiment."""
    def __init__(self, spectra: Sequence[Spectrum], failure: SimulationFailure | None = None):
        self._spectra = {s.label: s for s in spectra}
        self.failure = failure
    def __getitem__(self, label: str) -> Spectrum:
        return self._spectra[label]
    def __iter__(self) -> Iterator[str]:
        return iter(self._spectra)
    def __len__(self) -> int:
        return len(self._spectra)
    @property
    def spectra(self) -> tuple[Spectrum, ...]:
        return tuple(self._spectra.values())
    @property
    def failed(self) -> bool:
        return self.failure is not None

@dataclass(frozen=True, slots=True)
class SimnraMethod:
    """A detector label and the SIMNRA reference file that implements it."""
    label: str
    reference_file: Path
    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_file", Path(self.reference_file))

def _input(experiment: Experiment):
    from ._native import native
    value = native.simulator.SimulationInput()
    value.sample = _to_native_sample(experiment.sample)
    value.setup = _to_native_setup(experiment.detectors)
    return value

def _result(value) -> SimulationResult:
    failure = None
    if value.failure is not None:
        failure = SimulationFailure(value.failure.sample_index, value.failure.method_label,
                                    value.failure.type, value.failure.message)
    return SimulationResult([Spectrum(x.label, x.counts) for x in value.spectra], failure)

class Simulator:
    """Common interface implemented by all public simulator backends."""
    _native: object
    def simulate(self, experiment: Experiment, *, continue_after_failure: bool = False) -> SimulationResult:
        """Simulate one experiment."""
        return self.simulate_many([experiment], continue_after_failure=continue_after_failure)[0]
    def simulate_many(self, experiments: Sequence[Experiment], *,
                      continue_after_failure: bool = False) -> list[SimulationResult]:
        """Simulate experiments in stable input order."""
        from ._native import native
        options = native.simulator.SimulationOptions()
        options.continue_after_failure = continue_after_failure
        try:
            raw = self._native.simulate_batch([_input(x) for x in experiments], options)
            return [_result(x) for x in raw]
        except Exception as error:
            raise SimulationError(str(error)) from error
    def request_stop(self) -> None:
        self._native.request_stop()
    def reset_stop(self) -> None:
        self._native.reset_stop()

class DummySimulator(Simulator):
    """Deterministic dependency-free backend for tests and tutorials."""
    def __init__(self, channels: int = 1024):
        if channels <= 0:
            raise ValueError("channels must be positive")
        from ._native import native
        self._native = native.simulator.DummySimulator(channels)

class SimnraSimulator(Simulator):
    """Persistent SIMNRA worker pool. SIMNRA is an optional dependency."""
    def __init__(self, methods: Sequence[SimnraMethod], *, workers: int = 1,
                 multithreaded_apartment: bool = True, thread_priority: int = 0,
                 fast_calculation: bool = True):
        if workers <= 0:
            raise ValueError("workers must be positive")
        from ._native import native
        config = native.simulator.SimnraSimulatorConfig()
        native_methods = []
        for method in methods:
            value = native.simulator.SimnraMethod()
            value.label, value.reference_file = method.label, method.reference_file
            native_methods.append(value)
        config.methods, config.workers = native_methods, workers
        config.multithreaded_apartment = multithreaded_apartment
        config.thread_priority, config.fast_calculation = thread_priority, fast_calculation
        try:
            self._native = native.simulator.SimnraSimulator(config)
        except Exception as error:
            raise SimnraUnavailableError(
                "SIMNRA could not be started. Verify its installation and COM registration: " + str(error)
            ) from error
    def close(self) -> None:
        self._native.close()
    def __enter__(self) -> "SimnraSimulator":
        return self
    def __exit__(self, *_args) -> None:
        self.close()

__all__ = ["DummySimulator", "SimnraMethod", "SimnraSimulator", "SimulationFailure", "SimulationResult", "Simulator", "Spectrum"]
