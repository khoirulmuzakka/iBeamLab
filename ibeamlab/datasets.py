"""Read generated datasets into ordinary Python and NumPy objects."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from .exceptions import DatasetError
from .simulator import SimulationResult, _result

@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    format_version: int
    complete: bool
    requested: int
    accepted: int
    invalid: int
    failed: int
    parameter_names: tuple[str, ...]
    spectrum_labels: tuple[str, ...]
    spectrum_lengths: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class FailureRecord:
    sample_index: int
    sample_id: str
    method_label: str
    type: str
    message: str

@dataclass(frozen=True, slots=True)
class DatasetRecord:
    sample_index: int
    sample_id: str
    parameters: NDArray[np.float64]
    result: SimulationResult

class Dataset:
    """An opened iBeamLab dataset directory.

    Records are loaded lazily on first access. Use ``parameters`` and
    ``spectra(label)`` for rectangular NumPy arrays.
    """
    def __init__(self, path: str | Path):
        from ._native import native
        self.path = Path(path)
        try:
            self._reader = native.datasets.DatasetReader(self.path)
            raw = self._reader.metadata
        except Exception as error:
            raise DatasetError(f"cannot open dataset {self.path}: {error}") from error
        self.metadata = DatasetMetadata(raw.format_version, raw.complete, raw.requested,
            raw.accepted, raw.invalid, raw.failed, tuple(raw.parameter_names),
            tuple(raw.spectrum_labels), tuple(raw.spectrum_lengths))
        self._records: tuple[DatasetRecord, ...] | None = None

    @property
    def records(self) -> tuple[DatasetRecord, ...]:
        if self._records is None:
            try:
                self._records = tuple(DatasetRecord(x.sample_index, x.sample_id,
                    np.asarray(x.open_parameters, dtype=np.float64), _result(x.result))
                    for x in self._reader.read_all())
            except Exception as error:
                raise DatasetError(f"cannot read dataset {self.path}: {error}") from error
        return self._records

    def __len__(self) -> int:
        return self.metadata.accepted

    def iter_batches(self, batch_size: int = 512):
        """Stream at most batch_size records without populating the records cache.

        Traverses shards in the same order as records/read_all. Each iteration
        opens an independent cursor, so memory use is bounded by one batch.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        from ._native import native
        try:
            stream = native.datasets.DatasetStream(self.path)
            while True:
                batch = stream.read_batch(batch_size)
                if not batch:
                    break
                yield tuple(DatasetRecord(x.sample_index, x.sample_id,
                    np.asarray(x.open_parameters, dtype=np.float64), _result(x.result))
                    for x in batch)
        except Exception as error:
            raise DatasetError(f"cannot stream dataset {self.path}: {error}") from error

    @property
    def parameters(self) -> NDArray[np.float64]:
        """Parameter matrix with shape ``(samples, parameters)``."""
        if not self.records:
            return np.empty((0, len(self.metadata.parameter_names)), dtype=np.float64)
        return np.stack([x.parameters for x in self.records])

    def spectra(self, label: str) -> NDArray[np.float32]:
        """Spectrum matrix with shape ``(samples, channels)`` for one label."""
        if label not in self.metadata.spectrum_labels:
            raise KeyError(f"unknown spectrum label {label!r}; choose from {self.metadata.spectrum_labels}")
        if not self.records:
            length = self.metadata.spectrum_lengths[self.metadata.spectrum_labels.index(label)]
            return np.empty((0, length), dtype=np.float32)
        return np.stack([x.result[label].counts for x in self.records])

    @property
    def failures(self) -> tuple[FailureRecord, ...]:
        try:
            return tuple(FailureRecord(x.sample_index, x.sample_id, x.method_label, x.type, x.message)
                         for x in self._reader.read_failures())
        except Exception as error:
            raise DatasetError(f"cannot read failures from {self.path}: {error}") from error

def open_dataset(path: str | Path) -> Dataset:
    """Open an iBeamLab dataset directory."""
    return Dataset(path)

__all__ = ["Dataset", "DatasetMetadata", "DatasetRecord", "FailureRecord", "open_dataset"]
