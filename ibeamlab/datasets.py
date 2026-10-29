"""Completed sharded datasets with independent streaming cursors and NumPy access."""
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import numpy as np
from ._engine import engine
from .exceptions import DatasetError
from .parameters import ParameterSpec
from .sample import experiment_from_core
from .simulator import result_from_core


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Reference, physical columns, provenance and completed outcome counts."""
    reference: object
    parameters: tuple[ParameterSpec, ...]
    provenance: str
    requested: int
    accepted: int
    failed: int
    discarded: int

    @property
    def parameter_names(self):
        return tuple(p.name for p in self.parameters)

    @property
    def measurement_labels(self):
        return tuple(m.label for m in self.reference.setup.measurements)


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Original row identity, physical parameter values and simulation outcome."""
    sample_index: int
    sample_id: str
    parameters: np.ndarray
    result: object


class Dataset:
    """A completed dataset. Iteration streams all stored success/failure records.

    Each iteration opens its own cursor. Matrix helpers include successful rows
    only, in the same order, and allocate arrays explicitly for training.
    """
    def __init__(self, path):
        self.path = Path(path)
        try:
            raw = engine().DatasetReader(self.path).metadata
            self.metadata = DatasetMetadata(experiment_from_core(raw.reference), tuple(
                ParameterSpec(p.name, p.target, p.unit) for p in raw.parameters),
                raw.provenance, raw.requested, raw.accepted, raw.failed, raw.discarded)
        except (ValueError, RuntimeError) as error:
            raise DatasetError(f"cannot open {self.path}: {error}") from error

    def __len__(self):
        return self.metadata.accepted + self.metadata.failed

    def __iter__(self):
        try:
            reader = engine().DatasetReader(self.path)
            for record in reader:
                yield DatasetRecord(record.sample_index, record.sample_id,
                                    np.asarray(record.parameters, dtype=np.float64),
                                    result_from_core(record.result))
        except (ValueError, RuntimeError) as error:
            raise DatasetError(f"cannot read {self.path}: {error}") from error

    def iter_batches(self, batch_size=64):
        """Yield tuples of records without retaining the full dataset in memory."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        cursor = iter(self)
        while batch := tuple(islice(cursor, batch_size)):
            yield batch

    @property
    def records(self):
        """Explicitly materialize all stored records, including failures."""
        return tuple(self)

    @property
    def parameters(self):
        """Successful physical rows as float64 [sample,parameter]."""
        rows = [r.parameters for r in self if not r.result.failed]
        return np.stack(rows) if rows else np.empty((0, len(self.metadata.parameters)))

    def spectra(self, measurement):
        """Successful spectra as float64 [sample,channel]; reject variable-length rows."""
        if measurement not in self.metadata.measurement_labels:
            raise KeyError(measurement)
        rows = [r.result[measurement].counts for r in self if not r.result.failed]
        if not rows:
            raise DatasetError("no successful spectra; channel length is unavailable")
        try:
            return np.stack(rows)
        except ValueError as error:
            raise DatasetError("spectra have different lengths; stream records instead") from error

    @property
    def failures(self):
        return tuple(r for r in self if r.result.failed)


def open_dataset(path):
    """Open a completed dataset without eagerly reading its shards."""
    return Dataset(path)
