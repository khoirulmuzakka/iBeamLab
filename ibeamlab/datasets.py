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
    """Describe the physical reference and outcome counts of a completed dataset.

    Parameters
    ----------
    reference : Experiment
        Reference sample and setup, including detector geometries.
    parameters : tuple[ParameterSpec, ...]
        Physical input columns in stored order.
    provenance : str
        Backend identity and caller-supplied generation provenance.
    requested : int
        Number of requested input rows.
    accepted : int
        Number of successful stored records.
    failed : int
        Number of stored failure records.
    discarded : int
        Number of failed inputs omitted under DISCARD.

    Notes
    -----
    For a completed dataset requested equals accepted + failed + discarded.
    Discarded inputs have no stored records; sample_index preserves original row identity.
    """
    reference: object
    parameters: tuple[ParameterSpec, ...]
    provenance: str
    requested: int
    accepted: int
    failed: int
    discarded: int

    @property
    def parameter_names(self):
        """Return physical column names in stored order.

        Returns
        -------
        tuple[str, ...]
            Names corresponding to columns of Dataset.parameters.
        """
        return tuple(p.name for p in self.parameters)

    @property
    def measurement_labels(self):
        """Return labels declared by the reference setup.

        Returns
        -------
        tuple[str, ...]
            Measurement identifiers in reference order.
        """
        return tuple(m.label for m in self.reference.setup.measurements)


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Represent one stored input row and its simulation outcome.

    Parameters
    ----------
    sample_index : int
        Original zero-based generation row index; may have gaps after discarded inputs.
    sample_id : str
        Stored sample identifier.
    parameters : numpy.ndarray
        Physical float64 parameter vector in metadata column order.
    result : SimulationResult
        Labelled spectra or an explicit failure.

    Notes
    -----
    Record arrays remain associated with their own values; iteration does not
    require retaining an entire shard or dataset in Python.
    """
    sample_index: int
    sample_id: str
    parameters: np.ndarray
    result: object


class Dataset:
    """Read a completed, checksummed, sharded dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Dataset directory containing a committed manifest.ibd.

    Raises
    ------
    DatasetError
        The directory is incomplete, missing, invalid or unreadable.

    Notes
    -----
    Construction reads metadata, not all spectral data. Each iteration opens an
    independent cursor and streams both successful and stored failure records.
    Matrix helpers allocate arrays for successful rows only; use streaming for large
    or variable-length spectra. Independent cursors can be consumed separately.
    """
    def __init__(self, path):
        """Open dataset metadata; see Dataset for the completion requirement.
        """
        self.path = Path(path)
        try:
            raw = engine().DatasetReader(self.path).metadata
            self.metadata = DatasetMetadata(experiment_from_core(raw.reference), tuple(
                ParameterSpec(p.name, p.target, p.unit) for p in raw.parameters),
                raw.provenance, raw.requested, raw.accepted, raw.failed, raw.discarded)
        except (ValueError, RuntimeError) as error:
            raise DatasetError(f"cannot open {self.path}: {error}") from error

    def __len__(self):
        """Return the number of stored records, including failures.

        Returns
        -------
        int
            accepted + failed; discarded rows are not stored.
        """
        return self.metadata.accepted + self.metadata.failed

    def maximum_spectrum_lengths(self):
        """Scan the maximum successful spectrum length for every label.

        Returns
        -------
        dict[str, int]
            Maximum channel count observed for each measurement with successful spectra.

        Raises
        ------
        DatasetError
            The scan encounters invalid dataset storage.

        Notes
        -----
        Uses native streaming and reusable record buffers. It does not materialize
        Python spectra or matrices and does not disturb existing iterators.
        """
        try:
            return dict(engine().DatasetReader(self.path).maximum_spectrum_lengths())
        except (ValueError, RuntimeError) as error:
            raise DatasetError(f"cannot scan {self.path}: {error}") from error

    def __iter__(self):
        """Stream stored records using a new independent cursor.

        Returns
        -------
        iterator of DatasetRecord
            Successful and failed records in original row order.

        Raises
        ------
        DatasetError
            A shard is unreadable or fails validation while iterating.
        """
        try:
            reader = engine().DatasetReader(self.path)
            for record in reader:
                yield DatasetRecord(record.sample_index, record.sample_id,
                                    np.asarray(record.parameters, dtype=np.float64),
                                    result_from_core(record.result))
        except (ValueError, RuntimeError) as error:
            raise DatasetError(f"cannot read {self.path}: {error}") from error

    def iter_batches(self, batch_size=64):
        """Stream bounded groups of stored records.

        Parameters
        ----------
        batch_size : int, default=64
            Positive maximum number of records per batch.

        Returns
        -------
        iterator of tuple[DatasetRecord, ...]
            Record tuples, including failures; the final batch may be smaller.

        Raises
        ------
        ValueError
            batch_size is not a positive integer.
        DatasetError
            Dataset iteration fails.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        cursor = iter(self)
        while batch := tuple(islice(cursor, batch_size)):
            yield batch

    @property
    def records(self):
        """Materialize every stored success and failure record.

        Returns
        -------
        tuple[DatasetRecord, ...]
            All stored records in original order.

        Notes
        -----
        This allocates memory proportional to the complete stored dataset.
        Use iteration or iter_batches() when working with production datasets.
        """
        return tuple(self)

    @property
    def parameters(self):
        """Materialize successful physical parameter rows.

        Returns
        -------
        numpy.ndarray
            New float64 array shaped (accepted, number_of_parameters), in stored order.

        Notes
        -----
        The row ordering matches spectra() for the same completed dataset.
        Failure and discarded rows are excluded. Reading this property scans the dataset.
        """
        rows = [r.parameters for r in self if not r.result.failed]
        return np.stack(rows) if rows else np.empty((0, len(self.metadata.parameters)))

    def spectra(self, measurement):
        """Materialize equal-length successful spectra for one measurement.

        Parameters
        ----------
        measurement : str
            Label declared by the reference setup.

        Returns
        -------
        numpy.ndarray
            New float64 array shaped (accepted, channels), aligned with parameters.

        Raises
        ------
        KeyError
            The label is unknown.
        DatasetError
            There are no successful spectra or their lengths differ.

        Notes
        -----
        This scans the dataset and allocates a dense matrix. Stream records when
        channel lengths vary or the matrix is too large for available memory.
        """
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
        """Materialize stored failure records.

        Returns
        -------
        tuple[DatasetRecord, ...]
            Failed records retained under RECORD; discarded inputs are absent.

        Notes
        -----
        Scans the dataset. Successful spectra are not included in the returned tuple.
        """
        return tuple(r for r in self if r.result.failed)


def open_dataset(path):
    """Open a completed dataset without eagerly reading its shards.

    Parameters
    ----------
    path : str or pathlib.Path
        Dataset directory with a committed manifest.

    Returns
    -------
    Dataset
        Streaming reader with physical metadata and optional matrix helpers.

    Raises
    ------
    DatasetError
        The dataset is incomplete, invalid or unavailable.
    """
    return Dataset(path)
