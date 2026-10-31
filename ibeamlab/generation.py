"""Physical parameter plans and C++ dataset generation with explicit failure policies."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
from ._engine import engine, to_core
from .exceptions import DatasetError, ValidationError


class FailurePolicy(Enum):
    """Choose how generation handles failed simulation results.

    Notes
    -----
    STOP aborts generation and leaves incomplete output. RECORD retains the
    failed input row and its diagnostic. DISCARD omits the failed row while
    preserving the original indices of retained records. Invalid nonfinite parameter
    rows cannot be stored even when a record/discard policy is selected.
    """
    STOP = "STOP"
    RECORD = "RECORD"
    DISCARD = "DISCARD"


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """Declare a reference experiment and ordered physical input columns.

    Parameters
    ----------
    reference : Experiment or native Experiment
        Sample and setup defining all fields not supplied by a parameter column.
    parameters : sequence of ParameterSpec or Parameter
        Ordered unique physical columns. Bounded Parameter values are required only
        for sample_uniform(); supplied matrices can use unbounded specifications.

    Notes
    -----
    Sampling columns independently does not enforce concentration normalization
    or coupled geometry. Supply coordinated rows for those constraints. Physical
    validation is performed after applying every column of each row.
    """
    reference: object
    parameters: tuple

    def __post_init__(self):
        object.__setattr__(self, "parameters", tuple(self.parameters))

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.GenerationPlan
            Native value for lower-level workflows.

        Notes
        -----
        Copies the reference and parameter descriptions, then validates their names,
        target identities and reference values. Python sampling bounds are not native
        generation constraints.
        """
        value = engine().GenerationPlan()
        value.reference = to_core(self.reference)
        value.parameters = [to_core(p) for p in self.parameters]
        value.validate()
        return value

    def sample_uniform(self, samples, *, seed=None):
        """Draw reproducible independent uniform parameter columns.

        Parameters
        ----------
        samples : int
            Nonnegative number of rows.
        seed : int or numpy.random seed input, optional
            Seed passed to numpy.random.default_rng; None selects fresh entropy.

        Returns
        -------
        numpy.ndarray
            Float64 matrix shaped (samples, number_of_parameters).

        Raises
        ------
        ValidationError
            samples is invalid, no columns exist, or columns have no bounds.

        Notes
        -----
        Each column uses its Parameter.bounds. Composition sums and angle consistency
        are not imposed by this sampler; use a coupled sampler where required.
        """
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
            raise ValidationError("samples must be a nonnegative integer")
        try:
            bounds = np.asarray([p.bounds for p in self.parameters], dtype=np.float64)
        except AttributeError as error:
            raise ValidationError("uniform sampling requires bounded Parameter columns") from error
        if not self.parameters:
            raise ValidationError("sampling requires parameter columns")
        return np.random.default_rng(seed).uniform(bounds[:, 0], bounds[:, 1], (samples, len(bounds)))

    def generate(self, rows, simulator, destination, **options):
        """Generate a dataset using this plan.

        Parameters
        ----------
        rows : array_like
            Physical parameter matrix shaped (samples, parameters).
        simulator : Simulator
            Built-in or Python-defined simulation backend.
        destination : str or pathlib.Path
            New dataset directory; it must not already exist.
        **options : keyword arguments
            Forwarded to generate(), including failure_policy, progress and batch_size.

        Returns
        -------
        GenerationSummary
            Counts, cancellation state and output path.

        Notes
        -----
        This is the instance-method equivalent of generate(plan, rows, simulator, destination).
        """
        return generate(self, rows, simulator, destination, **options)


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Report the outcome of a generation request.

    Parameters
    ----------
    requested : int
        Number of requested input rows.
    accepted : int
        Successful records committed by this invocation.
    failed : int
        Stored failure records.
    discarded : int
        Failed inputs omitted under DISCARD.
    cancelled : bool
        Whether cooperative cancellation stopped generation.
    path : pathlib.Path
        Output dataset directory.

    Notes
    -----
    Cancellation preserves incomplete output. A dataset becomes readable through
    Dataset only after its final manifest has been committed.
    """
    requested: int
    accepted: int
    failed: int
    discarded: int
    cancelled: bool
    path: Path


def generate(plan, rows, simulator, destination, *, failure_policy=FailurePolicy.STOP,
             progress=None, provenance="", records_per_shard=64, batch_size=256):
    """Generate a sharded dataset from physical parameter rows.

    Parameters
    ----------
    plan : GenerationPlan
        Reference experiment and ordered physical columns.
    rows : array_like, shape (samples, parameters)
        Converted to contiguous float64; compatible arrays are passed without a copy.
    simulator : Simulator or native Simulator
        Built-in native backend or a Python subclass implementing the public interface.
    destination : str or pathlib.Path
        New directory for output. Existing destinations are rejected.
    failure_policy : FailurePolicy, default=FailurePolicy.STOP
        Stop, store failures, or discard failed inputs.
    progress : callable(completed, total), optional
        Progress callback. It may call simulator.request_stop(). Progress can precede
        shard publication; it is not a durability notification.
    provenance : str, default=""
        Caller description appended to backend provenance.
    records_per_shard : int, default=64
        Maximum buffered records per shard, independent of the execution batch size.
    batch_size : int, default=256
        Number of inputs offered to the backend per call; native range is 1..4096.

    Returns
    -------
    GenerationSummary
        Requested/stored/discarded counts, cancellation state and output path.

    Raises
    ------
    ValidationError
        rows is not a two-dimensional matrix or a public option is invalid.
    DatasetError
        Native validation, simulation under STOP, or dataset I/O fails.

    Notes
    -----
    The reference is copied into a bounded set of candidates. Each row is applied
    fully before physical validation. The Python sampling bounds are not enforced on
    supplied rows. Nonfinite failed rows cannot be serialized.

    Cancellation and STOP preserve incomplete output; readers require a completed
    manifest. There is no resume operation on an incomplete destination. Calls on a
    simulator instance must be serialized. A Python batch callback reports progress
    after completing its callback, and per-row failed results follow failure_policy.
    """
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValidationError("rows must have shape [sample,parameter]")
    policy = getattr(engine().FailurePolicy, FailurePolicy(failure_policy).value)
    path = Path(destination)
    try:
        result = engine().generate(to_core(plan), rows, to_core(simulator), path, policy,
                                   progress, provenance, records_per_shard, batch_size)
    except (ValueError, RuntimeError) as error:
        raise DatasetError(str(error)) from error
    return GenerationSummary(len(rows), result.accepted, result.failed, result.discarded, result.cancelled, path)
