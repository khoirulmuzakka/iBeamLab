"""Physical parameter plans and C++ dataset generation with explicit failure policies."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
from ._engine import engine, to_core
from .exceptions import DatasetError, ValidationError


class FailurePolicy(Enum):
    """Stop, store a failure record, or discard a failed input row."""
    STOP = "STOP"
    RECORD = "RECORD"
    DISCARD = "DISCARD"


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """Reference experiment and ordered physical parameter columns.

    sample_uniform samples bounded columns independently. Supply coupled rows
    yourself for concentrations; generation never silently normalizes them.
    """
    reference: object
    parameters: tuple

    def __post_init__(self):
        object.__setattr__(self, "parameters", tuple(self.parameters))

    def to_core(self):
        value = engine().GenerationPlan()
        value.reference = to_core(self.reference)
        value.parameters = [to_core(p) for p in self.parameters]
        value.validate()
        return value

    def sample_uniform(self, samples, *, seed=None):
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
        return generate(self, rows, simulator, destination, **options)


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Generation outcome; cancellation leaves an incomplete dataset."""
    requested: int
    accepted: int
    failed: int
    discarded: int
    cancelled: bool
    path: Path


def generate(plan, rows, simulator, destination, *, failure_policy=FailurePolicy.STOP,
             progress=None, provenance="", records_per_shard=64, batch_size=256):
    """Generate from a contiguous float64 [sample,parameter] matrix.

    Compatible NumPy arrays are passed without copying. progress(completed,total)
    may request cooperative cancellation through simulator.request_stop().
    Bounded batches keep backend workers busy; batch_size is independent of
    records_per_shard. Progress precedes buffered dataset publication.
    Destination must not exist; failures under STOP leave an incomplete directory.
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
