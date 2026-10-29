"""Count-preserving spectrum operations backed by the C++ implementation."""
import numpy as np
from ._engine import engine


def _input(values):
    result = np.ascontiguousarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("spectrum processing requires one-dimensional arrays")
    return result


def _output(output, size):
    if output is None:
        return np.empty(size, dtype=np.float64)
    if (not isinstance(output, np.ndarray) or output.dtype != np.float64 or
            output.shape != (size,) or not output.flags.c_contiguous or not output.flags.writeable):
        raise ValueError("output must be a writable contiguous float64 array of the required length")
    return output


def rebin(counts, old_edges, new_edges, *, output=None, require_full_coverage=False):
    """Redistribute integrated counts; reuse output when supplied, without silent conversion."""
    counts, old_edges, new_edges = map(_input, (counts, old_edges, new_edges))
    if len(new_edges) < 2:
        raise ValueError("at least two destination edges are required")
    output = _output(output, len(new_edges) - 1)
    core = engine()
    coverage = core.Coverage.REQUIRE_FULL if require_full_coverage else core.Coverage.ZERO_OUTSIDE
    core.rebin_into(counts, old_edges, new_edges, output, coverage)
    return output


def apply_pileup(counts, *, real_time, live_time, tau, output=None, workspace=None):
    """Apply full-support pair-sum pileup; reuse output and FFT workspace if supplied."""
    counts = _input(counts)
    if not len(counts):
        raise ValueError("pileup requires a nonempty spectrum")
    output = _output(output, 2 * len(counts) - 1)
    core = engine()
    workspace = core.PileupWorkspace() if workspace is None else workspace
    core.apply_pileup_into(counts, output, real_time, live_time, tau, workspace)
    return output


def remove_pileup(counts, *, real_time, live_time, tau, output=None):
    """Recover N bins from complete 2*N-1 pileup support; reject incompatible spectra."""
    counts = _input(counts)
    if not len(counts) or len(counts) % 2 == 0:
        raise ValueError("complete odd-length pileup support is required")
    output = _output(output, len(counts) // 2 + 1)
    engine().remove_pileup_into(counts, output, real_time, live_time, tau)
    return output
