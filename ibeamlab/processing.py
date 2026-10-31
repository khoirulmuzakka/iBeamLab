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
    """Redistribute integrated counts by energy-bin overlap.

    Parameters
    ----------
    counts : array_like, shape (N,)
        Finite nonnegative integrated source-bin counts.
    old_edges : array_like, shape (N+1,)
        Finite strictly increasing source energy edges.
    new_edges : array_like, shape (M+1,)
        Finite strictly increasing destination energy edges, in the same units.
    output : numpy.ndarray, optional
        Writable contiguous float64 destination shaped (M,); returned directly when supplied.
    require_full_coverage : bool, default=False
        Require the source energy interval to cover the destination interval.
        Otherwise uncovered destination regions receive zero.

    Returns
    -------
    numpy.ndarray
        Float64 counts shaped (M,). A supplied output buffer is reused.

    Raises
    ------
    ValueError
        Input values, grids, times, dimensions, buffer layout or overlap are invalid.

    Notes
    -----
    Counts outside the destination range are not retained. This is overlap-based
    integrated-count redistribution, not point interpolation or bin-width normalization.

    Inputs are converted to contiguous float64 only when needed. Output buffers
    are never silently converted and must not overlap input storage.
    """
    counts, old_edges, new_edges = map(_input, (counts, old_edges, new_edges))
    if len(new_edges) < 2:
        raise ValueError("at least two destination edges are required")
    output = _output(output, len(new_edges) - 1)
    core = engine()
    coverage = core.Coverage.REQUIRE_FULL if require_full_coverage else core.Coverage.ZERO_OUTSIDE
    core.rebin_into(counts, old_edges, new_edges, output, coverage)
    return output


def apply_pileup(counts, *, real_time, live_time, tau, output=None, workspace=None):
    """Apply the full-support pair-sum pileup model.

    Parameters
    ----------
    counts : array_like, shape (N,)
        Nonempty finite nonnegative input counts on a compatible uniform energy grid.
    real_time : float
        Positive real acquisition time in seconds.
    live_time : float
        Positive live acquisition time in seconds, not greater than real_time.
    tau : float
        Finite nonnegative resolving time in seconds.
    output : numpy.ndarray, optional
        Writable contiguous float64 output shaped (2*N-1,).
    workspace : PileupWorkspace, optional
        Reusable FFT workspace. Use a separate workspace per concurrent caller.

    Returns
    -------
    numpy.ndarray
        Float64 counts shaped (2*N-1,). A supplied output buffer is reused.

    Raises
    ------
    ValueError
        Input values, grids, times, dimensions, buffer layout or overlap are invalid.

    Notes
    -----
    The complete pair-sum support is returned, including channels beyond the
    original spectrum. Retaining the full support is required for exact inversion.

    Inputs are converted to contiguous float64 only when needed. Output buffers
    are never silently converted and must not overlap input storage.
    """
    counts = _input(counts)
    if not len(counts):
        raise ValueError("pileup requires a nonempty spectrum")
    output = _output(output, 2 * len(counts) - 1)
    core = engine()
    workspace = core.PileupWorkspace() if workspace is None else workspace
    core.apply_pileup_into(counts, output, real_time, live_time, tau, workspace)
    return output


def remove_pileup(counts, *, real_time, live_time, tau, output=None):
    """Recover the bare spectrum from complete pair-sum pileup support.

    Parameters
    ----------
    counts : array_like, shape (2*N-1,)
        Nonempty odd-length piled-up spectrum with its complete support.
    real_time : float
        Positive real acquisition time in seconds.
    live_time : float
        Positive live acquisition time in seconds, not greater than real_time.
    tau : float
        Resolving time in seconds, matching the forward operation.
    output : numpy.ndarray, optional
        Writable contiguous float64 output shaped (N,).

    Returns
    -------
    numpy.ndarray
        Float64 counts shaped (N,). A supplied output buffer is reused.

    Raises
    ------
    ValueError
        Input values, grids, times, dimensions, buffer layout or overlap are invalid.

    Notes
    -----
    Truncated support or spectra incompatible with the forward model are rejected.
    The native reconstruction checks its forward consistency; this operation does
    not provide a generic deconvolution for arbitrary pileup models.

    Inputs are converted to contiguous float64 only when needed. Output buffers
    are never silently converted and must not overlap input storage.
    """
    counts = _input(counts)
    if not len(counts) or len(counts) % 2 == 0:
        raise ValueError("complete odd-length pileup support is required")
    output = _output(output, len(counts) // 2 + 1)
    engine().remove_pileup_into(counts, output, real_time, live_time, tau)
    return output
