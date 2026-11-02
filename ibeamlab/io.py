"""Experimental spectrum readers; numerical data never implies detector geometry."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ._engine import engine, to_core
from .exceptions import ValidationError, DataReadError


def _array(values, dtype, *, optional=False):
    if values is None and optional:
        return None
    result = np.array(values, dtype=dtype, order="C", copy=True)
    if result.ndim != 1:
        raise ValidationError("observation arrays must be one-dimensional")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MeasuredSpectrum:
    """Own observed counts and optional masks/variances.

    Parameters
    ----------
    measurement : str
        Label of the corresponding measurement configuration.
    counts : array_like
        Finite, nonnegative counts in zero-based channel order.
    valid : array_like of bool, optional
        False excludes missing or unusable channels, including gaps in a file.
    variances : array_like, optional
        Known nonnegative variances. Selected channels must have positive variance
        when the fitting loss is explicitly set to ``variance_chi_square``.
    source : str, default=""
        Source identifier for diagnostics; it does not supply calibration.

    Notes
    -----
    Arrays are copied once and made read-only. An absent mask means all channels
    are observed. Counts at masked positions must still be finite/nonnegative.
    """
    measurement: str
    counts: object
    valid: object = None
    variances: object = None
    source: str = ""

    def __post_init__(self):
        object.__setattr__(self, "counts", _array(self.counts, np.float64))
        object.__setattr__(self, "valid", _array(self.valid, bool, optional=True))
        object.__setattr__(self, "variances", _array(self.variances, np.float64, optional=True))
        if not self.measurement or not self.counts.size or not np.all(np.isfinite(self.counts)) or np.any(self.counts < 0):
            raise ValidationError("observation requires a label and finite nonnegative counts")
        for values in (self.valid, self.variances):
            if values is not None and values.shape != self.counts.shape:
                raise ValidationError("observation mask/variance dimensions differ")
        if self.variances is not None and (not np.all(np.isfinite(self.variances)) or np.any(self.variances < 0)):
            raise ValidationError("variances must be finite and nonnegative")

    def to_core(self):
        """Copy this observation into an owning native value."""
        value = engine().MeasuredSpectrum()
        value.measurement, value.counts, value.source = self.measurement, self.counts, self.source
        if self.valid is not None:
            value.valid = self.valid.astype(np.uint8)
        if self.variances is not None:
            value.variances = self.variances
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MeasuredData:
    """Associate observations with an explicit setup, independently of a sample.

    Parameters
    ----------
    setup : ExperimentalSetup
        Beam, geometry, calibration and acquisition settings for each observation.
    spectra : sequence of MeasuredSpectrum
        Exactly one observation for every setup measurement, matched by label.
    """
    setup: object
    spectra: tuple

    def __post_init__(self):
        object.__setattr__(self, "spectra", tuple(self.spectra))

    def to_core(self):
        """Validate and copy observations and setup into native storage."""
        value = engine().MeasuredData()
        value.setup, value.spectra = to_core(self.setup), [to_core(s) for s in self.spectra]
        value.validate()
        return value


_READERS = {}


def register_spectrum_reader(name, reader, *, replace=False):
    """Register ``reader(path, measurement=..., **options) -> MeasuredSpectrum``.

    Names of built-in readers are reserved. Existing custom names require
    ``replace=True``. The reader must preserve missing-channel validity itself.
    """
    if not isinstance(name, str) or not name or name in {"counts", "channel_counts", "csv"}:
        raise ValueError("custom reader name must be nonempty and unreserved")
    if not callable(reader):
        raise TypeError("reader must be callable")
    if name in _READERS and not replace:
        raise ValueError(f"reader already registered: {name}")
    _READERS[name] = reader


def read_spectrum(path, *, measurement, format="channel_counts", **options):
    """Read a measured spectrum without inferring experimental settings.

    Parameters
    ----------
    path : path_like
        Text file to read.
    measurement : str
        Explicit setup measurement label.
    format : {"counts", "channel_counts", "csv"} or registered name
        Counts-only uses column zero. Channel/counts defaults to whitespace
        columns zero/one; CSV defaults to a comma delimiter.
    **options
        Native reader controls: ``counts_column=1``, ``channel_column=0``
        (None for sequential rows), ``variance_column=None``, ``channel_base=0``
        (or 1), ``header=False``, ``skip_rows=0``, ``delimiter`` and ``comment``
        (single character; empty disables), ``maximum_channels=1000000``.
        Column indices are always zero-based. Explicit channels must be unique
        integers. Blank/comment lines are skipped; skip_rows counts physical lines.

    Returns
    -------
    MeasuredSpectrum
        Read-only counts and a validity mask preserving gaps.

    Raises
    ------
    DataReadError
        Invalid data or an unreadable file; parse errors identify path and line.
    """
    path = Path(path)
    if format in _READERS:
        result = _READERS[format](path, measurement=measurement, **options)
        if not isinstance(result, MeasuredSpectrum) or result.measurement != measurement:
            raise DataReadError("custom reader must return MeasuredSpectrum with the requested label")
        return result
    formats = {"counts": "COUNTS", "channel_counts": "CHANNEL_COUNTS", "csv": "CSV"}
    if format not in formats:
        raise ValueError(f"unknown spectrum format: {format}")
    native = engine()
    settings = native.SpectrumReadOptions()
    settings.format = getattr(native.SpectrumFormat, formats[format])
    for name, value in options.items():
        if not hasattr(settings, name) or name == "format":
            raise TypeError(f"unknown reader option: {name}")
        if name in {"delimiter", "comment"} and value == "":
            value = "\0"
        setattr(settings, name, value)
    try:
        result = native.read_spectrum(path, measurement, settings)
    except (ValueError, RuntimeError) as error:
        raise DataReadError(str(error)) from error
    return MeasuredSpectrum(result.measurement, result.counts, result.valid or None,
                            result.variances or None, result.source)


@dataclass(frozen=True, slots=True)
class TimeField:
    """Header marker and unit conversion; fallback is already in seconds."""
    marker: str = ""
    seconds_per_unit: float = 1.
    fallback: object = None

    def to_core(self):
        """Copy explicit header-time settings into the native parser."""
        value = engine().SpectrumTimeField()
        value.marker, value.seconds_per_unit, value.fallback = self.marker, self.seconds_per_unit, self.fallback
        return value


@dataclass(frozen=True, slots=True)
class SpectrumParser:
    """Instrument text layout with optional data markers and acquisition times.

    records supplies read_spectrum options (including format). start_marker and
    end_marker exclude their own lines. Missing channels remain masked unless
    missing_channels_are_zero is explicitly enabled. Markers are case-sensitive.
    """
    records: object = None
    start_marker: str = ""
    end_marker: str = ""
    real_time: TimeField = TimeField()
    live_time: TimeField = TimeField()
    missing_channels_are_zero: bool = False

    def to_core(self):
        """Create native parsing settings; column indices are zero-based."""
        n = engine()
        value = n.SpectrumParser()
        options = dict(self.records or {})
        records = n.SpectrumReadOptions()
        format_name = options.pop("format", "channel_counts")
        if format_name not in {"counts", "channel_counts", "csv"}:
            raise ValueError("unknown instrument record format")
        records.format = getattr(n.SpectrumFormat, format_name.upper())
        for name, setting in options.items():
            if not hasattr(records, name):
                raise TypeError(f"unknown reader option: {name}")
            setattr(records, name, "\0" if name in {"delimiter", "comment"} and setting == "" else setting)
        value.records = records
        value.start_marker, value.end_marker = self.start_marker, self.end_marker
        value.real_time, value.live_time = self.real_time.to_core(), self.live_time.to_core()
        value.missing_channels_are_zero = self.missing_channels_are_zero
        return value


@dataclass(frozen=True, slots=True)
class ImportedSpectrum:
    """Observed spectrum and optional acquisition times, expressed in seconds."""
    spectrum: MeasuredSpectrum
    real_time: object = None
    live_time: object = None


def read_instrument_spectrum(path, measurement, parser):
    """Read an explicit instrument layout; errors include file context."""
    try:
        result = engine().read_instrument_spectrum(Path(path), measurement, to_core(parser))
        s = result.spectrum
        return ImportedSpectrum(MeasuredSpectrum(s.measurement, s.counts, s.valid or None,
                                                s.variances or None, s.source),
                                result.real_time, result.live_time)
    except (ValueError, RuntimeError) as error:
        raise DataReadError(f"{path}: {error}") from error


def write_fit_result(path, result, *, provenance=""):
    """Write scientific CSV tables into a new directory, refusing overwrite."""
    engine().write_fit_result(Path(path), result._core, provenance)
