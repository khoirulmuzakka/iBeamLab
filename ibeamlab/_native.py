"""Lazy access to the private compiled engine."""

from .exceptions import NativeExtensionError

try:
    from . import _ibeamlab_cpp as native
except (ImportError, OSError) as error:
    raise NativeExtensionError(
        "The iBeamLab native engine could not be loaded. Install a wheel matching "
        "your Python version and platform, or build with `pip install .`."
    ) from error

__all__ = ["native"]
