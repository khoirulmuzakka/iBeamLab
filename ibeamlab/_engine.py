"""Load the compiled extension on first use, preserving dependency load errors."""
from functools import lru_cache
from importlib import import_module
from .exceptions import NativeExtensionError


@lru_cache(maxsize=1)
def engine():
    """Resolve an installed extension or a CMake build directory on sys.path."""
    try:
        return import_module("._ibeamlab", __package__)
    except ModuleNotFoundError as error:
        if error.name != __package__ + "._ibeamlab":
            raise
    try:
        return import_module("_ibeamlab")
    except ModuleNotFoundError as error:
        if error.name != "_ibeamlab":
            raise
        raise NativeExtensionError(
            "The iBeamLab extension is unavailable. Install the package or add "
            "the CMake build/bin directory to PYTHONPATH."
        ) from error


def to_core(value):
    """Accept a Python wrapper or an already materialized C++ value."""
    convert = getattr(value, "to_core", None)
    return convert() if convert is not None else value
