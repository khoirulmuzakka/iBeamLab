"""Python interface to the standalone iBeamLab C++ library."""
try:
    from ._ibeamlab import *
except ModuleNotFoundError as error:
    if error.name != __name__ + "._ibeamlab":
        raise
    from _ibeamlab import *  # CMake build-tree use
from .export import export_package
