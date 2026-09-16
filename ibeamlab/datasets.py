"""Native dataset reader API."""

from ._ibeamlab_cpp import datasets as _native

__all__ = [name for name in dir(_native) if not name.startswith("_")]
globals().update({name: getattr(_native, name) for name in __all__})
