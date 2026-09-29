"""Model loading; prefer importing these names from :mod:`ibeamlab`."""

from . import models as _models

__all__ = _models.__all__


def __getattr__(name: str):
    return getattr(_models, name)
