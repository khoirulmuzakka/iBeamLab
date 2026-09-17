"""Native operations on physical spectra and channel grids."""

from ._ibeamlab_cpp.spectrum import (
    clip,
    concatenate,
    crop_or_pad,
    energy_to_channel_and_pileup,
    pileup,
    rebin,
)

__all__ = [
    "clip",
    "concatenate",
    "crop_or_pad",
    "energy_to_channel_and_pileup",
    "pileup",
    "rebin",
]
