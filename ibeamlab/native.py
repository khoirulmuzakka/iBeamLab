"""Advanced, unstable access to the compiled API.

Normal applications should use the public objects exported by :mod:`ibeamlab`.
"""
from ._native import native

sample = native.sample
simulator = native.simulator
generation = native.generation
datasets = native.datasets
spectrum = native.spectrum
transforms = native.transforms
model = native.model
inference = native.inference
