from __future__ import annotations

import pathlib
import sys

import numpy as np


root = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "examples"))

from generate import build_generation_config
from ibeamlab.sampling import sample_uniform_parameters


config = build_generation_config()
first = sample_uniform_parameters(config, 64, 123)
second = sample_uniform_parameters(config, 64, 123)
different = sample_uniform_parameters(config, 64, 124)

assert first.shape == (64, 5)
assert np.array_equal(first, second)
assert not np.array_equal(first, different)
assert np.all(first >= 0.0) and np.all(first <= 1.0)
assert np.allclose(first.sum(axis=1), 1.0)
