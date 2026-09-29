from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
import ibeamlab as ibl

experiment = ibl.Experiment(
    ibl.Sample([ibl.Layer(100, {"Li": 0.25, "Ni": 0.25, "O": 0.5})]),
    [ibl.Detector("RBS", ibl.Beam("H", 2974))],
)
study = ibl.GenerationStudy(experiment, [
    ibl.vary.concentration("Li", layer=0, bounds=(0, 0.5)),
    ibl.vary.concentration("Ni", layer=0, bounds=(0, 0.5)),
])
first = study.sample(64, seed=123)
second = study.sample(64, seed=123)
different = study.sample(64, seed=124)
assert first.shape == (64, 2)
assert np.array_equal(first, second)
assert not np.array_equal(first, different)
assert np.allclose(first.sum(axis=1), 0.5)
