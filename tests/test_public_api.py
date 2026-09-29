import numpy as np
import pytest
import ibeamlab as ibl

def experiment():
    return ibl.Experiment(
        sample=ibl.Sample([ibl.Layer(500_000, {"Li": 0.25, "Ni": 0.25, "O": 0.5})]),
        detectors=[ibl.Detector("RBS", ibl.Beam("H", 2974), resolution=20)],
    )

def test_domain_objects_are_pythonic():
    value = experiment()
    assert value.sample.layers[0].composition["Li"] == 0.25
    assert "RBS" in repr(value)

def test_validation_explains_normalization():
    with pytest.raises(ibl.ValidationError, match="sum to"):
        ibl.Layer(1, {"Li": 0.2, "O": 0.2})

def test_dummy_simulator_returns_mapping():
    result = ibl.DummySimulator(channels=32).simulate(experiment())
    assert result["RBS"].counts.shape == (32,)
    assert result["RBS"].counts.dtype == np.float32

def test_study_sampling_and_native_config(tmp_path):
    study = ibl.GenerationStudy(experiment(), [
        ibl.vary.concentration("Li", layer=0, bounds=(0.1, 0.4)),
        ibl.vary.concentration("Ni", layer=0, bounds=(0.1, 0.4)),
    ])
    rows = study.sample(10, seed=7)
    assert rows.shape == (10, 2)
    assert np.allclose(rows.sum(axis=1), 0.5)
    study._config()
    summary = study.generate(tmp_path / "dataset", simulator=ibl.DummySimulator(32), samples=rows)
    dataset = ibl.open_dataset(summary.path)
    assert dataset.parameters.shape == (10, 2)
    assert dataset.spectra("RBS").shape == (10, 32)
