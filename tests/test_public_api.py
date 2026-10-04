import numpy as np
import pytest
import ibeamlab as ibl
from ibeamlab import transforms

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


def test_lrn_input_transform_normalizes_each_layer_and_scales_thickness():
    study = ibl.GenerationStudy(experiment(), [
        ibl.vary.layer_thickness(layer=0, bounds=(1_000, 90_000)),
        ibl.vary.concentration("Li", layer=0, bounds=(0.0, 1.0)),
        ibl.vary.concentration("Ni", layer=0, bounds=(0.0, 1.0)),
    ])
    transform = transforms.build_lrn_input_transform(study)
    values = np.asarray([[50_000.0, 2.0, 6.0]], dtype=np.float32)

    transformed = transform.apply(values)

    assert np.allclose(transformed, [[(50_000 - 1_000) / (90_000 - 1_000), 0.25, 0.75]])


def test_lrn_thickness_uses_each_parameter_bounds():
    study = ibl.GenerationStudy(
        ibl.Experiment(
            ibl.Sample([ibl.Layer(1, {"Li": 1}), ibl.Layer(1, {"Li": 1})]),
            experiment().detectors,
        ),
        [ibl.vary.layer_thickness(layer=0, bounds=(1_000, 200_000)),
         ibl.vary.concentration("Li", layer=0, bounds=(0, 1)),
         ibl.vary.layer_thickness(layer=1, bounds=(5_000, 800_000)),
         ibl.vary.concentration("Li", layer=1, bounds=(0, 1))],
    )
    rows = np.array([[200_000, 1, 200_000, 1], [1_000, 1, 5_000, 1]], dtype=np.float32)
    default = transforms.build_lrn_input_transform(study).apply(rows)
    np.testing.assert_allclose(default, [[1, 1, (200_000 - 5_000) / (800_000 - 5_000), 1], [0, 1, 0, 1]])
    override = transforms.build_lrn_input_transform(study, thickness_bounds=(0, 400_000)).apply(rows)
    np.testing.assert_allclose(override[0], [0.5, 1, 0.5, 1])
    nonzero_min = transforms.build_lrn_input_transform(study, thickness_bounds=(1_000, 400_000)).apply(rows)
    assert nonzero_min[1, 0] == pytest.approx(0)
    with pytest.raises(ValueError, match="high > low"):
        transforms.build_lrn_input_transform(study, thickness_bounds=(1_000, 1_000))


def test_forward_correction_bindings_and_seconds():
    from ibeamlab._native import native
    metadata = native.model.ForwardModelMetadata()
    assert metadata.Apply_pileup_on_inference is True
    assert metadata.pileup_fudge_factor_seconds == pytest.approx(0.4e-6)
    options = native.inference.InferenceOptions()
    assert options.correction_threads == 1
    options.correction_threads = 16
    assert options.correction_threads == 16
    assert options.apply_pileup_on_inference is None
    options.apply_pileup_on_inference = False
    assert options.apply_pileup_on_inference is False
    tau, real, live = 0.4e-6, 10.0, 8.0
    factor = tau / real * np.exp(-6.0 / real * tau)
    expected = live / real * (np.array([2., 4., 0.]) -
        2 * factor * 6 * np.array([2., 4., 0.]) + factor * np.array([4., 16., 16.]))
    assert np.allclose(native.spectrum.pileup([2., 4.], real, live, tau), expected, rtol=1e-12)
