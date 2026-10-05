import math
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import ibeamlab as ibl
from ibeamlab.ibanet import IBAnet, IBAnetLoss, IBAnetPrediction


def make_model():
    study = ibl.GenerationStudy(
        ibl.Experiment(
            ibl.Sample([ibl.Layer(10, {"Si": 0.5, "O": 0.5}) for _ in range(2)]),
            [ibl.Detector("RBS", ibl.Beam("He", 2000), particles_sr=1e8),
             ibl.Detector("PIXE", ibl.Beam("H", 1000), particles_sr=1e8)],
        ), [],
    )
    model = IBAnet(study, {"PIXE": 13, "RBS": 17}, cnn_channels=(4, 8),
                   head_hidden_sizes=(12,), kernel_size=3,
                   input_mean=np.linspace(0, 1, 30), input_std=np.linspace(1, 2, 30))
    return study, model


def test_shapes_moments_and_gradients():
    _, model = make_model()
    result = model(torch.rand(3, 30) * 10)
    assert result.Y.shape == result.R.shape == result.P.shape == (3, 2, 2)
    assert (result.Y > 0).all() and (result.R > model.minimum_std).all()
    assert ((result.P > 0) & (result.P < 1)).all()
    target = torch.tensor([[[0., 1.], [2., 0.]]] * 3)
    loss = IBAnetLoss()(result, target)
    loss.backward()
    for head in (model.mean_head, model.std_head, model.presence_head):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    for encoder in model.encoders:
        assert encoder[0].weight.grad.abs().sum() > 0
    torch.testing.assert_close(result.mean, result.P * result.Y)
    torch.testing.assert_close(result.variance,
        result.P * (result.Y.square() + result.R.square()) - result.mean.square())
    assert result.sample(5).shape == (5, 3, 2, 2)
    assert model.training
    model.predict(torch.zeros(1, 30))
    assert model.training


def test_likelihood_matches_draft_and_masks_absent_regression():
    mean = torch.tensor([[[999., 2.]]], requires_grad=True)
    sigma = torch.tensor([[[1e-20, 0.5]]], requires_grad=True)
    logits = torch.tensor([[[0., math.log(3)]]], requires_grad=True)
    target = torch.tensor([[[0., 3.]]])
    loss = IBAnetLoss()(IBAnetPrediction(mean, sigma, logits), target)
    expected = -math.log(0.5) - math.log(0.75) + math.log(0.5) + 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert mean.grad[0, 0, 0] == sigma.grad[0, 0, 0] == 0
    torch.testing.assert_close(logits.grad, torch.tensor([[[0.5, -0.25]]]))
    assert mean.grad[0, 0, 1].item() == pytest.approx(-4)
    assert sigma.grad[0, 0, 1].item() == pytest.approx(-6)
    absent = IBAnetLoss()(IBAnetPrediction(mean, sigma, logits), torch.zeros_like(target))
    assert torch.isfinite(absent)
    with pytest.raises(ValueError, match="nonnegative"):
        IBAnetLoss()(IBAnetPrediction(mean, sigma, logits), -torch.ones_like(target))


def test_presence_loss_matches_explicit_logs_and_extreme_logits():
    logits = torch.tensor([[[-2., 0., 2., 1000., -1000.]]], requires_grad=True)
    target = torch.tensor([[[0., 1., 1., 1., 0.]]])
    # Y=A and R=1 make the Gaussian loss zero after removing its constant.
    prediction = IBAnetPrediction(target.clone(), torch.ones_like(target), logits)
    loss = IBAnetLoss()(prediction, target)
    p = torch.sigmoid(logits[:, :, :3])
    explicit = -torch.log1p(-p[0, 0, 0]) - torch.log(p[0, 0, 1]) - torch.log(p[0, 0, 2])
    torch.testing.assert_close(loss, explicit)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_invalid_spectrum_inputs():
    _, model = make_model()
    for bad in (torch.zeros(2, 29), -torch.ones(2, 30), torch.full((2, 30), float("nan"))):
        with pytest.raises(ValueError):
            model(bad)


def test_export_matches_native_multi_detector_edp(tmp_path):
    pytest.importorskip("onnx")
    _, model = make_model()
    counts = torch.rand(2, 30) * 30
    expected = model.predict(counts).mean.numpy() / 0.01
    original = {k: v.clone() for k, v in model.state_dict().items()}
    path = model.export(tmp_path / "ibanet.zip", output_inverse_factor=0.01)
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    raw = ibl._native.native.model.ModelPackage.open(path).metadata
    assert raw.format_version == 3 and raw.class_name == "IBAnet"
    assert raw.inverse.need_pileup_subtraction
    assert raw.inverse.pileup_fudge_factor_seconds == pytest.approx(0.4e-6)
    assert list(raw.inverse.output_edp.elements) == ["Si", "O"]
    batches = [{"PIXE": ibl.Spectrum("PIXE", row[17:].numpy()),
                "RBS": ibl.Spectrum("RBS", row[:17].numpy())} for row in counts]
    actual = ibl.InverseModel(path).predict_prepared(batches)
    np.testing.assert_allclose(np.stack([x.values for x in actual]), expected, rtol=2e-5, atol=1e-5)
    no_subtraction = model.export(tmp_path / "ibanet_with_pileup.zip", need_pileup_subtraction=False)
    assert not ibl._native.native.model.ModelPackage.open(no_subtraction).metadata.inverse.need_pileup_subtraction
    with pytest.raises(Exception):
        model.export(path)


def test_experimental_pipeline_matches_prepared_and_pytorch(tmp_path):
    pytest.importorskip("onnx")
    from ibeamlab.spectrum import pileup
    study, model = make_model()
    reference = np.linspace(2, 30, 30, dtype=np.float32)
    expected = model.predict(torch.from_numpy(reference[None])).mean.numpy()[0] / 0.01
    path = model.export(tmp_path / "corrected.zip", output_inverse_factor=0.01,
                        pileup_fudge_factor_seconds=0.003)
    deployed = ibl.InverseModel(path)
    assert deployed.metadata.inverse.pileup_fudge_factor_seconds == pytest.approx(0.003)
    measured_spectra, bare_spectra, detectors = {}, {}, []
    offset = 0
    for index, detector in enumerate(study.experiment.detectors):
        width = model.input_spectra_lengths[detector.label]
        exposure = index + 2
        # Finer experimental calibration: split each reference bin into two.
        bare = np.repeat(reference[offset:offset + width] * exposure / 2, 2)
        measured = pileup(bare.tolist(), 10, 8, 0.003)
        measured_spectra[detector.label] = ibl.Spectrum(detector.label, measured)
        bare_spectra[detector.label] = ibl.Spectrum(detector.label, bare)
        detectors.append(replace(detector, calibration=ibl.LinearCalibration(linear=0.5),
                                  particles_sr=detector.particles_sr * exposure,
                                  real_time=10, live_time=8))
        offset += width
    measurement = ibl.InverseInput(measured_spectra, detectors)
    removed = ibl.InverseInput(bare_spectra, detectors, pileup_already_removed=True)
    for output in deployed.predict([measurement, removed]):
        np.testing.assert_allclose(output.values, expected, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(deployed.predict(measurement).values, expected, rtol=2e-5, atol=1e-5)
    with pytest.raises(ibl.InferenceError, match="InverseInput"):
        deployed.predict(measured_spectra)
    bad_detectors = [replace(detectors[0], particles_sr=0), detectors[1]]
    with pytest.raises(ibl.InferenceError, match="ParticlesSr"):
        deployed.predict(ibl.InverseInput(measured_spectra, bad_detectors))
    with pytest.raises(ibl.InferenceError, match="missing experimental detector"):
        deployed.predict(ibl.InverseInput(measured_spectra, detectors[1:]))


def test_notebook_end_to_end_on_small_generated_dataset(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    root = Path(__file__).parents[1]
    configuration = ibl.load_generation_configuration(root / "examples" / "multilayer-generation.toml")
    for layers in (1, 2):
        study = configuration.study(layers)
        study.generate(tmp_path / "datasets" / f"layers_{layers:02}",
                       simulator=ibl.DummySimulator(channels=32),
                       samples=study.sample(20, seed=layers))
    notebook = json.loads((root / "examples" / "train_ibanet.ipynb").read_text())
    context = {"__name__": "__main__"}
    monkeypatch.chdir(root)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if index == 3:
            source = source.replace('DATASET_ROOT = EXAMPLES / "datasets"',
                                    f"DATASET_ROOT = Path({str(tmp_path / 'datasets')!r})")
            source = source.replace("MAX_LAYERS = 10", "MAX_LAYERS = 2")
            source = source.replace('OUTPUT_DIR = EXAMPLES / "artifacts" / "ibanet_multilayer"',
                                    f"OUTPUT_DIR = Path({str(tmp_path / 'artifacts')!r})")
        if index == 7:
            source = source.replace("CNN_CHANNELS = (16, 32, 64, 64)", "CNN_CHANNELS = (4, 8)")
            source = source.replace("HEAD_HIDDEN_SIZES = (512, 256)", "HEAD_HIDDEN_SIZES = (16,)")
        if index == 9:
            source = source.replace('"epochs": 20', '"epochs": 1').replace('"epochs": 10', '"epochs": 1')
        exec(compile(source, f"train_ibanet.ipynb cell {index}", "exec"), context)
    assert context["native_edp"].shape[1:] == (2, len(configuration.elements))
    assert context["weights_path"].is_file()
    checkpoint = torch.load(context["weights_path"], weights_only=True)
    assert "input_mean" in checkpoint["model_state_dict"]
    assert context["package_path"].is_file()
