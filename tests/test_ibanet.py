import math
import json
import re
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import ibeamlab as ibl
from ibeamlab.ibeamnet import iBeamNet, iBeamNetLoss, iBeamNetPrediction


def make_model(layers=2, elements=("Si", "O")):
    study = ibl.GenerationStudy(
        ibl.Experiment(
            ibl.Sample([ibl.Layer(10, {e: 1 / len(elements) for e in elements})
                        for _ in range(layers)]),
            [ibl.Detector("RBS", ibl.Beam("He", 2000), particles_sr=1e8),
             ibl.Detector("PIXE", ibl.Beam("H", 1000), particles_sr=1e8)],
        ), [],
    )
    model = iBeamNet(study, {"PIXE": 13, "RBS": 17}, version=1, input_mean=np.linspace(0, 1, 30), input_std=np.linspace(1, 2, 30), model_options=dict(cnn_channels=(4, 8), compression_channels=4, head_hidden_sizes=(12,), kernel_size=3, decoder_hidden_size=16, decoder_layers=2, layer_embedding_size=4))
    return study, model


def test_shapes_moments_and_gradients():
    _, model = make_model()
    result = model(torch.rand(3, 30) * 10)
    assert result.Y.shape == result.R.shape == result.P.shape == (3, 2, 2)
    assert (result.Y > 0).all() and (result.R > model.minimum_std).all()
    assert ((result.P > 0) & (result.P < 1)).all()
    target = torch.tensor([[[0., 1.], [2., 0.]]] * 3)
    loss = iBeamNetLoss()(result, target)
    loss.backward()
    for head in (model.mean_head, model.std_head, model.presence_head):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    for encoder in model.encoders:
        assert encoder[0].conv1.weight.grad.abs().sum() > 0
        assert encoder[-2].weight.grad.abs().sum() > 0
    assert model.decoder.weight_hh_l0.grad.abs().sum() > 0
    assert model.layer_embedding.weight.grad.abs().sum() > 0
    torch.testing.assert_close(result.mean, result.P * result.Y)
    torch.testing.assert_close(result.variance,
        result.P * (result.Y.square() + result.R.square()) - result.mean.square())
    assert result.sample(5).shape == (5, 3, 2, 2)
    assert model.training
    model.predict(torch.zeros(1, 30))
    assert model.training


def test_decoder_is_causal_and_supports_ten_by_ten_outputs():
    elements = ("Li", "Ni", "Mn", "Co", "O", "C", "F", "P", "Cu", "Al")
    _, model = make_model(layers=10, elements=elements)
    model.eval()
    counts = torch.rand(3, 30) * 10
    before = model(counts)
    assert before.Y.shape == before.R.shape == before.logits.shape == (3, 10, 10)
    assert not model.decoder.bidirectional
    # Changing a deeper layer's input cannot change shallower predictions.
    with torch.no_grad():
        model.layer_embedding.weight[5].add_(torch.arange(4) * 10)
    after = model(counts)
    for name in ("Y", "R", "logits"):
        torch.testing.assert_close(getattr(before, name)[:, :5], getattr(after, name)[:, :5])
        assert not torch.allclose(getattr(before, name)[:, 5:], getattr(after, name)[:, 5:])
    # Later predictions must also receive information from earlier layer states.
    model.zero_grad()
    model(counts).Y[:, -1].sum().backward()
    assert model.layer_embedding.weight.grad[0].abs().sum() > 0


def test_checkpoint_reconstruction_and_eval_dropout():
    study, model = make_model()
    counts = torch.rand(2, 30) * 10
    model.eval()
    expected = model(counts)
    restored = iBeamNet(study, model.input_spectra_lengths, version=1, model_options=dict(**model.architecture))
    restored.load_state_dict(model.state_dict())
    restored.eval()
    for name in ("Y", "R", "logits"):
        torch.testing.assert_close(getattr(expected, name), getattr(restored(counts), name))
        torch.testing.assert_close(getattr(expected, name), getattr(model(counts), name))
    model.train()
    assert not torch.allclose(model(counts).Y, model(counts).Y)


def test_bidirectional_transfer_and_joint_layer_context():
    study, original = make_model(layers=4)
    original.eval()
    source_state = {key: value.clone() for key, value in original.state_dict().items()}
    model = original.as_bidirectional()
    assert model.decoder.bidirectional and model.decoder_directions == 2
    assert model.initial_hidden.out_features == 2 * model.decoder_layers * model.decoder_hidden_size
    torch.testing.assert_close(model.decoder.weight_hh_l0_reverse, original.decoder.weight_hh_l0)
    torch.testing.assert_close(model.input_mean, original.input_mean)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, source_state[key])
    counts = torch.rand(3, 30)
    before = model(counts)
    assert before.Y.shape == (3, 4, 2)
    with torch.no_grad():
        model.layer_embedding.weight[-1].add_(torch.arange(4) * 10)
    after = model(counts)
    assert not torch.allclose(before.Y[:, 0], after.Y[:, 0])
    model.zero_grad()
    model(counts).Y[:, 0].sum().backward()
    assert model.layer_embedding.weight.grad[-1].abs().sum() > 0
    assert model.decoder.weight_hh_l0_reverse.grad.abs().sum() > 0
    restored = iBeamNet(study, model.input_spectra_lengths, version=1, elements=model.elements, model_options=dict(**model.architecture)).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(counts).mean, restored(counts).mean)


def test_legacy_checkpoint_without_direction_option_still_loads():
    study, model = make_model()
    architecture = dict(model.architecture)
    architecture.pop("bidirectional")
    restored = iBeamNet(study, model.input_spectra_lengths, version=1, model_options=dict(**architecture))
    restored.load_state_dict(model.state_dict())
    assert not restored.decoder.bidirectional


def test_channel_compression_preserves_energy_positions_and_reduces_projection():
    study, _ = make_model()
    lengths = {"RBS": 2997, "PIXE": 17}
    compressed = iBeamNet(study, lengths, version=1)
    uncompressed = iBeamNet(study, lengths, version=1, model_options=dict(compression_channels=None))
    for encoder, length in zip(compressed.encoders, compressed.input_spectra_lengths.values()):
        reduced_length = (length + 15) // 16
        with torch.no_grad():
            before = encoder[:-2](torch.rand(1, 1, length))
            after = encoder[-2](before)
        assert before.shape == (1, 128, reduced_length)
        assert after.shape == (1, 16, reduced_length)
        assert encoder[-2].kernel_size == (1,)
    assert compressed.context[0].in_features == 16 * (188 + 2)
    assert uncompressed.context[0].weight.numel() == 8 * compressed.context[0].weight.numel()
    # An earlier GRU checkpoint has no compression option in its architecture.
    old_architecture = dict(uncompressed.architecture)
    old_architecture.pop("compression_channels")
    restored = iBeamNet(study, lengths, version=1, model_options=dict(compression_channels=None, **old_architecture))
    restored.load_state_dict(uncompressed.state_dict())
    inputs = torch.rand(1, sum(lengths.values()))
    torch.testing.assert_close(restored.predict(inputs).mean, uncompressed.predict(inputs).mean)


@pytest.mark.parametrize("kwargs", [
    {"decoder_hidden_size": 0}, {"decoder_layers": 0},
    {"layer_embedding_size": 0}, {"decoder_layers": 1.5},
    {"dropout": -0.1}, {"dropout": 1.0}, {"dropout": float("nan")},
    {"compression_channels": 0}, {"compression_channels": 1.5},
    {"compression_channels": True},
])
def test_invalid_decoder_configuration(kwargs):
    study, model = make_model()
    with pytest.raises(ValueError):
        iBeamNet(study, model.input_spectra_lengths, version=1, model_options=dict(**kwargs))


def test_likelihood_matches_draft_and_masks_absent_regression():
    mean = torch.tensor([[[999., 2.]]], requires_grad=True)
    sigma = torch.tensor([[[1e-20, 0.5]]], requires_grad=True)
    logits = torch.tensor([[[0., math.log(3)]]], requires_grad=True)
    target = torch.tensor([[[0., 3.]]])
    loss = iBeamNetLoss()(iBeamNetPrediction(mean, sigma, logits), target)
    expected = -math.log(0.5) - math.log(0.75) + math.log(0.5) + 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert mean.grad[0, 0, 0] == sigma.grad[0, 0, 0] == 0
    torch.testing.assert_close(logits.grad, torch.tensor([[[0.5, -0.25]]]))
    assert mean.grad[0, 0, 1].item() == pytest.approx(-4)
    assert sigma.grad[0, 0, 1].item() == pytest.approx(-6)
    absent = iBeamNetLoss()(iBeamNetPrediction(mean, sigma, logits), torch.zeros_like(target))
    assert torch.isfinite(absent)
    with pytest.raises(ValueError, match="nonnegative"):
        iBeamNetLoss()(iBeamNetPrediction(mean, sigma, logits), -torch.ones_like(target))


def test_presence_loss_matches_explicit_logs_and_extreme_logits():
    logits = torch.tensor([[[-2., 0., 2., 1000., -1000.]]], requires_grad=True)
    target = torch.tensor([[[0., 1., 1., 1., 0.]]])
    # Y=A and R=1 make the Gaussian loss zero after removing its constant.
    prediction = iBeamNetPrediction(target.clone(), torch.ones_like(target), logits)
    loss = iBeamNetLoss()(prediction, target)
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
    posterior = model.predict(counts)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    path = model.export(tmp_path / "ibanet.zip", output_inverse_factor=0.01)
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    raw = ibl._native.native.model.ModelPackage.open(path).metadata
    assert raw.format_version == 4 and raw.class_name == "iBeamNet"
    assert raw.inverse.need_pileup_subtraction
    assert raw.inverse.pileup_fudge_factor_seconds == pytest.approx(0.4e-6)
    assert list(raw.inverse.output_edp.elements) == ["Si", "O"]
    batches = [{"PIXE": ibl.Spectrum("PIXE", row[17:].numpy()),
                "RBS": ibl.Spectrum("RBS", row[:17].numpy())} for row in counts]
    actual = ibl.InverseModel(path).predict_prepared(batches)
    np.testing.assert_allclose(np.stack([x.values for x in actual]), expected, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(np.stack([x.presence_probability for x in actual]), posterior.P.numpy(), rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(np.stack([x.posterior_std for x in actual]), posterior.variance.sqrt().numpy() / 0.01, rtol=2e-5, atol=1e-5)
    assert all(x.uncertainty_predicted for x in actual)
    for result in actual:
        result.edp.validate()
    no_subtraction = model.export(tmp_path / "ibanet_with_pileup.zip", need_pileup_subtraction=False)
    assert not ibl._native.native.model.ModelPackage.open(no_subtraction).metadata.inverse.need_pileup_subtraction
    with pytest.raises(Exception):
        model.export(path)


def test_ten_layer_export_handles_dynamic_batch_sizes(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import zipfile
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    elements = ("Li", "Ni", "Mn", "Co", "O", "C", "F", "P", "Cu", "Al")
    _, model = make_model(layers=10, elements=elements)
    package = model.export(tmp_path / "ten_layers.zip", output_inverse_factor=0.01)
    with zipfile.ZipFile(package) as archive:
        metadata = tomllib.loads(archive.read("package.toml").decode())
        session = ort.InferenceSession(archive.read(metadata["model"]["onnx_file"]),
                                      providers=["CPUExecutionProvider"])
    for batch_size in (1, 3, 8):
        inputs = torch.rand(batch_size, 30) * 10
        expected = model.predict(inputs).mean.flatten(1).numpy()
        actual = session.run(None, {"inputs": inputs.numpy()})[0]
        assert actual.shape == (batch_size, 100)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)


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
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    from test_encoder_notebook import make_sources
    from test_train_ibeamnet_notebook import run_notebook
    dataset,_,_=make_sources(tmp_path)
    context=run_notebook(dataset,tmp_path/'v1',version=1)
    assert context['model'].version==1 and context['completed_epochs']==2
    np.testing.assert_allclose(context['native_edp'],context['export_reference'],rtol=2e-4,atol=1e-3)
    np.testing.assert_array_equal(context['x_val'],context['fixed_poisson_counts'](
        context['x_raw'][context['val_indices']],context['SEED']+1))


def test_edp_requires_probability_and_std_matrices():
    native = ibl._native.native
    edp = native.inference.EdpMap()
    edp.elements = ["Si"]
    edp.values = [[2.0]]
    with pytest.raises(ValueError):
        edp.validate()
    edp.presence_probability = [[0.4]]
    edp.posterior_std = [[0.5]]
    edp.validate()
    for probability, deviation in ((1.1, 0.5), (float("nan"), 0.5), (0.4, -1.0)):
        edp.presence_probability = [[probability]]
        edp.posterior_std = [[deviation]]
        with pytest.raises(ValueError):
            edp.validate()


def test_legacy_inverse_export_has_deterministic_defaults(tmp_path):
    onnx = pytest.importorskip("onnx")
    import zipfile
    _, model = make_model()
    package = model.export(tmp_path / "posterior.zip")
    native = ibl._native.native
    metadata = native.model.ModelPackage.open(package).metadata
    with zipfile.ZipFile(package) as archive:
        graph = onnx.load_from_string(archive.read("model.onnx"))
    del graph.graph.output[1:]
    graph_path = tmp_path / "legacy.onnx"
    onnx.save(graph, graph_path)
    inverse = metadata.inverse
    inverse.presence_probability_output_name = ""
    inverse.posterior_std_output_name = ""
    inverse.uncertainty_predicted = False
    metadata.inverse = inverse
    legacy = native.model.ModelPackage.from_onnx(graph_path, metadata)
    assert legacy.metadata.format_version == 3
    legacy_path = tmp_path / "legacy.zip"
    legacy.write(legacy_path)
    result = ibl.InverseModel(legacy_path).predict_prepared(
        {"RBS": ibl.Spectrum("RBS", np.ones(17)), "PIXE": ibl.Spectrum("PIXE", np.ones(13))})
    np.testing.assert_array_equal(result.presence_probability, (result.values > 0).astype(np.float32))
    np.testing.assert_array_equal(result.posterior_std, np.zeros_like(result.values))
    assert not result.uncertainty_predicted


def test_ordered_density_prefix_gradients_and_roundtrip():
    _, source = make_model(layers=4)
    model = source.as_bidirectional().as_ordered_density().eval()
    counts = torch.rand(3, 30) * 10
    # Continue into layer 2, stop before layer 3; later decisions cannot reopen.
    hook = model.layer_activity_head.register_forward_hook(
        lambda module, inputs, output: output * 0 + output.new_tensor([2., -2., 2.]).reshape(1, 3, 1))
    prediction = model(counts)
    torch.testing.assert_close(prediction.layer_gates[0, :, 0], torch.tensor([1., 1., 0., 0.]))
    assert torch.count_nonzero(prediction.mean[:, 2:]) == 0
    assert (prediction.mean[:, :2] > 0).all()
    assert torch.count_nonzero(prediction.variance) == 0
    assert prediction.sample_logits.shape == (3, 2)
    hook.remove()
    prediction = model(counts)
    (prediction.mean.sum() + prediction.soft_layer_gates.sum() + prediction.sample_logits.sum()).backward()
    for head in (model.layer_activity_head, model.mean_head, model.sample_presence_head):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert model.std_head.weight.grad is None
    assert not source.ordered_density
    restored = iBeamNet(model.study, model.input_spectra_lengths, version=1, elements=model.elements, model_options=dict(**model.architecture)).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(counts).mean, model(counts).mean)
    with pytest.raises(ValueError, match="Deterministic"):
        prediction.sample()



def test_discrete_presence_uses_conditional_mean_and_retains_uncertainty():
    _, source = make_model()
    model = source.as_discrete_presence().eval()
    with torch.no_grad():
        model.presence_head.weight.zero_()
        model.presence_head.bias.copy_(torch.tensor([-2., 2.]))
    prediction = model(torch.ones(3, 30))
    assert torch.count_nonzero(prediction.mean[:, :, 0]) == 0
    torch.testing.assert_close(prediction.mean[:, :, 1], prediction.Y[:, :, 1])
    assert torch.all(prediction.variance > 0)
    assert not source.discrete_presence
    restored = iBeamNet(model.study, model.input_spectra_lengths, version=1, elements=model.elements, model_options=dict(**model.architecture)).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(torch.ones(3, 30)).mean, prediction.mean)


def test_reference_pileup_preprocessing_matches_prepared(tmp_path):
    """Measured pileup -> bare reference -> reference pileup, for two detectors."""
    from ibeamlab.spectrum import pileup, rebin
    _, original = make_model()
    detectors = [replace(d, real_time=20., live_time=16.)
                 for d in original.study.experiment.detectors]
    study = ibl.GenerationStudy(
        ibl.Experiment(original.study.experiment.sample, detectors), [])
    model = iBeamNet(study, {"RBS": 17, "PIXE": 13}, version=4,
        model_options=dict(cnn_channels=(4, 8), kernel_size=3, attention_size=8,
            attention_heads=2, attention_blocks=1, dropout=0.))
    tau = .003
    path = model.export(tmp_path / "reference_pileup.zip",
                        apply_reference_pileup=True, pileup_fudge_factor_seconds=tau)
    runtime = ibl.InverseModel(path)
    assert runtime.metadata.inverse.apply_reference_pileup
    prepared, measured, bare_measured, measured_detectors = {}, {}, {}, []
    for detector, width in zip(detectors, (9, 7)):
        bare = np.arange(1., width+1)
        reference = pileup(bare.tolist(), 20., 16., tau)
        prepared[detector.label] = ibl.Spectrum(detector.label, reference)
        # Different calibration, exposure and timing; fine grid preserves coverage.
        source = rebin(np.arange(width+1).tolist(),
                       (np.arange(2*width+1)/2).tolist(), bare.tolist())
        source = (np.asarray(source)*3).tolist()
        settings = replace(detector, particles_sr=detector.particles_sr*3,
            calibration=ibl.LinearCalibration(linear=.5), real_time=30., live_time=27.)
        measured_detectors.append(settings)
        measured[detector.label] = ibl.Spectrum(detector.label, pileup(source, 30., 27., tau))
        bare_measured[detector.label] = ibl.Spectrum(detector.label, source)
    expected = runtime.predict_prepared(prepared)
    for item in (ibl.InverseInput(measured, measured_detectors),
                 ibl.InverseInput(bare_measured, measured_detectors, pileup_already_removed=True)):
        actual = runtime.predict(item)
        np.testing.assert_allclose(actual.values, expected.values, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(actual.presence_probability, expected.presence_probability,
                                   rtol=2e-5, atol=1e-6)
    with pytest.raises(ValueError, match="requires measured"):
        model.export(tmp_path / "invalid.zip", apply_reference_pileup=True,
                     need_pileup_subtraction=False)
    _, no_timing = make_model()
    with pytest.raises(Exception, match="positive reference times"):
        no_timing.export(tmp_path / "no_timing.zip", apply_reference_pileup=True)
