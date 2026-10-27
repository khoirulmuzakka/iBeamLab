import numpy as np
import pytest

torch = pytest.importorskip("torch")
import ibeamlab as ibl


def make_model():
    study = ibl.GenerationStudy(
        ibl.Experiment(ibl.Sample([ibl.Layer(50_000, {"Li": .25, "C": .75})]),
                       [ibl.Detector("RBS", ibl.Beam("H", 2974), resolution=20)]),
        [ibl.vary.layer_thickness(layer=0, bounds=(0, 100_000)),
         ibl.vary.concentration("C", layer=0, bounds=(0, 1)),
         ibl.vary.concentration("Li", layer=0, bounds=(0, 1))])
    model = ibl.LRN_AD(study, {"RBS": 16}, elements=("Li", "C"),
                       hidden_size=8, contribution_size=8, setup_embedding_dim=4,
                       layer_embedding_dim=8, block_hidden_sizes=(8,),
                       decoder_hidden_sizes=(8,), refiner_hidden_channels=2)
    return study, model


def test_conversion_gradient_and_padding():
    _, model = make_model()
    physical = torch.tensor([[50_000., .75, .25]])
    scaled = model.physical_to_scaled_ad(physical)
    torch.testing.assert_close(scaled, torch.tensor([[.125, .375]]))
    scaled.requires_grad_()
    model(scaled).sum().backward()
    assert torch.isfinite(scaled.grad).all() and scaled.grad.abs().sum() > 0
    model.eval()
    torch.testing.assert_close(model(scaled), model(torch.cat((scaled, torch.zeros_like(scaled)), 1)))


def test_export_native_parity(tmp_path):
    pytest.importorskip("onnx")
    study, model = make_model()
    package = model.export(tmp_path / "lrn_ad.zip", output_inverse_factor=.01)
    expected = model.predict(torch.tensor([[.125, .375]])).numpy()[0] / .01
    actual = ibl.ForwardModel(package).predict(study.experiment)["RBS"].counts
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-4)
    from ibeamlab._native import native
    metadata = native.model.ModelPackage.open(package).metadata
    assert metadata.class_name == "LRN_AD"
    assert metadata.input_dimension == 3
    assert metadata.input_transform.type == "identity"


def test_dynamic_export_variable_layers_and_padding(tmp_path):
    import zipfile
    onnx = pytest.importorskip("onnx")
    study, model = make_model()
    model.eval()
    package = model.export(tmp_path / "dynamic.zip", layer_layout="dynamic", output_inverse_factor=.01)
    assert not model.training
    with zipfile.ZipFile(package) as archive:
        graph = onnx.load_from_string(archive.read("model.onnx"))
    assert any(node.op_type == "Loop" for node in graph.graph.node)
    runtime = ibl.ForwardModel(package)
    assert runtime.metadata.format_version == 5
    assert runtime.metadata.forward.layer_input.elements == ["Li", "C"]
    experiments, expected = [], []
    for count in (1, 2, 8, 10, 16, 32):
        layers = [ibl.Layer(1000 + i * 100, {"Li": .25, "C": .75}) for i in range(count)]
        experiments.append(ibl.Experiment(ibl.Sample(layers), study.experiment.detectors))
        ad = torch.tensor([[[layer.thickness * .25, layer.thickness * .75] for layer in layers]])
        expected.append(model.predict_ad(ad).numpy()[0] / .01)
    for result, truth in zip(runtime.predict(experiments), expected):
        np.testing.assert_allclose(result["RBS"].counts, truth, rtol=3e-4, atol=1e-4)
    # Zero layers anywhere preserve state; an interior zero must not stop the loop.
    layers = list(experiments[1].sample.layers)
    padded = ibl.Experiment(ibl.Sample([ibl.Layer(0, {"C": 1}), layers[0],
        ibl.Layer(0, {"C": 1}), layers[1], ibl.Layer(0, {"C": 1})]), study.experiment.detectors)
    np.testing.assert_allclose(runtime.predict(padded)["RBS"].counts, expected[1], rtol=3e-4, atol=1e-4)
    assert runtime.predict([]) == []
    zero = ibl.Experiment(ibl.Sample([ibl.Layer(0, {"C": 1})]), study.experiment.detectors)
    np.testing.assert_allclose(runtime.predict(zero)["RBS"].counts,
                              model.predict_ad(torch.zeros(1, 1, 2)).numpy()[0] / .01,
                              rtol=3e-4, atol=1e-4)
    unknown = ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Fe": 1})]), study.experiment.detectors)
    with pytest.raises(ibl.InferenceError, match="unsupported element"):
        runtime.predict(unknown)
    for invalid in (torch.zeros(1, 0, 2), torch.full((1, 2, 2), -1.),
                    torch.full((1, 2, 2), float("nan"))):
        with pytest.raises(ValueError):
            model.predict_ad(invalid)

    from ibeamlab._native import native
    metadata = native.model.ModelPackage.open(package).metadata
    schema = metadata.forward.layer_input
    schema.elements = ["Li", "Li"]
    forward = metadata.forward
    forward.layer_input = schema
    metadata.forward = forward
    graph_file = tmp_path / "model.onnx"
    onnx.save(graph, graph_file)
    with pytest.raises(ValueError, match="unique"):
        native.model.ModelPackage.from_onnx(graph_file, metadata)
