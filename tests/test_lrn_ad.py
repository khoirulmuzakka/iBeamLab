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
