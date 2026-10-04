import numpy as np
import pytest

torch = pytest.importorskip("torch")

import ibeamlab as ibl
from ibeamlab import transforms
from ibeamlab.lrn import LRNModel


def make_model():
    study = ibl.GenerationStudy(
        ibl.Experiment(
            ibl.Sample([ibl.Layer(50_000, {"Li": 0.25, "C": 0.75})]),
            [ibl.Detector("RBS", ibl.Beam("H", 2974), resolution=20)],
        ),
        [ibl.vary.layer_thickness(layer=0, bounds=(0, 100_000)),
         ibl.vary.concentration("Li", layer=0, bounds=(0, 1)),
         ibl.vary.concentration("C", layer=0, bounds=(0, 1))],
    )
    model = LRNModel(study, {"RBS": 16}, hidden_size=8, contribution_size=8,
                     setup_embedding_dim=4, layer_embedding_dim=8,
                     block_hidden_sizes=(8,), decoder_hidden_sizes=(8,),
                     refiner_hidden_channels=2, refiner_kernel_size=3)
    return study, model


def test_separate_layer_features_and_padding():
    _, model = make_model()
    inputs = torch.tensor([[0.5, 0.25, 0.75], [0, 0, 0]])
    captured = {}
    def capture_encoder(module, args):
        captured["inputs"] = args[0]
    hook = model.layer_block.layer_encoder.register_forward_pre_hook(capture_encoder)
    latent, _ = model.encode_latent(inputs)
    hook.remove()
    torch.testing.assert_close(captured["inputs"], inputs, rtol=0, atol=0)
    assert model.layer_block.layer_encoder[0].in_features == 3
    assert torch.count_nonzero(latent[1, :16]) == 0
    model.eval()
    single = inputs[:1]
    with_empty_tail = torch.cat((single, inputs[1:]), dim=1)
    torch.testing.assert_close(model(single), model(with_empty_tail))


def test_standard_gru_update_and_gradients():
    _, model = make_model()
    block = model.layer_block
    captured = {}
    def capture_gru(module, inputs, output):
        captured["candidate"] = output
    hook = block.gru.register_forward_hook(capture_gru)
    hidden = torch.randn(1, 8)
    _, updated = block(torch.tensor([[0.5, 0.25, 0.75]]), torch.empty(1, 0), hidden)
    hook.remove()
    torch.testing.assert_close(updated, captured["candidate"], rtol=0, atol=0)
    assert not hasattr(block, "thickness_gate")
    model(torch.tensor([[0.5, 0.25, 0.75]])).sum().backward()
    assert block.gru.weight_ih.grad is not None
    assert torch.isfinite(block.gru.weight_ih.grad).all()


def test_export_matches_native_prediction(tmp_path):
    pytest.importorskip("onnx")
    study, model = make_model()
    transform = transforms.build_lrn_input_transform(study)
    package = model.export(tmp_path / "lrn.zip", input_transform=transform)
    experiment = study.experiment
    expected = model.predict(torch.from_numpy(transform.apply(
        np.array([[50_000, 0.25, 0.75]], dtype=np.float32)))).numpy()[0]
    actual = ibl.ForwardModel(package).predict([experiment])[0]["RBS"].counts
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
