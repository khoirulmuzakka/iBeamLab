"""Attention surrogate invariants and physical-input export parity."""
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import ibeamlab as ibl


def make_model(layer_count=1):
    study = ibl.GenerationStudy(
        ibl.Experiment(ibl.Sample([ibl.Layer(50_000, {"Li": .25, "C": .75}) for _ in range(layer_count)]),
                       [ibl.Detector("RBS", ibl.Beam("H", 2974), resolution=20)]),
        [p for layer in range(layer_count) for p in (
            ibl.vary.layer_thickness(layer=layer, bounds=(0, 100_000)),
            ibl.vary.concentration("C", layer=layer, bounds=(0, 1)),
            ibl.vary.concentration("Li", layer=layer, bounds=(0, 1)))])
    model = ibl.LRN_AD_ATT(study, {"RBS": 16}, elements=("Li", "C"),
        hidden_size=8, contribution_size=8, setup_embedding_dim=4,
        attention_heads=2, attention_layers=2, attention_feedforward=16,
        decoder_hidden_sizes=(8,), refiner_hidden_channels=2)
    return study, model


def test_gradients_empty_samples_and_padding():
    _, model = make_model()
    inputs = torch.tensor([[.125, .375], [0., 0.]], requires_grad=True)
    outputs = model(inputs)
    assert torch.isfinite(outputs).all() and (outputs > 0).all()
    outputs.sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad[0].abs().sum() > 0
    assert all(p.grad is not None for p in model.parameters())
    model.eval()
    torch.testing.assert_close(model(inputs), model(torch.cat((inputs, torch.zeros_like(inputs)), 1)))
    # Empty slots must not change depth/order cues or pooling.
    torch.testing.assert_close(model(inputs), model(torch.cat((torch.zeros_like(inputs), inputs), 1)))
    extended = torch.tensor([[.125, .375, .4, .1]])
    assert torch.isfinite(model(extended)).all()
    assert not torch.allclose(model(extended), model(extended.reshape(1, 2, 2).flip(1).flatten(1)))
    rebuilt = ibl.LRN_AD_ATT(model.study, {"RBS": 16}, elements=model.elements,
                            ad_input_factor=float(model.ad_input_factor), **model.architecture)
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(inputs), rebuilt.eval()(inputs))


@pytest.mark.parametrize("layer_count", [1, 2])
def test_export_native_parity(tmp_path, layer_count):
    pytest.importorskip("onnx")
    study, model = make_model(layer_count)
    package = model.export(tmp_path / "lrn_ad_att.zip", output_inverse_factor=.01)
    expected = model.predict(torch.tensor([[.125, .375] * layer_count])).numpy()[0] / .01
    actual = ibl.ForwardModel(package).predict(study.experiment)["RBS"].counts
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-4)
    batch = ibl.ForwardModel(package).predict([study.experiment, study.experiment])
    for result in batch:
        np.testing.assert_allclose(result["RBS"].counts, expected, rtol=2e-4, atol=1e-4)


def test_notebook_model_configuration():
    notebook = json.loads((Path(__file__).parents[1] / "examples/train_lrn_ad_att.ipynb").read_text(encoding="utf8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in cells:
        compile(source, "attention_notebook", "exec")
    _, model = make_model()
    source = next(s for s in cells if "model = LRN_AD_ATT(" in s)
    # Execute the actual constructor settings against a tiny physical study.
    env = dict(torch=torch, np=np, SEED=7, LRN_AD_ATT=ibl.LRN_AD_ATT,
        study=model.study, METHOD="RBS", ELEMENTS=model.elements, AD_INPUT_FACTOR=1e-5,
        y_train=np.zeros((2, 16), np.float32))
    exec(source, env)
    assert env["model"](torch.ones(2, 2, device=env["device"])).shape == (2, 16)


def test_notebook_training_and_resume(tmp_path):
    from test_lrn_ad_notebook_training import run_training
    _, model = make_model()
    _, env = run_training(tmp_path)
    env.update(model=model, ELEMENTS=model.elements, study=model.study,
               y_train=np.ones((4, 16), np.float32), y_val=np.ones((2, 16), np.float32))
    notebook = json.loads((Path(__file__).parents[1] / "examples/train_lrn_ad_att.ipynb").read_text(encoding="utf8"))
    source = "".join(notebook["cells"][13]["source"])
    source = source.replace("CPU_THREADS = 8", "CPU_THREADS = 1").replace("BATCH_SIZE = 512", "BATCH_SIZE = 2")
    start = source.index("TRAINING_SCHEDULE = [")
    stop = source.index("TOTAL_EPOCHS =", start)
    source = source[:start] + 'TRAINING_SCHEDULE = [{"learning_rate": 1e-4, "epochs": 2}]\n' + source[stop:]
    exec(source, env)
    checkpoint = torch.load(env["resume_path"], weights_only=False)
    assert checkpoint["completed_epochs"] == 2
    assert checkpoint["model_type"] == "LRN_AD_ATT"
    assert checkpoint["architecture"] == model.architecture
    exec(source, env)
    assert env["completed_epochs"] == 2
    assert len(env["history"]) == 2
