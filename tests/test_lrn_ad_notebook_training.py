"""Exercise the notebook on tiny data; never read or write real checkpoints."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def run_training(tmp_path, model=None):
    notebook = json.loads((Path(__file__).parents[1] / "examples" /
                           "train_lrn_ad_multilayer.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][13]["source"])
    source = source.replace("CPU_THREADS = os.cpu_count() or 1", "CPU_THREADS = 1")
    source = source.replace("BATCH_SIZE = 512", "BATCH_SIZE = 2")
    start = source.index("TRAINING_SCHEDULE = [")
    end = source.index("TOTAL_EPOCHS =", start)
    source = source[:start] + 'TRAINING_SCHEDULE = [{"learning_rate": .05, "epochs": 2}]\n' + source[end:]
    if model is None:
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(1)
    model.architecture = {}
    env = dict(EXAMPLES=tmp_path, torch=torch, np=np, nn=nn,
               DataLoader=DataLoader, TensorDataset=TensorDataset, tqdm=tqdm,
               device=torch.device("cpu"), model=model, MAX_LAYERS=1, METHOD="RBS",
               ELEMENTS=("C",), AD_INPUT_FACTOR=1e-5, OUTPUT_SCALE=.01,
               study=SimpleNamespace(parameter_names=["x", "y"]), SEED=7,
               VALIDATION_FRACTION=.1, TEST_FRACTION=.1, MAX_VALIDATION_SAMPLES=2,
               MAX_TEST_SAMPLES=2, LAYER_DATASETS={},
               x_train=np.ones((4, 2), dtype=np.float32),
               y_train=np.full((4, 1), 10, dtype=np.float32),
               x_val=np.ones((2, 2), dtype=np.float32),
               y_val=np.ones((2, 1), dtype=np.float32))
    import os
    env["os"] = os
    torch.manual_seed(7)
    return source, env


def test_worse_training_and_resume_preserve_best(tmp_path):
    source, env = run_training(tmp_path)
    exec(source, env)
    best = torch.load(env["best_path"], weights_only=False)
    latest = torch.load(env["resume_path"], weights_only=False)
    assert best["best_validation_loss"] == 0
    assert latest["completed_epochs"] == 2
    assert latest["optimizer_state_dict"]["state"]
    assert latest["history"][-1][1] > best["best_validation_loss"]
    assert not env["weights_path"].exists()
    for name, tensor in best["model_state_dict"].items():
        torch.testing.assert_close(env["model"].state_dict()[name], tensor)
    source, resumed = run_training(tmp_path)
    exec(source, resumed)
    assert resumed["completed_epochs"] == 2
    assert len(resumed["history"]) == 2
    assert resumed["restored_loss"] == 0


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_interruption_before_first_epoch_preserves_baseline(tmp_path, monkeypatch, error):
    source, env = run_training(tmp_path)
    step = torch.optim.AdamW.step

    def interrupted_step(optimizer, *args, **kwargs):
        step(optimizer, *args, **kwargs)
        raise error("simulated failure after changing live weights")

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupted_step)
    if error is RuntimeError:
        with pytest.raises(RuntimeError, match="simulated failure"):
            exec(source, env)
    else:
        exec(source, env)
    best = torch.load(env["best_path"], weights_only=False)
    latest = torch.load(env["resume_path"], weights_only=False)
    assert best["best_validation_loss"] == 0
    assert latest["completed_epochs"] == 0
    for name, tensor in best["model_state_dict"].items():
        torch.testing.assert_close(env["model"].state_dict()[name], tensor)


def test_partial_epoch_replay_restores_optimizer_and_schedule(tmp_path, monkeypatch):
    source, reference = run_training(tmp_path / "reference")
    exec(source, reference)
    expected = torch.load(reference["resume_path"], weights_only=False)
    source, interrupted = run_training(tmp_path / "interrupted")
    step = torch.optim.AdamW.step
    calls = 0

    def interrupt_second_epoch(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        step(optimizer, *args, **kwargs)
        if calls == 3:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt_second_epoch)
        exec(source, interrupted)
    partial = torch.load(interrupted["resume_path"], weights_only=False)
    assert partial["completed_epochs"] == 1
    source, resumed = run_training(tmp_path / "interrupted")
    exec(source, resumed)
    actual = torch.load(resumed["resume_path"], weights_only=False)
    assert actual["completed_epochs"] == 2
    assert len(actual["history"]) == 2
    for name, tensor in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][name], tensor)
    for key, state in expected["optimizer_state_dict"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(actual["optimizer_state_dict"]["state"][key][field], value)


def test_improvement_persists_paired_best_optimizer(tmp_path):
    source, env = run_training(tmp_path)
    env["y_val"].fill(10)
    exec(source, env)
    best = torch.load(env["best_path"], weights_only=False)
    latest = torch.load(env["resume_path"], weights_only=False)
    assert best["best_validation_loss"] < 81 / 11
    assert best["best_validation_loss"] == min(v for _, v in env["history"])
    assert best["history"][-1][1] == best["best_validation_loss"]
    assert best["best_epoch"] == len(best["history"])
    for key, state in best["optimizer_state_dict"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(latest["optimizer_state_dict"]["state"][key][field], value)


def test_original_checkpoint_is_read_only(tmp_path):
    source, env = run_training(tmp_path)
    checkpoint = dict(model_state_dict=env["model"].state_dict(), max_layers=1,
                      method="RBS", elements=["C"], ad_input_factor=1e-5,
                      output_scale=.01, layer_representation="areal_density",
                      history=[(.01, 0)], architecture={})
    original = tmp_path / "artifacts" / "lrn_ad_rbs_10layers" / "lrn_ad_state.pt"
    original.parent.mkdir(parents=True)
    torch.save(checkpoint, original)
    before = original.read_bytes()
    exec(source, env)
    assert original.read_bytes() == before
    assert env["restored_loss"] == 0
