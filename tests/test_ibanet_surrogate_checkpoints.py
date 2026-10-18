import re
"""Checkpoint failures and stochastic resume on small notebook training inputs."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def training_context(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    tmp_path.mkdir(parents=True, exist_ok=True)
    inverse_path, surrogate_path = tmp_path / "inverse.pt", tmp_path / "surrogate.pt"
    inverse_path.write_bytes(b"read-only inverse")
    surrogate_path.write_bytes(b"read-only surrogate")
    notebook = json.loads((Path(__file__).parents[1] / "examples" /
                           "finetune_ibanet_surrogate.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][16]["source"])
    # Keep lightweight fake-model loss/evaluation; the real end-to-end test
    # exercises section 5's self-contained definitions on iBeamNet outputs.
    source = source[source.index("OUTPUT_DIR ="):]
    start = source.index("# Refresh metric definitions")
    end = source.index("# Fixed subsets", start)
    source = source[:start] + source[end:]
    source = re.sub(r"(?m)^BATCH_SIZE = \d+", "BATCH_SIZE = 2", source)
    source = source.replace("RAMP_EPOCHS = 5", "RAMP_EPOCHS = 1")
    start = source.index("TRAINING_SCHEDULE = [")
    end = source.index("TOTAL_EPOCHS =", start)
    source = source[:start] + ('TRAINING_SCHEDULE = [{"learning_rate": .05, "epochs": 1}, '
                              '{"learning_rate": .01, "epochs": 1}]\n') + source[end:]
    model = nn.Sequential(nn.Linear(1, 1), nn.Dropout(.1))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(1)
    env = dict(EXAMPLES=tmp_path, torch=torch, np=np, copy=copy, math=math,
               hashlib=hashlib, DataLoader=DataLoader, TensorDataset=TensorDataset,
               tqdm=tqdm, device=torch.device("cpu"), model=model, lrn=nn.Identity(),
               IBANET_CHECKPOINT=inverse_path, LRN_AD_CHECKPOINT=surrogate_path,
               LRN_AD_ARCHITECTURE={}, LRN_AD_INPUT_FACTOR=1e-5, LRN_AD_OUTPUT_SCALE=.01,
               SEED=7, POISSON_NOISE=True, base_signature={},
               base_checkpoint={key: torch.tensor([0, 1]) for key in
                                ("train_indices", "val_indices", "test_indices")},
               x_train=np.ones((4, 1), dtype=np.float32),
               a_train=np.full((4, 1), 10, dtype=np.float32),
               x_val=np.ones((2, 1), dtype=np.float32),
               a_val=np.ones((2, 1), dtype=np.float32),
               x_val_clean=np.ones((2, 1), dtype=np.float32),
               baseline_validation=dict(presence_bce=0, spectrum=0), baseline_test={},
               surrogate_floor=0, SPECTRUM_NORMALIZER=1,
               presence_loss=lambda prediction, target: (prediction-target).square().mean(),
               reconstruct=lambda prediction: prediction,
               spectrum_loss=lambda prediction, target: (prediction-target).square().mean())

    @torch.no_grad()
    def evaluate(inputs, targets, clean, batch_size=2048):
        model.eval()
        result = model(torch.from_numpy(inputs))
        return dict(presence_bce=float((result-torch.from_numpy(targets)).square().mean()),
                    spectrum=float((result-torch.from_numpy(clean)).square().mean()),
                    edp_mae=0, presence_brier=0)

    env['conditional_gaussian_loss'] = lambda prediction, target: -env['presence_loss'](prediction, target)
    env['gaussian_nll'] = lambda prediction, target: torch.zeros((), device=prediction.device)
    env['sampled_reconstruction_loss'] = lambda prediction, target, forward, **kwargs: kwargs['reconstruction_weight'] * (prediction-target).square().mean()/kwargs.get('normalizer', 1.0)
    env["evaluate"] = evaluate
    env["selection_score"] = lambda value: value["presence_bce"] + env["SPECTRUM_WEIGHT"] * value["spectrum"]
    return source, env


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_failure_after_update_protects_baseline_and_sources(tmp_path, monkeypatch, failure):
    source, env = training_context(tmp_path)
    original_step = torch.optim.AdamW.step

    def failing_step(optimizer, *args, **kwargs):
        original_step(optimizer, *args, **kwargs)
        raise failure("simulated optimizer failure")

    monkeypatch.setattr(torch.optim.AdamW, "step", failing_step)
    if failure is RuntimeError:
        with pytest.raises(RuntimeError, match="simulated optimizer failure"):
            exec(source, env)
    else:
        exec(source, env)
    best = torch.load(env["weights_path"], weights_only=True)
    latest = torch.load(env["resume_path"], weights_only=True)
    assert best["best_validation_score"] == 0
    assert latest["completed_epochs"] == 0
    for name, tensor in best["model_state_dict"].items():
        torch.testing.assert_close(env["model"].state_dict()[name], tensor)
    assert env["IBANET_CHECKPOINT"].read_bytes() == b"read-only inverse"
    assert env["LRN_AD_CHECKPOINT"].read_bytes() == b"read-only surrogate"


def test_stochastic_resume_replays_partial_epoch_and_keeps_best(tmp_path, monkeypatch):
    source, reference = training_context(tmp_path / "reference")
    exec(source, reference)
    expected = torch.load(reference["resume_path"], weights_only=True)
    source, interrupted = training_context(tmp_path / "interrupted")
    original_step = torch.optim.AdamW.step
    calls = 0

    def interrupt_second_epoch(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        original_step(optimizer, *args, **kwargs)
        if calls == 3:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt_second_epoch)
        exec(source, interrupted)
    assert torch.load(interrupted["resume_path"], weights_only=True)["completed_epochs"] == 1
    source, resumed = training_context(tmp_path / "interrupted")
    exec(source, resumed)
    actual = torch.load(resumed["resume_path"], weights_only=True)
    assert actual["completed_epochs"] == 2
    assert len(actual["history"]) == 2
    assert actual["history"][1]["weight"] == 1.0
    assert [row["stage"] for row in actual["history"]] == [1, 2]
    assert [row["learning_rate"] for row in actual["history"]] == [.05, .01]
    assert actual["optimizer_state_dict"]["param_groups"][0]["lr"] == .01
    def without_timings(history):
        return [{key: value for key, value in row.items() if not key.endswith('_seconds')} for row in history]
    assert without_timings(actual['history']) == without_timings(expected['history'])
    assert all(row[key] >= 0 for row in actual['history']
               for key in ('training_seconds', 'evaluation_seconds') if key in row)
    for name, tensor in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][name], tensor, rtol=0, atol=0)
    for name in ("noise_rng_state", "loader_rng_state", "torch_rng_state"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert resumed["protected"]["best_validation_score"] == 0


def test_changed_data_settings_cannot_replace_best(tmp_path):
    source, env = training_context(tmp_path)
    exec(source, env)
    before = env["weights_path"].read_bytes()
    source, resumed = training_context(tmp_path)
    resumed["x_train"] = np.ones((6, 1), dtype=np.float32)
    resumed["a_train"] = np.ones((6, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="Resume data/noise settings differ"):
        exec(source, resumed)
    assert resumed["weights_path"].read_bytes() == before


def test_batch_size_change_resumes_with_optimizer_state(tmp_path, monkeypatch):
    source, env = training_context(tmp_path)
    original_step = torch.optim.AdamW.step
    calls = 0

    def interrupt_second_epoch(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        original_step(optimizer, *args, **kwargs)
        if calls == 3:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt_second_epoch)
        exec(source, env)
    previous = torch.load(env["resume_path"], weights_only=True)
    assert previous["completed_epochs"] == 1
    source, resumed = training_context(tmp_path)
    source = source.replace("BATCH_SIZE = 2", "BATCH_SIZE = 4")
    exec(source, resumed)
    latest = torch.load(resumed["resume_path"], weights_only=True)
    assert latest["completed_epochs"] == 2
    assert len(latest["history"]) == 2
    assert latest["resume_settings"]["batch_size"] == 4
    for key, state in previous["optimizer_state_dict"]["state"].items():
        assert latest["optimizer_state_dict"]["state"][key]["step"] == state["step"] + 1
    assert resumed["protected"]["best_validation_score"] == 0


def test_final_weight_training_metrics_and_reconstruction_selection(tmp_path):
    source, env = training_context(tmp_path)
    exec(source, env)
    latest = torch.load(env["resume_path"], weights_only=True)
    env["model"].load_state_dict(latest["model_state_dict"])
    expected = env["evaluate"](env["x_train_eval"], env["a_train_eval"], env["x_train_eval_clean"])
    final_row = latest["history"][-1]
    assert final_row["train_metrics"] == expected
    assert final_row["train_loss"] == expected["presence_bce"] + expected["spectrum"]
    assert final_row["optimization_loss"] != final_row["train_loss"]
    assert all(row["score"] == row["spectrum"] for row in latest["history"])
    assert env["selection_score"](dict(presence_bce=-1e9, spectrum=10)) == 10
    assert env["selection_score"](dict(presence_bce=1e9, spectrum=1)) == 1


def test_presence_loss_has_no_gaussian_gradient():
    from types import SimpleNamespace
    notebook = json.loads((Path(__file__).parents[1] / "examples" /
                           "finetune_ibanet_surrogate.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][13]["source"]).split("@torch.no_grad()", 1)[0]
    env = {"torch": torch}
    exec(source, env)
    logits = torch.randn(2, 2, 3, requires_grad=True)
    mean = torch.randn(2, 2, 3, requires_grad=True)
    sigma = torch.ones(2, 2, 3, requires_grad=True)
    targets = torch.tensor([[[1., 0., 2.], [0., 0., 0.]], [[0., 3., 0.], [1., 0., 0.]]])
    loss = env["presence_loss"](SimpleNamespace(logits=logits, Y=mean, R=sigma), targets)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, (targets > 0).float(), reduction="none").flatten(1).sum(1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert mean.grad is None and sigma.grad is None




def test_reconstruction_weight_change_preserves_resume_and_best(tmp_path):
    source, env = training_context(tmp_path)
    exec(source.replace('SPECTRUM_WEIGHT = 1.0', 'SPECTRUM_WEIGHT = 0.1'), env)
    previous = torch.load(env['resume_path'], weights_only=True)
    source, resumed = training_context(tmp_path)
    exec(source, resumed)
    assert resumed['completed_epochs'] == previous['completed_epochs']
    assert resumed['history'] == previous['history']
    assert resumed['signature']['spectrum_weight'] == 1.0
    assert resumed['protected']['best_validation_score'] == 0.0
    for name, value in previous['model_state_dict'].items():
        torch.testing.assert_close(torch.load(resumed['resume_path'], weights_only=True)['model_state_dict'][name], value)


def test_changed_reconstruction_loss_remeasures_and_preserves_original_checkpoint(tmp_path):
    source, env = training_context(tmp_path)
    exec(source, env)
    saved_bytes = {}
    for path in (env['weights_path'], env['resume_path']):
        checkpoint = torch.load(path, weights_only=True)
        signature = checkpoint['finetuning_signature']
        signature['reconstruction_loss'] = 'count_chi2_mean'
        signature['spectrum_normalizer'] = 5.277
        for key in ('reconstruction_scale', 'tail_channel_start', 'tail_weight'):
            signature.pop(key)
        checkpoint['history'] = [dict(row, reconstruction_loss='count_chi2_mean') for row in checkpoint['history']]
        torch.save(checkpoint, path)
        saved_bytes[path.name] = path.read_bytes()
    source, resumed = training_context(tmp_path)
    exec(source, resumed)
    assert resumed['completed_epochs'] == 2
    assert resumed['signature']['reconstruction_loss'] == 'lrn_ad_scaled_tail_chi2_mean'
    for path in (resumed['weights_path'], resumed['resume_path']):
        backup = path.with_name(path.stem + '_before_lrn_ad_loss.pt')
        assert backup.read_bytes() == saved_bytes[path.name]
    assert resumed['protected']['best_validation_score'] == 0.0


def test_removing_layer_penalty_resumes_existing_checkpoint(tmp_path):
    source, env = training_context(tmp_path)
    exec(source, env)
    for path in (env['weights_path'], env['resume_path']):
        value = torch.load(path, weights_only=True)
        value['finetuning_signature']['layer_count_weight'] = 0.01
        torch.save(value, path)
    source, resumed = training_context(tmp_path)
    exec(source, resumed)
    assert resumed['completed_epochs'] == 2
    assert 'layer_count_weight' not in resumed['signature']
    assert resumed['protected']['best_validation_score'] == 0.0
