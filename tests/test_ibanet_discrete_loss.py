import itertools

import pytest

torch = pytest.importorskip("torch")
from ibeamlab.ibeamnet import iBeamNetPrediction
from ibeamlab.ibanet_surrogate import sampled_reconstruction_loss


def test_score_gradient_matches_exact_bernoulli_expectation(monkeypatch):
    # Exhaustively average all two-draw outcomes with their actual probabilities.
    # The surrogate has a nondifferentiable zero-layer skip, as LRN_AD does.
    p = 0.3
    expected_gradient = p * (1 - p) * (2.0 - 0.5)
    gradient = 0.0
    density_gradient = 0.0
    expected_value = 0.0
    for outcomes in itertools.product((0, 1), repeat=2):
        logits = torch.tensor([[[p / (1-p)]]]).log().requires_grad_()
        mean = torch.tensor([[[3.0]]], requires_grad=True)
        prediction = iBeamNetPrediction(mean.expand(2, -1, -1), torch.ones(2, 1, 1), logits.expand(2, -1, -1))
        monkeypatch.setattr(torch, "rand_like", lambda tensor: tensor.new_tensor([0.0 if value else 0.99 for value in outcomes]).reshape(2, 1, 1))
        monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.zeros_like(tensor))
        loss = sampled_reconstruction_loss(
            prediction, torch.ones(2, 1),
            lambda density: torch.where(density != 0, density, 0).flatten(1),
            spectrum_scale=1.0, tail_weight=1.0)
        loss.backward()
        probability = 1.0
        for outcome in outcomes:
            probability *= p if outcome else 1 - p
        gradient += probability * logits.grad.item()
        density_gradient += probability * mean.grad.item()
        expected_value += probability * loss.item()
    assert gradient == pytest.approx(expected_gradient, abs=1e-6)
    assert density_gradient == pytest.approx(2*p, abs=1e-6)
    assert expected_value == pytest.approx(p*2 + (1-p)*0.5, abs=1e-6)


def test_sampled_reconstruction_has_std_gradient_and_nonnegative_inputs():
    torch.manual_seed(12)
    logits = torch.full((4, 2, 2), 1.0, requires_grad=True)
    mean = torch.full_like(logits, 0.5, requires_grad=True)
    std = torch.full_like(logits, 0.2, requires_grad=True)
    prediction = iBeamNetPrediction(mean, std, logits)
    def forward(density):
        assert (density >= 0).all()
        return density.flatten(1)
    loss = sampled_reconstruction_loss(prediction, torch.ones(4, 4), forward)
    loss.backward()
    for value in (logits, mean, std):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0



def test_one_surrogate_call_per_batch_and_single_row_gradient():
    logits = torch.zeros(1, 1, 1, requires_grad=True)
    mean = torch.ones_like(logits, requires_grad=True)
    prediction = iBeamNetPrediction(mean, torch.ones_like(mean)*0.1, logits)
    calls = []
    def forward(density):
        calls.append(density.shape[0])
        return density.flatten(1)
    loss = sampled_reconstruction_loss(prediction, torch.ones(1, 1), forward)
    loss.backward()
    assert calls == [1]
    assert torch.isfinite(logits.grad).all()


def test_reconstruction_matches_notebook_lrn_ad_loss_and_gradients():
    import json
    from pathlib import Path
    from ibeamlab.ibanet_surrogate import reconstruction_per_sample
    notebook = json.loads((Path(__file__).parents[1] / 'examples/train_lrn_ad_multilayer.ipynb').read_text(encoding='utf8'))
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'class Chi2Loss' in ''.join(cell['source']))
    start = source.index('TAIL_CHANNEL_START =')
    end = source.index('loss_function = Chi2Loss()') + len('loss_function = Chi2Loss()')
    target = torch.rand(3, 2997) * 1000
    target[:, :5] = 0
    prediction = (torch.rand(3, 2997) * 1000).requires_grad_()
    env = dict(torch=torch, nn=torch.nn, device=torch.device('cpu'), y_train=target)
    exec(source[start:end], env)
    expected = env['loss_function'](target*0.01, prediction*0.01)
    actual = reconstruction_per_sample(prediction, target).mean()
    torch.testing.assert_close(actual, expected)
    expected_gradient = torch.autograd.grad(expected, prediction, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_dense_gaussian_matches_original_loss_and_all_head_gradients():
    from ibeamlab.ibeamnet import iBeamNetLoss
    from ibeamlab.ibanet_surrogate import conditional_gaussian_loss
    torch.manual_seed(18)
    mean = torch.rand(3, 10, 10).requires_grad_()
    std = (torch.rand(3, 10, 10)+0.01).requires_grad_()
    logits = torch.randn(3, 10, 10, requires_grad=True)
    target = torch.rand(3, 10, 10)
    target[target < 0.5] = 0
    prediction = iBeamNetPrediction(mean, std, logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, (target > 0).float(), reduction='none').flatten(1).sum(1).mean()
    actual = bce + conditional_gaussian_loss(prediction, target)
    expected = iBeamNetLoss()(prediction, target)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, (mean, std, logits), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (mean, std, logits))
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e)
