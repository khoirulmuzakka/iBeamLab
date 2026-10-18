"""One-pass discrete spike-and-slab reconstruction through a frozen model."""

import torch
from torch.nn import functional as F


def reconstruction_per_sample(reconstructed, target, *, spectrum_scale=0.01,
                              tail_channel_start=1100, tail_weight=5.0):
    """Exact LRN_AD scaled-target, mean-normalized tail-weighted Chi2Loss."""
    scaled_target = target * spectrum_scale
    scaled_prediction = reconstructed * spectrum_scale
    weights = torch.ones(target.shape[1], device=target.device, dtype=target.dtype)
    weights[tail_channel_start:] = tail_weight
    weights = weights / weights.mean()
    error = (scaled_prediction - scaled_target).square() / (scaled_target + 1.0)
    return (error * weights).mean(dim=1)


def conditional_gaussian_loss(prediction, target):
    """Present-cell Gaussian anchor without CUDA boolean-index synchronizations.

    Dataset validation already checks nonnegative, finite targets. Select safe
    inputs before arithmetic so absent cells cannot create NaN gradients.
    """
    present = target > 0
    mean = torch.where(present, prediction.Y, torch.zeros_like(prediction.Y))
    std = torch.where(present, prediction.R, torch.ones_like(prediction.R))
    gaussian = std.log() + 0.5 * ((target - mean) / std).square()
    return gaussian.flatten(1).sum(1).mean()


def sampled_reconstruction_loss(prediction, target, forward_density, *,
                                spectrum_scale=0.01, tail_channel_start=1100, tail_weight=5.0,
                                reconstruction_weight=1.0):
    """One hypothesis per sample; other batch rows provide the policy baseline.

    The baseline excludes the current row and is detached, so it does not depend
    on that row's Bernoulli draw. A one-row batch uses a zero baseline. Exactly
    one forward call supplies density/std gradients and score-function presence
    gradients, including for layers skipped by the surrogate. Gaussian forward
    draws are clipped to nonnegative AD; NLL remains a weak Gaussian anchor.
    """
    mask = (torch.rand_like(prediction.P) < prediction.P).to(prediction.Y.dtype)
    slab = (prediction.Y + prediction.R * torch.randn_like(prediction.Y)).clamp_min(0)
    output = forward_density(mask * slab)
    reconstruction = reconstruction_per_sample(output, target,
        spectrum_scale=spectrum_scale, tail_channel_start=tail_channel_start, tail_weight=tail_weight)
    costs = reconstruction_weight * reconstruction
    log_probability = -F.binary_cross_entropy_with_logits(
        prediction.logits, mask, reduction="none").flatten(1).sum(dim=1)
    detached = costs.detach()
    if costs.shape[0] > 1:
        baseline = (detached.sum() - detached) / (costs.shape[0] - 1)
    else:
        baseline = torch.zeros_like(detached)
    policy = ((detached - baseline) * log_probability).mean()
    return costs.mean() + (policy - policy.detach())
