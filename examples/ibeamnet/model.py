"""Example recurrent forward surrogate over scaled elemental areal densities.

Physical densities are in 1e15 atoms/cm2. The network accepts scaled densities
[batch, layers, elements]. The C++ input schema packs and scales physical AD.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .validation import validate_model_options, validate_loss, integer, number


class SpectrumHead(nn.Module):
    """Detector-specific decoder and local residual refinement; output is nonnegative."""
    def __init__(self, inputs, channels, decoder_hidden_sizes, refiner_channels, kernel_size):
        super().__init__()
        decoder, previous = [], inputs
        for width in decoder_hidden_sizes:
            decoder.extend((nn.Linear(previous, width), nn.LeakyReLU()))
            previous = width
        decoder.append(nn.Linear(previous, channels))
        self.decoder = nn.Sequential(*decoder)
        self.refiner = None
        if refiner_channels:
            self.refiner = LocalSpectrumRefiner(refiner_channels, kernel_size)
        self.output_scale = nn.Parameter(torch.ones(channels))

    def forward(self, latent):
        coarse = F.softplus(self.decoder(latent))
        if self.refiner is not None:
            coarse = F.softplus(self.refiner(coarse))
        return coarse * F.softplus(self.output_scale)


class LocalSpectrumRefiner(nn.Module):
    """Residual spectral refiner without setup conditioning."""
    def __init__(self, channels, kernel):
        super().__init__()
        self.in_conv = nn.Conv1d(1, channels, kernel, padding=kernel // 2)
        self.residual_conv = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.out_conv = nn.Conv1d(channels, 1, 1)

    def forward(self, spectra):
        features = F.leaky_relu(self.in_conv(spectra.unsqueeze(1)))
        residual = F.leaky_relu(self.residual_conv(features) + features)
        return spectra + self.out_conv(residual).squeeze(1)


class RecurrentSurrogate(nn.Module):
    """Shared layer recurrence with independent detector heads for this example.

    Every element is represented by its scaled areal density, not a separate
    thickness/fraction pair. All-zero AD layers preserve recurrent state and
    contribute nothing. Contributions are summed without extra density weights.
    A learned AD embedding feeds the GRU alongside the previous hidden state.
    Contributions use current AD and the updated hidden state. The fixed
    experiment is learned through ordinary network weights.
    """
    def __init__(self, elements, spectra_channels, *, hidden_size=256, ad_embedding_size=64,
                 contribution_size=512, decoder_hidden_sizes=(512, 512), refiner_channels=32,
                 refiner_kernel=8):
        super().__init__()
        if not elements or len(set(elements)) != len(elements):
            raise ValueError("elements must be nonempty and unique")
        if not spectra_channels or any(c < 1 for c in spectra_channels):
            raise ValueError("each detector must have a positive channel count")
        validate_model_options(dict(hidden_size=hidden_size, ad_embedding_size=ad_embedding_size,
            contribution_size=contribution_size, decoder_hidden_sizes=decoder_hidden_sizes,
            refiner_channels=refiner_channels, refiner_kernel=refiner_kernel))
        # Preserve spectrum length, matching the reference's even-to-odd rounding.
        refiner_kernel = refiner_kernel if refiner_kernel % 2 else refiner_kernel + 1
        self.elements = tuple(elements)
        self.spectra_channels = tuple(spectra_channels)
        self.hidden_size, self.contribution_size = hidden_size, contribution_size
        self.ad_embedding = nn.Sequential(nn.Linear(len(elements), ad_embedding_size), nn.LeakyReLU())
        self.gru = nn.GRUCell(ad_embedding_size, hidden_size)
        self.contribution = nn.Sequential(nn.Linear(len(elements) + hidden_size, contribution_size),
            nn.LeakyReLU(), nn.Linear(contribution_size, contribution_size))
        self.heads = nn.ModuleList(SpectrumHead(hidden_size + contribution_size, channels,
            decoder_hidden_sizes, refiner_channels, refiner_kernel) for channels in spectra_channels)

    def forward(self, inputs):
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if inputs.ndim != 3 or inputs.shape[1] < 1 or inputs.shape[2] != len(self.elements):
                raise ValueError("inputs must be [batch,layers,elements]")
        hidden = inputs.new_zeros((inputs.shape[0], self.hidden_size))
        total = inputs.new_zeros((inputs.shape[0], self.contribution_size))
        for layer in inputs.unbind(1):
            active = layer.ne(0).any(dim=1, keepdim=True).to(inputs.dtype)
            next_hidden = self.gru(self.ad_embedding(layer), hidden)
            total = total + self.contribution(torch.cat((layer, next_hidden), dim=1)) * active
            hidden = hidden * (1 - active) + next_hidden * active
        latent = torch.cat((total, hidden), dim=1)
        return torch.cat([head(latent) for head in self.heads], dim=1)


class SpectrumLoss(nn.Module):
    """Mean detector-balanced chi-square or MSE in the declared training count units."""
    def __init__(self, spectra, kind="chi_square", noise_floor=1.0):
        super().__init__()
        validate_loss(dict(kind=kind, noise_floor=noise_floor))
        for index, spec in enumerate(spectra):
            integer(spec["channels"], f"spectra[{index}].channels")
            number(spec["loss_weight"], f"spectra[{index}].loss_weight")
        self.spectra, self.kind, self.noise_floor = spectra, kind, noise_floor

    def forward(self, predicted, target):
        offset, loss = 0, predicted.new_zeros(())
        for spec in self.spectra:
            stop = offset + spec["channels"]
            error = (predicted[:, offset:stop] - target[:, offset:stop]).square()
            if self.kind == "chi_square":
                error = error / (target[:, offset:stop] + self.noise_floor)
            loss = loss + spec["loss_weight"] * error.mean()
            offset = stop
        return loss / sum(s["loss_weight"] for s in self.spectra)
