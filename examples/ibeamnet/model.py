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
    """Reference residual spectral refiner with fixed-setup FiLM conditioning."""
    def __init__(self, channels, kernel):
        super().__init__()
        self.in_conv = nn.Conv1d(1, channels, kernel, padding=kernel // 2)
        self.residual_conv = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.out_conv = nn.Conv1d(channels, 1, 1)
        self.film = nn.Sequential(nn.Linear(1, 2 * channels), nn.LeakyReLU(),
                                  nn.Linear(2 * channels, 2 * channels))

    def forward(self, spectra):
        features = self.in_conv(spectra.unsqueeze(1))
        gamma, beta = self.film(spectra.new_ones((spectra.shape[0], 1))).chunk(2, 1)
        features = F.leaky_relu(features * (torch.tanh(gamma).unsqueeze(-1) + 1) + beta.unsqueeze(-1))
        residual = F.leaky_relu(self.residual_conv(features) + features)
        return spectra + self.out_conv(residual).squeeze(1)


class RecurrentSurrogate(nn.Module):
    """Shared layer recurrence with independent detector heads for this example.

    Every element is represented by its scaled areal density, not a separate
    thickness/fraction pair. All-zero AD layers preserve recurrent state and
    contribute nothing. Contributions are summed without extra density weights.
    Fixed setups use the reference's learned constant setup conditioning.
    """
    def __init__(self, elements, spectra_channels, *, hidden_size=256, embedding_size=128,
                 contribution_size=512, decoder_hidden_sizes=(512, 512), refiner_channels=32,
                 refiner_kernel=8, setup_embedding_size=32, block_hidden_sizes=(512, 512)):
        super().__init__()
        if not elements or len(set(elements)) != len(elements):
            raise ValueError("elements must be nonempty and unique")
        if not spectra_channels or any(c < 1 for c in spectra_channels):
            raise ValueError("each detector must have a positive channel count")
        validate_model_options(dict(hidden_size=hidden_size, embedding_size=embedding_size,
            contribution_size=contribution_size, decoder_hidden_sizes=decoder_hidden_sizes,
            refiner_channels=refiner_channels, refiner_kernel=refiner_kernel,
            setup_embedding_size=setup_embedding_size, block_hidden_sizes=block_hidden_sizes))
        # Preserve spectrum length, matching the reference's even-to-odd rounding.
        refiner_kernel = refiner_kernel if refiner_kernel % 2 else refiner_kernel + 1
        self.elements = tuple(elements)
        self.spectra_channels = tuple(spectra_channels)
        self.hidden_size, self.contribution_size = hidden_size, contribution_size
        self.layer_encoder = nn.Sequential(nn.Linear(len(elements), embedding_size), nn.LeakyReLU())
        self.setup_encoder = nn.Sequential(nn.Linear(1, embedding_size), nn.LeakyReLU())
        self.setup_to_film = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.LeakyReLU(),
                                         nn.Linear(embedding_size, 2 * embedding_size))
        self.state_projection = nn.Sequential(nn.Linear(hidden_size, embedding_size), nn.LeakyReLU())
        tower, previous = [], 3 * embedding_size
        for width in block_hidden_sizes:
            tower.extend((nn.Linear(previous, width), nn.LeakyReLU()))
            previous = width
        self.fusion = nn.Sequential(*tower)
        self.gru = nn.GRUCell(previous, hidden_size)
        self.contribution = nn.Sequential(nn.Linear(previous + hidden_size, max(contribution_size, embedding_size)),
            nn.LeakyReLU(), nn.Linear(max(contribution_size, embedding_size), contribution_size))
        self.setup_context = nn.Sequential(nn.Linear(1, setup_embedding_size), nn.LeakyReLU(),
            nn.Linear(setup_embedding_size, setup_embedding_size), nn.LeakyReLU())
        self.heads = nn.ModuleList(SpectrumHead(hidden_size + contribution_size + setup_embedding_size, channels,
            decoder_hidden_sizes, refiner_channels, refiner_kernel) for channels in spectra_channels)

    def forward(self, inputs):
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if inputs.ndim != 3 or inputs.shape[1] < 1 or inputs.shape[2] != len(self.elements):
                raise ValueError("inputs must be [batch,layers,elements]")
        hidden = inputs.new_zeros((inputs.shape[0], self.hidden_size))
        total = inputs.new_zeros((inputs.shape[0], self.contribution_size))
        setup = inputs.new_ones((inputs.shape[0], 1))
        setup_features = self.setup_encoder(setup)
        gamma, beta = self.setup_to_film(setup_features).chunk(2, 1)
        for layer in inputs.unbind(1):
            active = layer.ne(0).any(dim=1, keepdim=True).to(inputs.dtype)
            encoded = self.layer_encoder(layer) * (torch.tanh(gamma) + 1) + beta
            fused = self.fusion(torch.cat((encoded, setup_features, self.state_projection(hidden)), dim=1))
            next_hidden = self.gru(fused, hidden)
            total = total + self.contribution(torch.cat((fused, next_hidden), dim=1)) * active
            hidden = hidden * (1 - active) + next_hidden * active
        latent = torch.cat((total, hidden, self.setup_context(setup)), dim=1)
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
