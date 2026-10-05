"""Optional PyTorch CNN/MLP spike-and-slab inverse model from docs/draft.tex."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .generation import GenerationStudy


@dataclass
class IBAnetPrediction:
    """Conditional Gaussian parameters in training target units.

    The Gaussian slab follows the draft and can produce negative draws even
    though its mean is constrained positive. Sampling does not clip those draws.
    Divide Y and R by the training target factor to recover physical units.
    """

    Y: torch.Tensor
    R: torch.Tensor
    logits: torch.Tensor

    @property
    def P(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    @property
    def mean(self) -> torch.Tensor:
        """Posterior mean E[A|S] = P * Y, including absence probability."""
        return self.P * self.Y

    @property
    def variance(self) -> torch.Tensor:
        p = self.P
        return p * self.R.square() + p * (1 - p) * self.Y.square()

    def sample(self, count: int = 1, *, generator=None) -> torch.Tensor:
        """Draw the draft's untruncated distribution: [draw, batch, layer, element]."""
        if count < 1:
            raise ValueError("count must be positive")
        shape = (count, *self.Y.shape)
        present = torch.rand(shape, device=self.Y.device, dtype=self.Y.dtype,
                             generator=generator) < self.P
        normal = torch.randn(shape, device=self.Y.device, dtype=self.Y.dtype,
                             generator=generator)
        return torch.where(present, self.Y + self.R * normal, 0.0)


class IBAnetLoss(nn.Module):
    """Spike-and-slab NLL up to a target-dependent constant, with M = (target > 0).

    Sum layer/element contributions per sample, then average the batch. Absent
    entries contribute only Bernoulli BCE; no Gaussian regression is applied.
    Negative NLL values are possible for continuous densities and are valid.
    Omits 0.5*log(2*pi) per present cell; this does not change gradients.
    """

    def forward(self, prediction: IBAnetPrediction, target: torch.Tensor) -> torch.Tensor:
        if target.shape != prediction.Y.shape or target.numel() == 0:
            raise ValueError("Targets must match nonempty [batch, layer, element] outputs")
        if not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError("EDP targets must be finite and nonnegative")
        mask = target > 0
        # P = sigmoid(q). This is exactly -M*log(P) - (1-M)*log(1-P),
        # evaluated from q without unstable logs of rounded probabilities.
        classification = F.binary_cross_entropy_with_logits(
            prediction.logits, mask.to(target.dtype), reduction="none")
        # Select present entries before division: absent means/sigmas never enter
        # the Gaussian term, even if their raw values would overflow.
        gaussian = torch.zeros_like(target)
        sigma = prediction.R[mask]
        residual = (target[mask] - prediction.Y[mask]) / sigma
        gaussian[mask] = sigma.log() + 0.5 * residual.square()
        return (classification + gaussian).flatten(1).sum(1).mean()


class IBAnet(nn.Module):
    """One 1D CNN per detector, fused MLP, and three layer/element heads.

    Inputs are raw nonnegative counts concatenated in ``spectrum_labels`` order.
    Log1p preprocessing and fitted per-channel standardization are embedded in
    the graph/state dict. Y uses softplus (a positive-mean restriction on the
    draft's Gaussian); R uses softplus + minimum_std; P uses sigmoid logits.
    """

    def __init__(self, study: GenerationStudy, input_spectra_lengths: Mapping[str, int], *,
                 elements: Sequence[str] | None = None,
                 cnn_channels: Sequence[int] = (16, 32, 64, 64), kernel_size: int = 7,
                 head_hidden_sizes: Sequence[int] = (512, 256), minimum_std: float = 1e-3,
                 input_mean=None, input_std=None) -> None:
        super().__init__()
        labels = tuple(d.label for d in study.experiment.detectors)
        if len(set(labels)) != len(labels) or set(input_spectra_lengths) != set(labels):
            raise ValueError("Spectrum lengths must match unique experiment detector labels")
        if not labels or any(not isinstance(n, int) or isinstance(n, bool) or n < 1
                             for n in input_spectra_lengths.values()):
            raise ValueError("Spectrum lengths must be positive integers")
        layers = study.experiment.sample.layers
        if not layers:
            raise ValueError("At least one layer is required")
        self.elements = tuple(elements if elements is not None else
                              (s.element for s in layers[0].species))
        if not self.elements or len(set(self.elements)) != len(self.elements):
            raise ValueError("Elements must be nonempty and unique")
        if any({s.element for s in layer.species} != set(self.elements) for layer in layers):
            raise ValueError("Every template layer must contain exactly the EDP elements")
        if any(p.layer is None for p in study.parameters):
            raise ValueError("Spectra-only IBAnet requires fixed experimental settings")
        if kernel_size < 1 or kernel_size % 2 != 1 or not cnn_channels:
            raise ValueError("Use a positive odd CNN kernel and nonempty channels")
        if any(n < 1 for n in (*cnn_channels, *head_hidden_sizes)):
            raise ValueError("Network widths must be positive")
        if not math.isfinite(minimum_std) or minimum_std <= 0:
            raise ValueError("minimum_std must be finite and positive")
        self.study = study
        self.spectrum_labels = labels
        self.input_spectra_lengths = {label: input_spectra_lengths[label] for label in labels}
        self.max_layers = len(layers)
        self.input_dimension = sum(self.input_spectra_lengths.values())
        self.output_size = self.max_layers * len(self.elements)
        self.minimum_std = minimum_std
        self.architecture = dict(cnn_channels=tuple(cnn_channels), kernel_size=kernel_size,
                                 head_hidden_sizes=tuple(head_hidden_sizes), minimum_std=minimum_std)
        mean = torch.zeros(self.input_dimension) if input_mean is None else torch.as_tensor(input_mean, dtype=torch.float32).clone()
        std = torch.ones(self.input_dimension) if input_std is None else torch.as_tensor(input_std, dtype=torch.float32).clone()
        if mean.shape != (self.input_dimension,) or std.shape != mean.shape:
            raise ValueError("Input normalization must have one value per channel")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Input normalization must be finite with positive std")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        self.encoders = nn.ModuleList()
        feature_count = 0
        for length in self.input_spectra_lengths.values():
            blocks = []
            previous = 1
            for channels in cnn_channels:
                blocks.extend([nn.Conv1d(previous, channels, kernel_size, stride=2,
                                         padding=kernel_size // 2), nn.SiLU()])
                previous = channels
                length = (length + 1) // 2
            blocks.append(nn.Flatten())
            self.encoders.append(nn.Sequential(*blocks))
            feature_count += previous * length
        head = []
        for width in head_hidden_sizes:
            head.extend([nn.Linear(feature_count, width), nn.SiLU()])
            feature_count = width
        self.head = nn.Sequential(*head)
        self.mean_head = nn.Linear(feature_count, self.output_size)
        self.std_head = nn.Linear(feature_count, self.output_size)
        self.presence_head = nn.Linear(feature_count, self.output_size)

    def forward(self, spectra: torch.Tensor) -> IBAnetPrediction:
        if not torch.onnx.is_in_onnx_export():
            if spectra.ndim != 2 or spectra.shape[1] != self.input_dimension:
                raise ValueError("Expected [batch, concatenated spectrum channels]")
            if not torch.isfinite(spectra).all() or (spectra < 0).any():
                raise ValueError("Spectra must be finite and nonnegative")
        x = (torch.log1p(spectra) - self.input_mean) / self.input_std
        features = []
        offset = 0
        for encoder, length in zip(self.encoders, self.input_spectra_lengths.values()):
            features.append(encoder(x[:, offset:offset + length].unsqueeze(1)))
            offset += length
        hidden = self.head(torch.cat(features, dim=1))
        shape = (-1, self.max_layers, len(self.elements))
        return IBAnetPrediction(F.softplus(self.mean_head(hidden)).reshape(shape),
                                (F.softplus(self.std_head(hidden)) + self.minimum_std).reshape(shape),
                                self.presence_head(hidden).reshape(shape))

    def predict(self, spectra: torch.Tensor) -> IBAnetPrediction:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self(spectra)
        finally:
            self.train(was_training)

    def export(self, path: str | Path, *, output_inverse_factor: float = 1.0,
               opset_version: int = 17, need_pileup_subtraction: bool = True,
               pileup_fudge_factor_seconds: float = 0.4e-6) -> Path:
        """Export posterior mean P*Y as a native version 3 EDP package.

        Native EDP packages hold a point estimate, not R/P or posterior samples.
        Input normalization is inside ONNX; metadata input transform is identity.
        The runtime divides the graph output by output_inverse_factor.
        need_pileup_subtraction declares pileup-free training inputs. Experimental
        spectra/settings are supplied through InverseInput for native correction.
        predict_prepared skips physical corrections for reference spectra.
        """
        if not math.isfinite(output_inverse_factor) or output_inverse_factor <= 0:
            raise ValueError("output_inverse_factor must be finite and positive")
        if not isinstance(need_pileup_subtraction, bool):
            raise ValueError("need_pileup_subtraction must be a boolean")
        if not math.isfinite(pileup_fudge_factor_seconds) or pileup_fudge_factor_seconds < 0:
            raise ValueError("pileup_fudge_factor_seconds must be finite and nonnegative")
        from ._native import native
        metadata = native.model.ModelMetadata()
        metadata.model_type = native.model.ModelType.INVERSE
        metadata.class_name = "IBAnet"
        metadata.input_dimension = self.input_dimension
        metadata.output_dimension = self.output_size
        metadata.opset_version = opset_version
        input_transform = native.model.TransformSpec()
        input_transform.input_dimension = self.input_dimension
        metadata.input_transform = input_transform
        output_transform = native.model.TransformSpec()
        output_transform.type = "constant_factor"
        output_transform.input_dimension = self.output_size
        output_transform.factor = output_inverse_factor
        metadata.output_transform = output_transform
        config = self.study._config()
        inverse = native.model.InverseModelMetadata()
        inverse.need_pileup_subtraction = need_pileup_subtraction
        inverse.pileup_fudge_factor_seconds = pileup_fudge_factor_seconds
        inverse.sample_template, inverse.setup_template = config.sample, config.setup
        specs = []
        for label, length in self.input_spectra_lengths.items():
            spec = native.model.SpectrumSpec()
            spec.label, spec.length = label, length
            specs.append(spec)
        inverse.input_spectra = specs
        edp = native.model.EdpSpec()
        edp.max_layers, edp.elements = self.max_layers, list(self.elements)
        inverse.output_edp = edp
        metadata.inverse = inverse

        class PointEstimate(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inputs):
                return self.model(inputs).mean.flatten(1)

        # Export a CPU copy so a failed export never moves or changes the live model.
        import copy
        exported = PointEstimate(copy.deepcopy(self).cpu().eval())
        kwargs = dict(input_names=["inputs"], output_names=["outputs"],
                      dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
                      opset_version=opset_version, do_constant_folding=True)
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kwargs["dynamo"] = False
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="ibeamlab-ibanet-") as directory:
            onnx_path = Path(directory) / "model.onnx"
            torch.onnx.export(exported, torch.zeros(1, self.input_dimension), str(onnx_path), **kwargs)
            native.model.ModelPackage.from_onnx(onnx_path, metadata).write(destination)
        return destination


__all__ = ["IBAnet", "IBAnetPrediction", "IBAnetLoss"]
