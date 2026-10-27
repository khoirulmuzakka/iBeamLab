"""Optional PyTorch CNN/GRU inverse model with a single Gaussian slab."""

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
class iBeamNetPrediction:
    """Conditional Gaussian parameters in training target units.

    The Gaussian slab follows the draft and can produce negative draws even
    though its mean is constrained positive. Sampling does not clip those draws.
    Divide Y and R by the training target factor to recover physical units.
    """

    Y: torch.Tensor
    R: torch.Tensor
    logits: torch.Tensor
    density: torch.Tensor | None = None
    layer_gates: torch.Tensor | None = None
    soft_layer_gates: torch.Tensor | None = None
    sample_logits: torch.Tensor | None = None

    @property
    def E(self):
        """Conditional density standard deviation; R is the stored parameter."""
        return self.R

    @property
    def P(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    @property
    def mean(self) -> torch.Tensor:
        """Configured density estimate: legacy P*Y, hard-presence Y, or ordered AD."""
        return self.density if self.density is not None else self.P * self.Y

    @property
    def variance(self) -> torch.Tensor:
        if self.layer_gates is not None:
            return torch.zeros_like(self.density)
        p = self.P
        return p * self.R.square() + p * (1 - p) * self.Y.square()

    def sample(self, count: int = 1, *, generator=None) -> torch.Tensor:
        """Draw the draft's untruncated distribution: [draw, batch, layer, element]."""
        if self.layer_gates is not None:
            raise ValueError("Deterministic density heads do not define uncertainty samples")
        if count < 1:
            raise ValueError("count must be positive")
        shape = (count, *self.Y.shape)
        present = torch.rand(shape, device=self.Y.device, dtype=self.Y.dtype,
                             generator=generator) < self.P
        normal = torch.randn(shape, device=self.Y.device, dtype=self.Y.dtype,
                             generator=generator)
        return torch.where(present, self.Y + self.R * normal, 0.0)


class iBeamNetLoss(nn.Module):
    """Spike-and-slab NLL up to a target-dependent constant, with M = (target > 0).

    Sum layer/element contributions per sample, then average the batch. Absent
    entries contribute only Bernoulli BCE; no Gaussian regression is applied.
    Negative NLL values are possible for continuous densities and are valid.
    include_constant adds 0.5*log(2*pi) per present cell. gaussian_depth_alpha
    optionally weights only the Gaussian term by normalized true midpoint depth;
    zero preserves the exact unweighted likelihood.
    """

    def __init__(self, *, include_constant=False, gaussian_depth_alpha=0.0):
        super().__init__()
        if not math.isfinite(gaussian_depth_alpha) or gaussian_depth_alpha < 0:
            raise ValueError("Gaussian depth decay must be finite and nonnegative")
        self.include_constant=include_constant
        self.gaussian_depth_alpha=float(gaussian_depth_alpha)

    def forward(self, prediction: iBeamNetPrediction, target: torch.Tensor) -> torch.Tensor:
        if target.shape != prediction.Y.shape or target.numel() == 0:
            raise ValueError("Targets must match nonempty [batch, layer, element] outputs")
        if not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError("EDP targets must be finite and nonnegative")
        mask = target > 0
        # P = sigmoid(q). This is exactly -M*log(P) - (1-M)*log(1-P),
        # evaluated from q without unstable logs of rounded probabilities.
        classification = F.binary_cross_entropy_with_logits(
            prediction.logits, mask.to(target.dtype), reduction="none")
        # Replace absent means/scales before arithmetic so their values cannot
        # overflow and they receive no conditional-Gaussian gradient.
        mean=torch.where(mask,prediction.Y,torch.zeros_like(target))
        sigma=torch.where(mask,prediction.R,torch.ones_like(target))
        gaussian=sigma.log()+0.5*((target-mean)/sigma).square()
        if self.include_constant:gaussian=gaussian+mask.to(target.dtype)*(0.5*math.log(2*math.pi))
        if self.gaussian_depth_alpha > 0:
            # True midpoint depth / total AD is invariant to the common AD scale.
            # Normalize over every layer slot to preserve mean weight one.
            thickness=target.detach().sum(dim=-1)
            midpoint=thickness.cumsum(dim=1)-0.5*thickness
            total=thickness.sum(dim=1,keepdim=True)
            relative_depth=midpoint/total.clamp_min(torch.finfo(target.dtype).tiny)
            weights=target.shape[1]*torch.softmax(-self.gaussian_depth_alpha*relative_depth,dim=1)
            gaussian=gaussian*weights.unsqueeze(-1)
        return (classification + gaussian).flatten(1).sum(1).mean()


def _normalization_groups(channels: int) -> int:
    return next(groups for groups in range(min(8, max(1, channels // 2)), 0, -1)
                if channels % groups == 0)


class _ResidualSpectrumBlock(nn.Module):
    """Downsample energy channels while retaining a learned residual path."""

    def __init__(self, inputs: int, outputs: int, kernel_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(inputs, outputs, kernel_size, stride=2,
                               padding=kernel_size // 2)
        self.norm1 = (nn.GroupNorm(_normalization_groups(outputs), outputs)
                      if outputs > 1 else nn.Identity())
        self.conv2 = nn.Conv1d(outputs, outputs, kernel_size,
                               padding=kernel_size // 2)
        self.norm2 = (nn.GroupNorm(_normalization_groups(outputs), outputs)
                      if outputs > 1 else nn.Identity())
        self.skip = nn.Conv1d(inputs, outputs, 1, stride=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(inputs)))
        return F.silu(self.norm2(self.conv2(hidden)) + self.skip(inputs))


class _CNNGRUV1(nn.Module):
    """Residual spectral CNNs and a configurable layer GRU.

    Inputs are raw nonnegative counts concatenated in ``spectrum_labels`` order.
    Log1p preprocessing and fitted per-channel standardization are embedded in
    the graph/state dict. Y uses softplus (a positive-mean restriction on the
    draft's Gaussian); R uses softplus + minimum_std; P uses sigmoid logits.
    Legacy heads predict Gaussian slabs. ``ordered_density=True`` instead
    predicts deterministic AD with straight-through ordered layer gates and
    sample-level element presence; the first layer is always active. Recurrent
    states carry context toward deeper layers, without previous sampled outputs
    or teacher forcing. The likelihood remains conditionally independent cells.
    """

    def __init__(self, study: GenerationStudy, input_spectra_lengths: Mapping[str, int], *,
                 elements: Sequence[str] | None = None,
                 cnn_channels: Sequence[int] = (32, 64, 128, 128), kernel_size: int = 7,
                 compression_channels: int | None = 16,
                 head_hidden_sizes: Sequence[int] = (128,), minimum_std: float = 1e-3,
                 decoder_hidden_size: int = 256, decoder_layers: int = 2,
                 layer_embedding_size: int = 32, dropout: float = 0.1, bidirectional: bool = False,
                 input_mean=None, input_std=None, ordered_density: bool = False,
                 discrete_presence: bool = False) -> None:
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
            raise ValueError("Spectra-only iBeamNet requires fixed experimental settings")
        if kernel_size < 1 or kernel_size % 2 != 1 or not cnn_channels:
            raise ValueError("Use a positive odd CNN kernel and nonempty channels")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (*cnn_channels, *head_hidden_sizes, decoder_hidden_size,
                         decoder_layers, layer_embedding_size)):
            raise ValueError("Network widths and decoder dimensions must be positive integers")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if compression_channels is not None and (
                not isinstance(compression_channels, int) or isinstance(compression_channels, bool)
                or compression_channels < 1):
            raise ValueError("compression_channels must be a positive integer or None")
        if not math.isfinite(minimum_std) or minimum_std <= 0:
            raise ValueError("minimum_std must be finite and positive")
        self.study = study
        self.spectrum_labels = labels
        self.input_spectra_lengths = {label: input_spectra_lengths[label] for label in labels}
        self.max_layers = len(layers)
        self.input_dimension = sum(self.input_spectra_lengths.values())
        self.output_size = self.max_layers * len(self.elements)
        self.minimum_std = minimum_std
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        if not isinstance(bidirectional, bool):
            raise ValueError("bidirectional must be a boolean")
        if not isinstance(ordered_density, bool):
            raise ValueError("ordered_density must be a boolean")
        if not isinstance(discrete_presence, bool) or (ordered_density and discrete_presence):
            raise ValueError("discrete_presence must be boolean and cannot combine with ordered_density")
        self.discrete_presence = discrete_presence
        self.ordered_density = ordered_density
        self.decoder_directions = 2 if bidirectional else 1
        self.architecture = dict(cnn_channels=tuple(cnn_channels), kernel_size=kernel_size,
                                 compression_channels=compression_channels,
                                 head_hidden_sizes=tuple(head_hidden_sizes), minimum_std=minimum_std,
                                 decoder_hidden_size=decoder_hidden_size, decoder_layers=decoder_layers,
                                 layer_embedding_size=layer_embedding_size, dropout=dropout, bidirectional=bidirectional, ordered_density=ordered_density, discrete_presence=discrete_presence)
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
                blocks.append(_ResidualSpectrumBlock(previous, channels, kernel_size))
                previous = channels
                length = (length + 1) // 2
            # Mix feature channels independently at each energy position.
            # None preserves the uncompressed CNN/GRU checkpoint layout.
            if compression_channels is not None:
                blocks.append(nn.Conv1d(previous, compression_channels, kernel_size=1))
                previous = compression_channels
            blocks.append(nn.Flatten())
            self.encoders.append(nn.Sequential(*blocks))
            feature_count += previous * length
        # Flattening preserves absolute energy position; global average pooling
        # would discard the peak positions needed by the inverse problem.
        self.context = nn.Sequential(nn.Linear(feature_count, decoder_hidden_size),
                                     nn.LayerNorm(decoder_hidden_size), nn.SiLU())
        self.layer_embedding = nn.Embedding(self.max_layers, layer_embedding_size)
        self.decoder_input_norm = nn.LayerNorm(decoder_hidden_size + layer_embedding_size)
        self.initial_hidden = nn.Linear(decoder_hidden_size, decoder_layers * self.decoder_directions * decoder_hidden_size)
        self.decoder = nn.GRU(decoder_hidden_size + layer_embedding_size,
                              decoder_hidden_size, num_layers=decoder_layers,
                              batch_first=True, bidirectional=bidirectional,
                              dropout=dropout if decoder_layers > 1 else 0.0)
        feature_count = decoder_hidden_size * self.decoder_directions
        head = []
        for width in head_hidden_sizes:
            head.extend([nn.Linear(feature_count, width), nn.SiLU(), nn.Dropout(dropout)])
            feature_count = width
        self.head = nn.Sequential(*head)
        self.mean_head = nn.Linear(feature_count, len(self.elements))
        self.std_head = nn.Linear(feature_count, len(self.elements))
        self.presence_head = nn.Linear(feature_count, len(self.elements))
        if ordered_density:
            self.layer_activity_head = nn.Linear(feature_count, 1)
            self.sample_presence_head = nn.Linear(decoder_hidden_size, len(self.elements))
            nn.init.zeros_(self.layer_activity_head.weight)
            nn.init.constant_(self.layer_activity_head.bias, 2.0)

    def forward(self, spectra: torch.Tensor) -> iBeamNetPrediction:
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
        context = self.context(torch.cat(features, dim=1))
        batch_size = spectra.shape[0]
        positions = self.layer_embedding(torch.arange(self.max_layers, device=spectra.device))
        decoder_inputs = self.decoder_input_norm(torch.cat([
            context.unsqueeze(1).expand(-1, self.max_layers, -1),
            positions.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1))
        initial = torch.tanh(self.initial_hidden(context)).reshape(
            batch_size, self.decoder_layers * self.decoder_directions, self.decoder_hidden_size).transpose(0, 1).contiguous()
        decoded, _ = self.decoder(decoder_inputs, initial)
        hidden = self.head(decoded)
        if self.ordered_density:
            # A hard contiguous prefix is used by the surrogate and ONNX graph.
            # Straight-through soft prefix gradients train the stopping head.
            continuation = torch.sigmoid(self.layer_activity_head(hidden[:, 1:]))
            first = torch.ones_like(hidden[:, :1, :1])
            soft_prefix, hard_prefix = [first], [first]
            for layer in range(self.max_layers - 1):
                probability = continuation[:, layer:layer + 1]
                soft_prefix.append(soft_prefix[-1] * probability)
                hard_prefix.append(hard_prefix[-1] * (probability >= 0.5).to(hidden.dtype))
            soft_gates = torch.cat(soft_prefix, dim=1)
            hard_gates = torch.cat(hard_prefix, dim=1)
            gates = hard_gates.detach() + (soft_gates - soft_gates.detach())
            density = gates * F.softplus(self.mean_head(hidden))
            sample_logits = self.sample_presence_head(context)
            logits = sample_logits.unsqueeze(1).expand_as(density)
            return iBeamNetPrediction(density, torch.zeros_like(density), logits,
                                    density, gates, soft_gates, sample_logits)
        prediction = iBeamNetPrediction(F.softplus(self.mean_head(hidden)),
                                      F.softplus(self.std_head(hidden)) + self.minimum_std,
                                      self.presence_head(hidden))
        if self.discrete_presence:
            # MAP presence configuration, with conditional means; never P * Y.
            prediction.density = (prediction.logits >= 0).to(hidden.dtype) * prediction.Y
        return prediction

    def as_discrete_presence(self):
        """Copy compatible weights into a probabilistic hard-presence model.

        Retains Gaussian/presence heads. Export uses the MAP presence mask times
        conditional means; posterior uncertainty remains available separately.
        """
        architecture = {**self.architecture, "ordered_density": False, "discrete_presence": True}
        converted = type(self)(self.study, self.input_spectra_lengths,
                               elements=self.elements, **architecture)
        state = {key: value for key, value in self.state_dict().items()
                 if key in converted.state_dict()}
        converted.load_state_dict(state)
        converted.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        converted.train(self.training)
        return converted

    def as_ordered_density(self):
        """Copy shared weights into a deterministic, ordered density model.

        New activity/presence heads need training; optimizer state is not copied.
        The original model and checkpoint remain untouched.
        """
        import copy
        if self.ordered_density:
            return copy.deepcopy(self)
        converted = type(self)(self.study, self.input_spectra_lengths,
                               elements=self.elements,
                               **{**self.architecture, "ordered_density": True, "discrete_presence": False})
        converted.load_state_dict(self.state_dict(), strict=False)
        converted.to(device=next(self.parameters()).device,
                     dtype=next(self.parameters()).dtype)
        converted.train(self.training)
        return converted

    def as_bidirectional(self):
        """Warm-start a bidirectional model from this model; predictions may change.

        Copy both recurrent directions from the original and split incoming
        weights equally across their concatenated outputs. Source is untouched.
        Optimizer state cannot transfer across these changed parameter shapes.
        """
        import copy
        if self.decoder.bidirectional:
            return copy.deepcopy(self)
        architecture = {**self.architecture, "bidirectional": True}
        converted = type(self)(self.study, self.input_spectra_lengths,
                               elements=self.elements, **architecture)
        original = self.state_dict()
        updated = converted.state_dict()
        for name, target in updated.items():
            source_name = name.removesuffix("_reverse") if name.startswith("decoder.") else name
            value = original[source_name]
            if name.startswith("initial_hidden."):
                value = value.reshape(self.decoder_layers, self.decoder_hidden_size, *value.shape[1:])
                value = value.repeat_interleave(2, dim=0).reshape(target.shape)
            elif value.shape != target.shape:
                # GRU upper-layer inputs and the first output-head projection
                # now receive forward and backward features.
                if value.ndim != 2 or target.shape != (value.shape[0], value.shape[1] * 2):
                    raise ValueError(f"Cannot transfer parameter {name}")
                value = torch.cat((value, value), dim=1) * 0.5
            updated[name] = value.detach().clone()
        converted.load_state_dict(updated)
        converted.to(next(self.parameters()).device)
        converted.train(self.training)
        return converted

    def predict(self, spectra: torch.Tensor) -> iBeamNetPrediction:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self(spectra)
        finally:
            self.train(was_training)

    def export(self, path: str | Path, *, output_inverse_factor: float = 1.0,
               opset_version: int = 17, need_pileup_subtraction: bool = True,
               pileup_fudge_factor_seconds: float = 0.4e-6,
               apply_reference_pileup: bool = False) -> Path:
        """Export density, presence probability and standard deviation (version 4).

        Discrete presence heads export the MAP presence mask times conditional
        means, with the underlying posterior std provided separately.
        Ordered density heads export sample-level element probabilities repeated
        across layers, and zero standard deviation as a compatibility placeholder.

        Posterior std includes both Gaussian variance and presence uncertainty.
        Input normalization is inside ONNX; metadata input transform is identity.
        The runtime divides the graph output by output_inverse_factor.
        need_pileup_subtraction requests measured pileup removal. Experimental
        spectra/settings are supplied through InverseInput for native correction.
        predict_prepared skips physical corrections for reference spectra.
        apply_reference_pileup=True adds reference pileup after measured pileup
        removal, rebinning and exposure scaling. Input lengths must include full
        2*N-1 support; predict_prepared then expects piled-up reference spectra.
        """
        if not math.isfinite(output_inverse_factor) or output_inverse_factor <= 0:
            raise ValueError("output_inverse_factor must be finite and positive")
        if not isinstance(need_pileup_subtraction, bool):
            raise ValueError("need_pileup_subtraction must be a boolean")
        if not math.isfinite(pileup_fudge_factor_seconds) or pileup_fudge_factor_seconds < 0:
            raise ValueError("pileup_fudge_factor_seconds must be finite and nonnegative")
        if not isinstance(apply_reference_pileup, bool):
            raise ValueError("apply_reference_pileup must be a boolean")
        if apply_reference_pileup and not need_pileup_subtraction:
            raise ValueError("reference pileup requires measured pileup subtraction")
        if apply_reference_pileup and any(n % 2 == 0 for n in self.input_spectra_lengths.values()):
            raise ValueError("reference pileup input lengths must be full odd 2*N-1 lengths")
        from ._native import native
        metadata = native.model.ModelMetadata()
        metadata.model_type = native.model.ModelType.INVERSE
        metadata.class_name = "iBeamNet"
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
        inverse.apply_reference_pileup = apply_reference_pileup
        inverse.pileup_fudge_factor_seconds = pileup_fudge_factor_seconds
        inverse.presence_probability_output_name = "presence_probability"
        inverse.posterior_std_output_name = "posterior_std"
        inverse.posterior_std_inverse_factor = output_inverse_factor
        inverse.uncertainty_predicted = True
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
                prediction = self.model(inputs)
                return (prediction.mean.reshape(inputs.shape[0], self.model.output_size),
                        prediction.P.reshape(inputs.shape[0], self.model.output_size),
                        prediction.variance.clamp_min(0).sqrt().reshape(inputs.shape[0], self.model.output_size))

        # Export a CPU copy so a failed export never moves or changes the live model.
        import copy
        exported = PointEstimate(copy.deepcopy(self).cpu().eval())
        kwargs = dict(input_names=["inputs"], output_names=["outputs", "presence_probability", "posterior_std"],
                      dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"},
                                    "presence_probability": {0: "batch"}, "posterior_std": {0: "batch"}},
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



class _AttentionV3(_CNNGRUV1):
    """128-wide CNN/layer-query attention with independent Gaussian slab heads."""

    def __init__(self, study, input_spectra_lengths, *, elements=None,
                 cnn_channels=(32,64,128,128), kernel_size=7, attention_size=128,
                 attention_heads=4, attention_blocks=2, dropout=0.0, minimum_std=1e-3,
                 initial_density=0.01, input_mean=None, input_std=None):
        nn.Module.__init__(self)
        from .spectrum_encoder import SpectrumAttentionEncoder
        self.study=study
        self.spectrum_labels=tuple(d.label for d in study.experiment.detectors)
        if set(input_spectra_lengths) != set(self.spectrum_labels) or len(set(self.spectrum_labels)) != len(self.spectrum_labels):
            raise ValueError('Spectrum lengths must match unique detector labels')
        if any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in input_spectra_lengths.values()):
            raise ValueError('Spectrum lengths must be positive integers')
        self.input_spectra_lengths={k:input_spectra_lengths[k] for k in self.spectrum_labels}
        layers=study.experiment.sample.layers
        self.elements=tuple(elements) if elements is not None else tuple(s.element for s in layers[0].species)
        if not self.elements or len(set(self.elements))!=len(self.elements) or any(set(s.element for s in l.species)!=set(self.elements) for l in layers):
            raise ValueError('Template layers must contain exactly the unique EDP elements')
        if any(p.layer is None for p in study.parameters):
            raise ValueError('Spectra-only model requires fixed experimental settings')
        if not math.isfinite(minimum_std) or minimum_std<=0:
            raise ValueError('minimum_std must be positive and finite')
        self.max_layers=len(layers)
        self.input_dimension=sum(self.input_spectra_lengths.values())
        self.output_size=self.max_layers*len(self.elements)
        self.minimum_std=minimum_std
        self.ordered_density=False
        self.backbone=SpectrumAttentionEncoder(self.input_dimension,self.output_size,
            max_layers=self.max_layers,elements_per_layer=len(self.elements),cnn_channels=tuple(cnn_channels),
            kernel_size=kernel_size,attention_size=attention_size,attention_heads=attention_heads,
            attention_blocks=attention_blocks,dropout=dropout,initial_density=initial_density,
            input_mean=input_mean,input_std=input_std)
        self.mean_head=self.backbone.output
        del self.backbone.output
        self.std_head=nn.Linear(attention_size,len(self.elements))
        self.presence_head=nn.Linear(attention_size,len(self.elements))
        self.architecture=dict(cnn_channels=tuple(cnn_channels),kernel_size=kernel_size,
            attention_size=attention_size,attention_heads=attention_heads,attention_blocks=attention_blocks,
            dropout=dropout,minimum_std=minimum_std,initial_density=initial_density)

    @property
    def input_mean(self): return self.backbone.input_mean

    @property
    def input_std(self): return self.backbone.input_std

    def forward(self, spectra):
        if not torch.onnx.is_in_onnx_export():
            if spectra.ndim!=2 or spectra.shape[1]!=self.input_dimension:
                raise ValueError('Expected [batch, concatenated spectrum channels]')
            if not torch.isfinite(spectra).all() or (spectra<0).any():
                raise ValueError('Spectra must be finite and nonnegative')
        features=self.backbone.encode_layers(spectra)
        prediction=iBeamNetPrediction(F.softplus(self.mean_head(features)),
            F.softplus(self.std_head(features))+self.minimum_std,self.presence_head(features))
        return prediction


def depth_weighted_ad_mse(predicted, target, *, alpha=2.0):
    """Per-cell MSE weighted by exp(-alpha * true midpoint depth / total AD).

    Weights sum to one over all layers per sample, including zero-padded layers.
    Both AD tensors use the same common scale. Empty targets use uniform weights.
    """
    if predicted.shape != target.shape or target.ndim != 3 or target.numel() == 0:
        raise ValueError('AD tensors must match nonempty [batch, layer, element] shapes')
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Depth decay alpha must be finite and nonnegative')
    thickness = target.detach().sum(dim=-1)
    midpoint = thickness.cumsum(dim=1) - 0.5 * thickness
    total = thickness.sum(dim=1, keepdim=True)
    relative_depth = midpoint / total.clamp_min(torch.finfo(target.dtype).tiny)
    weights = torch.softmax(-alpha * relative_depth, dim=1)
    return (weights * (predicted - target).square().mean(dim=-1)).sum(dim=1).mean()


class _PosteriorV4(_AttentionV3):
    """Attention with sigmoid presence and conditional Gaussian mean/std heads."""

    def __init__(self, study, input_spectra_lengths, *, elements=None,
                 cnn_channels=(32,64,128,128), kernel_size=7, attention_size=128,
                 attention_heads=4, attention_blocks=2, dropout=0.0,
                 initial_density=0.01, minimum_std=1e-5, presence_mapping='sigmoid',
                 input_mean=None, input_std=None):
        if presence_mapping!='sigmoid':raise ValueError('Version 4 requires sigmoid presence for exact NLL')
        super().__init__(study,input_spectra_lengths,elements=elements,cnn_channels=cnn_channels,
            kernel_size=kernel_size,attention_size=attention_size,attention_heads=attention_heads,
            attention_blocks=attention_blocks,dropout=dropout,initial_density=initial_density,
            minimum_std=minimum_std,input_mean=input_mean,input_std=input_std)
        self.architecture['presence_mapping']=presence_mapping


class iBeamNet(nn.Module):
    """Versioned inverse model: v1 CNN/GRU, v3 attention posterior, v4 sigmoid/Gaussian posterior.

    Architecture-specific arguments belong in model_options. Each implementation
    validates its own options; unsupported versions and unknown options fail explicitly.
    """
    _implementations={1:_CNNGRUV1,3:_AttentionV3,4:_PosteriorV4}

    def __init__(self, study, input_spectra_lengths, *, version=3, model_options=None,
                 elements=None, input_mean=None, input_std=None):
        super().__init__()
        if not isinstance(version,int) or isinstance(version,bool) or version not in self._implementations:
            raise ValueError(f'Unsupported iBeamNet version: {version}; supported: {tuple(self._implementations)}')
        options={} if model_options is None else dict(model_options)
        reserved={'study','input_spectra_lengths','elements','input_mean','input_std'} & options.keys()
        if reserved: raise ValueError(f'Reserved model_options keys: {sorted(reserved)}')
        implementation=self._implementations[version]
        allowed=set(inspect.signature(implementation.__init__).parameters)-{'self'}
        unknown=set(options)-allowed
        if unknown: raise ValueError(f'Unknown options for iBeamNet v{version}: {sorted(unknown)}')
        self.version=version
        self.implementation=implementation(study,input_spectra_lengths,elements=elements,
            input_mean=input_mean,input_std=input_std,**options)
        self.model_options=dict(self.implementation.architecture)

    def __getattr__(self,name):
        try: return super().__getattr__(name)
        except AttributeError:
            implementation=super().__getattr__('implementation')
            return getattr(implementation,name)

    def forward(self,spectra): return self.implementation(spectra)

    def predict(self,spectra): return self.implementation.predict(spectra)

    def state_dict(self,*args,**kwargs): return self.implementation.state_dict(*args,**kwargs)

    def load_state_dict(self,*args,**kwargs): return self.implementation.load_state_dict(*args,**kwargs)

    def _convert(self,method):
        if self.version!=1: raise ValueError(f'{method} is only available for iBeamNet v1')
        converted=getattr(self.implementation,method)()
        result=type(self)(self.study,self.input_spectra_lengths,version=1,
            elements=self.elements,model_options=converted.architecture)
        result.implementation=converted
        result.model_options=dict(converted.architecture)
        result.train(self.training)
        return result

    def as_bidirectional(self): return self._convert('as_bidirectional')
    def as_ordered_density(self): return self._convert('as_ordered_density')
    def as_discrete_presence(self): return self._convert('as_discrete_presence')

    def checkpoint(self):
        return dict(version=self.version,model_options=dict(self.model_options),
            elements=list(self.elements),input_spectra_lengths=dict(self.input_spectra_lengths),
            parameter_names=list(self.study.parameter_names),max_layers=self.max_layers,
            model_state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()})

    def save(self,path):
        destination=Path(path)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix(destination.suffix+'.tmp')
        torch.save(self.checkpoint(),temporary)
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls,path,*,study,map_location='cpu'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if list(study.parameter_names)!=checkpoint['parameter_names'] or len(study.experiment.sample.layers)!=checkpoint['max_layers']:
            raise ValueError('Study layout differs from saved iBeamNet')
        options=dict(checkpoint['model_options'])
        if checkpoint['version']==3 and options.get('discrete_presence') is False:
            options.pop('discrete_presence')  # Removing an unused option preserves P*Y exactly.
        model=cls(study,checkpoint['input_spectra_lengths'],version=checkpoint['version'],
            elements=checkpoint['elements'],model_options=options)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(map_location)

    def export(self,path,**options):
        import json,zipfile,tomli_w
        try: import tomllib
        except ImportError: import tomli as tomllib
        destination=self.implementation.export(path,**options)
        with zipfile.ZipFile(destination) as archive:
            contents={name:archive.read(name) for name in archive.namelist()}
        metadata=tomllib.loads(contents['package.toml'].decode())
        metadata['ibeamnet']=dict(version=self.version,model_options_json=json.dumps(self.model_options))
        contents['package.toml']=tomli_w.dumps(metadata).encode()
        temporary=destination.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,data in contents.items(): archive.writestr(name,data)
        temporary.replace(destination)
        return destination


__all__=['iBeamNet','iBeamNetPrediction','iBeamNetLoss','depth_weighted_ad_mse']
