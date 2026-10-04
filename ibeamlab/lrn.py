"""Trainable layerwise recurrent forward surrogate (LRN).

This module is optional and requires the ``training`` extra.  The eager PyTorch
model can infer on a variable number of layers when its input features use the
same setup-plus-layer-block prefix layout.  ONNX packages retain the fixed input
width of the training schema so the native C++ ``ForwardModel`` can validate
and materialize physical parameters from its package metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .generation import GenerationStudy


@dataclass(frozen=True, slots=True)
class _InputLayout:
    setup_indices: tuple[int, ...]
    layer_indices: tuple[tuple[int, ...], ...]
    layer_names: tuple[str, ...]

    @property
    def setup_size(self) -> int:
        return len(self.setup_indices)

    @property
    def layer_size(self) -> int:
        return len(self.layer_indices[0]) if self.layer_indices else 0

    @property
    def layer_count(self) -> int:
        return len(self.layer_indices)


def _feature_signature(parameter) -> tuple[str, str, str | None]:
    return (parameter.kind, parameter.element or "", parameter.detector)


def _infer_layout(study: GenerationStudy) -> _InputLayout:
    setup_indices: list[int] = []
    layer_groups: dict[int, list[tuple[int, object]]] = {}
    for index, parameter in enumerate(study.parameters):
        if parameter.layer is None:
            setup_indices.append(index)
        else:
            layer_groups.setdefault(parameter.layer, []).append((index, parameter))

    if not layer_groups:
        return _InputLayout(tuple(setup_indices), (), ())
    layer_numbers = sorted(layer_groups)
    if layer_numbers != list(range(layer_numbers[-1] + 1)):
        raise ValueError("LRN requires contiguous zero-based layer parameter groups.")

    template = layer_groups[layer_numbers[0]]
    signature = tuple(_feature_signature(parameter) for _, parameter in template)
    if not signature:
        raise ValueError("LRN layer feature groups cannot be empty.")
    layer_indices = []
    for number in layer_numbers:
        current = layer_groups[number]
        if tuple(_feature_signature(parameter) for _, parameter in current) != signature:
            raise ValueError("LRN requires the same ordered parameter features in every layer.")
        layer_indices.append(tuple(index for index, _ in current))

    return _InputLayout(
        tuple(setup_indices),
        tuple(layer_indices),
        tuple(parameter.name for _, parameter in template),
    )


class _LayerwiseGRUBlock(nn.Module):
    """Shared recurrent layer encoder that emits a latent contribution."""

    def __init__(self, layer_size: int, setup_size: int, hidden_size: int,
                 contribution_size: int, embedding_dim: int,
                 block_hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.dummy_layer = layer_size <= 0
        self.dummy_setup = setup_size <= 0
        self.layer_encoder = nn.Sequential(nn.Linear(max(layer_size, 1), embedding_dim), nn.LeakyReLU())
        self.setup_encoder = nn.Sequential(nn.Linear(max(setup_size, 1), embedding_dim), nn.LeakyReLU())
        self.setup_to_film = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
            nn.Linear(embedding_dim, 2 * embedding_dim),
        )
        self.hidden_projection = nn.Sequential(nn.Linear(hidden_size, embedding_dim), nn.LeakyReLU())
        tower: list[nn.Module] = []
        previous = 3 * embedding_dim
        for width in block_hidden_sizes:
            tower.extend((nn.Linear(previous, width), nn.LeakyReLU()))
            previous = width
        self.fused_tower = nn.Sequential(*tower) if tower else nn.Identity()
        self.gru = nn.GRUCell(previous, hidden_size)
        contribution_hidden = max(contribution_size, embedding_dim)
        self.thickness_gate = nn.Linear(previous + hidden_size, hidden_size)
        self.contribution_head = nn.Sequential(
            nn.Linear(previous + hidden_size, contribution_hidden), nn.LeakyReLU(),
            nn.Linear(contribution_hidden, contribution_size),
        )

    def forward(self, layer, setup, hidden, thickness):
        if self.dummy_layer:
            layer = layer.new_ones((layer.shape[0], 1))
        if self.dummy_setup:
            setup = setup.new_ones((setup.shape[0], 1))
        layer_features = self.layer_encoder(layer)
        setup_features = self.setup_encoder(setup)
        gamma, beta = self.setup_to_film(setup_features).chunk(2, dim=1)
        conditioned = layer_features * (torch.tanh(gamma) + 1.0) + beta
        fused = self.fused_tower(torch.cat((conditioned, setup_features,
                                            self.hidden_projection(hidden)), dim=1))
        candidate_hidden = self.gru(fused, hidden)
        # thickness is t / t_ref, using the same reference as areal densities.
        rate = F.softplus(self.thickness_gate(torch.cat((fused, hidden), dim=1)))
        gate = 1.0 - torch.exp(-thickness * rate)
        next_hidden = hidden + gate * (candidate_hidden - hidden)
        contribution = self.contribution_head(torch.cat((fused, next_hidden), dim=1))
        return contribution, next_hidden


class _LocalSpectrumRefiner(nn.Module):
    def __init__(self, setup_size: int, hidden_channels: int, kernel_size: int) -> None:
        super().__init__()
        if hidden_channels < 1 or kernel_size < 1:
            raise ValueError("refiner hidden_channels and kernel_size must be positive.")
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.dummy_setup = setup_size <= 0
        self.in_conv = nn.Conv1d(1, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.residual_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                                       padding=kernel_size // 2)
        self.out_conv = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        effective_setup = max(setup_size, 1)
        self.film = nn.Sequential(
            nn.Linear(effective_setup, hidden_channels * 2), nn.LeakyReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )

    def forward(self, spectra, setup):
        if self.dummy_setup:
            setup = setup.new_ones((spectra.shape[0], 1))
        features = self.in_conv(spectra.unsqueeze(1))
        gamma, beta = self.film(setup).chunk(2, dim=1)
        features = F.leaky_relu(features * (torch.tanh(gamma).unsqueeze(-1) + 1.0)
                                + beta.unsqueeze(-1))
        residual = self.residual_conv(features)
        residual = F.leaky_relu(residual + features)
        return spectra + self.out_conv(residual).squeeze(1)


class LRNModel(nn.Module):
    """Layerwise recurrent forward network for an iBeamLab generation study.

    Parameters are taken in the exact order of ``study.parameters``. A shared
    GRU block receives elemental areal densities in place of thickness and
    concentrations. Its state update vanishes continuously with thickness.
    Inputs must use normalized concentrations and thickness scaled as t/t_ref
    (for example, build_lrn_input_transform with zero lower thickness bound).
    The public input width retains the physical parameter schema for export.
    Latent contributions are summed and
    decoded once into the concatenated output spectra. Train this module with
    ordinary PyTorch optimizers, then call :meth:`export` to create a package
    consumable by iBeamLab's native C++ ``ForwardModel``.
    """

    def __init__(self, study: GenerationStudy,
                 output_spectra_lengths: Mapping[str, int], *,
                 hidden_size: int = 256, contribution_size: int = 256,
                 setup_embedding_dim: int = 128, layer_embedding_dim: int = 256,
                 block_hidden_sizes: Sequence[int] = (512, 512),
                 decoder_hidden_sizes: Sequence[int] = (1024, 1024),
                 refiner_hidden_channels: int = 32, refiner_kernel_size: int = 17) -> None:
        super().__init__()
        labels = tuple(detector.label for detector in study.experiment.detectors)
        if set(output_spectra_lengths) != set(labels):
            raise ValueError("output_spectra_lengths must have exactly the experiment detector labels.")
        if any(int(length) <= 0 for length in output_spectra_lengths.values()):
            raise ValueError("Every output spectrum length must be positive.")
        if min(hidden_size, contribution_size, setup_embedding_dim, layer_embedding_dim) < 1:
            raise ValueError("LRN dimensions must be positive.")

        self.study = study
        self.layout = _infer_layout(study)
        self.setup_param_size = self.layout.setup_size
        self.layer_param_size = self.layout.layer_size
        self.layer_count = self.layout.layer_count
        self.thickness_index = None
        self.concentration_indices = ()
        if self.layer_count:
            layer_parameters = [study.parameters[i] for i in self.layout.layer_indices[0]]
            thickness_indices = [i for i, p in enumerate(layer_parameters)
                                 if p.kind in {"thickness", "layer_thickness"}]
            self.concentration_indices = tuple(i for i, p in enumerate(layer_parameters)
                                               if p.kind == "concentration")
            if len(thickness_indices) != 1 or not self.concentration_indices:
                raise ValueError("LRN layers require one thickness and elemental concentrations.")
            self.thickness_index = thickness_indices[0]
        self.layer_feature_size = self.layer_param_size - int(self.thickness_index is not None)
        self.input_dimension = len(study.parameters)
        if self.input_dimension <= 0:
            raise ValueError("LRN requires at least one study parameter.")
        self.output_spectra_lengths = {str(k): int(v) for k, v in output_spectra_lengths.items()}
        self.spectrum_labels = tuple(label for label in labels)
        self.output_size = sum(self.output_spectra_lengths[label] for label in labels)
        self.hidden_size = int(hidden_size)
        self.contribution_size = int(contribution_size)
        self.setup_indices = torch.tensor(self.layout.setup_indices, dtype=torch.int64)
        self.layer_indices = tuple(torch.tensor(item, dtype=torch.int64)
                                   for item in self.layout.layer_indices)
        self.runtime_prefix_layout = self._has_prefix_layout()

        self.setup_context_encoder = nn.Sequential(
            nn.Linear(max(self.setup_param_size, 1), setup_embedding_dim), nn.LeakyReLU(),
            nn.Linear(setup_embedding_dim, setup_embedding_dim), nn.LeakyReLU(),
        )
        self.layer_block = _LayerwiseGRUBlock(
            self.layer_feature_size, self.setup_param_size, hidden_size, contribution_size,
            layer_embedding_dim, block_hidden_sizes,
        )
        decoder_layers: list[nn.Module] = []
        previous = contribution_size + hidden_size + setup_embedding_dim
        for width in decoder_hidden_sizes:
            decoder_layers.extend((nn.Linear(previous, width), nn.LeakyReLU()))
            previous = width
        decoder_layers.append(nn.Linear(previous, self.output_size))
        self.spectrum_decoder = nn.Sequential(*decoder_layers)
        self.spectrum_refiner = _LocalSpectrumRefiner(
            self.setup_param_size, refiner_hidden_channels, refiner_kernel_size,
        )
        self.output_scale = nn.Parameter(torch.ones(self.output_size))

    def _has_prefix_layout(self) -> bool:
        offset = self.setup_param_size
        if self.layout.setup_indices != tuple(range(offset)):
            return False
        for indices in self.layout.layer_indices:
            if indices != tuple(range(offset, offset + self.layer_param_size)):
                return False
            offset += self.layer_param_size
        return True

    def validate_input_shape(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 2:
            raise ValueError("Model inputs must have shape (batch, features).")
        if inputs.shape[1] == self.input_dimension:
            return
        if self.training:
            raise ValueError(f"Expected {self.input_dimension} input features during training, got {inputs.shape[1]}.")
        if not self.runtime_prefix_layout or self.layer_param_size <= 0:
            raise ValueError("Variable-layer inference requires setup features followed by contiguous layer blocks.")
        if inputs.shape[1] < self.setup_param_size or (inputs.shape[1] - self.setup_param_size) % self.layer_param_size:
            raise ValueError("Variable-layer input must be setup features plus N complete layer blocks.")

    def split_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs.shape[1] == self.input_dimension:
            setup = inputs.index_select(1, self.setup_indices.to(inputs.device)) if self.setup_param_size else inputs[:, :0]
            layers = [inputs.index_select(1, indices.to(inputs.device)) for indices in self.layer_indices]
            return setup, layers
        setup = inputs[:, :self.setup_param_size] if self.setup_param_size else inputs[:, :0]
        count = (inputs.shape[1] - self.setup_param_size) // self.layer_param_size
        layers = [inputs[:, self.setup_param_size + i * self.layer_param_size:
                         self.setup_param_size + (i + 1) * self.layer_param_size]
                  for i in range(count)]
        return setup, layers

    def _encode_setup(self, setup: torch.Tensor) -> torch.Tensor:
        if self.setup_param_size <= 0:
            setup = setup.new_ones((setup.shape[0], 1))
        return self.setup_context_encoder(setup)

    def layer_areal_features(self, layer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert zero-based scaled thickness and fractions to scaled areal amounts.

        The public input schema stays unchanged for datasets/native packages.
        With t scaled by t_ref, each elemental feature is (t / t_ref) * c_e.
        Other layer properties retain their existing representation.
        """
        thickness = layer[:, self.thickness_index:self.thickness_index + 1].clamp_min(0.0)
        features = [layer[:, i:i + 1] * thickness if i in self.concentration_indices
                    else layer[:, i:i + 1]
                    for i in range(self.layer_param_size) if i != self.thickness_index]
        return torch.cat(features, dim=1), thickness

    def encode_latent(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_input_shape(inputs)
        setup, layers = self.split_inputs(inputs)
        hidden = inputs.new_zeros((inputs.shape[0], self.hidden_size))
        contributions = inputs.new_zeros((inputs.shape[0], self.contribution_size))
        setup_context = self._encode_setup(setup)
        for layer in layers:
            features, thickness = self.layer_areal_features(layer)
            mask = (thickness > 0).to(inputs.dtype)
            contribution, next_hidden = self.layer_block(features, setup, hidden, thickness)
            contributions = contributions + contribution * mask
            hidden = hidden * (1.0 - mask) + next_hidden * mask
        return torch.cat((contributions, hidden, setup_context), dim=1), setup

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent, setup = self.encode_latent(inputs)
        logits = self.spectrum_decoder(latent)
        coarse = F.softplus(logits)
        refined = F.softplus(self.spectrum_refiner(coarse, setup))
        return refined * F.softplus(self.output_scale).to(device=logits.device, dtype=logits.dtype)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference without gradients, including shorter prefix layer layouts."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self(inputs)
        finally:
            if was_training:
                self.train()

    def export(self, path: str | Path, *, opset_version: int = 17,
               input_minimum: Sequence[float] | None = None,
               input_scale: Sequence[float] | None = None,
               input_transform=None,
               output_inverse_factor: float = 1.0,
               bare_spectrum_corrections: bool = False,
               Apply_pileup_on_inference: bool = True,
               pileup_fudge_factor_seconds: float = 0.4e-6) -> Path:
        """Export ONNX and wrap it in the native iBeamLab forward package format.

        ``input_minimum`` and ``input_scale`` describe the optional min-max
        input transform ``(x - minimum) / scale`` used before the network.
        ``input_transform`` accepts a fitted native transform or transform
        pipeline and is mutually exclusive with those legacy arrays.
        ``output_inverse_factor`` is applied by the package runtime as
        ``network_output / factor``; use it when training against scaled targets.
        ``bare_spectrum_corrections=True`` declares pileup-free reference-channel
        spectra at a fixed study setup. Native inference rebins and scales to
        the requested setup before optionally applying pileup. The fudge factor
        is in seconds (0.4e-6 matches AutoNRA's default).
        """
        # Bare corrections require a fixed setup and spectra generated without pileup.
        # The reference channel grid and exposure come from the study setup template.
        from ._native import native
        import math

        if input_transform is not None and (input_minimum is not None or input_scale is not None):
            raise ValueError("input_transform is mutually exclusive with input_minimum/input_scale.")
        if (input_minimum is None) != (input_scale is None):
            raise ValueError("input_minimum and input_scale must be supplied together.")
        if input_minimum is not None:
            if len(input_minimum) != self.input_dimension or len(input_scale) != self.input_dimension:
                raise ValueError("Input transform arrays must match the model input dimension.")
            if any(not math.isfinite(float(value)) for value in input_minimum):
                raise ValueError("input_minimum values must be finite.")
            if any(not math.isfinite(float(value)) or float(value) <= 0 for value in input_scale):
                raise ValueError("input_scale values must be finite and positive.")
        if not math.isfinite(output_inverse_factor) or output_inverse_factor <= 0:
            raise ValueError("output_inverse_factor must be finite and positive.")

        config = self.study._config()
        metadata = native.model.ModelMetadata()
        metadata.model_type = native.model.ModelType.FORWARD
        metadata.class_name = type(self).__name__
        metadata.input_dimension = self.input_dimension
        metadata.output_dimension = self.output_size
        metadata.opset_version = int(opset_version)
        identity_input = native.model.TransformSpec()
        identity_input.type = "identity"
        identity_input.input_dimension = self.input_dimension
        metadata.input_transform = identity_input
        identity_output = native.model.TransformSpec()
        identity_output.type = "identity"
        identity_output.input_dimension = self.output_size
        metadata.output_transform = identity_output
        if input_transform is not None:
            if int(input_transform.input_dimension) != self.input_dimension:
                raise ValueError("input_transform dimension must match the model input dimension.")

            def transform_spec(value):
                spec = native.model.TransformSpec()
                spec.input_dimension = int(value.input_dimension)
                if isinstance(value, native.transforms.LayerwiseConcentrationNormalizer):
                    spec.type = "layerwise_concentration_normalizer"
                    spec.concentration_groups = [list(group) for group in value.concentration_groups]
                elif isinstance(value, native.transforms.ParameterBoundMinMaxScaler):
                    spec.type = "parameter_bound_min_max_scaler"
                    spec.minimum = list(value.minimum)
                    spec.scale = list(value.scale)
                    spec.low = float(value.low)
                    spec.high = float(value.high)
                elif isinstance(value, native.transforms.MinMaxScaler):
                    spec.type = "min_max_scaler"
                    spec.minimum = list(value.minimum)
                    spec.scale = list(value.scale)
                    spec.low = float(value.low)
                    spec.high = float(value.high)
                elif isinstance(value, native.transforms.TransformPipeline):
                    spec.type = "pipeline"
                    spec.transforms = [transform_spec(child) for child in value.transforms]
                else:
                    raise TypeError(f"Unsupported native input transform: {type(value).__name__}")
                return spec

            metadata.input_transform = transform_spec(input_transform)
        elif input_minimum is not None:
            transform = native.model.TransformSpec()
            transform.type = "min_max_scaler"
            transform.input_dimension = self.input_dimension
            transform.minimum = [float(x) for x in input_minimum]
            transform.scale = [float(x) for x in input_scale]
            transform.low = 0.0
            transform.high = 1.0
            metadata.input_transform = transform
        if output_inverse_factor != 1.0:
            transform = native.model.TransformSpec()
            transform.type = "constant_factor"
            transform.input_dimension = self.output_size
            transform.factor = float(output_inverse_factor)
            metadata.output_transform = transform
        forward_metadata = native.model.ForwardModelMetadata()
        forward_metadata.bare_spectrum_corrections = bare_spectrum_corrections
        forward_metadata.Apply_pileup_on_inference = Apply_pileup_on_inference
        forward_metadata.pileup_fudge_factor_seconds = pileup_fudge_factor_seconds
        forward_metadata.sample_template = config.sample
        forward_metadata.setup_template = config.setup
        forward_metadata.input_parameters = config.parameters
        spectra = []
        for label in self.spectrum_labels:
            spectrum = native.model.SpectrumSpec()
            spectrum.label = label
            spectrum.length = self.output_spectra_lengths[label]
            spectra.append(spectrum)
        forward_metadata.output_spectra = spectra
        metadata.forward = forward_metadata
        with TemporaryDirectory(prefix="ibeamlab-lrn-") as directory:
            onnx_path = Path(directory) / "model.onnx"
            was_training = self.training
            try:
                original_device = next(self.parameters()).device
            except StopIteration:
                original_device = torch.device("cpu")
            self.to("cpu")
            self.eval()
            try:
                torch.onnx.export(
                    self,
                    self.example_input(),
                    str(onnx_path),
                    input_names=["inputs"],
                    output_names=["outputs"],
                    dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
                    opset_version=opset_version,
                    do_constant_folding=True,
                    dynamo=False,
                )
            finally:
                self.to(original_device)
                if was_training:
                    self.train()
            package = native.model.ModelPackage.from_onnx(onnx_path, metadata)
            output_path = Path(path)
            package.write(output_path)
            return output_path

    def example_input(self, batch_size: int = 1) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        return torch.zeros((batch_size, self.input_dimension), dtype=torch.float32)


__all__ = ["LRNModel"]
