"""Deterministic CNN/MLP mapping spectra to frozen LRN_AD input coordinates."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpectrumEncoder(nn.Module):
    """Keep energy positions; emit nonnegative scaled, layer-major elemental AD."""

    def __init__(self, input_size, output_size, *, cnn_channels=(32, 64, 128, 128),
                 compression_channels=16, hidden_sizes=(512, 256), kernel_size=7,
                 initial_density=0.01, input_mean=None, input_std=None):
        super().__init__()
        dimensions = (input_size, output_size, compression_channels, *cnn_channels, *hidden_sizes)
        if not cnn_channels or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in dimensions):
            raise ValueError("Encoder dimensions must be positive integers")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("Use a positive odd kernel size")
        if not math.isfinite(initial_density) or initial_density <= 0:
            raise ValueError("initial_density must be finite and positive")
        self.input_size, self.output_size = input_size, output_size
        self.architecture = dict(cnn_channels=tuple(cnn_channels), compression_channels=compression_channels,
                                 hidden_sizes=tuple(hidden_sizes), kernel_size=kernel_size,
                                 initial_density=initial_density)
        mean = torch.zeros(input_size) if input_mean is None else torch.as_tensor(input_mean).float().clone()
        std = torch.ones(input_size) if input_std is None else torch.as_tensor(input_std).float().clone()
        if mean.shape != (input_size,) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Input normalization must be finite, positive and match channel count")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        blocks, previous, width = [], 1, input_size
        for channels in cnn_channels:
            groups = next(g for g in range(min(8, channels), 0, -1) if channels % g == 0)
            blocks.extend([nn.Conv1d(previous, channels, kernel_size, stride=2, padding=kernel_size//2),
                           nn.GroupNorm(groups, channels), nn.SiLU()])
            previous, width = channels, (width + 1)//2
        blocks.extend([nn.Conv1d(previous, compression_channels, 1), nn.SiLU(), nn.Flatten()])
        self.cnn = nn.Sequential(*blocks)
        head, previous = [], width * compression_channels
        for hidden in hidden_sizes:
            head.extend([nn.Linear(previous, hidden), nn.SiLU()])
            previous = hidden
        self.mlp = nn.Sequential(*head)
        self.output = nn.Linear(previous, output_size)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.constant_(self.output.bias, initial_density + math.log(-math.expm1(-initial_density)))

    def forward(self, counts):
        normalized = (torch.log1p(counts) - self.input_mean) / self.input_std
        return F.softplus(self.output(self.mlp(self.cnn(normalized.unsqueeze(1)))))

    def export(self, path, *, study, method, elements, ad_input_factor,
               opset_version=17, need_pileup_subtraction=True,
               pileup_fudge_factor_seconds=0.4e-6):
        """Package encoder-only ONNX as a native inverse model yielding physical AD.

        Normalization is embedded. Output inverse scaling converts surrogate
        coordinates to AD. This deterministic encoder predicts no uncertainty.
        Export a CPU copy so live training weights/device remain untouched.
        """
        import copy
        import inspect
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from ._native import native

        if not math.isfinite(ad_input_factor) or ad_input_factor <= 0:
            raise ValueError("ad_input_factor must be finite and positive")
        if self.output_size != len(study.experiment.sample.layers) * len(elements):
            raise ValueError("Output dimensions must match the layer/element layout")
        if [detector.label for detector in study.experiment.detectors] != [method]:
            raise ValueError("Encoder export requires the matching single-detector study")
        if not isinstance(need_pileup_subtraction, bool) or not math.isfinite(pileup_fudge_factor_seconds) or pileup_fudge_factor_seconds < 0:
            raise ValueError("Invalid pileup correction settings")
        metadata = native.model.ModelMetadata()
        metadata.model_type = native.model.ModelType.INVERSE
        metadata.class_name = type(self).__name__
        metadata.input_dimension, metadata.output_dimension = self.input_size, self.output_size
        metadata.opset_version = opset_version
        transform = native.model.TransformSpec()
        transform.input_dimension = self.input_size
        metadata.input_transform = transform
        transform = native.model.TransformSpec()
        transform.type = "constant_factor"
        transform.input_dimension, transform.factor = self.output_size, ad_input_factor
        metadata.output_transform = transform
        inverse = native.model.InverseModelMetadata()
        config = study._config()
        inverse.sample_template, inverse.setup_template = config.sample, config.setup
        inverse.need_pileup_subtraction = need_pileup_subtraction
        inverse.pileup_fudge_factor_seconds = pileup_fudge_factor_seconds
        inverse.presence_probability_output_name = ""
        inverse.posterior_std_output_name = ""
        inverse.uncertainty_predicted = False
        spectrum = native.model.SpectrumSpec()
        spectrum.label, spectrum.length = method, self.input_size
        inverse.input_spectra = [spectrum]
        edp = native.model.EdpSpec()
        edp.max_layers, edp.elements = len(study.experiment.sample.layers), list(elements)
        inverse.output_edp = edp
        metadata.inverse = inverse
        exported = copy.deepcopy(self).cpu().float().eval()
        kwargs = dict(input_names=["inputs"], output_names=["outputs"],
                      dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
                      opset_version=opset_version, do_constant_folding=True)
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kwargs["dynamo"] = False
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="ibeamlab-encoder-") as directory:
            onnx_path = Path(directory) / "model.onnx"
            torch.onnx.export(exported, torch.zeros(1, self.input_size), str(onnx_path), **kwargs)
            native.model.ModelPackage.from_onnx(onnx_path, metadata).write(destination)
        return destination


class SpectrumGRUEncoder(SpectrumEncoder):
    """CNN layer-slot maps, LayerNorm, and a bidirectional layer GRU AD head."""

    def __init__(self, input_size, output_size, *, max_layers, elements_per_layer,
                 cnn_channels=(32, 64, 128, 128), compression_channels=None,
                 gru_hidden_size=256, layer_embedding_size=16, bidirectional=True,
                 dropout=0.0, kernel_size=7, initial_density=0.01,
                 input_mean=None, input_std=None):
        dimensions = (max_layers, elements_per_layer, gru_hidden_size, layer_embedding_size)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in dimensions):
            raise ValueError("Layer/GRU dimensions must be positive integers")
        if output_size != max_layers * elements_per_layer:
            raise ValueError("Output width must equal layers times elements")
        if compression_channels is None:
            compression_channels = max_layers
        if compression_channels != max_layers:
            raise ValueError("Use one CNN output channel per layer slot")
        if not isinstance(bidirectional, bool):
            raise ValueError("bidirectional must be a boolean")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        super().__init__(input_size, output_size, cnn_channels=cnn_channels,
                         compression_channels=compression_channels, hidden_sizes=(),
                         kernel_size=kernel_size, initial_density=initial_density,
                         input_mean=input_mean, input_std=input_std)
        slot_width = self.output.in_features // max_layers
        self.cnn = self.cnn[:-1]  # Keep [batch, layer slots, downsampled energy].
        del self.mlp
        self.directions = 2 if bidirectional else 1
        self.gru_hidden_size, self.max_layers = gru_hidden_size, max_layers
        self.slot_norm = nn.LayerNorm(slot_width)
        self.dropout = nn.Dropout(dropout)
        self.layer_embedding = nn.Embedding(max_layers, layer_embedding_size)
        self.gru = nn.GRU(slot_width + layer_embedding_size, gru_hidden_size,
                          num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.output = nn.Linear(self.directions * gru_hidden_size, elements_per_layer)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.constant_(self.output.bias, initial_density + math.log(-math.expm1(-initial_density)))
        self.architecture = dict(max_layers=max_layers, elements_per_layer=elements_per_layer,
            cnn_channels=tuple(cnn_channels), compression_channels=compression_channels,
            gru_hidden_size=gru_hidden_size, layer_embedding_size=layer_embedding_size,
            bidirectional=bidirectional, dropout=dropout,
            kernel_size=kernel_size, initial_density=initial_density)

    def forward(self, counts):
        normalized = (torch.log1p(counts) - self.input_mean) / self.input_std
        slots = self.dropout(self.slot_norm(self.cnn(normalized.unsqueeze(1))))
        positions = self.layer_embedding(torch.arange(self.max_layers, device=counts.device))
        inputs = torch.cat((slots, positions.unsqueeze(0).expand(counts.shape[0], -1, -1)), dim=-1)
        initial = counts.new_zeros((self.directions, counts.shape[0], self.gru_hidden_size))
        hidden, _ = self.gru(inputs, initial)
        return F.softplus(self.output(self.dropout(hidden))).reshape(counts.shape[0], self.output_size)


class _QueryAttention(nn.Module):
    """Multihead attention with fused eager execution and portable ONNX math."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.head_size, self.dropout = heads, width // heads, dropout
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, query, memory):
        def split(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_size).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(memory)), split(self.value(memory))
        if torch.onnx.is_in_onnx_export():
            weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size), dim=-1)
            attended = torch.matmul(weights, v)
        else:
            attended = F.scaled_dot_product_attention(q, k, v,
                dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.heads*self.head_size)
        return self.output(attended)


class _LayerQueryBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.cross_norm, self.self_norm, self.feed_norm = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.cross_attention = _QueryAttention(width, heads, dropout)
        self.self_attention = _QueryAttention(width, heads, dropout)
        self.feedforward = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, memory):
        queries = queries + self.dropout(self.cross_attention(self.cross_norm(queries), memory))
        normalized = self.self_norm(queries)
        queries = queries + self.dropout(self.self_attention(normalized, normalized))
        return queries + self.dropout(self.feedforward(self.feed_norm(queries)))


class SpectrumAttentionEncoder(SpectrumEncoder):
    """CNN energy tokens; learned layer queries attend to spectrum and each other."""

    def __init__(self, input_size, output_size, *, max_layers, elements_per_layer,
                 cnn_channels=(32,64,128,128), attention_size=128, attention_heads=4,
                 attention_blocks=2, dropout=0.0, kernel_size=7, initial_density=0.01,
                 input_mean=None, input_std=None):
        dimensions = (max_layers, elements_per_layer, attention_size, attention_heads, attention_blocks)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in dimensions):
            raise ValueError("Attention/layer dimensions must be positive integers")
        if output_size != max_layers*elements_per_layer or attention_size % attention_heads:
            raise ValueError("Output layout must match layers/elements; attention width must divide by head count")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        super().__init__(input_size, output_size, cnn_channels=cnn_channels,
                         compression_channels=attention_size, hidden_sizes=(),
                         kernel_size=kernel_size, initial_density=initial_density,
                         input_mean=input_mean, input_std=input_std)
        tokens = self.output.in_features//attention_size
        self.cnn = self.cnn[:-1]
        del self.mlp
        self.energy_positions = nn.Parameter(torch.randn(1,tokens,attention_size)*0.02)
        self.layer_queries = nn.Parameter(torch.randn(1,max_layers,attention_size)*0.02)
        self.memory_norm = nn.LayerNorm(attention_size)
        self.blocks = nn.ModuleList([_LayerQueryBlock(attention_size,attention_heads,dropout) for _ in range(attention_blocks)])
        self.output_norm = nn.LayerNorm(attention_size)
        self.output = nn.Linear(attention_size,elements_per_layer)
        nn.init.normal_(self.output.weight,std=0.001)
        nn.init.constant_(self.output.bias,initial_density+math.log(-math.expm1(-initial_density)))
        self.architecture = dict(max_layers=max_layers,elements_per_layer=elements_per_layer,
            cnn_channels=tuple(cnn_channels),attention_size=attention_size,attention_heads=attention_heads,
            attention_blocks=attention_blocks,dropout=dropout,kernel_size=kernel_size,initial_density=initial_density)

    def encode_layers(self,counts):
        normalized = (torch.log1p(counts)-self.input_mean)/self.input_std
        memory = self.memory_norm(self.cnn(normalized.unsqueeze(1)).transpose(1,2)+self.energy_positions)
        queries = self.layer_queries.expand(counts.shape[0],-1,-1)
        for block in self.blocks:
            queries = block(queries,memory)
        return self.output_norm(queries)

    def forward(self,counts):
        return F.softplus(self.output(self.encode_layers(counts))).reshape(counts.shape[0],self.output_size)


class SpectrumPresenceEncoder(SpectrumAttentionEncoder):
    """Sample-level element gates shared by every layer, with straight-through training."""

    def __init__(self, input_size, output_size, **architecture):
        super().__init__(input_size, output_size, **architecture)
        width = self.architecture['attention_size']
        elements = self.architecture['elements_per_layer']
        self.presence_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, elements))
        # Initially retain the protected encoder's complete AD prediction.
        nn.init.zeros_(self.presence_head[-1].weight)
        nn.init.constant_(self.presence_head[-1].bias, math.log(99.0))
        self.register_buffer('presence_thresholds', torch.full((elements,), 0.5))

    def forward_components(self, counts):
        features = self.encode_layers(counts)
        conditional_ad = F.softplus(self.output(features))
        logits = self.presence_head(features.mean(dim=1))
        probabilities = logits.sigmoid()
        hard = (probabilities >= self.presence_thresholds).to(probabilities.dtype)
        gates = hard.detach() - probabilities.detach() + probabilities if self.training else hard
        gated_ad = conditional_ad * gates.unsqueeze(1)
        return gated_ad.reshape(counts.shape[0], self.output_size), logits, probabilities

    def forward(self, counts):
        return self.forward_components(counts)[0]


def binary_entmax15(logits):
    """Exact 1.5-entmax for independent [absent, present] score pairs."""
    difference = ((logits[..., 1] - logits[..., 0]) / 2).clamp(-1, 1)
    present = 0.5 + 0.5 * difference * torch.sqrt(2 - difference.square())
    return torch.stack((1 - present, present), dim=-1)


def binary_entmax15_loss(logits, presence):
    """Mean Fenchel–Young loss; score gradient is entmax(logits) - target."""
    probabilities = binary_entmax15(logits)
    target = torch.stack((1 - presence, presence), dim=-1)
    entropy = (4 / 3) * (1 - probabilities.pow(1.5).sum(dim=-1))
    return ((logits * (probabilities - target)).sum(dim=-1) + entropy).mean()


def sample_element_presence(cell_presence):
    """Smooth OR over layers, a surrogate for distinct-element presence."""
    return 1 - (1 - cell_presence).prod(dim=1)


def distinct_element_penalty(cell_presence):
    """Mean expected element count per sample, not a count of occupied cells."""
    return sample_element_presence(cell_presence).sum(dim=-1).mean()


class SpectrumEntmaxEncoder(SpectrumAttentionEncoder):
    """Per-cell sparse presence times conditional AD; no rounding or stochastic gates."""

    def __init__(self, input_size, output_size, **architecture):
        super().__init__(input_size, output_size, **architecture)
        self.presence_head = nn.Linear(self.architecture['attention_size'],
                                       2 * self.architecture['elements_per_layer'])
        # Start within the differentiable support, rather than saturating all gates.
        nn.init.normal_(self.presence_head.weight, std=0.001)
        nn.init.zeros_(self.presence_head.bias)

    def forward_components(self, counts):
        features = self.encode_layers(counts)
        conditional_ad = F.softplus(self.output(features))
        logits = self.presence_head(features).reshape(
            counts.shape[0], self.architecture['max_layers'],
            self.architecture['elements_per_layer'], 2)
        presence = binary_entmax15(logits)[..., 1]
        ad = conditional_ad * presence
        return ad.reshape(counts.shape[0], self.output_size), logits, presence

    def forward(self, counts):
        return self.forward_components(counts)[0]
