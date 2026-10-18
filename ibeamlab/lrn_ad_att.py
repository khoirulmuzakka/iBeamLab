"""Attention encoder for the elemental areal-density forward surrogate."""

import math
import torch
from torch import nn
from .lrn_ad import LRN_AD


class _AttentionBlock(nn.Module):
    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(feedforward, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, active):
        batch, count, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, count, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        keys = active[:, None, None, :]
        weights = torch.softmax(scores.masked_fill(~keys, -1e4), dim=-1)
        # Renormalization also handles an entirely empty sample without NaNs.
        weights = weights * keys.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        context = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, count, width)
        tokens = tokens + self.dropout(self.projection(context))
        tokens = tokens + self.dropout(self.ff(self.norm2(tokens)))
        return tokens * active.unsqueeze(-1).to(tokens.dtype)


class LRN_AD_ATT(LRN_AD):
    """Bidirectional attention over layers, with cumulative AD and order features.

    Inputs, scaling and physical-input ONNX export follow LRN_AD. Empty layers
    are ignored; no learned maximum-length positional table is required.
    """

    def __init__(self, study, output_spectra_lengths, *, elements,
                 ad_input_factor=1e-5, hidden_size=256, contribution_size=512,
                 setup_embedding_dim=32, attention_heads=8, attention_layers=2,
                 attention_feedforward=768, dropout=0.0,
                 decoder_hidden_sizes=(768, 768), refiner_hidden_channels=32,
                 refiner_kernel_size=9):
        if min(hidden_size, attention_heads, attention_layers, attention_feedforward) < 1:
            raise ValueError("Attention dimensions and block count must be positive")
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        architecture = dict(hidden_size=hidden_size, contribution_size=contribution_size,
            setup_embedding_dim=setup_embedding_dim, attention_heads=attention_heads,
            attention_layers=attention_layers, attention_feedforward=attention_feedforward,
            dropout=dropout, decoder_hidden_sizes=tuple(decoder_hidden_sizes),
            refiner_hidden_channels=refiner_hidden_channels, refiner_kernel_size=refiner_kernel_size)
        super().__init__(study, output_spectra_lengths, elements=elements,
            ad_input_factor=ad_input_factor, hidden_size=hidden_size,
            contribution_size=contribution_size, setup_embedding_dim=setup_embedding_dim,
            layer_embedding_dim=hidden_size, block_hidden_sizes=(),
            decoder_hidden_sizes=decoder_hidden_sizes,
            refiner_hidden_channels=refiner_hidden_channels, refiner_kernel_size=refiner_kernel_size)
        del self.layer_block
        self.architecture = architecture
        self.token_encoder = nn.Linear(len(self.elements) + 3, hidden_size)
        self.attention = nn.ModuleList([
            _AttentionBlock(hidden_size, attention_heads, attention_feedforward, dropout)
            for _ in range(attention_layers)])
        self.token_norm = nn.LayerNorm(hidden_size)
        self.contribution_head = nn.Linear(hidden_size, contribution_size)

    def encode_latent(self, inputs):
        self.validate_input_shape(inputs)
        setup, layers = self.split_inputs(inputs)
        densities = torch.stack(layers, dim=1)
        active = (densities != 0).any(dim=-1)
        thickness = densities.sum(-1)
        above = thickness.cumsum(1) - thickness
        # Physical depth cues use scaled AD; order counts only nonempty layers.
        order = active.to(inputs.dtype).cumsum(1) / 10.0
        features = torch.cat((densities, torch.log1p(above.clamp_min(0)).unsqueeze(-1),
            torch.log1p(thickness.clamp_min(0)).unsqueeze(-1), order.unsqueeze(-1)), dim=-1)
        mask = active.unsqueeze(-1).to(inputs.dtype)
        tokens = self.token_encoder(features) * mask
        for block in self.attention:
            tokens = block(tokens, active)
        tokens = self.token_norm(tokens) * mask
        pooled = tokens.sum(1) / mask.sum(1).clamp_min(1)
        contributions = (self.contribution_head(tokens) * mask).sum(1)
        return torch.cat((contributions, pooled, self._encode_setup(setup)), dim=1), setup
