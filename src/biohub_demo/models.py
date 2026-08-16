"""Checkpoint-compatible Temporal 3-D U-Net and node transformer."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
    )


class _TemporalAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, times, channels = x.shape[:3]
        spatial = x.shape[3:]
        size = math.prod(spatial)
        values = x.reshape(batch, times, channels, size).permute(0, 3, 1, 2).reshape(batch * size, times, channels)
        values, _ = self.attn(self.norm(values), self.norm(values), self.norm(values), need_weights=False)
        values = values.reshape(batch, size, times, channels).permute(0, 2, 3, 1).reshape(batch, times, channels, *spatial)
        return x + values


class TemporalUNet3D(nn.Module):
    def __init__(
        self, in_channels: int = 1, out_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128), skip_fullres_temporal: bool = True,
    ) -> None:
        super().__init__()
        widths = list(layers)
        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        previous = in_channels
        for index, width in enumerate(widths):
            self.encoder_blocks.append(_conv_block(previous, width))
            self.temporal_blocks.append(nn.Identity() if skip_fullres_temporal and index == 0 else _TemporalAttention(width))
            previous = width
        self.pool = nn.MaxPool3d(2, 2)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for index in range(len(widths) - 1, 0, -1):
            self.upsamples.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            self.decoder_blocks.append(_conv_block(widths[index] + widths[index - 1], widths[index - 1]))
        self.head = nn.Conv3d(widths[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, times = x.shape[:2]
        value = x.reshape(batch * times, *x.shape[2:])
        skips: list[torch.Tensor] = []
        for index, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):
            if index:
                value = self.pool(value)
            value = block(value)
            value = temporal(value.reshape(batch, times, *value.shape[1:])).reshape(batch * times, *value.shape[1:])
            if index < len(self.encoder_blocks) - 1:
                skips.append(value)
        for upsample, block, skip in zip(self.upsamples, self.decoder_blocks, reversed(skips)):
            value = upsample(value)
            if value.shape[2:] != skip.shape[2:]:
                value = F.interpolate(value, size=skip.shape[2:], mode="trilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        value = self.head(value)
        return value.reshape(batch, times, *value.shape[1:])


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attention, _ = self.cross_attn(
            self.norm1(query), self.norm1(key_value), self.norm1(key_value), need_weights=False
        )
        query = query + attention
        return query + self.mlp(self.norm2(query))


class SimpleNodeTransformer(nn.Module):
    def __init__(
        self, feat_dim: int = 64, hidden_dim: int = 128, heads: int = 4,
        blocks: int = 4, dropout: float = 0.0, pair_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, heads, dropout) for _ in range(blocks)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, feat_t, feat_t1, coords_t, coords_t1):
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t, feat_t1 = feat_t[None], feat_t1[None]
            coords_t, coords_t1 = coords_t[None], coords_t1[None]
        query, key = self.norm_in(self.proj(feat_t)), self.norm_in(self.proj(feat_t1))
        for block in self.blocks:
            query = block(query, key)
            key = block(key, query)
        query, key = self.norm_out(query), self.norm_out(key)
        chunks = []
        for start in range(0, query.shape[1], self.pair_chunk_size):
            query_chunk = query[:, start:start + self.pair_chunk_size]
            coord_chunk = coords_t[:, start:start + self.pair_chunk_size]
            target_chunks = []
            for target_start in range(0, key.shape[1], self.pair_chunk_size):
                key_chunk = key[:, target_start:target_start + self.pair_chunk_size]
                target_coord_chunk = coords_t1[:, target_start:target_start + self.pair_chunk_size]
                query_expanded = query_chunk[:, :, None].expand(-1, -1, key_chunk.shape[1], -1)
                key_expanded = key_chunk[:, None].expand(-1, query_chunk.shape[1], -1, -1)
                relative = (coord_chunk[:, :, None] - target_coord_chunk[:, None]) / 100.0
                target_chunks.append(
                    self.pair_mlp(torch.cat((query_expanded, key_expanded, relative), -1)).squeeze(-1)
                )
            chunks.append(torch.cat(target_chunks, dim=2))
        logits = torch.cat(chunks, dim=1)
        return logits.squeeze(0) if unbatched else logits


class BiohubModel(nn.Module):
    """Names intentionally match the public ``MyUnet`` checkpoint."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.D = nn.Parameter(torch.ones(1))
        channels = int(config.get("unet_out_channels", 32))
        self.unet = TemporalUNet3D(1, channels, tuple(config.get("unet_layers", (32, 64, 128))))
        self.unet_out_channels = channels
        self.detect_head = nn.Conv3d(channels, 1, 1)
        self.transformer = SimpleNodeTransformer(
            feat_dim=channels + 32, hidden_dim=int(config.get("transformer_hidden_dim", 128)),
            heads=int(config.get("transformer_heads", 4)), blocks=int(config.get("transformer_blocks", 4)),
            dropout=0.0, pair_chunk_size=int(config.get("pair_chunk_size", 32)),
        )

    def forward_unet(self, image: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features = self.unet(image[:, :, None])
        return [features[:, 0], features[:, 1]], [self.detect_head(features[:, 0]), self.detect_head(features[:, 1])]

    def forward_transformer(self, feature0, feature1, coord0, coord1, pos0, pos1):
        return self.transformer(torch.cat((feature0, pos0), -1), torch.cat((feature1, pos1), -1), coord0, coord1)


def position_embedding(coords: torch.Tensor, t: int, image_shape: Sequence[int], time_length: int, width: int = 8) -> torch.Tensor:
    z, y, x = coords.float().unbind(dim=1)
    values = (
        torch.full_like(z, float(t) / max(1, time_length)),
        z / image_shape[0], y / image_shape[1], x / image_shape[2],
    )
    frequencies = 2.0 ** torch.arange(width // 2, dtype=coords.dtype, device=coords.device)
    encoded = []
    for value in values:
        angles = value[:, None] * frequencies[None] * torch.pi
        encoded.append(torch.cat((torch.sin(angles), torch.cos(angles)), dim=1))
    return torch.cat(encoded, dim=1)


def select_features(feature: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    _, z_size, y_size, x_size = feature.shape
    z = coords[:, 0].long().clamp(0, z_size - 1)
    y = coords[:, 1].long().clamp(0, y_size - 1)
    x = coords[:, 2].long().clamp(0, x_size - 1)
    return feature[:, z, y, x].permute(1, 0).contiguous()
