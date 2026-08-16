"""Training augmentations for grayscale Biohub volumes.

The competition images are single-channel, so colour saturation is not a
meaningful augmentation.  These transforms instead cover the observable
train/test intensity shift with gain, gamma, bias and noise while preserving
the temporal relationship between frames in a window.
"""

from __future__ import annotations

import numpy as np
import torch


def xy_dihedral_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an XY rotation/reflection and transform node coordinates."""
    if imgs.shape[-2] != imgs.shape[-1]:
        # A 90-degree rotation changes the tensor shape for non-square planes;
        # fall back to independently sampled XY reflections in that case.
        k = 0
    else:
        k = int(rng.integers(0, 4))
    flip_y = bool(rng.integers(0, 2))
    flip_x = bool(rng.integers(0, 2))

    output = torch.rot90(imgs, k, dims=(-2, -1)) if k else imgs
    transformed = coords.clone()
    height, width = int(imgs.shape[-2]), int(imgs.shape[-1])
    y = transformed[..., 1].clone()
    x = transformed[..., 2].clone()
    if k == 1:
        transformed[..., 1] = width - 1 - x
        transformed[..., 2] = y
        height, width = width, height
    elif k == 2:
        transformed[..., 1] = height - 1 - y
        transformed[..., 2] = width - 1 - x
    elif k == 3:
        transformed[..., 1] = x
        transformed[..., 2] = height - 1 - y
        height, width = width, height

    if flip_y:
        output = output.flip(-2)
        transformed[..., 1] = height - 1 - transformed[..., 1]
    if flip_x:
        output = output.flip(-1)
        transformed[..., 2] = width - 1 - transformed[..., 2]

    # Padded coordinates must remain zero.
    transformed = torch.where(masks[..., None], transformed, torch.zeros_like(transformed))
    return output, transformed, masks


def intensity_distribution_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: np.random.Generator,
    gain_range: tuple[float, float] = (0.75, 1.30),
    gamma_range: tuple[float, float] = (0.70, 1.45),
    bias_range: tuple[float, float] = (-0.10, 0.10),
    noise_std_range: tuple[float, float] = (0.0, 0.025),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perturb grayscale intensity while keeping all frames temporally coherent."""
    gain = float(rng.uniform(*gain_range))
    gamma = float(rng.uniform(*gamma_range))
    bias = float(rng.uniform(*bias_range))
    noise_std = float(rng.uniform(*noise_std_range))
    output = imgs.float().clamp_min(0.0).pow(gamma).mul(gain).add(bias)
    if noise_std > 0:
        seed = int(rng.integers(0, 2**31 - 1))
        generator = torch.Generator(device=output.device).manual_seed(seed)
        noise = torch.randn(
            output.shape, dtype=output.dtype, device=output.device, generator=generator
        )
        output = output + noise_std * noise
    return output.clamp_(0.0, 4.0), coords, masks


def robust_augmentations() -> list:
    """Return the stronger local-training recipe used by method search."""
    return [xy_dihedral_augment, intensity_distribution_augment]
