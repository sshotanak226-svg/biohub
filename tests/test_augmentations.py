import numpy as np
import torch

from biohub_demo.augmentations import intensity_distribution_augment, xy_dihedral_augment


def test_xy_augmentation_preserves_valid_coordinate_domain() -> None:
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    coords = torch.tensor([
        [[1.0, 0.0, 0.0], [2.0, 3.0, 3.0]],
        [[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
    ])
    masks = torch.tensor([[True, True], [True, False]])
    output, transformed, output_masks = xy_dihedral_augment(
        images, coords, masks, rng=np.random.default_rng(4)
    )
    assert output.shape == images.shape
    assert torch.equal(output_masks, masks)
    selected = transformed[masks]
    assert torch.all(selected[:, 1:] >= 0)
    assert torch.all(selected[:, 1:] <= 3)
    assert torch.equal(transformed[~masks], torch.zeros_like(transformed[~masks]))


def test_intensity_augmentation_is_finite_and_bounded() -> None:
    images = torch.linspace(0, 4, 64).reshape(1, 4, 4, 4)
    coords = torch.zeros(1, 1, 3)
    masks = torch.ones(1, 1, dtype=torch.bool)
    output, output_coords, output_masks = intensity_distribution_augment(
        images, coords, masks, rng=np.random.default_rng(3)
    )
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0
    assert float(output.max()) <= 4
    assert output_coords is coords
    assert output_masks is masks
