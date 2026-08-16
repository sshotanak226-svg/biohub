from __future__ import annotations

import torch

from biohub_demo.models import BiohubModel, position_embedding, select_features


def test_temporal_model_interfaces() -> None:
    model = BiohubModel({
        "unet_out_channels": 4, "unet_layers": [4, 8],
        "transformer_hidden_dim": 16, "transformer_heads": 4,
        "transformer_blocks": 1, "pair_chunk_size": 2,
    }).eval()
    image = torch.zeros((1, 2, 8, 8, 8))
    with torch.inference_mode():
        features, logits = model.forward_unet(image)
        assert logits[0].shape == (1, 1, 8, 8, 8)
        coords = torch.tensor([
            [1.0, 1.0, 1.0], [2.0, 2.0, 3.0], [3.0, 4.0, 4.0],
            [5.0, 5.0, 6.0], [6.0, 6.0, 6.0],
        ])
        selected = select_features(features[0][0], coords)
        positions = position_embedding(coords, 0, (8, 8, 8), 2)
        edge_logits = model.forward_transformer(selected, selected, coords, coords, positions, positions)
    assert edge_logits.shape == (5, 5)
