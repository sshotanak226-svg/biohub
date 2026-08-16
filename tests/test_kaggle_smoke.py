from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
import zarr

from biohub_demo.kaggle import run_kaggle
from biohub_demo.models import BiohubModel


def test_kaggle_cpu_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    store = zarr.open_group(str(input_dir / "tiny.zarr"), mode="w")
    store.create_array("0", data=np.zeros((3, 8, 8, 8), dtype=np.uint16))
    store.attrs["multiscales"] = [{
        "datasets": [{"coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1, 1]}]}]
    }]
    store.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}

    model_config = {
        "unet_out_channels": 4, "unet_layers": [4, 8],
        "transformer_hidden_dim": 16, "transformer_heads": 4,
        "transformer_blocks": 1, "pair_chunk_size": 8,
    }
    model = BiohubModel(model_config)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.detect_head.bias.data.fill_(10.0)
    weight_path = tmp_path / "weights.pth"
    torch.save(model.state_dict(), weight_path)
    model_config_path = tmp_path / "config.json"
    model_config_path.write_text(__import__("json").dumps(model_config), encoding="utf-8")
    config = {
        "seed": 1, "input_dir": str(input_dir), "output_dir": str(tmp_path / "output"),
        "weights": [str(weight_path)], "model_config": str(model_config_path), "device": "cpu",
        "precision": "fp32", "subsample": [1, 1, 1], "tta": 1,
        "point_threshold": 0.99, "nms_um": 3.0,
        "edges": {"strong_threshold": 0.5, "min_probability": 0.0, "top_k_parents": 3, "max_distance_um": 100.0},
        "ilp": {"edge_weight": -1.0, "appearance_weight": 0.0, "disappearance_weight": 1.4, "division_weight": 1.0},
        "gap": {"max_added_fraction": 0.005, "max_added_absolute": 2, "max_link_um": 6.0},
        "smooth": {"weight": 0.15, "max_shift_um": 1.5},
        "division": {"parent_child_um": 7.5, "child_child_um": 8.5, "min_probability": 0.0},
        "runtime": {"target_hours": 1, "warning_hours": 1, "stop_hours": 1, "max_vram_gb": 1},
    }
    config_path = tmp_path / "kaggle.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = run_kaggle(config_path, model_config_override=model_config_path)
    assert manifest["validation"]["valid"]
    assert (tmp_path / "output" / "submission.csv").is_file()
    assert (tmp_path / "output" / "geff" / "tiny.geff").exists()
