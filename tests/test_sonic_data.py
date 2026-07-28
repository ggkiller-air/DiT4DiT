from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from DiT4DiT.dataloader.gr00t_lerobot.data_config import UnitreeG1SonicDataConfig
from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset

DATASET_PATH = Path("/root/Projects/data/carry-bucket-stereo")


def _data_config(**overrides):
    values = {
        "data_root_dir": str(DATASET_PATH.parent),
        "data_mix": "carry_bucket_stereo",
        "lerobot_version": "v2.0",
        "action_mode": "abs",
        "include_state": True,
        "tactile_mode": "dream",
        "dream_horizon": 4,
        "dream_state": True,
        "dream_vision": True,
        "vision_horizon": 4,
        "video_delta_indices": [0, 1, 2, 3, 4],
        "action_video_freq_ratio": 1,
        "max_state_dim": 46,
        "max_action_dim": 78,
        "video_backend": "torchvision_av",
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_sonic_modalities_follow_ablation_mode():
    config = UnitreeG1SonicDataConfig()
    notac = config.modality_config_for(_data_config(tactile_mode="notac"))
    assert "tactile" not in notac
    assert notac["state"].delta_indices == [0]

    input_only = config.modality_config_for(_data_config(tactile_mode="input"))
    assert input_only["tactile"].delta_indices == [0]

    dream = config.modality_config_for(_data_config())
    assert dream["tactile"].delta_indices == list(range(5))
    assert dream["state"].delta_indices == list(range(5))
    assert dream["video"].delta_indices == list(range(5))
    assert dream["action"].delta_indices == list(range(40))

    longer_vision = config.modality_config_for(
        _data_config(video_delta_indices=None, dream_horizon=2, vision_horizon=4)
    )
    assert longer_vision["tactile"].delta_indices == [0, 1, 2]
    assert longer_vision["video"].delta_indices == [0, 1, 2, 3, 4]


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="carry-bucket-stereo is not installed")
def test_real_sonic_sample_and_episode_tail_padding():
    mixture = get_vla_dataset(_data_config(), mode="eval")
    single = mixture.datasets[0]
    trajectory_id = int(single.trajectory_ids[0])
    mixture.sample_step = lambda _index: (single, trajectory_id, 0)
    sample = mixture[0]
    assert sample["action"].shape == (40, 78)
    assert sample["state"].shape == (5, 46)
    assert sample["tactile"].shape == (5, 256)
    assert sample["tactile"].dtype == np.uint8
    assert len(sample["image"]) == 5
    assert all(tuple(image.shape) == (3, 224, 448) for image in sample["image"])

    raw = single.get_step_data(trajectory_id, 0)

    def resized(frame):
        tensor = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float() / 255.0
        return F.interpolate(
            tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)

    assert torch.equal(sample["image"][0][:, :, :224], resized(raw["video.ego_view_left"][0]))
    assert torch.equal(sample["image"][0][:, :, 224:], resized(raw["video.ego_view_right"][0]))

    tail_index = int(single.trajectory_lengths[0]) - 1
    mixture.sample_step = lambda _index: (single, trajectory_id, tail_index)
    tail = mixture[0]
    assert all(np.array_equal(tail["tactile"][0], value) for value in tail["tactile"][1:])
    assert all(np.array_equal(tail["state"][0], value) for value in tail["state"][1:])
    current = tail["image"][0].numpy()
    assert all(np.array_equal(current, frame.numpy()) for frame in tail["image"][1:])
