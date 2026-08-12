from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from DiT4DiT.dataloader.gr00t_lerobot.data_config import UnitreeG1SonicDataConfig
from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset
from DiT4DiT.model.modules.action_model.tactile_jepa import REGION_GRIDS, REGION_SIZES, VALID_IDX

DATASET_PATH = Path("/home/wzh/Projects/Uni_VLaT/data/desk_sweep")
RECOMMENDED_CONFIG = Path("DiT4DiT/config/real_robot/dit4dit_g1_sonic_jepa.yaml")
MODE_CONFIGS = {
    "notactile": Path("DiT4DiT/config/real_robot/dit4dit_g1_sonic_notactile.yaml"),
    "htd": Path("DiT4DiT/config/real_robot/dit4dit_g1_sonic_htd.yaml"),
    "jepa": RECOMMENDED_CONFIG,
}


def _data_config(**overrides):
    values = {
        "data_root_dir": str(DATASET_PATH.parent),
        "data_mix": "desk_sweep",
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

    with pytest.raises(ValueError, match="same consecutive deltas"):
        config.modality_config_for(_data_config(video_delta_indices=[0, 2, 4, 6, 8]))
    with pytest.raises(ValueError, match="action_video_freq_ratio=1"):
        config.modality_config_for(_data_config(action_video_freq_ratio=2))


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="desk_sweep is not installed")
def test_train_val_split_has_no_episode_overlap():
    cfg = _data_config(val_ratio=0.05)
    train = get_vla_dataset(cfg, mode="train", seed=42)
    val = get_vla_dataset(cfg, mode="val", seed=42)
    train_ids = {trajectory_id for trajectory_id, _ in train.datasets[0].all_steps}
    val_ids = {trajectory_id for trajectory_id, _ in val.datasets[0].all_steps}

    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(train.datasets[0].trajectory_ids)


def test_recommended_config_pins_the_isaac_tactile_layout():
    action_config = OmegaConf.load(RECOMMENDED_CONFIG).framework.action_model
    assert tuple(action_config.get("tactile_valid_idx", VALID_IDX)) == VALID_IDX
    assert tuple(action_config.get("tactile_region_sizes", REGION_SIZES)) == REGION_SIZES
    configured_grids = tuple(
        zip(
            action_config.get("tactile_region_rows", [rows for rows, _ in REGION_GRIDS]),
            action_config.get("tactile_region_cols", [cols for _, cols in REGION_GRIDS]),
            strict=True,
        )
    )
    assert configured_grids == REGION_GRIDS


def test_fixed_mode_configs_share_sonic_contract_and_have_distinct_targets():
    configs = {name: OmegaConf.load(path) for name, path in MODE_CONFIGS.items()}
    for config in configs.values():
        action = config.framework.action_model
        data = config.datasets.vla_data
        assert (action.state_dim, action.action_dim, action.action_horizon) == (46, 78, 40)
        assert action.future_action_window_size == 39
        assert (data.max_state_dim, data.max_action_dim) == (46, 78)
        assert data.action_video_freq_ratio == 1

    assert configs["notactile"].framework.action_model.tactile_mode == "notac"
    assert configs["notactile"].datasets.vla_data.video_delta_indices == [0]

    htd_model = configs["htd"].framework.action_model
    htd_data = configs["htd"].datasets.vla_data
    assert htd_model.tactile_mode == htd_data.tactile_mode == "dream"
    assert not htd_model.dream_state and not htd_model.dream_vision
    assert not htd_data.dream_state and not htd_data.dream_vision
    assert htd_data.video_delta_indices == [0]

    jepa_model = configs["jepa"].framework.action_model
    jepa_data = configs["jepa"].datasets.vla_data
    assert jepa_model.tactile_mode == jepa_data.tactile_mode == "dream"
    assert jepa_model.dream_state and jepa_model.dream_vision
    assert jepa_data.dream_state and jepa_data.dream_vision
    assert jepa_data.video_delta_indices == list(range(5))


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="desk_sweep is not installed")
def test_real_sonic_sample_and_episode_tail_padding():
    mixture = get_vla_dataset(_data_config(), mode="eval")
    single = mixture.datasets[0]
    trajectory_id = int(single.trajectory_ids[0])
    mixture.sample_step = lambda _index: (single, trajectory_id, 0)
    sample = mixture[0]
    assert sample["action"].shape == (40, 78)
    assert sample["state"].shape == (5, 46)
    assert sample["tactile"].shape == (5, 768)
    assert sample["tactile"].dtype == np.uint8
    assert len(sample["image"]) == 5
    assert all(tuple(image.shape) == (3, 224, 448) for image in sample["image"])
    assert sample["action_mask"].all()
    assert sample["tactile_future_mask"].all()
    assert sample["state_future_mask"].all()
    assert sample["vision_future_mask"].all()

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
    assert tail["action_mask"][0].all()
    assert not tail["action_mask"][1:].any()
    assert not tail["tactile_future_mask"].any()
    assert not tail["state_future_mask"].any()
    assert not tail["vision_future_mask"].any()
