from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from DiT4DiT.model.framework.DiT4DiT import DiT4DiT
from DiT4DiT.model.framework.dit4dit_jepa import DiT4DiTJEPAFrameworkMixin
from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead
from DiT4DiT.model.modules.vlm.Cosmos25 import _Cosmos25_Interface


class _CaptureAction(nn.Module):
    def __init__(self, dream_state=False):
        super().__init__()
        self.dream_state = dream_state
        self.dream_vision = False
        self.use_tactile = False
        self.tactile_mode = "notac"
        self.dream_horizon = 2
        self.vision_horizon = 2
        self.action_horizon = 4
        self.captured_state = None
        self.captured_future_state = None

    def forward(self, hidden, actions, action_mask, *, state, future_state, **kwargs):
        self.captured_state = None if state is None else state.detach().clone()
        self.captured_future_state = None if future_state is None else future_state.detach().clone()
        return hidden.sum() * 0.0 + actions.sum() * 0.0 + action_mask.sum() * 0.0

    def predict_action(self, hidden, *, state, **kwargs):
        self.captured_state = None if state is None else state.detach().clone()
        return hidden.new_zeros(hidden.shape[0], 4, 6)


class _StateHarness(DiT4DiTJEPAFrameworkMixin, nn.Module):
    def __init__(self, dream_state=False):
        super().__init__()
        self.action_model = _CaptureAction(dream_state=dream_state)
        self.config = OmegaConf.create(
            {
                "trainer": {"repeated_diffusion_steps": 1},
                "datasets": {"vla_data": {}},
            }
        )


def _examples(state):
    return [
        {
            "action": np.zeros((4, 6), dtype=np.float32),
            "action_mask": np.ones((4, 6), dtype=np.float32),
            "state": state[index],
        }
        for index in range(state.shape[0])
    ]


def test_notac_preserves_full_state_history_for_train_and_inference():
    harness = _StateHarness()
    state = np.arange(2 * 16 * 5, dtype=np.float32).reshape(2, 16, 5)
    examples = _examples(state)
    hidden = torch.randn(2, 3, 8)
    harness._forward_dit4dit_action(examples, hidden, None)
    assert torch.equal(harness.action_model.captured_state, torch.from_numpy(state))
    harness._predict_dit4dit_action(examples, hidden)
    assert torch.equal(harness.action_model.captured_state, torch.from_numpy(state))


def test_dream_condition_sees_current_state_and_teacher_sees_future():
    harness = _StateHarness(dream_state=True)
    state = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    harness._forward_dit4dit_action(_examples(state), torch.randn(2, 3, 8), None)
    assert torch.equal(harness.action_model.captured_state, torch.from_numpy(state[:, :1]))
    assert torch.equal(harness.action_model.captured_future_state, torch.from_numpy(state[:, 1:]))


def test_input_ablation_ignores_dream_flags_left_in_dataset_config():
    harness = _StateHarness()
    harness.action_model.tactile_mode = "input"
    harness.config.datasets.vla_data = OmegaConf.create(
        {
            "tactile_mode": "input",
            "dream_horizon": 2,
            "vision_horizon": 2,
            "dream_state": True,
            "dream_vision": True,
        }
    )
    harness._validate_dataset_jepa_config()


def test_video_only_jepa_lifecycle_hooks_are_noops():
    harness = _StateHarness()
    harness.action_model = None
    assert harness.jepa_loss_weights() == {}
    harness.sync_jepa_teachers()
    harness.update_jepa_teachers()


def test_cosmos_input_split_keeps_future_pixels_out_of_condition_video():
    interface = object.__new__(_Cosmos25_Interface)
    current = torch.zeros(3, 2, 2)
    future_a = torch.ones(3, 2, 2)
    future_b = torch.full((3, 2, 2), 2.0)
    first = interface.build_cosmos_inputs([[current, future_a]], ["move"])
    second = interface.build_cosmos_inputs([[current, future_b]], ["move"])
    assert torch.equal(first["videos"], second["videos"])
    assert not torch.equal(first["future_videos"], second["future_videos"])


def test_ablation_mode_effectively_disables_dataset_dream_flags():
    harness = _StateHarness()
    harness.action_model.tactile_mode = "input"
    harness.config.datasets.vla_data = OmegaConf.create(
        {
            "tactile_mode": "input",
            "dream_horizon": 2,
            "dream_state": True,
            "dream_vision": True,
            "vision_horizon": 2,
        }
    )
    harness._validate_dataset_jepa_config()
    harness.config.datasets.vla_data.tactile_mode = "notac"
    with pytest.raises(ValueError, match="tactile_mode"):
        harness._validate_dataset_jepa_config()


class _FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


class _FakeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = _FakeVAE()
        self.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=5))

    @staticmethod
    def _coerce_videos_to_bcthw(videos, *, height, width):
        return videos.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _encode_video_to_latents_norm(video):
        pooled = video.float().mean(dim=(1, 2, 3, 4))
        return pooled[:, None, None, None, None].repeat(1, 4, 1, 1, 1)


class _VisionHarness(DiT4DiTJEPAFrameworkMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.action_model = SimpleNamespace(dream_vision=True, vision_horizon=2, vision_target_dim=4)
        self.backbone_interface = SimpleNamespace(extractor=_FakeExtractor())


def test_future_vision_target_preserves_sample_and_time_order():
    harness = _VisionHarness()
    harness._init_dit4dit_jepa()
    assert not harness.backbone_interface.extractor.vae.training
    assert all(not parameter.requires_grad for parameter in harness.backbone_interface.extractor.vae.parameters())
    future = torch.tensor([[1.0, 3.0], [11.0, 13.0]]).reshape(2, 2, 1, 1, 1)
    future = future.repeat(1, 1, 3, 1, 1)
    target = harness._encode_future_vision_targets({"future_videos": future}, torch.float32)
    assert target.shape == (2, 2, 4)
    assert torch.equal(target[:, :, 0], torch.tensor([[1.0, 3.0], [11.0, 13.0]]))


class _SplitBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = _FakeExtractor()

    @staticmethod
    def build_cosmos_inputs(images, instructions):
        del instructions
        videos = torch.stack([sample[0] for sample in images]).unsqueeze(1)
        future = torch.stack([torch.stack(sample[1:]) for sample in images])
        return {"videos": videos, "future_videos": future}

    def forward(self, videos, future_videos, **kwargs):
        del kwargs
        hidden = videos.mean(dim=(1, 2, 3, 4)).reshape(-1, 1, 1).repeat(1, 3, 8)
        return SimpleNamespace(
            hidden_states=[hidden],
            future_video_loss=future_videos.float().mean(),
        )


class _SplitAction(nn.Module):
    dream_vision = True
    dream_state = False
    use_tactile = False
    tactile_mode = "dream"
    dream_horizon = 2
    vision_horizon = 2
    vision_target_dim = 4
    action_horizon = 2
    lambda_tactile = 0.5
    lambda_state = 0.5
    lambda_vision = 0.5

    def forward(self, hidden, actions, action_mask, *, future_vision_target, **kwargs):
        del actions, action_mask, kwargs
        return {
            "action_loss": hidden.float().mean(),
            "vision_jepa_loss": future_vision_target.float().mean(),
        }

    def update_jepa_teachers(self):
        pass


def _split_framework():
    model = object.__new__(DiT4DiT)
    nn.Module.__init__(model)
    model.video_fm_only = False
    model.backbone_interface = _SplitBackbone()
    model.action_model = _SplitAction()
    model.config = OmegaConf.create(
        {"trainer": {"repeated_diffusion_steps": 1}, "datasets": {"vla_data": {}}}
    )
    return model


def test_future_ground_truth_cannot_change_action_conditioning():
    current = torch.ones(3, 1, 1)

    def examples(future_value):
        return [
            {
                "image": [current, torch.full_like(current, future_value), torch.full_like(current, future_value)],
                "lang": "task",
                "action": np.zeros((2, 2), dtype=np.float32),
                "action_mask": np.ones((2, 2), dtype=np.float32),
                "vision_future_mask": np.ones(2, dtype=bool),
            }
        ]

    model = _split_framework()
    low = model(examples(2.0))
    high = model(examples(9.0))
    assert torch.equal(low["action_loss"], high["action_loss"])
    assert not torch.equal(low["future_video_loss"], high["future_video_loss"])
    assert not torch.equal(low["vision_jepa_loss"], high["vision_jepa_loss"])


def _action_config(mode="dream"):
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 32,
                    "action_dim": 6,
                    "state_dim": 5,
                    "future_action_window_size": 3,
                    "action_horizon": 4,
                    "num_inference_timesteps": 2,
                    "add_pos_embed": True,
                    "max_seq_len": 32,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 100,
                    "tactile_mode": mode,
                    "n_tactile_tokens": 2,
                    "tactile_hidden_dim": 16,
                    "tactile_num_heads": 2,
                    "dream_horizon": 2,
                    "vision_horizon": 2,
                    "dream_hidden_dim": 32,
                    "dream_state": True,
                    "dream_vision": True,
                    "vision_target_dim": 64,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 64,
                        "output_dim": 64,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "positional_embeddings": None,
                    },
                }
            }
        }
    )


class _CheckpointHarness(DiT4DiTJEPAFrameworkMixin, nn.Module):
    def __init__(self, mode="dream"):
        super().__init__()
        self.action_model = FlowmatchingActionHead(_action_config(mode))


def test_checkpoint_validator_accepts_only_coherent_generations():
    harness = _CheckpointHarness()
    all_jepa = {
        key
        for key in harness.state_dict()
        if any(key.startswith(prefix) for prefix in harness._JEPA_CHECKPOINT_PREFIXES)
    }
    dream_only = {
        key
        for key in all_jepa
        if any(key.startswith(prefix) for prefix in harness._JEPA_DREAM_PREFIXES)
    }
    harness.validate_jepa_checkpoint_keys([], [])
    notac_result = harness.load_state_dict(_CheckpointHarness("notac").state_dict(), strict=False)
    assert set(notac_result.missing_keys) == all_jepa
    harness.validate_jepa_checkpoint_keys(notac_result.missing_keys, notac_result.unexpected_keys)
    input_result = harness.load_state_dict(_CheckpointHarness("input").state_dict(), strict=False)
    assert set(input_result.missing_keys) == dream_only
    harness.validate_jepa_checkpoint_keys(input_result.missing_keys, input_result.unexpected_keys)
    corrupted = dict(harness.state_dict())
    corrupted.pop(next(key for key in all_jepa if key.startswith("action_model.tactile_encoder.")))
    result = harness.load_state_dict(corrupted, strict=False)
    with pytest.raises(RuntimeError, match="partial/invalid"):
        harness.validate_jepa_checkpoint_keys(result.missing_keys, result.unexpected_keys)
