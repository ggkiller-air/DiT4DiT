"""Canonical SONIC websocket adapter for DiT4DiT checkpoints."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

PROTOCOL = "sonic_vla_v1"
STATE_DIM = 46
ACTION_HORIZON = 40
ACTION_DIM = 78
TACTILE_DIM = 256
VIDEO_KEYS = ("ego_view_left", "ego_view_right")


def validate_observation(observation: Mapping[str, Any], *, requires_tactile: bool) -> dict:
    state = np.asarray(observation.get("state"))
    if state.dtype != np.float32 or state.shape != (STATE_DIM,):
        raise ValueError(f"state must be float32[{STATE_DIM}], got {state.dtype} {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or infinity")

    result = {"state": state}
    image_shape = None
    for key in VIDEO_KEYS:
        image = np.asarray(observation.get(key))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be uint8[H, W, 3], got {image.dtype} {image.shape}")
        image_shape = image.shape if image_shape is None else image_shape
        if image.shape != image_shape:
            raise ValueError("Stereo images must have identical shapes")
        result[key] = image

    prompt = observation.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    result["prompt"] = prompt

    tactile_value = observation.get("tactile")
    if requires_tactile and tactile_value is None:
        raise ValueError("This DiT4DiT checkpoint requires tactile uint8[256]")
    if tactile_value is not None:
        tactile = np.asarray(tactile_value)
        if tactile.dtype != np.uint8 or tactile.shape != (TACTILE_DIM,):
            raise ValueError(
                f"tactile must be uint8[{TACTILE_DIM}], got {tactile.dtype} {tactile.shape}"
            )
        result["tactile"] = tactile
    return result


def validate_actions(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"actions must have shape ({ACTION_HORIZON}, {ACTION_DIM}), got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN or infinity")
    return actions


class SonicQ99Normalizer:
    """Apply the q01/q99 transforms used by the SONIC training DataConfig."""

    def __init__(self, state_stats: Mapping[str, Any], action_stats: Mapping[str, Any]) -> None:
        self._state_q01, self._state_q99 = self._bounds(state_stats, STATE_DIM, "state")
        self._action_q01, self._action_q99 = self._bounds(action_stats, ACTION_DIM, "action")

    @staticmethod
    def _bounds(stats: Mapping[str, Any], dim: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        q01 = np.asarray(stats.get("q01"), dtype=np.float32)
        q99 = np.asarray(stats.get("q99"), dtype=np.float32)
        if q01.shape != (dim,) or q99.shape != (dim,):
            raise ValueError(
                f"SONIC {name} q01/q99 must both have shape ({dim},), "
                f"got {q01.shape} and {q99.shape}"
            )
        if not np.isfinite(q01).all() or not np.isfinite(q99).all():
            raise ValueError(f"SONIC {name} statistics contain NaN or infinity")
        return q01, q99

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        varying = self._state_q01 != self._state_q99
        normalized = state.copy()
        normalized[varying] = (
            2.0
            * (state[varying] - self._state_q01[varying])
            / (self._state_q99[varying] - self._state_q01[varying])
            - 1.0
        )
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = validate_actions(actions)
        return (
            (actions + 1.0) / 2.0 * (self._action_q99 - self._action_q01)
            + self._action_q01
        ).astype(np.float32, copy=False)


def make_stereo_condition(left: np.ndarray, right: np.ndarray) -> torch.Tensor:
    """Reproduce training's 224x224-per-eye horizontal stereo packing."""
    views = []
    for image in (left, right):
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        views.append(tensor)
    return torch.cat(views, dim=-1).contiguous()


class SonicPolicyAdapter:
    def __init__(self, policy, model_config: Mapping[str, Any], norm_stats, unnorm_key=None) -> None:
        self._policy = policy
        action_cfg = model_config["framework"]["action_model"]
        if int(action_cfg["action_dim"]) != ACTION_DIM:
            raise ValueError(f"SONIC checkpoint action_dim must be {ACTION_DIM}")
        if int(action_cfg["action_horizon"]) != ACTION_HORIZON:
            raise ValueError(f"SONIC checkpoint action_horizon must be {ACTION_HORIZON}")
        if int(action_cfg["state_dim"]) != STATE_DIM:
            raise ValueError(f"SONIC checkpoint state_dim must be {STATE_DIM}")
        self.requires_tactile = str(action_cfg.get("tactile_mode", "notac")).lower() != "notac"

        resolved_key = policy._check_unnorm_key(norm_stats, unnorm_key)
        selected_stats = norm_stats[resolved_key]
        self._normalizer = SonicQ99Normalizer(
            selected_stats["state"], selected_stats["action"]
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "backend": "dit4dit",
            "state_dim": STATE_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "video_keys": list(VIDEO_KEYS),
            "requires_tactile": self.requires_tactile,
            "action_layout": {
                "motion_token": [0, 64],
                "left_hand_joints": [64, 71],
                "right_hand_joints": [71, 78],
            },
        }

    def infer(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        obs = validate_observation(observation, requires_tactile=self.requires_tactile)
        example = {
            "image": [make_stereo_condition(obs["ego_view_left"], obs["ego_view_right"])],
            "lang": obs["prompt"],
            "state": self._normalizer.normalize_state(obs["state"]),
        }
        if "tactile" in obs:
            example["tactile"] = obs["tactile"]

        output = self._policy.predict_action(examples=[example])
        batched_actions = np.asarray(output["normalized_actions"])
        if batched_actions.shape != (1, ACTION_HORIZON, ACTION_DIM):
            raise ValueError(
                "DiT4DiT policy must return normalized actions with shape "
                f"(1, {ACTION_HORIZON}, {ACTION_DIM}), got {batched_actions.shape}"
            )
        actions = self._normalizer.unnormalize_actions(batched_actions[0])
        return {"actions": validate_actions(actions)}
