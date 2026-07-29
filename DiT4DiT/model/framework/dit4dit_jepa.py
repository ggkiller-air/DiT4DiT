"""Tactile/state/vision JEPA plumbing for the DiT4DiT framework."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class DiT4DiTJEPAFrameworkMixin:
    _JEPA_CHECKPOINT_PREFIXES = (
        "action_model.tactile_encoder.",
        "action_model.tactile_target_encoder.",
        "action_model.tactile_dream_head.",
        "action_model.state_target_encoder.",
        "action_model.state_dream_head.",
        "action_model.vision_dream_head.",
    )
    _JEPA_DREAM_PREFIXES = (
        "action_model.tactile_target_encoder.",
        "action_model.tactile_dream_head.",
        "action_model.state_target_encoder.",
        "action_model.state_dream_head.",
        "action_model.vision_dream_head.",
    )

    def _init_dit4dit_jepa(self) -> None:
        if not self.action_model.dream_vision:
            return
        vae = self._vision_teacher()
        vae.requires_grad_(False)
        vae.eval()
        extractor = self.backbone_interface.extractor
        actual_dim = int(extractor.transformer.config.in_channels) - 1
        if actual_dim != self.action_model.vision_target_dim:
            raise ValueError(
                f"Cosmos VAE latent width {actual_dim} does not match "
                f"vision_target_dim {self.action_model.vision_target_dim}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.action_model is not None and self.action_model.dream_vision:
            vae = self._vision_teacher()
            vae.requires_grad_(False)
            vae.eval()
        return self

    def _vision_teacher(self):
        extractor = getattr(self.backbone_interface, "extractor", None)
        vae = getattr(extractor, "vae", None)
        if vae is None:
            raise RuntimeError("vision-JEPA requires the Cosmos VAE encoder")
        return vae

    def _validate_dataset_jepa_config(self) -> None:
        data_config = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        if data_config is None or data_config.get("tactile_mode", None) is None:
            return
        data_tactile_mode = str(data_config.get("tactile_mode", "notac")).lower()
        expected = {
            "tactile_mode": self.action_model.tactile_mode,
            "dream_horizon": self.action_model.dream_horizon,
            "dream_state": self.action_model.dream_state,
            "dream_vision": self.action_model.dream_vision,
            "vision_horizon": self.action_model.vision_horizon,
        }
        for key, model_value in expected.items():
            data_value = data_config.get(key, model_value)
            if isinstance(model_value, bool):
                data_value = data_tactile_mode == "dream" and bool(data_value)
            elif isinstance(model_value, int):
                data_value = int(data_value)
            else:
                data_value = str(data_value).lower()
            if data_value != model_value:
                raise ValueError(f"Dataset/model JEPA config mismatch for {key}: {data_value!r} != {model_value!r}")

    @torch.no_grad()
    def _encode_future_vision_targets(
        self,
        backbone_inputs: dict[str, Any],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        future = backbone_inputs.get("future_videos")
        expected_horizon = self.action_model.vision_horizon
        if future is None or future.ndim != 5 or future.shape[1] < expected_horizon:
            shape = None if future is None else tuple(future.shape)
            raise ValueError(
                f"vision-JEPA requires future video [B, >= {expected_horizon}, 3, H, W], got {shape}"
            )
        future = future[:, :expected_horizon]
        batch, horizon, channels, height, width = future.shape
        if channels != 3:
            raise ValueError(f"vision-JEPA expected RGB future frames, got {future.shape}")

        flat = future.reshape(batch * horizon, 1, channels, height, width)
        extractor = self.backbone_interface.extractor
        video = extractor._coerce_videos_to_bcthw(flat, height=height, width=width)
        latents = extractor._encode_video_to_latents_norm(video)
        pooled = latents.float().mean(dim=(2, 3, 4)).reshape(batch, horizon, -1)
        expected_dim = self.action_model.vision_target_dim
        if pooled.shape[-1] != expected_dim:
            raise ValueError(f"Visual target width {pooled.shape[-1]} does not match {expected_dim}")
        return pooled.to(dtype=dtype).detach()

    @staticmethod
    def _tensorize_optional(examples, key: str, device, dtype):
        if key not in examples[0]:
            if any(key in example for example in examples[1:]):
                raise ValueError(f"Inconsistent optional key {key!r} within a batch")
            return None
        if any(key not in example for example in examples):
            raise ValueError(f"Inconsistent optional key {key!r} within a batch")
        return torch.as_tensor(np.asarray([example[key] for example in examples]), device=device, dtype=dtype)

    def _forward_dit4dit_action(
        self,
        examples,
        last_hidden: torch.Tensor,
        future_vision_target: torch.Tensor | None,
    ):
        self._validate_dataset_jepa_config()
        device, dtype = last_hidden.device, last_hidden.dtype
        actions = torch.as_tensor(np.asarray([example["action"] for example in examples]), device=device, dtype=dtype)
        actions = actions[:, -self.action_model.action_horizon :]
        action_mask = torch.as_tensor(
            np.asarray([example["action_mask"] for example in examples]),
            device=device,
            dtype=dtype,
        )
        action_mask = action_mask[:, -self.action_model.action_horizon :]
        repeats = int(self.config.trainer.get("repeated_diffusion_steps", 4))

        state_window = self._tensorize_optional(examples, "state", device, dtype)
        current_state = None
        future_state = None
        if state_window is not None:
            if state_window.ndim == 2:
                current_state = state_window.unsqueeze(1)
            elif state_window.ndim == 3:
                current_state = state_window[:, :1] if self.action_model.dream_state else state_window
            else:
                raise ValueError(f"Expected state [B, D] or [B, T, D], got {state_window.shape}")
            if self.action_model.dream_state:
                expected = self.action_model.dream_horizon + 1
                if state_window.ndim != 3 or state_window.shape[1] != expected:
                    raise ValueError(f"state-JEPA requires [B, {expected}, D], got {state_window.shape}")
                future_state = state_window[:, 1:]

        tactile = self._tensorize_optional(examples, "tactile", device, dtype)
        if self.action_model.use_tactile and tactile is None:
            raise ValueError("Active tactile mode requires the tactile key in every sample")
        tactile_future_mask = self._tensorize_optional(
            examples, "tactile_future_mask", device, dtype
        )
        state_future_mask = self._tensorize_optional(
            examples, "state_future_mask", device, dtype
        )
        vision_future_mask = self._tensorize_optional(
            examples, "vision_future_mask", device, dtype
        )

        def repeat(value):
            return None if value is None else value.repeat(repeats, *([1] * (value.ndim - 1)))

        output = self.action_model(
            repeat(last_hidden),
            repeat(actions),
            repeat(action_mask),
            state=repeat(current_state),
            tactile=repeat(tactile),
            future_state=repeat(future_state),
            future_vision_target=repeat(future_vision_target),
            tactile_future_mask=repeat(tactile_future_mask),
            state_future_mask=repeat(state_future_mask),
            vision_future_mask=repeat(vision_future_mask),
        )
        return {"action_loss": output} if torch.is_tensor(output) else output

    def _predict_dit4dit_action(self, examples, last_hidden):
        device, dtype = last_hidden.device, last_hidden.dtype
        state = self._tensorize_optional(examples, "state", device, dtype)
        if state is not None:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            elif state.ndim == 3:
                if self.action_model.dream_state:
                    state = state[:, :1]
            else:
                raise ValueError(f"Expected state [B, D] or [B, T, D], got {state.shape}")
        tactile = self._tensorize_optional(examples, "tactile", device, dtype)
        return self.action_model.predict_action(last_hidden, state=state, tactile=tactile)

    def sync_jepa_teachers(self) -> None:
        if self.action_model is not None:
            self.action_model.sync_jepa_teachers()

    def update_jepa_teachers(self) -> None:
        if self.action_model is not None:
            self.action_model.update_jepa_teachers()

    def jepa_loss_weights(self) -> dict[str, float]:
        if self.action_model is None:
            return {}
        return {
            "tactile_loss": self.action_model.lambda_tactile,
            "state_jepa_loss": self.action_model.lambda_state,
            "vision_jepa_loss": self.action_model.lambda_vision,
        }

    def validate_jepa_checkpoint_keys(self, missing_keys, unexpected_keys) -> None:
        if self.action_model is None:
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    f"Video-only checkpoint mismatch: missing={list(missing_keys)}, "
                    f"unexpected={list(unexpected_keys)}"
                )
            return
        missing = set(missing_keys)
        all_jepa = {
            key
            for key in self.state_dict()
            if any(key.startswith(prefix) for prefix in self._JEPA_CHECKPOINT_PREFIXES)
        }
        dream_only = {
            key
            for key in all_jepa
            if any(key.startswith(prefix) for prefix in self._JEPA_DREAM_PREFIXES)
        }
        valid_missing_sets = {frozenset(), frozenset(all_jepa), frozenset(dream_only)}
        if frozenset(missing) not in valid_missing_sets or unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible with this JEPA model: "
                f"partial/invalid missing={sorted(missing)}, unexpected={list(unexpected_keys)}"
            )
        self._jepa_teacher_keys_missing = bool(missing)
