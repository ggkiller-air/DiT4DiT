# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Teli Ma/ HKUST GZ] in [2025]. 


import sys
from pathlib import Path
# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from DiT4DiT.training.trainer_utils import initialize_overwatch


logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.framework.dit4dit_jepa import DiT4DiTJEPAFrameworkMixin
from DiT4DiT.model.modules.vlm import get_backbone_model
from DiT4DiT.model.modules.action_model.ActionDiT import get_action_model, FlowmatchingActionHead
from DiT4DiT.training.trainer_utils.trainer_tools import resize_images
from DiT4DiT.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("DiT4DiT")
class DiT4DiT(DiT4DiTJEPAFrameworkMixin, baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config

        # Determine training mode from config: "video", "action", or "joint"
        training_mode = config.framework.cosmos25.training.lower() if config is not None else "action"
        self.video_fm_only = (training_mode == "video")

        self.backbone_interface = get_backbone_model(config=self.config)

        # -------- Align DiT cross-attention dim with backbone output dim --------
        # GR00T ActionHead uses `diffusion_model_cfg.cross_attention_dim` to match vl_embs' last dim.
        vl_hidden_dim = None
        if hasattr(self.backbone_interface, "model") and hasattr(self.backbone_interface.model, "config"):
            vl_hidden_dim = getattr(self.backbone_interface.model.config, "hidden_size", None)
        if vl_hidden_dim is None and hasattr(self.backbone_interface, "extractor"):
            vl_hidden_dim = getattr(self.backbone_interface.extractor, "hidden_size", None)
        if vl_hidden_dim is None:
            vl_hidden_dim = getattr(self.config.framework.cosmos25, "vl_hidden_dim", None)

        if not self.video_fm_only:
            if vl_hidden_dim is None:
                raise ValueError(
                    "Cannot infer `vl_hidden_dim` for the selected backbone. "
                    "Please set `framework.cosmos25.vl_hidden_dim` in your config."
                )
            self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

            self.future_action_window_size = config.framework.action_model.future_action_window_size
            self.past_action_window_size = config.framework.action_model.past_action_window_size
            self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
            self._init_dit4dit_jepa()
        else:
            # Video-only mode: skip action model entirely
            self.action_model = None
            self.future_action_window_size = 0
            self.past_action_window_size = 0
            self.chunk_len = 0
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B, [frame_0, frame_1, ..., frame_T-1]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        # Step 1: backbone input format
        # All video frames (condition + future) are already in batch_images;
        # build_cosmos_inputs splits them into videos (cond) and future_videos internally.
        backbone_inputs = self.backbone_interface.build_cosmos_inputs(images=batch_images, instructions=instructions)
        if backbone_inputs.get("future_videos") is not None:
            has_future_masks = ["vision_future_mask" in example for example in examples]
            if any(has_future_masks):
                if not all(has_future_masks):
                    raise ValueError("vision_future_mask must be present for every sample in a batch")
                backbone_inputs["future_video_mask"] = torch.as_tensor(
                    np.asarray([example["vision_future_mask"] for example in examples]),
                    dtype=torch.bool,
                )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone_interface(
                **backbone_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            if not self.video_fm_only:
                last_hidden = backbone_outputs.hidden_states[-1]  # [B, L, H] ##2560-4b
            else:
                last_hidden = None
            future_video_loss = getattr(backbone_outputs, "future_video_loss", None)

        # Video-only FM training: no action branch.
        if self.video_fm_only:
            if future_video_loss is None:
                raise ValueError(
                    "video_fm_only is enabled (cosmos25.training='video') but `future_video_loss` is None. "
                    "Please provide `image_next` (or future_images) and set "
                    "`framework.cosmos25.future_loss_type=flow_matching`."
                )
            return {"future_video_loss": future_video_loss}

        future_vision_target = None
        if self.action_model.dream_vision:
            future_vision_target = self._encode_future_vision_targets(
                backbone_inputs,
                dtype=last_hidden.dtype,
            )

        with torch.autocast("cuda", dtype=torch.float32):
            out = self._forward_dit4dit_action(examples, last_hidden, future_vision_target)
        if future_video_loss is not None:
            out["future_video_loss"] = future_video_loss
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with backbone (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = []
        for ex in examples:
            img = ex["image"]
            if isinstance(img, (list, tuple)) and len(img) > 0:
                batch_images.append(img)
            else:
                batch_images.append([img])
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        # Step 1: backbone input format
        backbone_inputs = self.backbone_interface.build_cosmos_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone_interface(
                **backbone_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = backbone_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self._predict_dit4dit_action(examples, last_hidden)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
