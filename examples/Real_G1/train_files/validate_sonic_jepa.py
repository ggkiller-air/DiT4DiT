"""Run one real-data Cosmos/SONIC JEPA forward or backward pass."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset  # noqa: E402
from DiT4DiT.model.framework import build_framework  # noqa: E402

LOSS_NAMES = (
    "action_loss",
    "tactile_loss",
    "state_jepa_loss",
    "vision_jepa_loss",
    "future_video_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("DiT4DiT/config/real_robot/dit4dit_g1_sonic_jepa.yaml"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    return parser.parse_args()


def require_finite_scalar(name: str, value: object) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError(f"{name} must be a scalar tensor, got {type(value)}")
    if not math.isfinite(float(value.detach())):
        raise RuntimeError(f"{name} is not finite: {float(value.detach())}")
    return value


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires CUDA")

    config = OmegaConf.load(args.config)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = get_vla_dataset(config.datasets.vla_data)
    example = dataset[args.sample_index]

    expected_shapes = {
        "action": (40, 78),
        "action_mask": (40, 78),
        "state": (5, 46),
        "state_mask": (5, 46),
        "tactile": (5, 256),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(example[key].shape)
        if actual != expected:
            raise RuntimeError(f"Unexpected {key} shape: {actual} != {expected}")
        if not torch.isfinite(torch.as_tensor(example[key]).float()).all():
            raise RuntimeError(f"Real-data sample contains non-finite {key} values")
    if len(example["image"]) != 5:
        raise RuntimeError(f"Expected 5 video frames, got {len(example['image'])}")
    for frame in example["image"]:
        if tuple(frame.shape) != (3, 224, 448) or not torch.isfinite(frame).all():
            raise RuntimeError(f"Invalid stereo video frame: {tuple(frame.shape)}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = build_framework(config).to(device)
    model.train()

    outputs = model([example])
    losses = {name: require_finite_scalar(name, outputs.get(name)) for name in LOSS_NAMES}
    total_loss = losses["action_loss"] + losses["future_video_loss"]
    for name, weight in model.jepa_loss_weights().items():
        total_loss = total_loss + float(weight) * losses[name]
    require_finite_scalar("total_loss", total_loss)

    print("losses")
    for name in LOSS_NAMES:
        print(f"  {name}={float(losses[name].detach()):.8f}")
    print(f"  total_loss={float(total_loss.detach()):.8f}")

    if args.backward:
        total_loss.backward()
        required_gradients = {
            "tactile_encoder": model.action_model.tactile_encoder,
            "action_model": model.action_model.model,
            "cosmos_transformer": model.backbone_interface.extractor.transformer,
        }
        for name, module in required_gradients.items():
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            if not gradients:
                raise RuntimeError(f"No gradients reached {name}")
            if not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError(f"Non-finite gradients reached {name}")
            print(f"finite_gradient_tensors[{name}]={len(gradients)}")
        if any(parameter.grad is not None for parameter in model._vision_teacher().parameters()):
            raise RuntimeError("Frozen vision teacher unexpectedly received gradients")

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"peak_cuda_memory_gib={peak_gib:.2f}")
    print("SONIC JEPA smoke passed")


if __name__ == "__main__":
    main()
