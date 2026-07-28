from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from DiT4DiT.model.framework.dit4dit_jepa import DiT4DiTJEPAFrameworkMixin
from DiT4DiT.training import train
from DiT4DiT.training.trainer_utils import trainer_tools
from DiT4DiT.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


def test_build_accelerator_wires_yaml_gradient_accumulation(monkeypatch):
    captured = {}

    class _FakeAccelerator:
        state = "fake-state"

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def print(self, value):
            captured["printed"] = value

    monkeypatch.setattr(train, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(train, "DeepSpeedPlugin", lambda **kwargs: ("plugin", kwargs))
    cfg = SimpleNamespace(trainer={"gradient_accumulation_steps": 4})
    accelerator = train.build_accelerator(cfg)
    assert isinstance(accelerator, _FakeAccelerator)
    assert captured["gradient_accumulation_steps"] == 4
    assert captured["deepspeed_plugin"][0] == "plugin"
    assert captured["printed"] == "fake-state"


class _LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.teacher_updates = 0

    def forward(self, _batch):
        loss = self.weight.square()
        return {"action_loss": loss, "tactile_loss": loss}

    @staticmethod
    def jepa_loss_weights():
        return {"tactile_loss": 0.5}

    def update_jepa_teachers(self):
        self.teacher_updates += 1


class _FakeAccelerator:
    num_processes = 1
    gradient_accumulation_steps = 2
    optimizer_step_was_skipped = False

    def __init__(self):
        self.sync_gradients = False

    @staticmethod
    def accumulate(_model):
        return nullcontext()

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def backward(loss):
        loss.backward()


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _VideoOnlyModel(DiT4DiTJEPAFrameworkMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.action_model = None
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, _batch):
        return {"future_video_loss": self.weight.square()}


def test_train_step_accumulates_weighted_loss_and_updates_teacher_once():
    model = _LossModel()
    accelerator = _FakeAccelerator()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _CountingScheduler()
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(gradient_clipping=None),
        datasets=SimpleNamespace(vla_data=SimpleNamespace(per_device_batch_size=1)),
    )
    trainer = train.VLATrainer(cfg, model, [], optimizer, scheduler, accelerator)
    first_metrics = trainer._train_step(None)
    assert first_metrics["total_loss"] == 1.5
    assert model.weight.item() == 1.0
    assert model.teacher_updates == 0
    assert scheduler.steps == 0
    accelerator.sync_gradients = True
    trainer._train_step(None)
    assert torch.isclose(model.weight, torch.tensor(0.4))
    assert model.teacher_updates == 1
    assert scheduler.steps == 1
    assert model.weight.grad is None


def test_video_only_train_step_has_no_jepa_dependency():
    model = _VideoOnlyModel()
    accelerator = _FakeAccelerator()
    accelerator.sync_gradients = True
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _CountingScheduler()
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(gradient_clipping=None),
        datasets=SimpleNamespace(vla_data=SimpleNamespace(per_device_batch_size=1)),
    )
    trainer = train.VLATrainer(cfg, model, [], optimizer, scheduler, accelerator)
    metrics = trainer._train_step(None)
    assert metrics["future_video_loss"] == 1.0
    assert metrics["total_loss"] == 1.0
    assert scheduler.steps == 1


def test_optimizer_groups_exclude_frozen_parameters():
    model = nn.Module()
    model.online = nn.Linear(2, 2)
    model.teacher = nn.Linear(2, 2)
    model.teacher.requires_grad_(False)
    cfg = OmegaConf.create(
        {"trainer": {
            "learning_rate": {"base": 1e-4},
            "freeze_modules": "",
        }}
    )
    groups = build_param_lr_groups(model, cfg)
    optimized = {id(parameter) for group in groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.online.parameters()}


def test_partial_reload_rejects_direct_ema_teacher_load(tmp_path, monkeypatch):
    model = nn.Module()
    model.action_model = nn.Module()
    model.action_model.tactile_target_encoder = nn.Linear(2, 2)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    with pytest.raises(ValueError, match="EMA teachers"):
        TrainerUtils.load_pretrained_backbones(
            model,
            str(checkpoint),
            reload_modules="action_model.tactile_target_encoder",
        )


class _CheckpointAccelerator:
    num_processes = 1
    gradient_accumulation_steps = 1
    is_main_process = True

    def __init__(self):
        self.saved = []
        self.messages = []

    def save_state(self, path):
        path = Path(path)
        path.mkdir(parents=True)
        (path / "optimizer.bin").write_bytes(b"state")
        self.saved.append(path)

    @staticmethod
    def wait_for_everyone():
        return None

    def print(self, message):
        self.messages.append(str(message))


def _checkpoint_trainer(tmp_path):
    model = nn.Linear(2, 2)
    accelerator = _CheckpointAccelerator()
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "trainer": {"is_resume": False, "pretrained_checkpoint": None},
        }
    )
    trainer = train.VLATrainer(
        cfg,
        model,
        [],
        torch.optim.SGD(model.parameters(), lr=0.1),
        _CountingScheduler(),
        accelerator,
    )
    trainer.checkpoint_dir = str(tmp_path / "checkpoints")
    Path(trainer.checkpoint_dir).mkdir()
    return trainer


def test_checkpoint_saves_full_state_directory_and_resolved_config(tmp_path):
    trainer = _checkpoint_trainer(tmp_path)
    trainer.completed_steps = 12
    trainer._save_checkpoint()
    checkpoint = tmp_path / "checkpoints" / "steps_12"
    assert trainer.accelerator.saved == [checkpoint]
    assert (checkpoint / "optimizer.bin").is_file()
    assert (checkpoint / "config.yaml").is_file()


def test_latest_checkpoint_prefers_full_state_at_same_step(tmp_path):
    trainer = _checkpoint_trainer(tmp_path)
    checkpoint_dir = Path(trainer.checkpoint_dir)
    (checkpoint_dir / "steps_5_pytorch_model.pt").write_bytes(b"legacy")
    (checkpoint_dir / "steps_5").mkdir()
    latest, step = trainer._get_latest_checkpoint(str(checkpoint_dir))
    assert Path(latest) == checkpoint_dir / "steps_5"
    assert step == 5


def test_partial_online_encoder_reload_marks_teachers_for_sync(tmp_path, monkeypatch):
    class _Action(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_encoder = nn.Linear(2, 2)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_model = _Action()

    source = _Model()
    checkpoint = tmp_path / "model.pt"
    torch.save(source.state_dict(), checkpoint)
    target = _Model()
    monkeypatch.setattr(trainer_tools.dist, "get_rank", lambda: 0)
    loaded = trainer_tools.TrainerUtils.load_pretrained_backbones(
        target,
        checkpoint,
        reload_modules="action_model.state_encoder",
    )
    assert loaded._jepa_teacher_keys_missing
