# DiT4DiT SONIC tactile training

The three fixed configs use the same stereo dataset, 46-D state, and `40 x 78` SONIC
action target. Future observations are training-only targets and never enter action
conditioning.

| Config | Current tactile | Future tactile | Future state | Future stereo |
|---|---:|---:|---:|---:|
| `dit4dit_g1_sonic_notactile.yaml` | no | no | no | no |
| `dit4dit_g1_sonic_htd.yaml` | yes | yes | no | no |
| `dit4dit_g1_sonic_jepa.yaml` (UniVLaT/JEPA) | yes | yes | yes | yes |

HTD is short for *Humanoid Transformer with Touch Dreaming* (arXiv:2604.13015). Here, HTD
mode means its current-tactile fusion and future-tactile latent objective, not a claim that
DiT4DiT reproduces the paper's full system. UniVLaT/JEPA adds future state and stereo latent
targets. Padded episode-tail targets are excluded by explicit time masks.

## Environment

```bash
cd /root/Projects/DiT4DiT
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .

source .venv/bin/activate
huggingface-cli login
wandb login
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained
```

If PyPI downloads are slow on this host, rerun the two `uv pip install` commands with
`--index-url https://pypi.tuna.tsinghua.edu.cn/simple` and unset local proxy variables for
that command. The requirements resolve to the tested PyTorch 2.7.1/CUDA 12.6 stack.

The configs use the Hugging Face cache (`local_files_only: true`) and expect the dataset at
`/root/Projects/data/carry-bucket-stereo`. Change `base_model`, `data_root_dir`, and the W&B
entity/project in all three YAMLs when running elsewhere.

`nvidia/Cosmos-Predict2.5-2B` is gated. The logged-in Hugging Face account must first be
approved on that model page; HTTP 403 means access has not been granted, not an environment
or proxy failure. After approval, run the download command above once before training.

## Full training

DiT4DiT requires BF16 ZeRO-3 for full joint Cosmos and action-model training on four A800
80GB GPUs. The fixed configs run 20k steps, save at 10k and 20k, and use the measured safe
batch of 1 per GPU (global batch 4).

```bash
cd /root/Projects/DiT4DiT
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file DiT4DiT/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 4 DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/real_robot/dit4dit_g1_sonic_notactile.yaml

accelerate launch --config_file DiT4DiT/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 4 DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/real_robot/dit4dit_g1_sonic_htd.yaml

accelerate launch --config_file DiT4DiT/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 4 DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/real_robot/dit4dit_g1_sonic_jepa.yaml
```

Training writes resumable states to `results/Checkpoints/<run_id>/checkpoints/steps_<N>/`.
The directly serveable checkpoint is
`results/Checkpoints/<run_id>/final_model/pytorch_model.pt`; its sibling run directory also
contains `config.yaml` and `dataset_statistics.json`.

## Backend and SONIC bridge

Start the DiT4DiT websocket backend with the checkpoint matching the ablation mode:

```bash
cd /root/Projects/DiT4DiT
source .venv/bin/activate
python -m deployment.model_server.server_sonic_policy \
  --ckpt-path results/Checkpoints/sonic_jepa/final_model/pytorch_model.pt \
  --device cuda:0 --use-bf16 --port 8000
```

Expose that websocket server through the common GR00T ZMQ policy interface:

```bash
cd /root/Projects/Isaac-GR00T
uv pip install --python .venv/bin/python -e /root/Projects/openpi/packages/openpi-client
uv run --no-sync python -m gr00t.eval.run_openpi_bridge_server \
  --openpi-host 127.0.0.1 --openpi-port 8000 --port 5550
```

Then launch the existing controller path:

```bash
cd /root/Projects/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_inference.py \
  --policy-host 127.0.0.1 --policy-port 5550 \
  --camera-host 192.168.123.164 --tactile-zmq-host 192.168.123.164 \
  --prompt "carry the bucket"
```

For a No Tactile checkpoint, omit `--tactile-zmq-host` and add `--no-use-tactile`.

The websocket handshake is `sonic_vla_v1`. Requests contain `state: float32[46]`,
`ego_view_left/right: uint8[H,W,3]`, `prompt: str`, and `tactile: uint8[256]` only for
HTD/JEPA. Responses are finite `actions: float32[40,78]` with
`motion_token[0:64] | left_hand[64:71] | right_hand[71:78]`. SONIC decodes the motion token;
DiT4DiT does not emit low-level whole-body joint commands.
