# SONIC Deployment

Serve the JEPA checkpoint through the same websocket contract consumed by the
Isaac-GR00T SONIC bridge:

```bash
python -m deployment.model_server.server_sonic_policy \
  --ckpt-path /path/to/checkpoints/model.pt \
  --device cuda \
  --use-bf16 \
  --port 8000
```

The adapter requires current stereo RGB, the canonical 46-dimensional G1
state, the prompt, and one `uint8[256]` tactile frame. It reproduces training's
224x224-per-eye horizontal stereo packing, applies the saved q01/q99 state
normalization, and returns an unnormalized finite `40 x 78` SONIC latent action
chunk. It does not use the legacy direct-WBC ZMQ path or its hard-coded 23/32
dimensional action slicing.
