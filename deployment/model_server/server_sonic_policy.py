"""Serve a DiT4DiT JEPA checkpoint through the canonical SONIC contract."""

import argparse
import logging

import torch

from deployment.model_server.sonic_policy import SonicPolicyAdapter
from deployment.model_server.tools.sonic_websocket_policy_server import (
    SonicWebsocketPolicyServer,
)
from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.framework.share_tools import read_mode_config


def main(args) -> None:
    model_config, norm_stats = read_mode_config(args.ckpt_path)
    policy = baseframework.from_pretrained(args.ckpt_path)
    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to(args.device).eval()

    adapter = SonicPolicyAdapter(
        policy,
        model_config=model_config,
        norm_stats=norm_stats,
        unnorm_key=args.unnorm_key,
    )
    logging.info("SONIC metadata: %s", adapter.metadata)
    SonicWebsocketPolicyServer(adapter, host=args.host, port=args.port).serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description="DiT4DiT SONIC websocket server")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--unnorm-key", default=None)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
