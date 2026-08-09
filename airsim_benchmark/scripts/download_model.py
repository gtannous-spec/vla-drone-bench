#!/usr/bin/env python3
"""
download_model.py — Download model weights from HuggingFace.

Supports OpenVLA, InternVL2, LLaVA, and any HuggingFace model.

Usage:
    python -m airsim_benchmark.scripts.download_model \
        --model-id openvla/openvla-7b \
        --output ~/models/openvla-7b

    python -m airsim_benchmark.scripts.download_model \
        --model-id OpenGVLab/InternVL2-8B

Requires: pip install huggingface_hub
"""

import argparse
import os
import sys
from pathlib import Path

KNOWN_MODELS = {
    "openvla-7b": "openvla/openvla-7b",
    "internvl2-8b": "OpenGVLab/InternVL2-8B",
    "llava-v1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
}


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights from HuggingFace Hub"
    )
    parser.add_argument(
        "--model-id",
        default="OpenGVLab/InternVL2-8B",
        help=(
            f"HuggingFace model ID or shortcut. "
            f"Shortcuts: {', '.join(KNOWN_MODELS.keys())}"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Local directory to save (default: HuggingFace cache)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    model_id = KNOWN_MODELS.get(args.model_id.lower(), args.model_id)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run:")
        print("  pip install --user huggingface_hub")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    download_kwargs = dict(
        repo_id=model_id,
        token=token,
        resume_download=True,
    )

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        download_kwargs["local_dir"] = str(output_dir)

    print(f"Downloading model: {model_id}")
    if args.output:
        print(f"Destination: {args.output}")
    else:
        print("Destination: HuggingFace cache (default)")
    print()

    try:
        path = snapshot_download(**download_kwargs)
        print(f"\nModel downloaded successfully to: {path}")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
