"""Resolve LoRA adapters saved beside a regression head (no model load)."""

from __future__ import annotations

import json
import os


def resolve_lora_dir(regression_head_path: str, lora_path: str = "") -> str:
    """Return the PEFT adapter directory to merge at eval.

    Training writes ``adapter_config.json`` + ``regression_head.pt`` into the
    same folder. Eval used to load only the MLP, so the 7B stayed stock and
    the head emitted a near-constant fly-forward action.
    """
    if lora_path:
        return lora_path
    if not regression_head_path:
        return ""
    directory = (
        regression_head_path
        if os.path.isdir(regression_head_path)
        else os.path.dirname(regression_head_path)
    )
    if directory and os.path.isfile(os.path.join(directory, "adapter_config.json")):
        return directory
    return ""


def adapter_saves_projector(adapter_dir: str) -> bool:
    """True when PEFT adapter_config lists the vision projector as saved."""
    if not adapter_dir:
        return False
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    saved = data.get("modules_to_save") or []
    return any("projector" in str(name).lower() for name in saved)
