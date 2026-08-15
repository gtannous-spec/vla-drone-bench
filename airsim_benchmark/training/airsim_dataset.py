"""
airsim_dataset.py — PyTorch Dataset for AirSim LoRA fine-tuning.

Reads the manifest.jsonl produced by collect_trajectories.py and yields
(input_ids, pixel_values, labels) tuples ready for causal-LM training.
Labels use IGNORE_INDEX (-100) for all non-action positions so the loss
is computed only over the 8 action tokens.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# OpenFly vln_norm statistics (baked into the checkpoint)
VLN_Q01 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
VLN_Q99 = np.array([1, 5, 15, 15, 2, 0, 0, 0], dtype=np.float32)


def action_to_token_ids(
    action_vec: np.ndarray,
    vocab_size: int = 32000,
    n_bins: int = 16,
) -> np.ndarray:
    """Convert an 8-D action vector to OpenFly action token IDs.

    Steps:
      1. Normalize to [-1, 1] using vln_norm q01/q99.
      2. Discretize into ``n_bins`` uniform bins via ``np.digitize``.
      3. Map bin indices to the *last* ``n_bins`` token IDs in the vocab.

    Default ``n_bins=16`` reduces the classification from 256-way to 16-way
    per dimension, which the model can realistically learn from ~3-30k samples.
    """
    q_range = VLN_Q99 - VLN_Q01
    safe_range = np.where(q_range > 0, q_range, 1.0)
    normalized = np.where(
        q_range > 0,
        2.0 * (action_vec - VLN_Q01) / safe_range - 1.0,
        0.0,
    )
    normalized = np.clip(normalized, -1.0, 1.0)

    bins = np.linspace(-1.0, 1.0, n_bins)
    discretized = np.digitize(normalized, bins)  # 1 .. n_bins
    token_ids = vocab_size - discretized           # 31744 .. 31999
    return token_ids.astype(np.int64)


class AirSimTrajectoryDataset(Dataset):
    """Loads trajectory samples from a manifest.jsonl file.

    Each sample returns a dict with:
      - ``input_ids``:    (seq_len + 8,) int64 — prompt tokens + action tokens
      - ``pixel_values``: (3, C, H, W)  float32 — 3-image triplet
      - ``labels``:       (seq_len + 8,) int64 — IGNORE for prompt, real for actions

    Parameters
    ----------
    manifest_path : str or Path
        Path to ``manifest.jsonl``.
    processor : Any
        A ``PrismaticProcessor`` instance (from OpenFly's custom code).
    vocab_size : int
        Effective vocab size for action tokenization (default 32000).
    """

    def __init__(
        self,
        manifest_path: str,
        processor: Any,
        vocab_size: int = 32000,
        n_bins: int = 16,
    ) -> None:
        self.data_dir = Path(manifest_path).parent
        self.processor = processor
        self.vocab_size = vocab_size
        self.n_bins = n_bins

        self.samples: List[Dict[str, Any]] = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from PIL import Image as PILImage

        sample = self.samples[idx]

        # Load triplet images
        images = []
        for rel_path in sample["images"]:
            img_path = self.data_dir / rel_path
            images.append(PILImage.open(str(img_path)).convert("RGB"))

        instruction = sample["instruction"]
        action_vec = np.array(sample["action"], dtype=np.float32)

        # Process through PrismaticProcessor → input_ids + pixel_values
        inputs = self.processor(instruction, images)
        input_ids = inputs["input_ids"].squeeze(0)         # (seq_len,)
        pixel_values = inputs["pixel_values"]              # (3, C, H, W)

        # Tokenize action
        action_token_ids = action_to_token_ids(
            action_vec, self.vocab_size, self.n_bins
        )
        action_tokens = torch.tensor(action_token_ids, dtype=torch.long)

        # Append trigger token (29871) + action tokens to input_ids
        trigger = torch.tensor([29871], dtype=torch.long)
        full_input_ids = torch.cat([input_ids, trigger, action_tokens])

        # Labels: IGNORE_INDEX for prompt + trigger, real IDs for action tokens
        prompt_len = input_ids.shape[0] + 1  # +1 for trigger
        labels = torch.full_like(full_input_ids, IGNORE_INDEX)
        labels[prompt_len:] = action_tokens

        return {
            "input_ids": full_input_ids,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad input_ids and labels to the same length within a batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list = []
    labels_list = []
    pixel_values_list = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        if pad_len > 0:
            input_ids_list.append(
                torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([b["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
            )
        else:
            input_ids_list.append(b["input_ids"])
            labels_list.append(b["labels"])
        pixel_values_list.append(b["pixel_values"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "pixel_values": torch.stack(pixel_values_list),
        "labels": torch.stack(labels_list),
    }
