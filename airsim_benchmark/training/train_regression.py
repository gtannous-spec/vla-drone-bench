#!/usr/bin/env python3
"""
train_regression.py — Regression-head fine-tuning of OpenFly-Agent-7B.

Instead of predicting 8 action tokens autoregressively, this approach adds
a small MLP on top of the LLM's last hidden state that directly outputs
8 continuous action values.  Training uses MSE loss against normalized
ground-truth actions.

This sidesteps the constant-token collapse seen with autoregressive
action-token prediction, because the output space is continuous and
the loss gradient flows directly into the action dimensions.

Usage:
    python -m airsim_benchmark.training.train_regression \
        --data ./data/lora_training_v3/manifest.jsonl \
        --output ./data/regression_checkpoints \
        --epochs 10 --lr 2e-4
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airsim_benchmark.core.action_space import VLN_Q01, VLN_Q99, normalize_action
from airsim_benchmark.core.checkpoint_resolve import adapter_saves_projector
from airsim_benchmark.training.airsim_dataset import filter_samples

logger = logging.getLogger(__name__)

ACTION_DIM = 8


def find_instruction_span(input_ids, processor, instruction):
    """Find the token span of the instruction in the full input_ids."""
    instr_ids = processor.tokenizer.encode(instruction, add_special_tokens=False)
    seq = input_ids.squeeze()
    instr_tensor = torch.tensor(instr_ids, device=seq.device)
    for i in range(len(seq) - len(instr_ids) + 1):
        if torch.equal(seq[i:i + len(instr_ids)], instr_tensor):
            return (i, i + len(instr_ids))
    return (0, min(20, len(seq)))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class ActionRegressionHead(nn.Module):
    """MLP that maps the LLM's last hidden state to 8 continuous action values."""

    def __init__(self, hidden_dim: int, action_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim),
            nn.Tanh(),  # output in [-1, 1], matching normalized action range
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)


class RegressionDataset(Dataset):
    """Loads trajectory samples for regression training.

    Returns (input_ids, pixel_values, action_normalized) where
    action_normalized is an 8-D vector in [-1, 1].
    """

    def __init__(self, manifest_path: str, processor, vocab_size: int = 32000,
                 filter_mode: str = "all"):
        self.data_dir = Path(manifest_path).parent
        self.processor = processor
        self.vocab_size = vocab_size

        raw_samples = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_samples.append(json.loads(line))

        self.samples = filter_samples(raw_samples, mode=filter_mode)
        logger.info(
            f"Loaded {len(self.samples)}/{len(raw_samples)} samples "
            f"from {manifest_path} (filter_mode={filter_mode!r})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image as PILImage

        sample = self.samples[idx]

        images = []
        for rel_path in sample["images"]:
            img_path = self.data_dir / rel_path
            images.append(PILImage.open(str(img_path)).convert("RGB"))

        instruction = sample["instruction"]
        action_vec = np.array(sample["action"], dtype=np.float32)
        normalized = normalize_action(action_vec)

        inputs = self.processor(instruction, images)
        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"]

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "action": torch.tensor(normalized, dtype=torch.float32),
            "instruction": instruction,
        }


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_list = []
    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        if pad_len > 0:
            input_ids_list.append(
                torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
            )
        else:
            input_ids_list.append(b["input_ids"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
    }


def prepare_model(model_path: str, device: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if os.path.isdir(model_path):
        openfly_dir = model_path
    else:
        openfly_dir = snapshot_download(model_path)
        logger.info(f"Model cached at {openfly_dir}")

    platform_hf = "/tmp/openfly-platform/train/extern/hf"
    for py_file in [
        "configuration_prismatic.py",
        "modeling_prismatic.py",
        "processing_prismatic.py",
    ]:
        src = os.path.join(platform_hf, py_file)
        dst = os.path.join(openfly_dir, py_file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    import json as _json
    config_path = os.path.join(openfly_dir, "config.json")
    with open(config_path) as f:
        cfg = _json.load(f)
    cfg["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenFlyConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }
    with open(config_path, "w") as f:
        _json.dump(cfg, f, indent=2)

    preproc_path = os.path.join(openfly_dir, "preprocessor_config.json")
    with open(preproc_path) as f:
        pcfg = _json.load(f)
    pcfg["auto_map"] = {
        "AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor",
        "AutoProcessor": "processing_prismatic.PrismaticProcessor",
    }
    with open(preproc_path, "w") as f:
        _json.dump(pcfg, f, indent=2)

    processor = AutoProcessor.from_pretrained(openfly_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        openfly_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
    )
    return model, processor, openfly_dir


def apply_lora(model, rank: int = 16, alpha: int = 32,
               unfreeze_projector: bool = True):
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["projector"] if unfreeze_projector else None,
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lora_" not in name and "modules_to_save" not in name:
            param.requires_grad = False

    if unfreeze_projector:
        proj_params = 0
        for name, param in model.named_parameters():
            if "projector" in name:
                param.requires_grad = True
                proj_params += param.numel()
        logger.info(f"Projector unfrozen: {proj_params:,} params")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Base trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def get_hidden_dim(model) -> int:
    """Extract the LLM hidden dimension from model config."""
    if hasattr(model.config, "text_config"):
        return model.config.text_config.hidden_size
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    for name, param in model.named_parameters():
        if "lm_head" in name and param.dim() == 2:
            return param.shape[1]
    raise ValueError("Cannot determine hidden_dim from model config")


def train(
    model,
    processor,
    manifest_path: str,
    output_dir: str,
    rank: int = 16,
    alpha: int = 32,
    epochs: int = 10,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-4,
    warmup_frac: float = 0.1,
    val_split: float = 0.1,
    patience: int = 5,
    device: str = "cuda:0",
    head_type: str = "last_token",
    filter_mode: str = "all",
) -> None:
    hidden_dim = get_hidden_dim(model)
    logger.info(f"LLM hidden dim: {hidden_dim}")

    if head_type == "film":
        from airsim_benchmark.training.instruction_head import InstructionConditionedHead
        reg_head = InstructionConditionedHead(hidden_dim=hidden_dim, action_dim=ACTION_DIM).to(device)
    else:
        reg_head = ActionRegressionHead(hidden_dim=hidden_dim, action_dim=ACTION_DIM).to(device)
    reg_head_params = sum(p.numel() for p in reg_head.parameters())
    logger.info(f"Regression head ({head_type}): {reg_head_params:,} trainable params")

    full_dataset = RegressionDataset(manifest_path, processor, filter_mode=filter_mode)

    if head_type == "film":
        n_check = min(5, len(full_dataset))
        rng = np.random.default_rng(42)
        check_indices = rng.choice(len(full_dataset), size=n_check, replace=False).tolist()
        mismatch_count = 0
        for idx in check_indices:
            raw_sample = full_dataset.samples[idx]
            instr_text = raw_sample["instruction"]
            item = full_dataset[idx]
            computed_span = find_instruction_span(item["input_ids"], processor, instr_text)
            stored_span = raw_sample.get("instruction_span")
            if stored_span is not None:
                stored_span = tuple(stored_span)
                if computed_span != stored_span:
                    mismatch_count += 1
                    logger.warning(
                        f"[SPAN-CHECK] sample {idx}: computed={computed_span} "
                        f"!= stored={stored_span}"
                    )
                else:
                    logger.info(f"[SPAN-CHECK] sample {idx}: span={computed_span} OK")
            else:
                logger.info(f"[SPAN-CHECK] sample {idx}: span={computed_span} (no stored span)")
        if mismatch_count > 0:
            rate = mismatch_count / n_check
            if rate > 0.2:
                raise ValueError(
                    f"Instruction span mismatch rate {rate:.0%} "
                    f"({mismatch_count}/{n_check}) exceeds 20% — aborting "
                    f"to prevent training on misaligned features"
                )
            logger.warning(
                f"[SPAN-CHECK] {mismatch_count}/{n_check} span mismatches ({rate:.0%})"
            )
        else:
            logger.info(f"[SPAN-CHECK] All {n_check} spans validated OK")

    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Dataset split: {n_train} train, {n_val} val")

    if batch_size != 1:
        logger.info(
            f"Overriding batch_size={batch_size} -> 1; "
            f"grad_accum {grad_accum} -> {grad_accum * batch_size}"
        )
        grad_accum *= batch_size
        batch_size = 1

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    model = model.to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    lora_params = []
    projector_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        elif "projector" in name:
            projector_params.append(param)

    param_groups = [
        {"params": lora_params, "lr": lr},
        {"params": projector_params, "lr": lr * 0.1},
        {"params": list(reg_head.parameters()), "lr": lr * 2.0},
    ]

    lr_summary = ", ".join(f"{g['lr']:.1e}({len(g['params'])}p)" for g in param_groups)
    logger.info(f"Optimizer param groups: {lr_summary}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    dim_names = ["stop", "fwd", "yawL", "yawR", "up", "dn", "L", "R"]

    logger.info(
        f"Training: {epochs} epochs, batch=1, grad_accum={grad_accum}, "
        f"lr={lr}, total_steps={total_steps}, train={n_train}, val={n_val}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        reg_head.train()
        epoch_loss = 0.0
        epoch_count = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].squeeze(0).to(device, dtype=torch.bfloat16)
            action_target = batch["action"].to(device, dtype=torch.float32)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1]
            if head_type == "film":
                instruction = batch["instruction"][0]
                instr_start, instr_end = find_instruction_span(
                    batch["input_ids"][0], processor, instruction,
                )
                instr_hidden = hidden_states[:, instr_start:instr_end, :].float()
                action_vec = hidden_states[:, -1, :].float()
                pred_action = reg_head(instr_hidden, action_vec).squeeze(0)
            else:
                last_token_hidden = hidden_states[0, -1, :].float()
                pred_action = reg_head(last_token_hidden)
            loss = F.mse_loss(pred_action, action_target[0]) / grad_accum
            loss.backward()

            epoch_loss += F.mse_loss(pred_action, action_target[0]).item()
            epoch_count += 1

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(reg_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (batch_idx + 1) % 10 == 0:
                avg = epoch_loss / max(1, epoch_count)
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                    f"mse_loss={avg:.6f} lr={current_lr:.2e}"
                )

        train_elapsed = time.time() - t0
        avg_train = epoch_loss / max(1, epoch_count)
        train_losses.append(avg_train)

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        reg_head.eval()
        val_loss = 0.0
        val_count = 0
        per_dim_mse = defaultdict(list)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].squeeze(0).to(device, dtype=torch.bfloat16)
                action_target = batch["action"].to(device, dtype=torch.float32)

                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[-1]
                if head_type == "film":
                    instruction = batch["instruction"][0]
                    instr_start, instr_end = find_instruction_span(
                        batch["input_ids"][0], processor, instruction,
                    )
                    instr_hidden = hidden_states[:, instr_start:instr_end, :].float()
                    action_vec = hidden_states[:, -1, :].float()
                    pred_action = reg_head(instr_hidden, action_vec).squeeze(0)
                else:
                    last_token_hidden = hidden_states[0, -1, :].float()
                    pred_action = reg_head(last_token_hidden)

                sample_mse = F.mse_loss(pred_action, action_target[0]).item()
                val_loss += sample_mse
                val_count += 1

                for d in range(ACTION_DIM):
                    dim_err = (pred_action[d] - action_target[0, d]).item() ** 2
                    per_dim_mse[d].append(dim_err)

                all_preds.append(pred_action.cpu().numpy())
                all_targets.append(action_target[0].cpu().numpy())

        avg_val = val_loss / max(1, val_count)
        val_losses.append(avg_val)

        dim_mse_str = ", ".join(
            f"{dim_names[d]}={np.mean(per_dim_mse[d]):.4f}" for d in range(ACTION_DIM)
        )
        logger.info(
            f"Epoch {epoch}/{epochs} done in {train_elapsed:.0f}s — "
            f"train_mse={avg_train:.6f}, val_mse={avg_val:.6f}"
        )
        logger.info(f"  per_dim_mse: {dim_mse_str}")

        # Diversity check: are predictions varied or constant?
        preds_np = np.array(all_preds)
        pred_std = preds_np.std(axis=0)
        targets_np = np.array(all_targets)
        target_std = targets_np.std(axis=0)
        diversity_str = ", ".join(
            f"{dim_names[d]}: pred_std={pred_std[d]:.3f} tgt_std={target_std[d]:.3f}"
            for d in range(ACTION_DIM)
        )
        logger.info(f"  [DIVERSITY] {diversity_str}")

        # Sample diagnostics
        if epoch == 1 or epoch == epochs or epoch % 3 == 0:
            for i in range(min(5, len(all_preds))):
                p = all_preds[i]
                t = all_targets[i]
                err = np.abs(p - t)
                logger.info(
                    f"  [DIAG] sample {i+1}: "
                    f"pred=[{', '.join(f'{v:.3f}' for v in p)}] "
                    f"tgt=[{', '.join(f'{v:.3f}' for v in t)}] "
                    f"abs_err=[{', '.join(f'{v:.3f}' for v in err)}]"
                )

        # ── Checkpoint ────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_without_improvement = 0
            save_dir = os.path.join(output_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            torch.save(reg_head.state_dict(), os.path.join(save_dir, "regression_head.pt"))
            if adapter_saves_projector(save_dir):
                logger.info("  Adapter includes vision projector (modules_to_save)")
            else:
                logger.error(
                    "  Adapter missing projector — eval will use stock OpenFly projector"
                )
            logger.info(f"  Best checkpoint saved (val_mse={avg_val:.6f})")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement for {epochs_without_improvement}/{patience} epochs"
            )
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping triggered at epoch {epoch}")
                break

    # Save final
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    torch.save(reg_head.state_dict(), os.path.join(final_dir, "regression_head.pt"))
    if not adapter_saves_projector(final_dir):
        logger.error(
            "  Final adapter missing projector — eval will use stock OpenFly projector"
        )
    logger.info(f"Final checkpoint saved to {final_dir}")

    meta = {
        "mode": "regression",
        "head_type": head_type,
        "epochs_completed": epoch,
        "epochs_requested": epochs,
        "early_stopped": epochs_without_improvement >= patience,
        "batch_size": 1,
        "grad_accum": grad_accum,
        "lr": lr,
        "rank": rank,
        "alpha": alpha,
        "hidden_dim": hidden_dim,
        "reg_head_params": reg_head_params,
        "best_val_mse": round(best_val_loss, 6),
        "train_losses": [round(l, 6) for l in train_losses],
        "val_losses": [round(l, 6) for l in val_losses],
        "final_per_dim_mse": {
            dim_names[d]: round(np.mean(per_dim_mse[d]), 6)
            for d in range(ACTION_DIM) if per_dim_mse[d]
        },
        "n_train": n_train,
        "n_val": n_val,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Regression-head fine-tuning of OpenFly on AirSim"
    )
    parser.add_argument("--data", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output", default="./data/regression_checkpoints")
    parser.add_argument("--model", default="IPEC-COMMUNITY/openfly-agent-7b")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--head-type", choices=["last_token", "film"], default="last_token",
                        help="Regression head type: last_token (original) or film (instruction-conditioned)")
    parser.add_argument("--filter-mode", choices=["all", "detected_only", "agreement_only"], default="all",
                        help="Training data filter mode")
    args = parser.parse_args()

    setup_logging()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    logger.info("Loading model...")
    model, processor, _ = prepare_model(args.model, device)

    logger.info("Applying LoRA adapters...")
    model = apply_lora(model, rank=args.rank, alpha=args.alpha)

    logger.info("Starting regression training...")
    train(
        model, processor,
        manifest_path=args.data,
        output_dir=args.output,
        rank=args.rank,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        patience=args.patience,
        device=device,
        head_type=args.head_type,
        filter_mode=args.filter_mode,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
