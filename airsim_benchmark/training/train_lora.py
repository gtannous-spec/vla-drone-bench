#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tuning of OpenFly-Agent-7B on AirSim trajectories.

Applies Low-Rank Adaptation to the LLM attention layers while freezing
the vision backbone and projector.  Trains on keyframe triplets +
language instructions, with loss computed only on action token positions.

Usage:
    python -m airsim_benchmark.training.train_lora \
        --data ./data/lora_training/manifest.jsonl \
        --output ./data/lora_checkpoints \
        --epochs 5 --batch-size 4 --lr 2e-4

Requires: peft>=0.12.0, torch, transformers, accelerate
"""

import argparse
import json
import logging
import re
import warnings

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def prepare_model(model_path: str, device: str):
    """Load OpenFly model + processor and apply the same config patches
    used by the inference controller."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if os.path.isdir(model_path):
        openfly_dir = model_path
    else:
        openfly_dir = snapshot_download(model_path)
        logger.info(f"Model cached at {openfly_dir}")

    # Copy architecture files from OpenFly-Platform
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

    # Inject auto_map into config files
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


def apply_lora(
    model,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    unfreeze_projector: bool = True,
    unfreeze_vision_layers: int = 0,
    unfreeze_lm_head: bool = False,
):
    """Wrap the model with LoRA adapters on the LLM attention layers,
    and optionally unfreeze the projector, vision blocks, and lm_head."""
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Freeze everything that isn't a LoRA parameter
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    # Unfreeze the projector MLP so it can learn AirSim-specific
    # vision-to-LLM mappings (this was the bottleneck in rounds 1-2)
    if unfreeze_projector:
        proj_params = 0
        for name, param in model.named_parameters():
            if "projector" in name:
                param.requires_grad = True
                proj_params += param.numel()
        logger.info(f"Projector unfrozen: {proj_params:,} params")

    # Optionally unfreeze the last N transformer blocks of each ViT
    if unfreeze_vision_layers > 0:
        # Find total number of blocks in the featurizer
        total_blocks = 0
        for name, _ in model.named_parameters():
            m = re.search(r"featurizer\.blocks\.(\d+)\.", name)
            if m:
                total_blocks = max(total_blocks, int(m.group(1)) + 1)

        cutoff = total_blocks - unfreeze_vision_layers
        vision_params = 0
        for name, param in model.named_parameters():
            if "vision_backbone" not in name:
                continue
            m = re.search(r"blocks\.(\d+)\.", name)
            if m and int(m.group(1)) >= cutoff:
                param.requires_grad = True
                vision_params += param.numel()
        logger.info(
            f"Vision backbone: last {unfreeze_vision_layers}/{total_blocks} "
            f"blocks unfrozen ({vision_params:,} params)"
        )

    # Unfreeze lm_head so the model can learn to redirect logits from
    # language tokens to action tokens (31744-31999).
    if unfreeze_lm_head:
        lm_head_params = 0
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
                lm_head_params += param.numel()
        logger.info(f"lm_head unfrozen: {lm_head_params:,} params")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Total trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model


def train(
    model,
    processor,
    manifest_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-4,
    warmup_frac: float = 0.1,
    val_split: float = 0.1,
    patience: int = 3,
    n_bins: int = 16,
    device: str = "cuda:0",
    use_wandb: bool = False,
    wandb_project: str = "openfly-lora",
) -> None:
    """Run the LoRA training loop with validation, early stopping, and
    optional Weights & Biases logging."""
    from airsim_benchmark.training.airsim_dataset import (
        AirSimTrajectoryDataset,
        collate_fn,
    )

    vocab_size = 32000
    pad_attr = getattr(model.config, "pad_to_multiple_of", 64)
    if hasattr(model.config, "text_config"):
        vocab_size = model.config.text_config.vocab_size - pad_attr

    full_dataset = AirSimTrajectoryDataset(
        manifest_path, processor, vocab_size=vocab_size, n_bins=n_bins,
    )
    logger.info(f"Action tokenization: {n_bins} bins, token range [{vocab_size - n_bins}, {vocab_size - 1}]")

    # Train/val split
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Dataset split: {n_train} train, {n_val} val")

    # The vision backbone processes exactly 3 images (pixel_values shape
    # (3, 6, 224, 224)) and doesn't support a batch dimension.  We must
    # use batch_size=1 and rely on gradient accumulation for effective
    # batch size.
    if batch_size != 1:
        logger.info(
            f"Overriding batch_size={batch_size} → 1 "
            f"(vision backbone requires unbatched pixel_values); "
            f"increasing grad_accum {grad_accum} → {grad_accum * batch_size}"
        )
        grad_accum = grad_accum * batch_size
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

    # Enable gradient checkpointing to reduce activation memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Differential learning rates: lower for projector/vision to avoid
    # destroying pretrained features, full LR for LoRA adapters
    lora_params = []
    projector_params = []
    vision_params = []
    lm_head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lm_head" in name:
            lm_head_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        elif "projector" in name:
            projector_params.append(param)
        elif "vision_backbone" in name:
            vision_params.append(param)

    param_groups = [{"params": lora_params, "lr": lr}]
    if projector_params:
        param_groups.append({"params": projector_params, "lr": lr * 0.1})
    if vision_params:
        param_groups.append({"params": vision_params, "lr": lr * 0.05})
    if lm_head_params:
        param_groups.append({"params": lm_head_params, "lr": lr * 0.5})

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

    # Optional W&B init
    if use_wandb:
        try:
            import wandb
            wandb.init(project=wandb_project, config={
                "epochs": epochs, "batch_size": batch_size,
                "grad_accum": grad_accum, "lr": lr,
                "val_split": val_split, "patience": patience,
                "n_train": n_train, "n_val": n_val,
            })
        except ImportError:
            logger.warning("wandb not installed, disabling W&B logging")
            use_wandb = False

    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    logger.info(
        f"Training: {epochs} epochs, batch={batch_size}, "
        f"grad_accum={grad_accum}, lr={lr}, "
        f"total_steps={total_steps}, train={n_train}, val={n_val}, "
        f"patience={patience}"
    )

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            # Squeeze batch dim: (1, 3, 6, H, W) → (3, 6, H, W)
            pixel_values = batch["pixel_values"].squeeze(0).to(device, dtype=torch.bfloat16)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            n_action_tokens = (labels != -100).sum().item()
            epoch_loss += outputs.loss.item() * n_action_tokens
            epoch_tokens += n_action_tokens

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (batch_idx + 1) % 10 == 0:
                avg = epoch_loss / max(1, epoch_tokens)
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                    f"train_loss={avg:.4f} lr={current_lr:.2e}"
                )
                if use_wandb:
                    import wandb
                    wandb.log({"train_loss_step": avg, "lr": current_lr,
                               "global_step": global_step})

        train_elapsed = time.time() - t0
        avg_train_loss = epoch_loss / max(1, epoch_tokens)
        train_losses.append(avg_train_loss)

        # ── Validation (with per-dimension loss tracking) ─────────────
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        per_dim_losses = defaultdict(list)  # type: dict[int, list[float]]
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].squeeze(0).to(device, dtype=torch.bfloat16)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                n_act = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_act
                val_tokens += n_act

                # Per-dimension loss: find action token positions and
                # compute CE loss for each of the 8 action dims separately
                action_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
                if len(action_positions) > 0 and outputs.logits is not None:
                    logits = outputs.logits
                    for dim_idx, pos in enumerate(action_positions):
                        if dim_idx >= 8:
                            break
                        # logits at pos-1 predict the token at pos
                        dim_loss = F.cross_entropy(
                            logits[0, pos - 1 : pos, :],
                            labels[0, pos : pos + 1],
                        )
                        per_dim_losses[dim_idx].append(dim_loss.item())

        avg_val_loss = val_loss / max(1, val_tokens)
        val_losses.append(avg_val_loss)

        dim_loss_str = ", ".join(
            f"{np.mean(per_dim_losses[d]):.4f}" if per_dim_losses[d] else "N/A"
            for d in range(8)
        )
        dim_names = "stop,fwd,yawL,yawR,up,dn,L,R"
        logger.info(
            f"Epoch {epoch}/{epochs} done in {train_elapsed:.0f}s — "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )
        logger.info(
            f"  val_loss_per_dim [{dim_names}]: [{dim_loss_str}]"
        )

        # ── Detailed diagnostics: predicted vs target tokens ─────────
        # Run on first 5 validation samples to see EXACTLY what the
        # model produces vs. what it should produce.
        if epoch == 1 or epoch == epochs or epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                diag_count = 0
                for batch in val_loader:
                    if diag_count >= 5:
                        break
                    diag_count += 1
                    inp = batch["input_ids"].to(device)
                    pv = batch["pixel_values"].squeeze(0).to(device, dtype=torch.bfloat16)
                    lab = batch["labels"].to(device)

                    out = model(input_ids=inp, pixel_values=pv, labels=lab)
                    action_pos = (lab[0] != -100).nonzero(as_tuple=True)[0]
                    if len(action_pos) < 8:
                        continue

                    target_toks = lab[0, action_pos].cpu().tolist()
                    pred_toks = []
                    in_action_range = []
                    for pos in action_pos:
                        logits_at = out.logits[0, pos - 1]
                        argmax_id = logits_at.argmax().item()
                        pred_toks.append(argmax_id)
                        in_action_range.append(31744 <= argmax_id <= 31999)

                        # On first sample only, log top-5 for each dim
                        if diag_count == 1:
                            top5 = torch.topk(logits_at, 5)
                            top5_ids = top5.indices.tolist()
                            top5_vals = [f"{v:.2f}" for v in top5.values.tolist()]
                            dim_i = (pos - action_pos[0]).item()
                            logger.info(
                                f"  [DIAG] dim{dim_i} top5: "
                                f"ids={top5_ids} vals={top5_vals} "
                                f"target={target_toks[dim_i]}"
                            )

                    n_in_range = sum(in_action_range)
                    match = sum(p == t for p, t in zip(pred_toks, target_toks))
                    logger.info(
                        f"  [DIAG] sample {diag_count}: "
                        f"target={target_toks} pred={pred_toks} "
                        f"match={match}/8 in_action_range={n_in_range}/8"
                    )
        if use_wandb:
            import wandb
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss,
                       "epoch": epoch})

        # ── Checkpointing + early stopping ────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_path = os.path.join(output_dir, "best")
            model.save_pretrained(save_path)
            logger.info(f"  Best checkpoint saved (val_loss={avg_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement for {epochs_without_improvement}/{patience} epochs"
            )
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping triggered at epoch {epoch}")
                break

    # Save final checkpoint
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    logger.info(f"Final checkpoint saved to {final_path}")

    # Compute final per-dim val losses for metadata
    final_per_dim = {
        f"dim{d}": round(np.mean(per_dim_losses[d]), 6) if per_dim_losses[d] else None
        for d in range(8)
    }

    meta = {
        "epochs_completed": epoch,
        "epochs_requested": epochs,
        "early_stopped": epochs_without_improvement >= patience,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "n_bins": n_bins,
        "total_steps": global_step,
        "best_val_loss": round(best_val_loss, 6),
        "train_losses": [round(l, 6) for l in train_losses],
        "val_losses": [round(l, 6) for l in val_losses],
        "final_val_loss_per_dim": final_per_dim,
        "n_train": n_train,
        "n_val": n_val,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if use_wandb:
        import wandb
        wandb.finish()

    # Auto-register training experiment
    try:
        from airsim_benchmark.experiments import ExperimentRegistry
        registry = ExperimentRegistry()
        registry.register_training(
            checkpoint_dir=output_dir,
            data_manifest=manifest_path,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            gpu=os.environ.get("SLURM_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES")),
            node=os.environ.get("SLURM_NODELIST", os.environ.get("HOSTNAME")),
        )
    except Exception as e:
        logger.warning(f"Could not register training experiment: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune OpenFly on AirSim")
    parser.add_argument("--data", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output", default="./data/lora_checkpoints")
    parser.add_argument("--model", default="IPEC-COMMUNITY/openfly-agent-7b")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction held out for validation (default 0.1)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience in epochs (default 3)")
    parser.add_argument("--unfreeze-projector", action="store_true", default=True,
                        help="Unfreeze the projector MLP (default: True)")
    parser.add_argument("--freeze-projector", action="store_true", default=False,
                        help="Keep projector frozen (overrides --unfreeze-projector)")
    parser.add_argument("--unfreeze-vision-layers", type=int, default=0,
                        help="Number of last vision backbone blocks to unfreeze (default 0)")
    parser.add_argument("--unfreeze-lm-head", action="store_true", default=False,
                        help="Unfreeze the lm_head layer so the model can learn to "
                             "produce action-range tokens instead of text tokens")
    parser.add_argument("--n-bins", type=int, default=16,
                        help="Number of discretization bins per action dim (default 16)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    setup_logging()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    logger.info("Loading model...")
    model, processor, _ = prepare_model(args.model, device)

    unfreeze_proj = args.unfreeze_projector and not args.freeze_projector
    logger.info("Applying LoRA adapters...")
    model = apply_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        unfreeze_projector=unfreeze_proj,
        unfreeze_vision_layers=args.unfreeze_vision_layers,
        unfreeze_lm_head=args.unfreeze_lm_head,
    )

    logger.info("Starting training...")
    train(
        model, processor,
        manifest_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        val_split=args.val_split,
        patience=args.patience,
        n_bins=args.n_bins,
        device=device,
        use_wandb=args.wandb,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
