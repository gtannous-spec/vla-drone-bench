"""
finetune_dino.py — LoRA fine-tuning of GroundingDINO for aerial drone views.

Adapts the IDEA-Research/grounding-dino-base model to detect objects that are
poorly recognized from top-down perspectives (rooftop, tree, fence, intersection,
garage, structure, mailbox) using parameter-efficient LoRA on the Swin backbone.

Training data: CSV annotations + JPEG frames from collect_dino_data.py.
"""

import argparse
import copy
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class DinoFinetuneDataset(Dataset):
    """Dataset for GroundingDINO fine-tuning from CSV annotations.

    Each sample corresponds to one image with all its annotated bounding boxes.
    Builds text prompts from the unique classes present in each image, and returns
    preprocessed inputs ready for the GroundingDINO forward pass.
    """

    def __init__(self, annotations: list, image_dir: Path, processor):
        self.processor = processor
        self.image_dir = image_dir

        grouped = defaultdict(list)
        for row in annotations:
            grouped[row["image_name"]].append(row)

        self.samples = []
        for image_name, rows in grouped.items():
            image_path = image_dir / image_name
            if not image_path.exists():
                logger.warning(f"Image not found, skipping: {image_path}")
                continue
            if len(rows) == 0:
                continue
            self.samples.append((image_name, rows))

        logger.info(f"Dataset: {len(self.samples)} images with annotations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, rows = self.samples[idx]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        classes = []
        for row in rows:
            if row["label_name"] not in classes:
                classes.append(row["label_name"])

        text_prompt = " . ".join(classes) + " ."

        boxes = []
        class_labels = []
        for row in rows:
            x1 = float(row["bbox_x1"]) / img_w
            y1 = float(row["bbox_y1"]) / img_h
            x2 = float(row["bbox_x2"]) / img_w
            y2 = float(row["bbox_y2"]) / img_h

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            boxes.append([cx, cy, w, h])
            class_labels.append(classes.index(row["label_name"]))

        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)

        labels = {
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(batch):
    """Custom collate — each image may have different text length and box count."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = [item["input_ids"] for item in batch]
    max_len = max(ids.size(0) for ids in input_ids)
    padded_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded_ids[i, : ids.size(0)] = ids
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "input_ids": padded_ids,
        "labels": labels,
    }


class EMA:
    """Exponential Moving Average of model parameters for training stability."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module):
        """Replace model params with EMA shadow params."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original params from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def load_annotations(csv_path: Path) -> list:
    """Load annotation rows from CSV file."""
    annotations = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(row)
    logger.info(f"Loaded {len(annotations)} annotation rows from {csv_path}")
    return annotations


def split_data(annotations: list, val_split: float, seed: int = 42):
    """Split annotations by image into train/val sets."""
    import random

    images = list({row["image_name"] for row in annotations})
    rng = random.Random(seed)
    rng.shuffle(images)

    n_val = max(1, int(len(images) * val_split))
    val_images = set(images[:n_val])
    train_images = set(images[n_val:])

    train_ann = [r for r in annotations if r["image_name"] in train_images]
    val_ann = [r for r in annotations if r["image_name"] in val_images]

    logger.info(
        f"Split: {len(train_images)} train images, {len(val_images)} val images"
    )
    return train_ann, val_ann


def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then linear decay scheduler."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, processor, val_dataset, device, num_samples=10):
    """Run detection on held-out frames and report per-class detection rates."""
    model.eval()
    num_samples = min(num_samples, len(val_dataset))
    if num_samples == 0:
        model.train()
        return {}

    class_hits = defaultdict(int)
    class_total = defaultdict(int)

    for i in range(num_samples):
        sample = val_dataset[i]
        image_name, rows = val_dataset.samples[i]
        image_path = val_dataset.image_dir / image_name
        image = Image.open(image_path).convert("RGB")

        classes = []
        for row in rows:
            if row["label_name"] not in classes:
                classes.append(row["label_name"])

        text_prompt = " . ".join(classes) + " ."

        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(
            device
        )

        with autocast(dtype=torch.float16):
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.25,
            text_threshold=0.20,
            target_sizes=[image.size[::-1]],
        )[0]

        detected_labels = set()
        for label in results.get("labels", []):
            detected_labels.add(label.strip().lower())

        for cls in classes:
            class_total[cls] += 1
            if cls.lower() in detected_labels:
                class_hits[cls] += 1

    detection_rates = {}
    for cls in sorted(class_total.keys()):
        rate = class_hits[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        detection_rates[cls] = rate
        logger.info(f"  Val detection rate [{cls}]: {rate:.1%}")

    avg_rate = (
        sum(detection_rates.values()) / len(detection_rates)
        if detection_rates
        else 0.0
    )
    logger.info(f"  Val average detection rate: {avg_rate:.1%}")

    model.train()
    return detection_rates


def train(args):
    """Main training loop."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "annotations.csv"
    image_dir = data_dir / "images"

    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    logger.info(f"Loading processor from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    logger.info(f"Loading model from {args.model_id}...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model_id, trust_remote_code=True
    )

    model.config.auxiliary_loss = False
    logger.info("Disabled auxiliary loss (prevents exploding encoder loss_ce_enc)")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = model.to(device)
    model.train()

    annotations = load_annotations(csv_path)
    train_ann, val_ann = split_data(annotations, args.val_split)

    train_dataset = DinoFinetuneDataset(train_ann, image_dir, processor)
    val_dataset = DinoFinetuneDataset(val_ann, image_dir, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    scaler = GradScaler()
    ema = EMA(model, decay=0.999)
    grad_accum_steps = 4

    best_avg_rate = 0.0
    global_step = 0

    logger.info(f"Training for {args.epochs} epochs, {total_steps} total steps")
    logger.info(f"Warmup steps: {warmup_steps}, grad accumulation: {grad_accum_steps}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = [
                {k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]
            ]

            with autocast(dtype=torch.float16):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels,
                )
                ld = outputs.loss_dict
                loss = (
                    ld["loss_ce"]
                    + ld["loss_bbox"] * 5.0
                    + ld["loss_giou"] * 2.0
                ) / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (
                batch_idx + 1
            ) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=0.1
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                ema.update(model)
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps
            n_batches += 1

            if global_step % 10 == 0 and global_step > 0:
                avg_loss = epoch_loss / n_batches
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}/{args.epochs} | Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                )

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch} complete — avg loss: {avg_epoch_loss:.4f}")

        if epoch % 10 == 0:
            logger.info(f"--- Validation at epoch {epoch} ---")
            ema.apply(model)
            detection_rates = validate(
                model, processor, val_dataset, device, num_samples=10
            )
            ema.restore(model)

            if detection_rates:
                avg_rate = sum(detection_rates.values()) / len(detection_rates)
                if avg_rate > best_avg_rate:
                    best_avg_rate = avg_rate
                    best_dir = output_dir / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    ema.apply(model)
                    model.save_pretrained(best_dir)
                    ema.restore(model)
                    logger.info(
                        f"New best model saved (avg rate: {avg_rate:.1%}) → {best_dir}"
                    )

            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            logger.info(f"Checkpoint saved → {ckpt_dir}")

    logger.info("Training complete. Saving final model...")
    ema.apply(model)

    final_lora_dir = output_dir / "final-lora"
    final_lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_lora_dir)
    logger.info(f"LoRA adapter saved → {final_lora_dir}")

    if args.save_merged:
        logger.info("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        logger.info(f"Merged model saved → {merged_dir}")

    logger.info(f"Best average detection rate achieved: {best_avg_rate:.1%}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GroundingDINO with LoRA for aerial drone detection"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/dino_finetune",
        help="Path to directory with images/ and annotations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/dino-aerial-lora",
        help="Where to save the fine-tuned model",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Base GroundingDINO HuggingFace model ID",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--save-merged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to also save a merged (full-weight) model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
