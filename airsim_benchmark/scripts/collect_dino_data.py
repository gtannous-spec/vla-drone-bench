#!/usr/bin/env python3
"""
collect_dino_data.py — Auto-annotate AirSim frames for GroundingDINO fine-tuning.

Flies a grid of poses over the AirSim Neighborhood, captures RGB + segmentation
mask pairs, extracts per-class bounding boxes from the segmentation, and writes
annotations in CSV format suitable for GroundingDINO / DINO fine-tuning.

Output:
    {output_dir}/images/*.jpg           — RGB frames
    {output_dir}/annotations.csv        — label_name,bbox_x1,...,image_name,...

Usage:
    python -m airsim_benchmark.scripts.collect_dino_data \\
        --output data/dino_finetune --weather clear
"""

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airsim_benchmark.core.sync_rpc import SyncClient, SyncAddress
import msgpackrpc
msgpackrpc.Client = SyncClient
msgpackrpc.Address = SyncAddress
import airsim

logger = logging.getLogger(__name__)

# ── Segmentation class mapping ──────────────────────────────────────────────
SEG_CLASSES = {
    10: {"pattern": r"Roof[\w]*",     "label": "rooftop",  "is_regex": True},
    20: {"pattern": r"Tree[\w]*",     "label": "tree",     "is_regex": True},
    30: {"pattern": r"Fence[\w]*",    "label": "fence",    "is_regex": True},
    40: {"pattern": r"Road[\w]*",     "label": "road",     "is_regex": True},
    50: {"pattern": r"Car[\w]*",      "label": "car",      "is_regex": True},
    60: {"pattern": r"Driveway[\w]*", "label": "driveway", "is_regex": True},
    70: {"pattern": r"Garage[\w]*",   "label": "garage",   "is_regex": True},
    80: {"pattern": r"House[\w]*",    "label": "house",    "is_regex": True},
    85: {"pattern": r"Wall[\w]*",     "label": "house",    "is_regex": True},
}

# ── Grid parameters ─────────────────────────────────────────────────────────
X_RANGE = range(-50, 131, 20)
Y_RANGE = range(-60, 81, 20)
YAW_ANGLES = [0, 90, 180, 270]
DEFAULT_ALTITUDES = [-8, -12, -18, -25]

MIN_BOX_AREA = 100


def setup_segmentation(client: airsim.MultirotorClient) -> None:
    """Reset all segmentation IDs to 0, then assign per-class IDs."""
    logger.info("Resetting all segmentation IDs to 0...")
    ok = client.simSetSegmentationObjectID(r"[\w]*", 0, True)
    logger.info("Reset all → 0: %s", "OK" if ok else "FAILED")

    for seg_id, info in SEG_CLASSES.items():
        ok = client.simSetSegmentationObjectID(
            info["pattern"], seg_id, info["is_regex"]
        )
        status = "found" if ok else "NOT found"
        logger.info(
            "  seg_id=%3d  %-20s  → %s  (%s)",
            seg_id, info["label"], info["pattern"], status,
        )


def extract_bboxes_binary(seg_mask: np.ndarray, label: str) -> list:
    """Extract bounding boxes from a binary segmentation mask.

    AirSim maps stencil IDs through a color palette, so we can't read the
    raw ID from the R channel. Instead, we use a per-class binary approach:
    the target class is set to seg_id 255 while everything else is 0. Any
    non-zero pixel belongs to the target class.

    Uses connected-component labeling to split disjoint objects of the same
    class into separate bounding boxes.

    Returns a list of dicts: {label, x1, y1, x2, y2}.
    """
    h, w = seg_mask.shape[:2]
    mask = np.any(seg_mask > 10, axis=2).astype(np.uint8)

    if not mask.any():
        return []

    try:
        import cv2
        n_labels, labels_map = cv2.connectedComponents(mask)
    except ImportError:
        n_labels, labels_map = 1, mask

    boxes = []
    if n_labels <= 1 and mask.any():
        ys, xs = np.where(mask > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        area = (x2 - x1) * (y2 - y1)
        if area >= MIN_BOX_AREA and not (x1 == 0 and y1 == 0 and x2 == w - 1 and y2 == h - 1):
            boxes.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return boxes

    for comp_id in range(1, n_labels):
        ys, xs = np.where(labels_map == comp_id)
        if len(ys) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            continue
        if x1 == 0 and y1 == 0 and x2 == w - 1 and y2 == h - 1:
            continue
        boxes.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return boxes


def build_pose_grid(altitudes: list, max_positions: int = 0) -> list:
    """Build the full grid of (x, y, z, yaw) poses to visit."""
    poses = []
    for x in X_RANGE:
        for y in Y_RANGE:
            for z in altitudes:
                for yaw in YAW_ANGLES:
                    poses.append((x, y, z, yaw))

    if max_positions > 0:
        poses = poses[:max_positions]

    return poses


def set_weather(client: airsim.MultirotorClient, weather: str) -> None:
    """Configure weather based on CLI flag."""
    if weather == "clear":
        return

    client.simEnableWeather(True)
    if weather == "fog":
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.3)
        logger.info("Weather: fog=0.3")
    elif weather == "rain":
        client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.4)
        logger.info("Weather: rain=0.4")


def collect(args: argparse.Namespace) -> None:
    """Main collection loop."""
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "annotations.csv"
    altitudes = args.altitudes

    # ── Connect to AirSim ────────────────────────────────────────────────
    logger.info("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name="Drone0")
    client.armDisarm(True, vehicle_name="Drone0")
    logger.info("Connected — API control enabled.")

    # ── Setup segmentation IDs ───────────────────────────────────────────
    setup_segmentation(client)

    # ── Weather ──────────────────────────────────────────────────────────
    set_weather(client, args.weather)

    # ── Build pose grid ──────────────────────────────────────────────────
    poses = build_pose_grid(altitudes, args.max_positions)
    total = len(poses)
    logger.info("Pose grid: %d positions to visit", total)

    # ── Build list of active classes (only those whose meshes exist) ────
    active_classes = {
        sid: info for sid, info in SEG_CLASSES.items()
        if client.simSetSegmentationObjectID(info["pattern"], 0, info["is_regex"])
    }
    logger.info("Active classes for collection: %s",
                [info["label"] for info in active_classes.values()])

    # ── Collection loop ──────────────────────────────────────────────────
    # Per-class binary approach: for each pose, iterate classes and capture
    # a segmentation image with only that class highlighted (seg_id=255).
    # This avoids the AirSim palette mapping issue entirely.
    all_rows = []
    class_counts = defaultdict(int)
    frames_saved = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "image_name", "image_width", "image_height",
        ])

        for idx, (x, y, z, yaw) in enumerate(poses):
            pose = airsim.Pose(
                airsim.Vector3r(x, y, z),
                airsim.to_quaternion(0, 0, np.radians(yaw)),
            )
            client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name="Drone0")
            client.enableApiControl(True, vehicle_name="Drone0")
            client.armDisarm(True, vehicle_name="Drone0")
            time.sleep(0.15)

            rgb_resp = client.simGetImages(
                [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)],
                vehicle_name="Drone0",
            )[0]
            if rgb_resp.width == 0:
                continue
            rgb = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(
                rgb_resp.height, rgb_resp.width, 3,
            )

            frame_boxes = []
            for seg_id, info in active_classes.items():
                client.simSetSegmentationObjectID(r"[\w]*", 0, True)
                client.simSetSegmentationObjectID(info["pattern"], 255, info["is_regex"])
                time.sleep(0.03)

                seg_resp = client.simGetImages(
                    [airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)],
                    vehicle_name="Drone0",
                )[0]
                if seg_resp.width == 0:
                    continue
                seg = np.frombuffer(seg_resp.image_data_uint8, dtype=np.uint8).reshape(
                    seg_resp.height, seg_resp.width, 3,
                )
                frame_boxes.extend(extract_bboxes_binary(seg, info["label"]))

            if not frame_boxes:
                continue

            image_name = f"frame_{idx:06d}.jpg"
            img = Image.fromarray(rgb)
            img.save(images_dir / image_name, quality=95)

            h, w = rgb.shape[:2]
            for box in frame_boxes:
                writer.writerow([
                    box["label"],
                    box["x1"], box["y1"], box["x2"], box["y2"],
                    image_name, w, h,
                ])
                class_counts[box["label"]] += 1
                all_rows.append(box)

            frames_saved += 1

            if (idx + 1) % 50 == 0 or idx == total - 1:
                logger.info(
                    "Progress: %d/%d poses | %d frames saved | %d annotations",
                    idx + 1, total, frames_saved, len(all_rows),
                )

    # ── Summary ──────────────────────────────────────────────────────────
    client.armDisarm(False, vehicle_name="Drone0")
    client.enableApiControl(False, vehicle_name="Drone0")

    logger.info("=" * 60)
    logger.info("Collection complete.")
    logger.info("  Frames saved : %d", frames_saved)
    logger.info("  Total annotations : %d", len(all_rows))
    logger.info("  Output dir   : %s", output_dir)
    logger.info("  CSV          : %s", csv_path)
    logger.info("  Detection counts per class:")
    for label in sorted(class_counts):
        logger.info("    %-12s : %d", label, class_counts[label])
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect AirSim RGB+segmentation data for GroundingDINO fine-tuning",
    )
    parser.add_argument(
        "--output", type=str, default="data/dino_finetune",
        help="Output directory (default: data/dino_finetune)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=0,
        help="Max number of poses to visit (0 = all, default: 0)",
    )
    parser.add_argument(
        "--altitudes", type=float, nargs="+", default=DEFAULT_ALTITUDES,
        help="NED altitudes to fly (default: -8 -12 -18 -25)",
    )
    parser.add_argument(
        "--weather", type=str, default="clear",
        choices=["clear", "fog", "rain"],
        help="Weather preset (default: clear)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    collect(args)


if __name__ == "__main__":
    main()
