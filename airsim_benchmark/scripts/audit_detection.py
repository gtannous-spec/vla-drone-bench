#!/usr/bin/env python3
"""
audit_detection.py — Test GroundingDINO detection rates on saved AirSim frames.

Runs every query against a sample of frames and reports detection rates,
scores, and bbox sizes. Outputs a CSV + summary table.

Usage:
    python -m airsim_benchmark.scripts.audit_detection \
        --frames-dir logs/airsim_output/detection_bias0.0/frames/mission_16 \
        --output detection_audit.csv

    # On the cluster (needs GPU):
    sbatch --export=ALL scripts/run_audit_detection.slurm
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airsim_benchmark.core.detection_inference import ObjectDetector

logger = logging.getLogger(__name__)

QUERIES = [
    "red car",
    "car",
    "vehicle",
    "truck",
    "rooftop",
    "building",
    "house",
    "roof",
    "structure",
    "intersection",
    "road",
    "street",
    "mailbox",
    "tree",
    "fence",
    "driveway",
    "sidewalk",
    "lawn",
    "white house",
    "two-story house",
    "swimming pool",
    "parking lot",
    "fire hydrant",
    "street light",
    "bench",
    "stop sign",
    "garage",
    "garden",
    "window",
    "door",
]


def load_frames(frames_dir: str, max_frames: int = 50, stride: int = 0):
    """Load a spread of frames from the directory."""
    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.jpg/png in {frames_dir}")

    if stride <= 0:
        stride = max(1, len(frame_files) // max_frames)

    selected = frame_files[::stride][:max_frames]
    logger.info(f"Selected {len(selected)}/{len(frame_files)} frames (stride={stride})")

    frames = []
    for fp in selected:
        img = np.array(Image.open(fp).convert("RGB"))
        frames.append((fp.name, img))
    return frames


def run_audit(detector, frames, queries):
    """Run every query against every frame, collect results."""
    results = []
    total = len(frames) * len(queries)
    done = 0

    for frame_name, frame_img in frames:
        for query in queries:
            detections = detector.detect(frame_img, query)
            best = detections[0] if detections else None

            results.append({
                "frame": frame_name,
                "query": query,
                "detected": best is not None,
                "score": round(best.score, 3) if best else 0.0,
                "bbox_area_pct": round(best.area_ratio(640, 480) * 100, 2) if best else 0.0,
                "center_x": round(best.center_x, 1) if best else None,
                "center_y": round(best.center_y, 1) if best else None,
                "num_detections": len(detections),
            })

            done += 1
            if done % 100 == 0:
                logger.info(f"  Progress: {done}/{total} ({done*100//total}%)")

    return results


def summarize(results, queries):
    """Print a summary table of detection rates per query."""
    print(f"\n{'='*75}")
    print(f"  GroundingDINO Detection Audit — {len(set(r['frame'] for r in results))} frames")
    print(f"{'='*75}")
    print(f"{'Query':<22} {'Det%':>6} {'AvgScore':>9} {'AvgArea%':>9} {'MaxScore':>9} {'Count':>6}")
    print(f"{'-'*75}")

    rows = []
    for query in queries:
        qr = [r for r in results if r["query"] == query]
        n_frames = len(qr)
        n_detected = sum(1 for r in qr if r["detected"])
        det_pct = (n_detected / n_frames * 100) if n_frames > 0 else 0

        scores = [r["score"] for r in qr if r["detected"]]
        areas = [r["bbox_area_pct"] for r in qr if r["detected"]]

        avg_score = np.mean(scores) if scores else 0
        max_score = max(scores) if scores else 0
        avg_area = np.mean(areas) if areas else 0

        rows.append((query, det_pct, avg_score, avg_area, max_score, n_detected))

    rows.sort(key=lambda r: r[1], reverse=True)

    for query, det_pct, avg_score, avg_area, max_score, n_det in rows:
        marker = "✓" if det_pct > 30 else "✗" if det_pct == 0 else "~"
        print(f"{marker} {query:<20} {det_pct:>5.1f}% {avg_score:>8.3f} {avg_area:>8.2f}% {max_score:>8.3f} {n_det:>5}")

    print(f"{'='*75}")

    good = sum(1 for _, dp, *_ in rows if dp > 30)
    partial = sum(1 for _, dp, *_ in rows if 0 < dp <= 30)
    failed = sum(1 for _, dp, *_ in rows if dp == 0)
    print(f"\n  ✓ Reliable (>30%): {good}  ~ Partial (1-30%): {partial}  ✗ Failed (0%): {failed}")
    print(f"  Total queries tested: {len(queries)}\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Audit GroundingDINO detection on AirSim frames")
    parser.add_argument("--frames-dir",
                        default="logs/airsim_output/detection_bias0.0/frames/mission_16",
                        help="Directory with frame_*.jpg files")
    parser.add_argument("--output", default="data/detection_audit.csv",
                        help="Output CSV path")
    parser.add_argument("--max-frames", type=int, default=40,
                        help="Max frames to test (evenly spaced)")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base",
                        help="GroundingDINO model ID")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Custom queries (default: built-in list of 30)")
    parser.add_argument("--lora-path", default=None,
                        help="Path to LoRA adapter for fine-tuned model")
    args = parser.parse_args()

    logger.info(f"Loading GroundingDINO: {args.model}")
    detector = ObjectDetector(model_id=args.model, lora_path=args.lora_path)

    logger.info(f"Loading frames from: {args.frames_dir}")
    frames = load_frames(args.frames_dir, max_frames=args.max_frames)

    queries = args.queries or QUERIES
    logger.info(f"Testing {len(queries)} queries × {len(frames)} frames = {len(queries)*len(frames)} detections")

    results = run_audit(detector, frames, queries)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to {args.output}")

    summarize(results, queries)


if __name__ == "__main__":
    main()
