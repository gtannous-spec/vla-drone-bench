#!/usr/bin/env python3
"""
plot_flights.py — Generate 2D and 3D flight-path plots from AirSim trajectory CSVs.

Usage:
    python -m airsim_benchmark.scripts.plot_flights <trajectory_dir> \
        [--output <plot_dir>] [--config <yaml>]

Produces per-task plots and a combined overview, matching the Gazebo PoC style:
    - Phase-colored trajectories (blue=takeoff, green=navigate, orange=landing)
    - Green dot = start, red star = goal
    - Semi-transparent building bounding boxes for spatial context
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import yaml

# Approximate building geometry from AirSim Neighborhood environment.
# Positions and sizes in NED (metres), estimated from the map.
BUILDINGS = [
    {"name": "House A",  "pos": (30, -20, -4),   "size": (12, 10, 8),  "color": "#8888aa"},
    {"name": "House B",  "pos": (60, 10, -3.5),  "size": (14, 12, 7),  "color": "#b08060"},
    {"name": "House C",  "pos": (90, -40, -4),   "size": (12, 10, 8),  "color": "#888880"},
    {"name": "House D",  "pos": (45, 50, -5),    "size": (16, 12, 10), "color": "#7899aa"},
    {"name": "House E",  "pos": (-20, 40, -3),   "size": (10, 10, 6),  "color": "#d8d0c0"},
    {"name": "House F",  "pos": (110, -10, -3),  "size": (12, 14, 6),  "color": "#c0b898"},
]

PHASE_COLORS = {
    "IDLE":     "#999999",
    "TAKEOFF":  "#2196F3",
    "NAVIGATE": "#4CAF50",
    "LAND":     "#FF9800",
    "DONE":     "#9E9E9E",
}


def load_trajectory(path: str):
    """Load a trajectory CSV into numpy arrays."""
    xs, ys, zs, phases = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            phases.append(row["phase"])
    return np.array(xs), np.array(ys), np.array(zs), phases


def load_goals(config_path: str) -> dict:
    """Return {task_id: goal_ned} from benchmark config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {t["id"]: t["goal"] for t in cfg.get("tasks", [])}


def draw_building_3d(ax, bld, alpha=0.12):
    """Draw a semi-transparent 3D box for a building."""
    cx, cy, cz = bld["pos"]
    sx, sy, sz = bld["size"]
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0 = 0
    z1 = cz - sz / 2  # NED: goes more negative = taller

    # Ensure z0 > z1 for proper rendering in NED
    if z0 < z1:
        z0, z1 = z1, z0

    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    col = Poly3DCollection(
        verts, alpha=alpha, facecolor=bld["color"],
        edgecolor=bld["color"], linewidth=0.4,
    )
    ax.add_collection3d(col)


def draw_building_2d(ax, bld, alpha=0.15):
    """Draw a 2D rectangle for a building (top-down view)."""
    cx, cy, _ = bld["pos"]
    sx, sy, _ = bld["size"]
    rect = Rectangle(
        (cx - sx / 2, cy - sy / 2), sx, sy,
        facecolor=bld["color"], edgecolor=bld["color"],
        alpha=alpha, linewidth=0.5,
    )
    ax.add_patch(rect)


def plot_phase_segments(ax, xs, ys, zs, phases, is_3d=True):
    """Plot trajectory colored by flight phase."""
    prev_phase = phases[0]
    seg_start = 0
    for i in range(1, len(phases)):
        if phases[i] != prev_phase or i == len(phases) - 1:
            end = i + 1 if i == len(phases) - 1 else i + 1
            c = PHASE_COLORS.get(prev_phase, "#333333")
            if is_3d:
                ax.plot(xs[seg_start:end], ys[seg_start:end], zs[seg_start:end],
                        color=c, linewidth=1.8, alpha=0.9)
            else:
                ax.plot(xs[seg_start:end], ys[seg_start:end],
                        color=c, linewidth=1.8, alpha=0.9)
            seg_start = i
            prev_phase = phases[i]


def plot_single_task(xs, ys, zs, phases, task_id, goal, out_path):
    """Generate combined 2D (top-down) and 3D plot for a single task."""
    fig = plt.figure(figsize=(16, 7))

    # --- 3D subplot ---
    ax3d = fig.add_subplot(121, projection="3d")
    for bld in BUILDINGS:
        draw_building_3d(ax3d, bld)
    plot_phase_segments(ax3d, xs, ys, zs, phases, is_3d=True)

    ax3d.scatter([xs[0]], [ys[0]], [zs[0]], color="#2E7D32", s=80,
                 marker="o", zorder=5, label="Start")
    if goal:
        ax3d.scatter([goal[0]], [goal[1]], [goal[2]], color="#D32F2F", s=120,
                     marker="*", zorder=5, label="Goal")

    ax3d.set_xlabel("North (m)")
    ax3d.set_ylabel("East (m)")
    ax3d.set_zlabel("Down (m)")
    ax3d.set_title(f"Task {task_id} — 3D Flight Path", fontsize=12, fontweight="bold")
    ax3d.legend(loc="upper left", fontsize=9)
    ax3d.view_init(elev=28, azim=-55)

    # --- 2D top-down subplot ---
    ax2d = fig.add_subplot(122)
    for bld in BUILDINGS:
        draw_building_2d(ax2d, bld)
    plot_phase_segments(ax2d, xs, ys, zs, phases, is_3d=False)

    ax2d.scatter([xs[0]], [ys[0]], color="#2E7D32", s=80,
                 marker="o", zorder=5, label="Start")
    if goal:
        ax2d.scatter([goal[0]], [goal[1]], color="#D32F2F", s=120,
                     marker="*", zorder=5, label="Goal")

    ax2d.set_xlabel("North (m)")
    ax2d.set_ylabel("East (m)")
    ax2d.set_title(f"Task {task_id} — Top-Down View", fontsize=12, fontweight="bold")
    ax2d.legend(loc="upper left", fontsize=9)
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_combined(all_data, goals, out_path):
    """Generate combined overview with all tasks on one figure."""
    fig = plt.figure(figsize=(16, 7))

    task_colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2", "#FBC02D"]

    # --- 3D combined ---
    ax3d = fig.add_subplot(121, projection="3d")
    for bld in BUILDINGS:
        draw_building_3d(ax3d, bld, alpha=0.08)

    for idx, (task_id, xs, ys, zs, _) in enumerate(all_data):
        c = task_colors[idx % len(task_colors)]
        ax3d.plot(xs, ys, zs, color=c, linewidth=1.5, alpha=0.85,
                  label=f"Task {task_id}")
        ax3d.scatter([xs[0]], [ys[0]], [zs[0]], color=c, s=50, marker="o", zorder=5)
        goal = goals.get(task_id)
        if goal:
            ax3d.scatter([goal[0]], [goal[1]], [goal[2]], color=c, s=100,
                         marker="*", zorder=5)

    ax3d.set_xlabel("North (m)")
    ax3d.set_ylabel("East (m)")
    ax3d.set_zlabel("Down (m)")
    ax3d.set_title("AirSim Benchmark — All Flights (3D)", fontsize=13, fontweight="bold")
    ax3d.legend(loc="upper left", fontsize=9)
    ax3d.view_init(elev=32, azim=-50)

    # --- 2D combined ---
    ax2d = fig.add_subplot(122)
    for bld in BUILDINGS:
        draw_building_2d(ax2d, bld, alpha=0.10)

    for idx, (task_id, xs, ys, zs, _) in enumerate(all_data):
        c = task_colors[idx % len(task_colors)]
        ax2d.plot(xs, ys, color=c, linewidth=1.5, alpha=0.85,
                  label=f"Task {task_id}")
        ax2d.scatter([xs[0]], [ys[0]], color=c, s=50, marker="o", zorder=5)
        goal = goals.get(task_id)
        if goal:
            ax2d.scatter([goal[0]], [goal[1]], color=c, s=100, marker="*", zorder=5)

    ax2d.set_xlabel("North (m)")
    ax2d.set_ylabel("East (m)")
    ax2d.set_title("AirSim Benchmark — All Flights (Top-Down)", fontsize=13, fontweight="bold")
    ax2d.legend(loc="upper left", fontsize=9)
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def add_phase_legend(fig):
    """Add a legend explaining the phase colors."""
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PHASE_COLORS["TAKEOFF"], lw=2, label="Takeoff"),
        Line2D([0], [0], color=PHASE_COLORS["NAVIGATE"], lw=2, label="Navigate"),
        Line2D([0], [0], color=PHASE_COLORS["LAND"], lw=2, label="Landing"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description="Plot AirSim drone flight trajectories")
    parser.add_argument("trajectory_dir",
                        help="Directory containing task_*_trajectory.csv files")
    parser.add_argument("--output", default=None,
                        help="Output directory for plots (default: <traj_dir>/../plots)")
    parser.add_argument("--config", default=None,
                        help="Path to benchmark_config.yaml for goal markers")
    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)
    if not traj_dir.is_dir():
        print(f"ERROR: {traj_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else traj_dir.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    goals = {}
    if args.config and Path(args.config).is_file():
        goals = load_goals(args.config)

    csv_files = sorted(traj_dir.glob("task_*_trajectory.csv"))
    if not csv_files:
        print(f"No trajectory CSVs found in {traj_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} trajectory file(s)")

    all_data = []
    for csv_path in csv_files:
        task_id = int(csv_path.stem.split("_")[1])
        xs, ys, zs, phases = load_trajectory(str(csv_path))
        goal = goals.get(task_id)
        plot_single_task(
            xs, ys, zs, phases, task_id, goal,
            str(out_dir / f"task_{task_id}_flight.png"),
        )
        all_data.append((task_id, xs, ys, zs, phases))

    if len(all_data) > 1:
        plot_combined(all_data, goals, str(out_dir / "all_flights.png"))

    print(f"\nDone — {len(all_data)} plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
