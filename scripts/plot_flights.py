#!/usr/bin/env python3
"""
plot_flights.py — Generate 3D flight-path plots from trajectory CSVs.

Usage:
    python3 plot_flights.py <trajectory_dir> [--output <plot_dir>] [--config <yaml>]

Reads task_*_trajectory.csv files and produces per-task 3D plots plus a
combined overview.  Uses the Agg backend so it works headless.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import yaml

# Building geometry extracted from urban_city.sdf (pose = centre, size = extents)
BUILDINGS = [
    {'name': 'Building A',  'pos': (20, -8, 10),    'size': (10, 10, 20), 'color': '#8888aa'},
    {'name': 'Building B',  'pos': (-18, 15, 6),    'size': (12, 8, 12),  'color': '#b08060'},
    {'name': 'Building C',  'pos': (-12, -15, 3),   'size': (14, 10, 6),  'color': '#888880'},
    {'name': 'Building D',  'pos': (30, 5, 12.5),   'size': (8, 8, 25),   'color': '#7899aa'},
    {'name': 'Building E',  'pos': (-30, -5, 4),    'size': (10, 12, 8),  'color': '#d8d0c0'},
    {'name': 'Building F',  'pos': (12, 25, 2.5),   'size': (6, 6, 5),    'color': '#c0b898'},
]

PHASE_COLORS = {
    'IDLE':     '#999999',
    'TAKEOFF':  '#2196F3',
    'NAVIGATE': '#4CAF50',
    'LAND':     '#FF9800',
    'DONE':     '#9E9E9E',
}


def load_trajectory(path: str):
    xs, ys, zs, phases = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row['x_enu']))
            ys.append(float(row['y_enu']))
            zs.append(float(row['z_enu']))
            phases.append(row['phase'])
    return np.array(xs), np.array(ys), np.array(zs), phases


def load_goals(config_path: str) -> dict:
    """Return {task_id: goal_enu} from benchmark config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {t['id']: t['goal'] for t in cfg.get('tasks', [])}


def draw_building(ax, bld, alpha=0.12):
    """Draw a semi-transparent box for a building."""
    cx, cy, cz = bld['pos']
    sx, sy, sz = bld['size']
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = 0, cz + sz / 2

    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    col = Poly3DCollection(verts, alpha=alpha, facecolor=bld['color'],
                           edgecolor=bld['color'], linewidth=0.4)
    ax.add_collection3d(col)


def plot_single_task(xs, ys, zs, phases, task_id, goal, out_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for bld in BUILDINGS:
        draw_building(ax, bld)

    # Color segments by phase
    prev_phase = phases[0]
    seg_start = 0
    for i in range(1, len(phases)):
        if phases[i] != prev_phase or i == len(phases) - 1:
            end = i + 1 if i == len(phases) - 1 else i + 1
            c = PHASE_COLORS.get(prev_phase, '#333333')
            ax.plot(xs[seg_start:end], ys[seg_start:end], zs[seg_start:end],
                    color=c, linewidth=1.8, alpha=0.9)
            seg_start = i
            prev_phase = phases[i]

    ax.scatter(*[[xs[0]], [ys[0]], [zs[0]]], color='#2E7D32', s=80,
              marker='o', zorder=5, label='Start')
    if goal:
        ax.scatter(*[[goal[0]], [goal[1]], [goal[2]]], color='#D32F2F', s=120,
                  marker='*', zorder=5, label='Goal')

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title(f'Task {task_id} — Flight Path', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=28, azim=-55)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_combined(all_data, goals, out_path):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for bld in BUILDINGS:
        draw_building(ax, bld, alpha=0.08)

    task_colors = ['#1976D2', '#E64A19', '#388E3C', '#7B1FA2', '#FBC02D']
    for idx, (task_id, xs, ys, zs, _phases) in enumerate(all_data):
        c = task_colors[idx % len(task_colors)]
        ax.plot(xs, ys, zs, color=c, linewidth=1.5, alpha=0.85,
                label=f'Task {task_id}')
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=c, s=50, marker='o', zorder=5)
        goal = goals.get(task_id)
        if goal:
            ax.scatter([goal[0]], [goal[1]], [goal[2]], color=c, s=100,
                      marker='*', zorder=5)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('Milestone 1 — All Flight Paths', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=32, azim=-50)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot drone flight trajectories')
    parser.add_argument('trajectory_dir', help='Directory containing task_*_trajectory.csv files')
    parser.add_argument('--output', default=None, help='Output directory for plots (default: <trajectory_dir>/../plots)')
    parser.add_argument('--config', default=None, help='Path to benchmark_config.yaml')
    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)
    if not traj_dir.is_dir():
        print(f'ERROR: {traj_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else traj_dir.parent / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    goals = {}
    if args.config and os.path.isfile(args.config):
        goals = load_goals(args.config)

    csv_files = sorted(traj_dir.glob('task_*_trajectory.csv'))
    if not csv_files:
        print(f'No trajectory CSVs found in {traj_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(csv_files)} trajectory file(s)')

    all_data = []
    for csv_path in csv_files:
        task_id = int(csv_path.stem.split('_')[1])
        xs, ys, zs, phases = load_trajectory(str(csv_path))
        goal = goals.get(task_id)
        plot_single_task(xs, ys, zs, phases, task_id, goal,
                         str(out_dir / f'task_{task_id}_flight.png'))
        all_data.append((task_id, xs, ys, zs, phases))

    if len(all_data) > 1:
        plot_combined(all_data, goals, str(out_dir / 'all_flights.png'))

    print(f'\nDone — {len(all_data)} plot(s) in {out_dir}')


if __name__ == '__main__':
    main()
