#!/usr/bin/env python3
"""
evaluate.py — Compute navigation-accuracy metrics from trajectory CSVs.

Reads task_*_trajectory.csv files produced by any planner (classical, VLA,
etc.), compares against the benchmark_config.yaml goals and constraints,
and writes a per-task + aggregate metrics.json.

Usage:
    python3 evaluate.py <trajectory_dir> \
        --config <benchmark_config.yaml> \
        [--baseline <classical_trajectory_dir>]
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml


# ── Trajectory I/O ──────────────────────────────────────────────────────────

def load_trajectory(path: str):
    """Return arrays (times, xs, ys, zs) and list of phase strings."""
    times, xs, ys, zs, phases = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            xs.append(float(row['x_enu']))
            ys.append(float(row['y_enu']))
            zs.append(float(row['z_enu']))
            phases.append(row['phase'])
    return (np.array(times), np.array(xs), np.array(ys), np.array(zs), phases)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Metric computation ─────────────────────────────────────────────────────

def euclidean(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def compute_path_length(xs, ys, zs):
    diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2)
    return float(np.sum(diffs))


def compute_smoothness(xs, ys, zs):
    """Mean heading-change rate in degrees per metre of travel.

    Only considers segments where horizontal displacement > 0.1 m to
    ignore hovering jitter that would inflate the metric.
    """
    dx = np.diff(xs)
    dy = np.diff(ys)
    headings = np.arctan2(dy, dx)
    heading_changes = np.abs(np.diff(headings))
    heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)

    seg_lengths = np.sqrt(dx[:-1]**2 + dy[:-1]**2)
    moving = seg_lengths > 0.1
    if not np.any(moving):
        return 0.0
    rate = np.degrees(heading_changes[moving]) / seg_lengths[moving]
    return float(np.mean(rate))


def count_constraint_violations(xs, ys, zs, constraints, origin=(0.0, 0.0)):
    """Count trajectory samples that breach altitude or geofence constraints."""
    min_alt = constraints.get('min_altitude', 0.0)
    max_alt = constraints.get('max_altitude', 100.0)
    geofence_r = constraints.get('geofence_radius', 1e6)

    violations = 0
    for x, y, z in zip(xs, ys, zs):
        if z < min_alt - 0.5 or z > max_alt + 0.5:
            violations += 1
        horiz_dist = math.sqrt((x - origin[0])**2 + (y - origin[1])**2)
        if horiz_dist > geofence_r:
            violations += 1
    return violations


def evaluate_task(times, xs, ys, zs, phases, task_cfg, global_cfg,
                  baseline_path_length=None):
    """Compute all metrics for a single task trajectory."""
    goal = task_cfg['goal']
    constraints = task_cfg['constraints']
    arrival_tol = global_cfg.get('arrival_tolerance', 0.5)

    # FDG: minimum distance to goal during NAVIGATE phase (before landing
    # moves the drone away from the aerial goal position).
    min_dist = float('inf')
    for i, ph in enumerate(phases):
        if ph == 'NAVIGATE':
            d = euclidean([float(xs[i]), float(ys[i]), float(zs[i])], goal)
            if d < min_dist:
                min_dist = d
    # Fallback: if NAVIGATE phase never occurred, use the last point
    if min_dist == float('inf'):
        min_dist = euclidean(
            [float(xs[-1]), float(ys[-1]), float(zs[-1])], goal)

    final_dist = min_dist
    success = final_dist < arrival_tol

    # Path length only during active flight (TAKEOFF + NAVIGATE)
    active_mask = np.array([p in ('TAKEOFF', 'NAVIGATE') for p in phases])
    active_xs = xs[active_mask]
    active_ys = ys[active_mask]
    active_zs = zs[active_mask]
    path_len = compute_path_length(active_xs, active_ys, active_zs)

    straight_line = euclidean(task_cfg['start'], goal)
    npl = path_len / straight_line if straight_line > 1e-6 else float('inf')

    opr = None
    if baseline_path_length is not None and baseline_path_length > 1e-6:
        opr = path_len / baseline_path_length

    takeoff_idx = None
    land_idx = None
    for i, ph in enumerate(phases):
        if ph == 'TAKEOFF' and takeoff_idx is None:
            takeoff_idx = i
        if ph == 'LAND' and land_idx is None:
            land_idx = i

    ttg = None
    if takeoff_idx is not None and land_idx is not None:
        ttg = float(times[land_idx] - times[takeoff_idx])

    # Constraint violations only during NAVIGATE (TAKEOFF/LAND naturally
    # breach altitude limits and should not be penalised)
    nav_mask = np.array([p == 'NAVIGATE' for p in phases])
    nav_xs = xs[nav_mask]
    nav_ys = ys[nav_mask]
    nav_zs = zs[nav_mask]

    violations = count_constraint_violations(nav_xs, nav_ys, nav_zs, constraints)
    smoothness = compute_smoothness(active_xs, active_ys, active_zs)

    return {
        'task_id': task_cfg['id'],
        'instruction': task_cfg['instruction'],
        'goal': goal,
        'success': success,
        'final_distance_to_goal': round(final_dist, 4),
        'path_length': round(path_len, 4),
        'straight_line_distance': round(straight_line, 4),
        'normalized_path_length': round(npl, 4),
        'oracle_path_ratio': round(opr, 4) if opr is not None else None,
        'time_to_goal_s': round(ttg, 2) if ttg is not None else None,
        'constraint_violations': violations,
        'smoothness_deg_per_m': round(smoothness, 4),
        'num_trajectory_points': len(xs),
    }


def aggregate_metrics(task_metrics: list) -> dict:
    n = len(task_metrics)
    successes = sum(1 for m in task_metrics if m['success'])
    fdgs = [m['final_distance_to_goal'] for m in task_metrics]
    npls = [m['normalized_path_length'] for m in task_metrics
            if m['normalized_path_length'] != float('inf')]
    ttgs = [m['time_to_goal_s'] for m in task_metrics
            if m['time_to_goal_s'] is not None]
    violations = sum(m['constraint_violations'] for m in task_metrics)
    oprs = [m['oracle_path_ratio'] for m in task_metrics
            if m['oracle_path_ratio'] is not None]

    return {
        'num_tasks': n,
        'success_rate': round(successes / n, 4) if n else 0.0,
        'mean_final_distance': round(float(np.mean(fdgs)), 4) if fdgs else None,
        'std_final_distance': round(float(np.std(fdgs)), 4) if fdgs else None,
        'mean_normalized_path_length': round(float(np.mean(npls)), 4) if npls else None,
        'mean_oracle_path_ratio': round(float(np.mean(oprs)), 4) if oprs else None,
        'mean_time_to_goal_s': round(float(np.mean(ttgs)), 2) if ttgs else None,
        'total_constraint_violations': violations,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate drone navigation trajectories')
    parser.add_argument('trajectory_dir',
                        help='Directory containing task_*_trajectory.csv')
    parser.add_argument('--config', required=True,
                        help='Path to benchmark_config.yaml')
    parser.add_argument('--baseline', default=None,
                        help='Directory with classical-planner CSVs (for OPR)')
    parser.add_argument('--output', default=None,
                        help='Output path for metrics.json (default: <traj_dir>/metrics.json)')
    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)
    if not traj_dir.is_dir():
        print(f'ERROR: {traj_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    tasks_by_id = {t['id']: t for t in cfg.get('tasks', [])}

    baseline_lengths = {}
    if args.baseline:
        bl_dir = Path(args.baseline)
        for csv_path in bl_dir.glob('task_*_trajectory.csv'):
            tid = int(csv_path.stem.split('_')[1])
            _, bxs, bys, bzs, _ = load_trajectory(str(csv_path))
            baseline_lengths[tid] = compute_path_length(bxs, bys, bzs)

    csv_files = sorted(traj_dir.glob('task_*_trajectory.csv'))
    if not csv_files:
        print(f'No trajectory CSVs found in {traj_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Evaluating {len(csv_files)} trajectory file(s) from {traj_dir}\n')

    task_metrics = []
    for csv_path in csv_files:
        tid = int(csv_path.stem.split('_')[1])
        if tid not in tasks_by_id:
            print(f'  WARNING: task {tid} not in config — skipping')
            continue
        times, xs, ys, zs, phases = load_trajectory(str(csv_path))
        bl = baseline_lengths.get(tid)
        m = evaluate_task(times, xs, ys, zs, phases,
                          tasks_by_id[tid], cfg, baseline_path_length=bl)
        task_metrics.append(m)

    agg = aggregate_metrics(task_metrics)

    result = {
        'aggregate': agg,
        'tasks': task_metrics,
    }

    out_path = args.output or str(traj_dir / 'metrics.json')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'{"Task":>6}  {"Success":>7}  {"FDG(m)":>8}  {"NPL":>7}  '
          f'{"OPR":>7}  {"TTG(s)":>7}  {"Viol":>5}  {"Smooth":>8}')
    print('-' * 68)
    for m in task_metrics:
        opr_str = f'{m["oracle_path_ratio"]:.2f}' if m['oracle_path_ratio'] else '   —'
        ttg_str = f'{m["time_to_goal_s"]:.1f}' if m['time_to_goal_s'] else '   —'
        print(f'{m["task_id"]:>6}  '
              f'{"  PASS" if m["success"] else "  FAIL":>7}  '
              f'{m["final_distance_to_goal"]:>8.3f}  '
              f'{m["normalized_path_length"]:>7.2f}  '
              f'{opr_str:>7}  '
              f'{ttg_str:>7}  '
              f'{m["constraint_violations"]:>5}  '
              f'{m["smoothness_deg_per_m"]:>8.2f}')

    print('-' * 68)
    print(f'SR={agg["success_rate"]:.0%}  '
          f'mean_FDG={agg["mean_final_distance"]:.3f}m  '
          f'mean_NPL={agg["mean_normalized_path_length"]:.2f}  '
          f'violations={agg["total_constraint_violations"]}')
    print(f'\nMetrics written to {out_path}')


if __name__ == '__main__':
    main()
