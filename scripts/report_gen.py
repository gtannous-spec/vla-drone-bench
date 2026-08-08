#!/usr/bin/env python3
"""
report_gen.py — Generate comparison tables and plots from ablation metrics.

Reads metrics.json files produced by evaluate.py for each planner mode
(classical, vla_only, vla_gt) and generates:
  - A comparison summary table (stdout + CSV)
  - Bar charts: SR, mean FDG, mean NPL per mode
  - Per-task scatter: FDG across modes
  - Box plots: path length distributions

Usage:
    python3 report_gen.py <base_trajectory_dir> \
        --output <report_dir> --config <benchmark_config.yaml>

Expects subdirectories like:
    <base_trajectory_dir>/classical/metrics.json
    <base_trajectory_dir>/vla_only/metrics.json
    <base_trajectory_dir>/vla_gt/metrics.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


MODE_LABELS = {
    'classical': 'Classical',
    'vla_only': 'VLA-only',
    'vla_gt': 'VLA + GT',
}

MODE_COLORS = {
    'classical': '#1976D2',
    'vla_only': '#E64A19',
    'vla_gt': '#388E3C',
}


def load_metrics(base_dir: Path) -> dict:
    """Load metrics.json from each mode subdirectory."""
    results = {}
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                results[subdir.name] = json.load(f)
    return results


def write_summary_csv(all_metrics: dict, out_path: str):
    """Write a CSV comparison table."""
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Mode', 'Success Rate', 'Mean FDG (m)', 'Std FDG (m)',
                     'Mean NPL', 'Mean OPR', 'Mean TTG (s)',
                     'Total Violations'])
        for mode, data in all_metrics.items():
            agg = data['aggregate']
            w.writerow([
                MODE_LABELS.get(mode, mode),
                agg.get('success_rate'),
                agg.get('mean_final_distance'),
                agg.get('std_final_distance'),
                agg.get('mean_normalized_path_length'),
                agg.get('mean_oracle_path_ratio', ''),
                agg.get('mean_time_to_goal_s', ''),
                agg.get('total_constraint_violations'),
            ])


def print_summary_table(all_metrics: dict):
    header = (f'{"Mode":<14} {"SR":>6} {"FDG(m)":>8} {"NPL":>7} '
              f'{"OPR":>7} {"TTG(s)":>7} {"Viol":>5}')
    print(header)
    print('-' * len(header))
    for mode, data in all_metrics.items():
        agg = data['aggregate']
        opr = agg.get('mean_oracle_path_ratio')
        ttg = agg.get('mean_time_to_goal_s')
        opr_str = f'{opr:>7.2f}' if opr is not None else f'{"—":>7}'
        ttg_str = f'{ttg:>7.1f}' if ttg is not None else f'{"—":>7}'
        print(f'{MODE_LABELS.get(mode, mode):<14} '
              f'{agg["success_rate"]:>5.0%} '
              f'{agg["mean_final_distance"]:>8.3f} '
              f'{agg["mean_normalized_path_length"]:>7.2f} '
              f'{opr_str} '
              f'{ttg_str} '
              f'{agg["total_constraint_violations"]:>5}')


def plot_bar_chart(all_metrics: dict, metric_key: str, ylabel: str,
                   title: str, out_path: str, fmt: str = '.2f'):
    modes = list(all_metrics.keys())
    vals = [all_metrics[m]['aggregate'].get(metric_key, 0) or 0 for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]
    labels = [MODE_LABELS.get(m, m) for m in modes]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='white',
                  linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:{fmt}}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_per_task_fdg(all_metrics: dict, out_path: str):
    """Grouped bar chart of FDG per task across modes."""
    modes = list(all_metrics.keys())
    all_task_ids = set()
    for data in all_metrics.values():
        for t in data['tasks']:
            all_task_ids.add(t['task_id'])
    task_ids = sorted(all_task_ids)

    fig, ax = plt.subplots(figsize=(12, 5))
    n_modes = len(modes)
    width = 0.8 / n_modes
    x = np.arange(len(task_ids))

    for i, mode in enumerate(modes):
        task_map = {t['task_id']: t for t in all_metrics[mode]['tasks']}
        fdgs = [task_map.get(tid, {}).get('final_distance_to_goal', 0)
                for tid in task_ids]
        offset = (i - n_modes / 2 + 0.5) * width
        ax.bar(x + offset, fdgs, width, label=MODE_LABELS.get(mode, mode),
               color=MODE_COLORS.get(mode, '#888888'), edgecolor='white',
               linewidth=0.8)

    ax.set_xlabel('Task ID', fontsize=12)
    ax.set_ylabel('Final Distance to Goal (m)', fontsize=12)
    ax.set_title('Per-Task Final Distance to Goal', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in task_ids])
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_path_length_boxplot(all_metrics: dict, out_path: str):
    """Box plot of normalized path lengths across modes."""
    modes = list(all_metrics.keys())
    data_lists = []
    labels = []
    for mode in modes:
        npls = [t['normalized_path_length'] for t in all_metrics[mode]['tasks']
                if t['normalized_path_length'] != float('inf')]
        if npls:
            data_lists.append(npls)
            labels.append(MODE_LABELS.get(mode, mode))

    if not data_lists:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data_lists, labels=labels, patch_artist=True, widths=0.4)

    colors = [MODE_COLORS.get(m, '#888888') for m in modes if
              MODE_LABELS.get(m, m) in labels]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Normalized Path Length', fontsize=12)
    ax.set_title('Path Length Distribution by Mode', fontsize=14,
                 fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_success_comparison(all_metrics: dict, out_path: str):
    """Per-task success/fail heatmap-style chart."""
    modes = list(all_metrics.keys())
    all_task_ids = set()
    for data in all_metrics.values():
        for t in data['tasks']:
            all_task_ids.add(t['task_id'])
    task_ids = sorted(all_task_ids)

    fig, ax = plt.subplots(figsize=(max(8, len(task_ids) * 0.8 + 2), 3))
    for row, mode in enumerate(modes):
        task_map = {t['task_id']: t for t in all_metrics[mode]['tasks']}
        for col, tid in enumerate(task_ids):
            success = task_map.get(tid, {}).get('success', False)
            color = '#4CAF50' if success else '#F44336'
            ax.add_patch(plt.Rectangle((col, row), 0.9, 0.8, facecolor=color,
                                       edgecolor='white', linewidth=2))
            ax.text(col + 0.45, row + 0.4,
                    'P' if success else 'F',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white')

    ax.set_xlim(-0.1, len(task_ids))
    ax.set_ylim(-0.1, len(modes))
    ax.set_xticks([i + 0.45 for i in range(len(task_ids))])
    ax.set_xticklabels([str(t) for t in task_ids])
    ax.set_yticks([i + 0.4 for i in range(len(modes))])
    ax.set_yticklabels([MODE_LABELS.get(m, m) for m in modes])
    ax.set_xlabel('Task ID', fontsize=12)
    ax.set_title('Success / Fail by Mode and Task', fontsize=14,
                 fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate ablation comparison report')
    parser.add_argument('base_dir',
                        help='Base trajectory directory containing mode subdirs')
    parser.add_argument('--output', default=None,
                        help='Output directory for report (default: <base_dir>/../report)')
    parser.add_argument('--config', default=None,
                        help='Path to benchmark_config.yaml (unused, reserved)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f'ERROR: {base_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else base_dir.parent / 'report'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = load_metrics(base_dir)
    if not all_metrics:
        print('No metrics.json files found in subdirectories of '
              f'{base_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Found metrics for modes: {", ".join(all_metrics.keys())}\n')

    # Summary table
    print_summary_table(all_metrics)
    csv_path = str(out_dir / 'comparison_summary.csv')
    write_summary_csv(all_metrics, csv_path)
    print(f'\n  Saved {csv_path}')

    # Bar charts
    plot_bar_chart(all_metrics, 'success_rate', 'Success Rate',
                   'Success Rate by Planner Mode',
                   str(out_dir / 'bar_success_rate.png'), fmt='.0%')

    plot_bar_chart(all_metrics, 'mean_final_distance',
                   'Mean Final Distance to Goal (m)',
                   'Mean FDG by Planner Mode',
                   str(out_dir / 'bar_mean_fdg.png'))

    plot_bar_chart(all_metrics, 'mean_normalized_path_length',
                   'Mean Normalized Path Length',
                   'Mean NPL by Planner Mode',
                   str(out_dir / 'bar_mean_npl.png'))

    # Per-task FDG
    plot_per_task_fdg(all_metrics, str(out_dir / 'per_task_fdg.png'))

    # Box plot
    plot_path_length_boxplot(all_metrics,
                             str(out_dir / 'boxplot_path_length.png'))

    # Success heatmap
    plot_success_comparison(all_metrics,
                            str(out_dir / 'success_heatmap.png'))

    print(f'\nDone — report saved to {out_dir}')


if __name__ == '__main__':
    main()
