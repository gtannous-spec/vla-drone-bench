#!/usr/bin/env python3
"""
compare_experiments.py — Compare experiment results for thesis reporting.

Reads the unified experiments.json and produces:
  - Side-by-side comparison tables (terminal + markdown)
  - CSV export for plotting in LaTeX / matplotlib
  - Per-controller and per-task breakdowns
  - LoRA training curve summaries

Usage:
    python -m airsim_benchmark.scripts.compare_experiments
    python -m airsim_benchmark.scripts.compare_experiments --type benchmark_task
    python -m airsim_benchmark.scripts.compare_experiments --controller openfly llamauav
    python -m airsim_benchmark.scripts.compare_experiments --export csv
    python -m airsim_benchmark.scripts.compare_experiments --export markdown --output report.md
"""

import argparse
import csv
import json
import logging
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airsim_benchmark.experiments import ExperimentRegistry

logger = logging.getLogger(__name__)


def format_table(headers: List[str], rows: List[List[Any]], fmt: str = "terminal") -> str:
    """Render a list of rows as a formatted table."""
    str_rows = [[str(c) if c is not None else "—" for c in row] for row in rows]
    col_widths = [
        max(len(h), *(len(r[i]) for r in str_rows))
        for i, h in enumerate(headers)
    ]

    if fmt == "markdown":
        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        sep_line = "| " + " | ".join("-" * w for w in col_widths) + " |"
        data_lines = [
            "| " + " | ".join(c.ljust(w) for c, w in zip(r, col_widths)) + " |"
            for r in str_rows
        ]
        return "\n".join([header_line, sep_line] + data_lines)

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "  ".join("-" * w for w in col_widths)
    data_lines = [
        "  ".join(c.ljust(w) for c, w in zip(r, col_widths))
        for r in str_rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def compare_task_benchmarks(experiments: List[Dict], fmt: str = "terminal") -> str:
    """Compare task-mode benchmark runs side by side."""
    lines = []
    lines.append("=" * 80)
    lines.append("TASK BENCHMARK COMPARISON")
    lines.append("=" * 80)

    headers = [
        "Experiment", "Controller", "Goal Bias", "LoRA",
        "SR", "Mean FDG (m)", "Mean NPL", "Mean Time (s)",
        "Collisions", "Tasks",
    ]
    rows = []
    for exp in experiments:
        s = exp.get("summary", {})
        lora = "yes" if exp.get("lora_path") else "—"
        gb = exp.get("goal_bias")
        gb_str = f"{gb:.1f}" if gb is not None else "—"
        rows.append([
            exp["id"][:25],
            exp.get("controller", "?"),
            gb_str,
            lora,
            f"{s.get('success_rate', 0):.0%}",
            f"{s.get('mean_final_distance_m', 0):.2f}" if s.get("mean_final_distance_m") else "—",
            f"{s.get('mean_npl', 0):.3f}" if s.get("mean_npl") else "—",
            f"{s.get('mean_time_s', 0):.1f}" if s.get("mean_time_s") else "—",
            str(s.get("total_collisions", 0)),
            str(s.get("num_tasks", 0)),
        ])

    lines.append(format_table(headers, rows, fmt))
    lines.append("")

    # Per-task breakdown
    lines.append("-" * 80)
    lines.append("PER-TASK BREAKDOWN")
    lines.append("-" * 80)
    task_headers = ["Experiment", "Task", "Success", "FDG (m)", "NPL", "Time (s)", "Collisions"]
    task_rows = []
    for exp in experiments:
        s = exp.get("summary", {})
        for t in s.get("tasks", []):
            task_rows.append([
                exp["id"][:20],
                f"T{t.get('task_id', '?')}",
                "PASS" if t.get("success") else "FAIL",
                f"{t.get('final_distance_m', 0):.2f}" if t.get("final_distance_m") else "—",
                f"{t.get('npl', 0):.3f}" if t.get("npl") else "—",
                f"{t.get('time_s', 0):.1f}" if t.get("time_s") else "—",
                str(t.get("collisions", 0)),
            ])
    if task_rows:
        lines.append(format_table(task_headers, task_rows, fmt))
    lines.append("")

    return "\n".join(lines)


def compare_mission_benchmarks(experiments: List[Dict], fmt: str = "terminal") -> str:
    """Compare mission-mode benchmark runs side by side."""
    lines = []
    lines.append("=" * 80)
    lines.append("MISSION BENCHMARK COMPARISON")
    lines.append("=" * 80)

    headers = [
        "Experiment", "Controller", "Goal Bias", "LoRA",
        "Missions", "Legs Done", "Path (m)", "Collisions", "Smoothness",
    ]
    rows = []
    for exp in experiments:
        s = exp.get("summary", {})
        lora = "yes" if exp.get("lora_path") else "—"
        gb = exp.get("goal_bias")
        gb_str = f"{gb:.1f}" if gb is not None else "—"
        rows.append([
            exp["id"][:25],
            exp.get("controller", "?"),
            gb_str,
            lora,
            str(s.get("num_missions", 0)),
            f"{s.get('completed_legs', 0)}/{s.get('total_legs', 0)}",
            f"{s.get('total_path_m', 0):.1f}",
            str(s.get("total_collisions", 0)),
            f"{s.get('mean_heading_smoothness', 0):.2f}" if s.get("mean_heading_smoothness") else "—",
        ])

    lines.append(format_table(headers, rows, fmt))
    lines.append("")

    # Per-mission breakdown
    lines.append("-" * 80)
    lines.append("PER-MISSION BREAKDOWN")
    lines.append("-" * 80)
    m_headers = ["Experiment", "Mission", "Name", "Legs", "Path (m)", "Time (s)", "Collisions"]
    m_rows = []
    for exp in experiments:
        s = exp.get("summary", {})
        for m in s.get("missions", []):
            m_rows.append([
                exp["id"][:16],
                f"M{m.get('mission_id', '?')}",
                m.get("name", "")[:30],
                str(m.get("legs_completed", "?")),
                f"{m.get('path_length_m', 0):.1f}" if m.get("path_length_m") else "—",
                f"{m.get('time_s', 0):.1f}" if m.get("time_s") else "—",
                str(m.get("collisions", 0)),
            ])
    if m_rows:
        lines.append(format_table(m_headers, m_rows, fmt))
    lines.append("")

    return "\n".join(lines)


def compare_training_runs(experiments: List[Dict], fmt: str = "terminal") -> str:
    """Compare LoRA training runs."""
    lines = []
    lines.append("=" * 80)
    lines.append("LORA TRAINING COMPARISON")
    lines.append("=" * 80)

    headers = [
        "Experiment", "Checkpoint", "Epochs", "Early Stop",
        "Best Val Loss", "Final Train Loss",
        "LR", "Samples", "Timestamp",
    ]
    rows = []
    for exp in experiments:
        s = exp.get("summary", {})
        ckpt = Path(exp.get("checkpoint_dir", "")).name
        rows.append([
            exp["id"][:25],
            ckpt,
            f"{s.get('epochs_completed', '?')}/{s.get('epochs_requested', '?')}",
            "YES" if s.get("early_stopped") else "no",
            f"{s.get('best_val_loss', 0):.6f}" if s.get("best_val_loss") else "—",
            f"{s.get('final_train_loss', 0):.6f}" if s.get("final_train_loss") else "—",
            f"{s.get('lr', 0):.1e}" if s.get("lr") else "—",
            str(s.get("n_train", "?")),
            exp.get("timestamp", "")[:19],
        ])

    lines.append(format_table(headers, rows, fmt))
    lines.append("")

    # Loss curves side by side
    lines.append("-" * 80)
    lines.append("TRAINING LOSS CURVES")
    lines.append("-" * 80)
    for exp in experiments:
        s = exp.get("summary", {})
        ckpt = Path(exp.get("checkpoint_dir", "")).name
        train_l = s.get("train_losses", [])
        val_l = s.get("val_losses", [])
        lines.append(f"\n  {ckpt}:")
        lines.append(f"    Train: {' → '.join(f'{l:.4f}' for l in train_l)}")
        lines.append(f"    Val:   {' → '.join(f'{l:.4f}' for l in val_l)}")

        per_dim = s.get("final_val_loss_per_dim")
        if per_dim:
            dim_names = ["stop", "fwd", "yawL", "yawR", "up", "dn", "L", "R"]
            dim_str = ", ".join(
                f"{dim_names[i]}={per_dim.get(f'dim{i}', 'N/A')}"
                for i in range(8)
            )
            lines.append(f"    Per-dim: [{dim_str}]")
    lines.append("")

    return "\n".join(lines)


def export_csv(experiments: List[Dict], output_path: str) -> None:
    """Export all experiments to a flat CSV for LaTeX/matplotlib."""
    if not experiments:
        return

    rows = []
    for exp in experiments:
        s = exp.get("summary", {})
        hw = exp.get("hardware", {})
        row = {
            "id": exp.get("id", ""),
            "type": exp.get("type", ""),
            "timestamp": exp.get("timestamp", ""),
            "controller": exp.get("controller", ""),
            "model": exp.get("model", exp.get("base_model", "")),
            "goal_bias": exp.get("goal_bias", ""),
            "lora_path": exp.get("lora_path", ""),
            "gpu": hw.get("gpu", ""),
            "node": hw.get("node", ""),
        }
        # Flatten summary keys
        for k, v in s.items():
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v)
            else:
                row[k] = v
        rows.append(row)

    all_keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} experiments to {output_path}")


def full_report(registry: ExperimentRegistry, fmt: str = "terminal") -> str:
    """Generate a full comparison report across all experiment types."""
    lines = []
    lines.append("")
    lines.append("╔" + "═" * 78 + "╗")
    lines.append("║" + " VLA DRONE NAVIGATION — EXPERIMENT REPORT".center(78) + "║")
    lines.append("║" + f" Generated: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}".center(78) + "║")
    lines.append("╚" + "═" * 78 + "╝")
    lines.append("")

    total = len(registry.experiments)
    types = registry.list_types()
    controllers = registry.list_controllers()
    lines.append(f"Total experiments: {total}")
    lines.append(f"Types: {', '.join(types)}")
    lines.append(f"Controllers: {', '.join(controllers)}")
    lines.append("")

    # Task benchmarks
    task_exps = registry.filter(type="benchmark_task")
    if task_exps:
        lines.append(compare_task_benchmarks(task_exps, fmt))

    # Mission benchmarks
    mission_exps = registry.filter(type="benchmark_mission")
    if mission_exps:
        lines.append(compare_mission_benchmarks(mission_exps, fmt))

    # Training runs
    training_exps = registry.filter(type="lora_training")
    if training_exps:
        lines.append(compare_training_runs(training_exps, fmt))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VLA navigation experiments for thesis reporting"
    )
    parser.add_argument(
        "--registry", default="./data/experiments",
        help="Path to experiment registry directory (default: ./data/experiments)",
    )
    parser.add_argument(
        "--type", choices=["benchmark_task", "benchmark_mission", "lora_training"],
        help="Filter by experiment type",
    )
    parser.add_argument(
        "--controller", nargs="+",
        help="Filter by controller name(s)",
    )
    parser.add_argument(
        "--export", choices=["csv", "markdown", "terminal"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Scan existing output dirs and register all experiments first",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    registry = ExperimentRegistry(args.registry)

    if args.backfill:
        n1 = registry.backfill_benchmarks("./logs/airsim_output")
        n2 = registry.backfill_training("./data")
        print(f"\nBackfilled {n1} benchmark + {n2} training experiments.\n")

    if args.export == "csv":
        exps = registry.experiments
        if args.type:
            exps = [e for e in exps if e["type"] == args.type]
        if args.controller:
            exps = [e for e in exps if e.get("controller") in args.controller]
        out_path = args.output or "./data/experiments/experiments_export.csv"
        export_csv(exps, out_path)
        print(f"Exported to {out_path}")
        return

    fmt = "markdown" if args.export == "markdown" else "terminal"

    if args.type or args.controller:
        exps = registry.experiments
        if args.type:
            exps = [e for e in exps if e["type"] == args.type]
        if args.controller:
            exps = [e for e in exps if e.get("controller") in args.controller]

        if args.type == "benchmark_task":
            report = compare_task_benchmarks(exps, fmt)
        elif args.type == "benchmark_mission":
            report = compare_mission_benchmarks(exps, fmt)
        elif args.type == "lora_training":
            report = compare_training_runs(exps, fmt)
        else:
            report = full_report(registry, fmt)
    else:
        report = full_report(registry, fmt)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
