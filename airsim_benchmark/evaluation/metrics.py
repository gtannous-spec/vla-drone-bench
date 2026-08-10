"""
metrics.py — Rich post-hoc evaluation metrics for VLA navigation.

Computes metrics from log files or metrics.json output that go beyond the
basic path-length and collision counts already in the benchmark runner.
These are designed for thesis-quality evaluation of LoRA-adapted models.

Usage:
    python -m airsim_benchmark.evaluation.metrics \
        --log logs/vla_XXXXX.out \
        --metrics logs/airsim_output/openfly_bias0.0/metrics.json
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_action_tokens_from_log(log_path: str) -> List[List[int]]:
    """Extract all raw action token ID sequences from a benchmark log."""
    pattern = re.compile(
        r"\[FIX-C\].*gen_tokens=\[([^\]]+)\]"
        r"|\[DIAG\] raw action token IDs: \[([^\]]+)\]"
        r"|\[FIX-B\] direct logit token IDs: \[([^\]]+)\]"
    )
    sequences = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                raw = m.group(1) or m.group(2) or m.group(3)
                tokens = [int(x.strip()) for x in raw.split(",")]
                sequences.append(tokens)
    return sequences


def parse_action_dims_from_log(log_path: str) -> List[List[float]]:
    """Extract decoded action dimension vectors from a benchmark log."""
    pattern = re.compile(r"\[DIAG\] inference #\d+.*all_dims=\[([^\]]+)\]")
    actions = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                dims = [float(x.strip()) for x in m.group(1).split(",")]
                actions.append(dims)
    return actions


def parse_hop_positions_from_log(log_path: str) -> List[Tuple[float, float, float]]:
    """Extract drone positions from hop log lines."""
    pattern = re.compile(
        r"OpenFly hop \d+.*pos=\(([^)]+)\)"
    )
    positions = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                coords = [float(x.strip()) for x in m.group(1).split(",")]
                if len(coords) == 3:
                    positions.append(tuple(coords))
    return positions


# ── Token-level metrics ────────────────────────────────────────────────────


def action_token_diversity(sequences: List[List[int]]) -> Dict[str, Any]:
    """Measure diversity of generated action tokens.

    Returns:
        unique_ratio: fraction of inferences with >1 unique token
        mean_unique: average number of unique tokens per 8-token sequence
        token_histogram: counts of most common tokens across all sequences
        collapsed_rate: fraction of sequences where all 8 tokens are identical
    """
    if not sequences:
        return {"unique_ratio": 0.0, "mean_unique": 0, "collapsed_rate": 1.0, "token_histogram": {}}

    unique_counts = [len(set(seq)) for seq in sequences]
    all_tokens = [t for seq in sequences for t in seq]
    collapsed = sum(1 for seq in sequences if len(set(seq)) <= 1)

    histogram = Counter(all_tokens).most_common(10)

    return {
        "unique_ratio": round(sum(1 for u in unique_counts if u > 1) / len(sequences), 4),
        "mean_unique_per_seq": round(sum(unique_counts) / len(sequences), 2),
        "collapsed_rate": round(collapsed / len(sequences), 4),
        "total_sequences": len(sequences),
        "token_histogram": {str(k): v for k, v in histogram},
    }


# ── Action-level metrics ──────────────────────────────────────────────────


def action_diversity_score(actions: List[List[float]]) -> Dict[str, Any]:
    """Measure the diversity of decoded action vectors.

    A model with good domain understanding should produce varied action
    magnitudes across different hops.
    """
    if not actions:
        return {"variance_per_dim": [], "overall_variance": 0.0, "n_actions": 0}

    import numpy as np
    arr = np.array(actions)  # (N, 8)
    var_per_dim = np.var(arr, axis=0).tolist()
    overall_var = float(np.var(arr))

    # Check if the stop flag varies (dim 0)
    stop_vals = arr[:, 0]
    stop_diversity = float(np.std(stop_vals))

    # Check if forward distance varies (dim 1)
    fwd_vals = arr[:, 1]
    fwd_diversity = float(np.std(fwd_vals))

    return {
        "variance_per_dim": [round(v, 6) for v in var_per_dim],
        "overall_variance": round(overall_var, 6),
        "stop_flag_std": round(stop_diversity, 4),
        "forward_dist_std": round(fwd_diversity, 4),
        "n_actions": len(actions),
        "n_distinct_patterns": len(set(tuple(round(v, 2) for v in a) for a in actions)),
    }


# ── Trajectory-level metrics ─────────────────────────────────────────────


def heading_responsiveness(positions: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    """Measure how much the drone changes heading during flight.

    A model that responds to visual cues should show varied heading
    changes, not fly in a straight line.
    """
    if len(positions) < 3:
        return {"mean_abs_heading_change": 0.0, "max_heading_change": 0.0, "n_turns": 0}

    heading_changes = []
    for i in range(1, len(positions) - 1):
        dx1 = positions[i][0] - positions[i - 1][0]
        dy1 = positions[i][1] - positions[i - 1][1]
        dx2 = positions[i + 1][0] - positions[i][0]
        dy2 = positions[i + 1][1] - positions[i][1]

        if abs(dx1) < 0.01 and abs(dy1) < 0.01:
            continue
        if abs(dx2) < 0.01 and abs(dy2) < 0.01:
            continue

        h1 = math.atan2(dy1, dx1)
        h2 = math.atan2(dy2, dx2)
        delta = math.degrees(h2 - h1)
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
        heading_changes.append(abs(delta))

    if not heading_changes:
        return {"mean_abs_heading_change": 0.0, "max_heading_change": 0.0, "n_turns": 0}

    n_turns = sum(1 for h in heading_changes if h > 10.0)

    return {
        "mean_abs_heading_change": round(sum(heading_changes) / len(heading_changes), 2),
        "max_heading_change": round(max(heading_changes), 2),
        "n_turns": n_turns,
        "n_significant_turns_gt30": sum(1 for h in heading_changes if h > 30.0),
        "straight_line_ratio": round(
            sum(1 for h in heading_changes if h < 5.0) / len(heading_changes), 4
        ),
    }


def path_efficiency(
    positions: List[Tuple[float, float, float]],
    start: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """Compute path efficiency metrics.

    Compares total distance flown to the displacement from start to end.
    Efficient navigation has ratio close to 1.0; circling or zigzagging
    increases the ratio.
    """
    if len(positions) < 2:
        return {"total_distance": 0.0, "displacement": 0.0, "efficiency": 0.0}

    total = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dz = positions[i][2] - positions[i - 1][2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)

    p0 = start if start else positions[0]
    pn = positions[-1]
    displacement = math.sqrt(
        (pn[0] - p0[0]) ** 2 + (pn[1] - p0[1]) ** 2 + (pn[2] - p0[2]) ** 2
    )

    efficiency = displacement / total if total > 0 else 0.0

    return {
        "total_distance": round(total, 2),
        "displacement": round(displacement, 2),
        "efficiency": round(efficiency, 4),
    }


# ── Thesis evaluation metrics ─────────────────────────────────────────────


def parse_mission_results(log_path: str) -> List[Dict[str, Any]]:
    """Extract per-mission results from a benchmark log.

    Parses mission completion status, leg counts, distances, collisions,
    and timing from the summary table in the log.
    """
    missions = []
    table_re = re.compile(
        r"^\s*(\d+)\s+(.+?)\s+(\d+)/\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = table_re.match(line.strip())
            if m:
                missions.append({
                    "id": int(m.group(1)),
                    "name": m.group(2).strip(),
                    "legs_completed": int(m.group(3)),
                    "legs_total": int(m.group(4)),
                    "path_length": float(m.group(5)),
                    "final_distance": float(m.group(6)),
                    "collisions": int(m.group(7)),
                    "time_sec": float(m.group(8)),
                })
    return missions


def thesis_metrics(log_path: str) -> Dict[str, Any]:
    """Compute the four thesis evaluation metrics from a benchmark log.

    Returns:
        SR:  Success Rate (fraction of missions with all legs completed)
        FDG: Final Distance to Goal (mean across missions, meters)
        PE:  Path Efficiency (displacement / total path, closer to 1 = better)
        CR:  Collision Rate (collisions per mission)
    """
    missions = parse_mission_results(log_path)
    if not missions:
        return {"error": "no mission results found in log"}

    n = len(missions)
    successes = sum(1 for m in missions if m["legs_completed"] == m["legs_total"])
    sr = successes / n

    fdg_mean = sum(m["final_distance"] for m in missions) / n

    positions = parse_hop_positions_from_log(log_path)
    pe = path_efficiency(positions)

    total_collisions = sum(m["collisions"] for m in missions)
    cr = total_collisions / n

    return {
        "n_missions": n,
        "success_rate": round(sr, 4),
        "final_distance_to_goal_mean": round(fdg_mean, 2),
        "path_efficiency": pe["efficiency"],
        "collision_rate": round(cr, 2),
        "total_collisions": total_collisions,
        "per_mission": missions,
    }


def thesis_comparison_table(
    logs: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Build a structured comparison across multiple methods.

    Args:
        logs: mapping of method name to log path, e.g.:
            {"Base OpenFly": "logs/vla_base.out",
             "LoRA v9": "logs/vla_v9.out",
             "Regression v11": "logs/vla_v11.out",
             "Oracle": "logs/oracle.out"}

    Returns:
        Dict with per-method metrics ready for a thesis table.
    """
    results = {}
    for name, path in logs.items():
        try:
            results[name] = thesis_metrics(path)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


# ── Aggregate evaluation ─────────────────────────────────────────────────


def evaluate_log(log_path: str) -> Dict[str, Any]:
    """Run all metrics on a benchmark log file."""
    tokens = parse_action_tokens_from_log(log_path)
    actions = parse_action_dims_from_log(log_path)
    positions = parse_hop_positions_from_log(log_path)

    return {
        "log_file": log_path,
        "token_diversity": action_token_diversity(tokens),
        "action_diversity": action_diversity_score(actions),
        "heading_responsiveness": heading_responsiveness(positions),
        "path_efficiency": path_efficiency(positions),
    }


def compare_runs(
    baseline_log: str, lora_log: str
) -> Dict[str, Any]:
    """Compare baseline (pre-LoRA) and LoRA-adapted run metrics."""
    baseline = evaluate_log(baseline_log)
    lora = evaluate_log(lora_log)

    def delta(key_path: str) -> Optional[float]:
        keys = key_path.split(".")
        b_val = baseline
        l_val = lora
        for k in keys:
            b_val = b_val.get(k, {}) if isinstance(b_val, dict) else None
            l_val = l_val.get(k, {}) if isinstance(l_val, dict) else None
        if isinstance(b_val, (int, float)) and isinstance(l_val, (int, float)):
            return round(l_val - b_val, 4)
        return None

    return {
        "baseline": baseline,
        "lora": lora,
        "improvements": {
            "collapsed_rate_delta": delta("token_diversity.collapsed_rate"),
            "unique_ratio_delta": delta("token_diversity.unique_ratio"),
            "action_variance_delta": delta("action_diversity.overall_variance"),
            "n_distinct_patterns_delta": delta("action_diversity.n_distinct_patterns"),
            "heading_change_delta": delta("heading_responsiveness.mean_abs_heading_change"),
            "n_turns_delta": delta("heading_responsiveness.n_turns"),
            "path_efficiency_delta": delta("path_efficiency.efficiency"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLA navigation quality")
    parser.add_argument("--log", required=True, help="Path to benchmark log file")
    parser.add_argument("--baseline", default=None, help="Baseline log for comparison")
    parser.add_argument("--output", default=None, help="Write JSON results to file")
    args = parser.parse_args()

    if args.baseline:
        results = compare_runs(args.baseline, args.log)
    else:
        results = evaluate_log(args.log)

    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
