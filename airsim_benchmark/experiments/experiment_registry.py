"""
experiment_registry.py — Unified experiment catalog for thesis benchmarking.

Tracks every benchmark run and LoRA training run in a single experiments.json
file.  Supports:
  - Auto-registration after each benchmark / training run
  - Backfill from existing output directories
  - Querying / filtering by controller, mode, goal_bias, etc.
  - Export to comparison tables (CSV / markdown)

Each experiment record stores:
  - Unique ID (timestamp-based)
  - Run type: "benchmark_task", "benchmark_mission", or "lora_training"
  - Controller, model, hyperparameters
  - Hardware info (GPU, node)
  - Aggregate metrics (SR, path length, collisions, val_loss, etc.)
  - Paths to raw artifacts (metrics.json, trajectories, checkpoints, videos)
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "experiments.json"


class ExperimentRegistry:
    """Append-only catalog of experiment records.

    Parameters
    ----------
    registry_dir : str or Path
        Directory where ``experiments.json`` lives (created if missing).
    """

    def __init__(self, registry_dir: str = "./data/experiments") -> None:
        self._dir = Path(registry_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / REGISTRY_FILENAME
        self._experiments: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                self._experiments = json.load(f)
            logger.info(
                f"Loaded {len(self._experiments)} experiments from {self._path}"
            )
        else:
            self._experiments = []

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._experiments, f, indent=2, default=str)

    @property
    def experiments(self) -> List[Dict[str, Any]]:
        return list(self._experiments)

    def _generate_id(self, run_type: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{run_type}_{ts}"

    def _find_existing(self, output_dir: str) -> Optional[int]:
        """Return index of experiment with matching output_dir, or None."""
        for i, exp in enumerate(self._experiments):
            if exp.get("output_dir") == str(output_dir):
                return i
        return None

    # ──────────────────────────────────────────────────────────────────
    # Registration methods
    # ──────────────────────────────────────────────────────────────────

    def register_benchmark(
        self,
        run_type: str,
        controller: str,
        model: str,
        metrics: Dict[str, Any],
        output_dir: str,
        goal_bias: Optional[float] = None,
        waypoint_scale: float = 15.0,
        lora_path: Optional[str] = None,
        slurm_job_id: Optional[str] = None,
        gpu: Optional[str] = None,
        node: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a benchmark run (task or mission mode).

        Returns the experiment ID.
        """
        exp_id = self._generate_id(run_type)

        agg = metrics.get("aggregate", {})

        if run_type == "benchmark_task":
            summary = {
                "num_tasks": agg.get("num_tasks", 0),
                "success_rate": agg.get("success_rate", 0.0),
                "mean_final_distance_m": agg.get("mean_final_distance", None),
                "mean_npl": agg.get("mean_normalized_path_length", None),
                "mean_time_s": agg.get("mean_time_to_goal_s", None),
                "total_collisions": agg.get("total_collisions", 0),
                "constraint_violations": agg.get("total_constraint_violations", 0),
            }
            per_task = []
            for t in metrics.get("tasks", []):
                per_task.append({
                    "task_id": t.get("task_id"),
                    "instruction": t.get("instruction", ""),
                    "success": t.get("success", False),
                    "final_distance_m": t.get("final_distance_to_goal"),
                    "npl": t.get("normalized_path_length"),
                    "path_length_m": t.get("path_length"),
                    "time_s": t.get("time_to_goal_s"),
                    "collisions": t.get("collision_count", 0),
                })
            summary["tasks"] = per_task

        elif run_type == "benchmark_mission":
            summary = {
                "num_missions": agg.get("num_missions", 0),
                "total_legs": agg.get("total_legs", 0),
                "completed_legs": agg.get("completed_legs", 0),
                "total_path_m": agg.get("total_path_length_m", 0),
                "total_collisions": agg.get("total_collisions", 0),
                "mean_heading_smoothness": agg.get("mean_heading_smoothness"),
            }
            per_mission = []
            for m in metrics.get("missions", []):
                per_mission.append({
                    "mission_id": m.get("mission_id"),
                    "name": m.get("name", ""),
                    "legs_completed": f"{m.get('legs_completed', 0)}/{m.get('num_legs', 0)}",
                    "path_length_m": m.get("path_length"),
                    "time_s": m.get("total_time_s"),
                    "collisions": m.get("collision_count", 0),
                    "heading_smoothness": m.get("heading_smoothness"),
                })
            summary["missions"] = per_mission
        else:
            summary = agg

        record = {
            "id": exp_id,
            "type": run_type,
            "timestamp": datetime.now().isoformat(),
            "controller": controller,
            "model": model,
            "goal_bias": goal_bias,
            "waypoint_scale": waypoint_scale,
            "lora_path": lora_path,
            "slurm_job_id": slurm_job_id,
            "hardware": {"gpu": gpu, "node": node},
            "output_dir": str(output_dir),
            "metrics_file": str(Path(output_dir) / "metrics.json"),
            "summary": summary,
        }
        if extra:
            record["extra"] = extra

        idx = self._find_existing(output_dir)
        if idx is not None:
            record["id"] = self._experiments[idx]["id"]
            self._experiments[idx] = record
            logger.info(f"Updated experiment {record['id']}")
        else:
            self._experiments.append(record)
            logger.info(f"Registered experiment {exp_id}")

        self._save()
        return record["id"]

    def register_training(
        self,
        checkpoint_dir: str,
        base_model: str = "IPEC-COMMUNITY/openfly-agent-7b",
        train_meta: Optional[Dict[str, Any]] = None,
        data_manifest: Optional[str] = None,
        slurm_job_id: Optional[str] = None,
        gpu: Optional[str] = None,
        node: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a LoRA training run."""
        exp_id = self._generate_id("lora_training")

        if train_meta is None:
            meta_path = Path(checkpoint_dir) / "train_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    train_meta = json.load(f)
            else:
                train_meta = {}

        summary = {
            "epochs_completed": train_meta.get("epochs_completed"),
            "epochs_requested": train_meta.get("epochs_requested"),
            "early_stopped": train_meta.get("early_stopped", False),
            "best_val_loss": train_meta.get("best_val_loss"),
            "final_train_loss": (
                train_meta["train_losses"][-1]
                if train_meta.get("train_losses")
                else None
            ),
            "n_train": train_meta.get("n_train"),
            "n_val": train_meta.get("n_val"),
            "lr": train_meta.get("lr"),
            "batch_size": train_meta.get("batch_size"),
            "grad_accum": train_meta.get("grad_accum"),
            "train_losses": train_meta.get("train_losses", []),
            "val_losses": train_meta.get("val_losses", []),
            "final_val_loss_per_dim": train_meta.get("final_val_loss_per_dim"),
        }

        record = {
            "id": exp_id,
            "type": "lora_training",
            "timestamp": train_meta.get("timestamp", datetime.now().isoformat()),
            "base_model": base_model,
            "data_manifest": data_manifest,
            "checkpoint_dir": str(checkpoint_dir),
            "slurm_job_id": slurm_job_id,
            "hardware": {"gpu": gpu, "node": node},
            "summary": summary,
        }
        if extra:
            record["extra"] = extra

        idx = None
        for i, exp in enumerate(self._experiments):
            if exp.get("checkpoint_dir") == str(checkpoint_dir):
                idx = i
                break
        if idx is not None:
            record["id"] = self._experiments[idx]["id"]
            self._experiments[idx] = record
        else:
            self._experiments.append(record)

        self._save()
        logger.info(f"Registered training experiment {record['id']}")
        return record["id"]

    # ──────────────────────────────────────────────────────────────────
    # Backfill — scan existing output dirs
    # ──────────────────────────────────────────────────────────────────

    def backfill_benchmarks(self, output_root: str) -> int:
        """Scan all subdirectories of output_root for metrics.json and
        register them."""
        count = 0
        root = Path(output_root)
        if not root.exists():
            return 0

        for sub in sorted(root.iterdir()):
            metrics_file = sub / "metrics.json"
            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                metrics = json.load(f)

            dirname = sub.name
            controller = dirname.split("_bias")[0].split("_mode")[0]

            goal_bias = None
            if "_bias" in dirname:
                try:
                    goal_bias = float(dirname.split("_bias")[1].split("_")[0])
                except ValueError:
                    pass

            agg = metrics.get("aggregate", {})
            if "missions" in metrics:
                run_type = "benchmark_mission"
            else:
                run_type = "benchmark_task"

            model = self._infer_model(controller)

            self.register_benchmark(
                run_type=run_type,
                controller=controller,
                model=model,
                metrics=metrics,
                output_dir=str(sub),
                goal_bias=goal_bias,
            )
            count += 1

        logger.info(f"Backfilled {count} benchmark runs from {output_root}")
        return count

    def backfill_training(self, data_root: str) -> int:
        """Scan for lora_checkpoints* directories and register them."""
        count = 0
        root = Path(data_root)
        if not root.exists():
            return 0

        for sub in sorted(root.iterdir()):
            if not sub.is_dir() or "lora_checkpoint" not in sub.name:
                continue
            meta_file = sub / "train_meta.json"
            if not meta_file.exists():
                continue

            self.register_training(checkpoint_dir=str(sub))
            count += 1

        logger.info(f"Backfilled {count} training runs from {data_root}")
        return count

    @staticmethod
    def _infer_model(controller: str) -> str:
        mapping = {
            "classical": "classical-waypoint",
            "openfly": "IPEC-COMMUNITY/openfly-agent-7b",
            "llamauav": "TravelUAV/LLaMA-UAV",
            "vlm": "OpenGVLab/InternVL2-8B",
            "navila": "OpenGVLab/InternVL2-8B",
            "vla": "openvla/openvla-7b",
        }
        return mapping.get(controller, controller)

    # ──────────────────────────────────────────────────────────────────
    # Query helpers
    # ──────────────────────────────────────────────────────────────────

    def filter(self, **kwargs) -> List[Dict[str, Any]]:
        """Filter experiments by any top-level key.

        Example: registry.filter(controller="openfly", type="benchmark_task")
        """
        results = self._experiments
        for key, val in kwargs.items():
            results = [e for e in results if e.get(key) == val]
        return results

    def get_by_id(self, exp_id: str) -> Optional[Dict[str, Any]]:
        for e in self._experiments:
            if e["id"] == exp_id:
                return e
        return None

    def list_controllers(self) -> List[str]:
        return sorted(set(e.get("controller", "") for e in self._experiments))

    def list_types(self) -> List[str]:
        return sorted(set(e["type"] for e in self._experiments))
