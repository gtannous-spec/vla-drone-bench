"""
drl_controller.py — Deep Reinforcement Learning Controller skeleton.

Provides the interface for integrating a trained DRL policy (PPO/SAC)
with the benchmark framework. Adapted from UAV_Navigation_DRL_AirSim
architecture patterns.

The DRL agent observes:
    - Drone position relative to goal (dx, dy, dz)
    - Current velocity (vx, vy, vz)
    - Distance to goal
    - Optional: flattened image features from a CNN encoder

And outputs:
    - 3D velocity command (vx, vy, vz) or target waypoint offset

Training loop is separate — this controller loads a trained checkpoint
and runs inference only.
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

from .base_controller import BaseController, ControlAction, DroneState

logger = logging.getLogger(__name__)

_OBS_DIM = 10  # dx,dy,dz, vx,vy,vz, dist, heading_to_goal, altitude, speed


def _build_observation(state: DroneState, goal: Tuple[float, float, float]) -> np.ndarray:
    """Encode drone state + goal into a fixed-size observation vector."""
    dx = goal[0] - state.position[0]
    dy = goal[1] - state.position[1]
    dz = goal[2] - state.position[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    heading = math.atan2(dy, dx)
    altitude = -state.position[2]
    speed = math.sqrt(
        state.velocity[0]**2 + state.velocity[1]**2 + state.velocity[2]**2
    )

    return np.array([
        dx, dy, dz,
        state.velocity[0], state.velocity[1], state.velocity[2],
        dist, heading, altitude, speed,
    ], dtype=np.float32)


class DRLController(BaseController):
    """DRL policy controller for learned navigation.

    Supports two modes:
        1. Inference: Load a trained policy checkpoint and run forward pass.
        2. Untrained fallback: Navigate toward goal using the observation
           vector directly (for testing the pipeline before training).

    Args:
        policy_path: Path to a trained policy checkpoint (.pt file).
            If empty or None, uses the untrained fallback.
        device: CUDA device for inference.
        arrival_tolerance: Distance threshold for goal check (m).
        nav_speed: Cruise speed (m/s).
        action_scale: Multiplier for policy output → waypoint offset.
    """

    def __init__(
        self,
        policy_path: str = "",
        device: str = "auto",
        arrival_tolerance: float = 1.5,
        nav_speed: float = 5.0,
        action_scale: float = 10.0,
    ):
        self._policy_path = policy_path
        self._arrival_tolerance = arrival_tolerance
        self._nav_speed = nav_speed
        self._action_scale = action_scale
        self._goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._constraints: dict = {}
        self._task_id: int = 0
        self._step_count: int = 0
        self._policy = None

        if policy_path:
            self._load_policy(policy_path, device)
        else:
            logger.info("DRLController: no policy loaded — using heuristic fallback")

    def _load_policy(self, path: str, device: str) -> None:
        """Load a trained policy network from checkpoint."""
        try:
            import torch
            import torch.nn as nn

            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Expected policy architecture: simple MLP
            # Users should replace this with their actual trained architecture
            class PolicyMLP(nn.Module):
                def __init__(self, obs_dim, act_dim=3):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(obs_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, act_dim),
                        nn.Tanh(),
                    )

                def forward(self, x):
                    return self.net(x)

            self._policy = PolicyMLP(_OBS_DIM).to(device)
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "policy" in checkpoint:
                self._policy.load_state_dict(checkpoint["policy"])
            else:
                self._policy.load_state_dict(checkpoint)
            self._policy.eval()
            self._device = device
            logger.info(f"DRL policy loaded from {path} on {device}")

        except Exception as e:
            logger.error(f"Failed to load DRL policy: {e}")
            self._policy = None

    def reset(self, task_config: dict) -> None:
        self._task_id = task_config["id"]
        self._goal = tuple(task_config["goal"])
        self._constraints = task_config.get("constraints", {})
        self._step_count = 0
        max_speed = self._constraints.get("max_speed", self._nav_speed)
        self._effective_speed = min(self._nav_speed, max_speed)
        logger.info(f"DRLController reset — task {self._task_id}, goal={self._goal}")

    def get_action(self, state: DroneState) -> ControlAction:
        self._step_count += 1
        obs = _build_observation(state, self._goal)

        if self._policy is not None:
            action = self._policy_inference(obs)
        else:
            action = self._heuristic_action(obs)

        target_x = state.position[0] + action[0] * self._action_scale
        target_y = state.position[1] + action[1] * self._action_scale
        target_z = state.position[2] + action[2] * self._action_scale
        target_z = self._clamp_altitude(target_z)

        return ControlAction(
            target_position=(target_x, target_y, target_z),
            velocity=self._effective_speed,
        )

    def is_goal_reached(self, state: DroneState) -> bool:
        dist = self._distance_to_goal(state)
        if dist < self._arrival_tolerance * 2.0:
            logger.info(f"DRL goal reached at {dist:.1f}m after {self._step_count} steps")
            return True
        if self._step_count >= 100:
            logger.info(f"DRL max steps (100) — dist={dist:.1f}m")
            return True
        return False

    def _policy_inference(self, obs: np.ndarray) -> np.ndarray:
        """Run forward pass through the trained policy."""
        import torch
        with torch.inference_mode():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self._device)
            action_t = self._policy(obs_t)
            return action_t.squeeze(0).cpu().numpy()

    def _heuristic_action(self, obs: np.ndarray) -> np.ndarray:
        """Simple proportional controller — navigate toward goal."""
        dx, dy, dz = obs[0], obs[1], obs[2]
        dist = obs[6]
        if dist < 1e-3:
            return np.zeros(3, dtype=np.float32)
        direction = np.array([dx, dy, dz], dtype=np.float32)
        direction /= max(np.linalg.norm(direction), 1e-6)
        speed_factor = min(1.0, dist / self._action_scale)
        return direction * speed_factor

    def _distance_to_goal(self, state: DroneState) -> float:
        dx = state.position[0] - self._goal[0]
        dy = state.position[1] - self._goal[1]
        dz = state.position[2] - self._goal[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _clamp_altitude(self, z_ned: float) -> float:
        min_alt = self._constraints.get("min_altitude", 2.0)
        max_alt = self._constraints.get("max_altitude", 50.0)
        z_min = -max_alt
        z_max = -min_alt
        return max(z_min, min(z_max, z_ned))
