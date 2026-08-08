"""
vla_planner.py — Milestone 2 OpenVLA-based drone controller.

Uses the same FSM as the classical planner but replaces the NAVIGATE phase
with end-to-end inference: camera image + language instruction -> action delta.

Supports two operating modes via the 'mode' ROS parameter:
  - vla_only : camera + language instruction only
  - vla_gt   : camera + language + ground-truth goal coordinates in prompt

Communication with PX4 happens over Micro-XRCE-DDS via px4_msgs.
"""

import csv
import math
import os
import time as _time
from enum import Enum, auto
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from sensor_msgs.msg import Image

import yaml
from ament_index_python.packages import get_package_share_directory

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    _HAS_VLA = True
except ImportError:
    _HAS_VLA = False


# ── Finite-state machine ────────────────────────────────────────────────────

class FlightPhase(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    NAVIGATE = auto()
    LAND = auto()
    DONE = auto()


# ── OpenVLA wrapper ─────────────────────────────────────────────────────────

class OpenVLABackend:
    """Thin wrapper around the OpenVLA HuggingFace model."""

    DEFAULT_MODEL_ID = 'openvla/openvla-7b'

    def __init__(self, model_id: str = '', device: str = 'cuda'):
        model_id = model_id or self.DEFAULT_MODEL_ID
        self.device = device

        if not _HAS_VLA:
            raise RuntimeError(
                'torch / transformers not installed — cannot load OpenVLA')

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

    @torch.inference_mode()
    def predict_action(self, image_np: np.ndarray, prompt: str) -> np.ndarray:
        """Run a single inference pass.

        Returns a 7-element numpy array (dx, dy, dz, droll, dpitch, dyaw,
        gripper).  For drone navigation we only use the first three.
        """
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image_np)

        inputs = self.processor(prompt, pil_img).to(
            self.device, dtype=torch.bfloat16)
        action = self.model.predict_action(**inputs, unnorm_key='bridge_orig')
        return np.array(action, dtype=np.float64)


# ── ROS 2 node ──────────────────────────────────────────────────────────────

class VLAPlanner(Node):

    # Scaling factor: OpenVLA action deltas (designed for tabletop robots)
    # are small; we multiply by this to get meaningful drone displacements.
    ACTION_SCALE = 2.0

    def __init__(self):
        super().__init__('vla_planner')

        # ----- ROS parameters ------------------------------------------------
        self.declare_parameter('task_id', 1)
        self.declare_parameter('config_file', '')
        self.declare_parameter('output_dir', '/tmp')
        self.declare_parameter('mode', 'vla_only')
        self.declare_parameter('model_id', '')
        self.declare_parameter('action_scale', self.ACTION_SCALE)
        self.declare_parameter('inference_hz', 2.0)

        task_id = self.get_parameter('task_id').value
        config_path = self.get_parameter('config_file').value
        self.output_dir = self.get_parameter('output_dir').value
        self.mode = self.get_parameter('mode').value
        model_id = self.get_parameter('model_id').value
        self.action_scale = self.get_parameter('action_scale').value
        inference_hz_param = self.get_parameter('inference_hz').value

        if self.mode not in ('vla_only', 'vla_gt'):
            self.get_logger().fatal(
                f'Unknown mode "{self.mode}" — expected vla_only or vla_gt')
            raise SystemExit(1)

        if not config_path:
            pkg_share = get_package_share_directory('vla_navigation')
            config_path = f'{pkg_share}/config/benchmark_config.yaml'

        # ----- Load config ----------------------------------------------------
        self.get_logger().info(f'Loading config from {config_path}')
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        task = self._find_task(cfg, task_id)
        if task is None:
            self.get_logger().fatal(f'Task {task_id} not found in config')
            raise SystemExit(1)

        self.goal = task['goal']
        self.instruction = task['instruction']
        self.constraints = task['constraints']
        self.arrival_tol = cfg.get('arrival_tolerance', 0.5)
        self.takeoff_alt = cfg.get('takeoff_altitude', 5.0)
        self.land_speed = cfg.get('land_speed', 0.5)
        rate_hz = cfg.get('setpoint_rate_hz', 20)
        inference_hz = inference_hz_param or cfg.get('vla_inference_hz', 2.0)

        self.get_logger().info(
            f'Task {task_id} [{self.mode}]: "{self.instruction}" — '
            f'goal={self.goal}, constraints={self.constraints}'
        )

        # ----- State ----------------------------------------------------------
        self.phase = FlightPhase.IDLE
        self.task_id = task_id
        self.offboard_setpoint_counter = 0
        self.arm_attempts = 0
        self.tick_count = 0
        self.max_ticks = rate_hz * cfg.get('mission_timeout', 120)
        self.trajectory: list = []
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        # Navigation state driven by VLA inference
        self.current_setpoint_enu = [0.0, 0.0, self.takeoff_alt]
        self.latest_frame: Optional[np.ndarray] = None
        self.last_inference_time = 0.0
        self.inference_period = 1.0 / inference_hz

        # ----- OpenVLA model ---------------------------------------------------
        self.get_logger().info(f'Loading OpenVLA model (id={model_id or "default"})...')
        self.vla = OpenVLABackend(model_id=model_id)
        self.get_logger().info('OpenVLA model loaded')

        # ----- cv_bridge for ROS Image → numpy --------------------------------
        if CvBridge is None:
            self.get_logger().fatal('cv_bridge not available')
            raise SystemExit(1)
        self.bridge = CvBridge()

        # ----- QoS profiles ---------------------------------------------------
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----- Subscribers ----------------------------------------------------
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self._vehicle_local_position_cb, px4_qos)
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self._vehicle_status_cb, px4_qos)
        self.create_subscription(
            Image, '/camera', self._camera_cb, sensor_qos)

        # ----- Publishers -----------------------------------------------------
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_qos)

        # ----- Timer (main control loop) --------------------------------------
        period = 1.0 / rate_hz
        self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f'VLA planner initialised (mode={self.mode}) — waiting for vehicle heartbeat')

    # ── Config helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _find_task(cfg: dict, task_id: int) -> Optional[dict]:
        for t in cfg.get('tasks', []):
            if t['id'] == task_id:
                return t
        return None

    # ── Subscriber callbacks ─────────────────────────────────────────────────

    def _vehicle_local_position_cb(self, msg: VehicleLocalPosition):
        self.vehicle_local_position = msg

    def _vehicle_status_cb(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def _camera_cb(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}', throttle_duration_sec=5.0)

    # ── PX4 command helpers (same as classical planner) ──────────────────────

    def _publish_offboard_mode(self, position: bool = True):
        msg = OffboardControlMode()
        msg.position = position
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def _publish_setpoint(self, x: float, y: float, z: float,
                          yaw: float = float('nan')):
        """Publish a position setpoint.  Input is ENU; converts to NED."""
        msg = TrajectorySetpoint()
        msg.position = np.array([x, -y, -z], dtype=np.float32)
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def _send_command(self, command: int, param1: float = 0.0,
                      param2: float = 0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def _arm(self):
        self._send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info('ARM command sent')

    def _disarm(self):
        self._send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.get_logger().info('DISARM command sent')

    def _engage_offboard(self):
        self._send_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('OFFBOARD mode requested')

    # ── Position helpers ─────────────────────────────────────────────────────

    def _current_position_enu(self):
        p = self.vehicle_local_position
        return (p.x, -p.y, -p.z)

    def _distance_to(self, target_enu) -> float:
        cx, cy, cz = self._current_position_enu()
        return math.sqrt((cx - target_enu[0])**2 +
                         (cy - target_enu[1])**2 +
                         (cz - target_enu[2])**2)

    def _clamp_altitude(self, alt: float) -> float:
        lo = self.constraints.get('min_altitude', 0.0)
        hi = self.constraints.get('max_altitude', 100.0)
        return max(lo, min(hi, alt))

    # ── VLA inference ────────────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        prompt = f'In: What action should I take to {self.instruction}?'
        if self.mode == 'vla_gt':
            prompt += f' Target is at {self.goal}.'
        return prompt

    def _run_inference(self):
        """Call OpenVLA and update self.current_setpoint_enu."""
        if self.latest_frame is None:
            return

        now = _time.monotonic()
        if now - self.last_inference_time < self.inference_period:
            return

        prompt = self._build_prompt()
        action = self.vla.predict_action(self.latest_frame, prompt)
        dx, dy, dz = (
            action[0] * self.action_scale,
            action[1] * self.action_scale,
            action[2] * self.action_scale,
        )

        cx, cy, cz = self._current_position_enu()
        new_x = cx + dx
        new_y = cy + dy
        new_z = self._clamp_altitude(cz + dz)
        self.current_setpoint_enu = [new_x, new_y, new_z]

        self.last_inference_time = now
        self.get_logger().info(
            f'VLA action: delta=({dx:.2f},{dy:.2f},{dz:.2f}) '
            f'→ setpoint=({new_x:.1f},{new_y:.1f},{new_z:.1f})',
            throttle_duration_sec=1.0)

    # ── Trajectory logging ───────────────────────────────────────────────────

    def _record_position(self):
        p = self.vehicle_local_position
        if p.timestamp == 0:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        self.trajectory.append((t, p.x, -p.y, -p.z, self.phase.name))

    def _write_trajectory_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir,
                            f'task_{self.task_id}_trajectory.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'x_enu', 'y_enu', 'z_enu', 'phase'])
            w.writerows(self.trajectory)
        self.get_logger().info(
            f'Trajectory saved to {path} ({len(self.trajectory)} points)')

    # ── Main control loop ────────────────────────────────────────────────────

    def _control_loop(self):
        self.tick_count += 1
        if (self.tick_count > self.max_ticks
                and self.phase not in (FlightPhase.DONE, FlightPhase.LAND)):
            self.get_logger().error(
                f'Mission timeout after {self.tick_count} ticks — aborting')
            self._send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self._disarm()
            self.phase = FlightPhase.DONE
            raise SystemExit(1)

        self._publish_offboard_mode()
        self._record_position()

        # ── IDLE ─────────────────────────────────────────────────────────────
        if self.phase == FlightPhase.IDLE:
            self._publish_setpoint(0.0, 0.0, self.takeoff_alt)
            self.offboard_setpoint_counter += 1

            if self.offboard_setpoint_counter >= 200:
                if self.offboard_setpoint_counter == 200:
                    self._engage_offboard()

                armed = (self.vehicle_status.arming_state
                         == VehicleStatus.ARMING_STATE_ARMED)
                if not armed:
                    if self.offboard_setpoint_counter % 40 == 0:
                        self.arm_attempts += 1
                        self.get_logger().info(
                            f'ARM attempt {self.arm_attempts}')
                        self._arm()
                else:
                    self.phase = FlightPhase.TAKEOFF
                    self.get_logger().info(
                        f'Armed after {self.arm_attempts} attempt(s) → TAKEOFF')

        # ── TAKEOFF ──────────────────────────────────────────────────────────
        elif self.phase == FlightPhase.TAKEOFF:
            self._publish_setpoint(0.0, 0.0, self.takeoff_alt)
            alt = -self.vehicle_local_position.z
            if abs(alt - self.takeoff_alt) < self.arrival_tol:
                self.phase = FlightPhase.NAVIGATE
                self.current_setpoint_enu = list(self._current_position_enu())
                self.get_logger().info('→ NAVIGATE (VLA active)')

        # ── NAVIGATE (VLA-driven) ────────────────────────────────────────────
        elif self.phase == FlightPhase.NAVIGATE:
            self._run_inference()
            sx, sy, sz = self.current_setpoint_enu
            self._publish_setpoint(sx, sy, sz)

            dist = self._distance_to(self.goal)
            if dist < self.arrival_tol:
                self.get_logger().info(f'Goal reached (dist={dist:.2f} m)')
                self.phase = FlightPhase.LAND
                self.get_logger().info('→ LAND')

        # ── LAND ─────────────────────────────────────────────────────────────
        elif self.phase == FlightPhase.LAND:
            self._send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            alt = -self.vehicle_local_position.z
            if alt < 0.3:
                self._disarm()
                self.phase = FlightPhase.DONE
                self.get_logger().info('→ DONE — mission complete')

        # ── DONE ─────────────────────────────────────────────────────────────
        elif self.phase == FlightPhase.DONE:
            self._write_trajectory_csv()
            self.get_logger().info(
                'Mission finished. Shutting down.', once=True)
            raise SystemExit(0)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VLAPlanner()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info('Node exiting')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
