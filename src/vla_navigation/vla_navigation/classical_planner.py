"""
classical_planner.py — Milestone 1 waypoint controller.

Reads a task from benchmark_config.yaml and flies the PX4-simulated drone
from start to goal using pure coordinate-based navigation (no VLA).

Communication with PX4 happens over Micro-XRCE-DDS via px4_msgs.
"""

import csv
import math
import os
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

import yaml
from ament_index_python.packages import get_package_share_directory


# ── Finite-state machine ────────────────────────────────────────────────────

class FlightPhase(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    NAVIGATE = auto()
    LAND = auto()
    DONE = auto()


# ── Node ─────────────────────────────────────────────────────────────────────

class ClassicalPlanner(Node):

    def __init__(self):
        super().__init__('classical_planner')

        # ----- ROS parameters ------------------------------------------------
        self.declare_parameter('task_id', 1)
        self.declare_parameter('config_file', '')
        self.declare_parameter('output_dir', '/tmp')

        task_id = self.get_parameter('task_id').value
        config_path = self.get_parameter('config_file').value
        self.output_dir = self.get_parameter('output_dir').value

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

        self.goal = task['goal']              # [x, y, z]  ENU
        self.constraints = task['constraints']
        self.arrival_tol = cfg.get('arrival_tolerance', 0.5)
        self.takeoff_alt = cfg.get('takeoff_altitude', 5.0)
        self.land_speed = cfg.get('land_speed', 0.5)
        rate_hz = cfg.get('setpoint_rate_hz', 20)

        self.get_logger().info(
            f'Task {task_id}: "{task["instruction"]}" — '
            f'goal={self.goal}, constraints={self.constraints}'
        )

        # ----- State ----------------------------------------------------------
        self.phase = FlightPhase.IDLE
        self.task_id = task_id
        self.offboard_setpoint_counter = 0
        self.arm_attempts = 0
        self.tick_count = 0
        self.max_ticks = rate_hz * cfg.get('mission_timeout', 120)
        self.trajectory: list[tuple] = []
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        # ----- QoS profile matching PX4 defaults ------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----- Subscribers ----------------------------------------------------
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self._vehicle_local_position_cb,
            qos,
        )
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self._vehicle_status_cb,
            qos,
        )

        # ----- Publishers -----------------------------------------------------
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos
        )

        # ----- Timer (main control loop) --------------------------------------
        period = 1.0 / rate_hz
        self.create_timer(period, self._control_loop)

        self.get_logger().info('Classical planner initialised — waiting for vehicle heartbeat')

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

    # ── PX4 command helpers ──────────────────────────────────────────────────

    def _publish_offboard_mode(self, position: bool = True):
        msg = OffboardControlMode()
        msg.position = position
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def _publish_setpoint(self, x: float, y: float, z: float, yaw: float = float('nan')):
        """Publish a position setpoint in NED frame (PX4 convention).

        The config uses ENU coordinates.  PX4 TrajectorySetpoint expects NED,
        so we convert:  NED_x = ENU_x,  NED_y = -ENU_y,  NED_z = -ENU_z.
        """
        msg = TrajectorySetpoint()
        msg.position = np.array([x, -y, -z], dtype=np.float32)
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def _send_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
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
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,   # base mode
            param2=6.0,   # PX4 custom mode: offboard
        )
        self.get_logger().info('OFFBOARD mode requested')

    # ── Position helpers ─────────────────────────────────────────────────────

    def _current_position_ned(self):
        p = self.vehicle_local_position
        return (p.x, p.y, p.z)

    def _distance_to(self, target_enu: list[float]) -> float:
        """Euclidean distance from current NED position to an ENU target."""
        cx, cy, cz = self._current_position_ned()
        tx, ty, tz = target_enu[0], -target_enu[1], -target_enu[2]
        return math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2 + (cz - tz) ** 2)

    def _clamp_altitude(self, alt_enu: float) -> float:
        lo = self.constraints.get('min_altitude', 0.0)
        hi = self.constraints.get('max_altitude', 100.0)
        return max(lo, min(hi, alt_enu))

    def _record_position(self):
        """Append current position (converted to ENU) and flight phase."""
        p = self.vehicle_local_position
        if p.timestamp == 0:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        self.trajectory.append((t, p.x, -p.y, -p.z, self.phase.name))

    def _write_trajectory_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f'task_{self.task_id}_trajectory.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'x_enu', 'y_enu', 'z_enu', 'phase'])
            w.writerows(self.trajectory)
        self.get_logger().info(f'Trajectory saved to {path} ({len(self.trajectory)} points)')

    # ── Main control loop ────────────────────────────────────────────────────

    def _control_loop(self):
        self.tick_count += 1
        if self.tick_count > self.max_ticks and self.phase not in (FlightPhase.DONE, FlightPhase.LAND):
            self.get_logger().error(f'Mission timeout after {self.tick_count} ticks — aborting')
            self._send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self._disarm()
            self.phase = FlightPhase.DONE
            raise SystemExit(1)

        # Always publish offboard heartbeat so PX4 doesn't fall back
        self._publish_offboard_mode()
        self._record_position()

        # ── IDLE: wait for PX4 sensors to stabilise, then offboard + arm ─────
        if self.phase == FlightPhase.IDLE:
            self._publish_setpoint(0.0, 0.0, self.takeoff_alt)
            self.offboard_setpoint_counter += 1

            if self.offboard_setpoint_counter >= 200:  # ~10 s at 20 Hz — sensors need time in SITL
                if self.offboard_setpoint_counter == 200:
                    self._engage_offboard()

                armed = self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                if not armed:
                    if self.offboard_setpoint_counter % 40 == 0:  # retry every 2 s
                        self.arm_attempts += 1
                        self.get_logger().info(f'ARM attempt {self.arm_attempts}')
                        self._arm()
                else:
                    self.phase = FlightPhase.TAKEOFF
                    self.get_logger().info(f'Armed after {self.arm_attempts} attempt(s) → TAKEOFF')

        # ── TAKEOFF: climb to takeoff altitude ───────────────────────────────
        elif self.phase == FlightPhase.TAKEOFF:
            self._publish_setpoint(0.0, 0.0, self.takeoff_alt)

            alt = -self.vehicle_local_position.z  # NED → AGL
            if abs(alt - self.takeoff_alt) < self.arrival_tol:
                self.phase = FlightPhase.NAVIGATE
                self.get_logger().info('→ NAVIGATE')

        # ── NAVIGATE: fly to goal ────────────────────────────────────────────
        elif self.phase == FlightPhase.NAVIGATE:
            gx, gy, gz = self.goal
            gz = self._clamp_altitude(gz)
            self._publish_setpoint(gx, gy, gz)

            dist = self._distance_to([gx, gy, gz])
            if dist < self.arrival_tol:
                self.get_logger().info(f'Goal reached (dist={dist:.2f} m)')
                self.phase = FlightPhase.LAND
                self.get_logger().info('→ LAND')

        # ── LAND: descend and disarm ─────────────────────────────────────────
        elif self.phase == FlightPhase.LAND:
            self._send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

            alt = -self.vehicle_local_position.z
            if alt < 0.3:
                self._disarm()
                self.phase = FlightPhase.DONE
                self.get_logger().info('→ DONE — mission complete')

        # ── DONE: nothing left to do ─────────────────────────────────────────
        elif self.phase == FlightPhase.DONE:
            self._write_trajectory_csv()
            self.get_logger().info('Mission finished. Shutting down.', once=True)
            raise SystemExit(0)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ClassicalPlanner()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info('Node exiting')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
