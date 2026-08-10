"""
airsim_client.py — Thin wrapper around airsim.MultirotorClient.

Provides a simplified, project-specific interface for drone control,
hiding AirSim API details from the FSM and controllers.
"""

import logging
import time
from typing import Optional, Tuple

# On Python 3.10+, msgpackrpc's tornado-based RPC deadlocks because Tornado 6
# wraps asyncio and run_forever() is not reentrant. Bypass it entirely by
# replacing msgpackrpc.Client/Address with our synchronous socket-based client
# BEFORE airsim imports msgpackrpc.
from airsim_benchmark.core.sync_rpc import SyncClient, SyncAddress
import msgpackrpc                       # noqa: E402 — must import before airsim
msgpackrpc.Client = SyncClient          # monkey-patch
msgpackrpc.Address = SyncAddress        # monkey-patch

import airsim                           # noqa: E402 — now uses SyncClient
import numpy as np

from airsim_benchmark.core.move_plan import plan_move_to, plan_vertical_velocity, yaw_from_quat

logger = logging.getLogger(__name__)


class AirSimClient:
    """Wrapper around airsim.MultirotorClient for multirotor control."""

    def __init__(self, vehicle_name: str = "Drone0", timeout_sec: float = 60.0):
        self._vehicle_name = vehicle_name
        self._timeout_sec = timeout_sec
        self._client: Optional[airsim.MultirotorClient] = None

    @property
    def vehicle_name(self) -> str:
        return self._vehicle_name

    @property
    def client(self) -> airsim.MultirotorClient:
        if self._client is None:
            raise RuntimeError("AirSim client not connected. Call connect() first.")
        return self._client

    def connect(self) -> None:
        """Establish connection to the AirSim simulator."""
        logger.info("Creating MultirotorClient...")
        self._client = airsim.MultirotorClient()
        logger.info("Calling confirmConnection() (timeout=%ss)...", self._timeout_sec)
        self._client.confirmConnection()
        logger.info("confirmConnection() returned — enabling API control...")
        self._client.enableApiControl(True, vehicle_name=self._vehicle_name)
        self._client.armDisarm(True, vehicle_name=self._vehicle_name)
        logger.info("Connected to AirSim — API control enabled, vehicle armed.")

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self.client.reset()
        time.sleep(0.5)
        self.client.enableApiControl(True, vehicle_name=self._vehicle_name)
        self.client.armDisarm(True, vehicle_name=self._vehicle_name)
        logger.info("Simulation reset.")

    def disconnect(self) -> None:
        """Disable API control and release the client."""
        if self._client is not None:
            self._client.armDisarm(False, vehicle_name=self._vehicle_name)
            self._client.enableApiControl(False, vehicle_name=self._vehicle_name)
            logger.info("Disconnected from AirSim.")
        self._client = None

    def teleport_to(self, x: float, y: float, z: float, yaw_deg: float = 0.0) -> None:
        """Teleport vehicle to a specific NED pose (used for task start positions)."""
        pose = airsim.Pose(
            airsim.Vector3r(x, y, z),
            airsim.to_quaternion(0, 0, np.radians(yaw_deg)),
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True,
                                      vehicle_name=self._vehicle_name)
        # Arm immediately so the 1s settle is a hover, not a fall.
        self.client.enableApiControl(True, vehicle_name=self._vehicle_name)
        self.client.armDisarm(True, vehicle_name=self._vehicle_name)
        time.sleep(0.2)
        logger.info(f"Teleported to ({x:.1f}, {y:.1f}, {z:.1f}), yaw={yaw_deg:.0f}°")

    def takeoff(self, altitude: float = -5.0) -> None:
        """Take off to specified altitude (NED, negative = up).

        Uses moveToZAsync for precise altitude control rather than the
        default takeoffAsync which only goes to a fixed height.
        """
        self.client.takeoffAsync(
            timeout_sec=self._timeout_sec,
            vehicle_name=self._vehicle_name,
        ).join()
        self.client.moveToZAsync(
            altitude, velocity=2.0,
            timeout_sec=self._timeout_sec,
            vehicle_name=self._vehicle_name,
        ).join()
        logger.info(f"Takeoff complete — hovering at z={altitude:.1f} m")

    def hold_altitude(self, altitude: float) -> None:
        """Settle at ``altitude`` after teleport (already airborne).

        Do not use a 0.5 m lookahead here: after teleport we are already at
        the target, so that lookahead never "arrives" and ``join()`` blocks
        for ``timeout_sec`` (60s) while the drone falls. Default lookahead
        (~3 m) plus a short timeout is a settle, not a minute-long wait.
        """
        self.client.moveToZAsync(
            altitude, velocity=2.0,
            timeout_sec=5.0,
            vehicle_name=self._vehicle_name,
        ).join()
        self.hover()
        now_z = self.get_position()[2]
        logger.info(f"Holding altitude at z={altitude:.1f} m (now {now_z:.1f})")

    def move_to(self, x: float, y: float, z: float, velocity: float = 5.0,
                timeout_sec: float = 30.0) -> None:
        """Fly to a target position (NED) at given velocity.

        SimpleFlight defaults ``lookahead=3``. Oracle search/ego/climb hops
        are 1.5–2.5 m, so the default treats those waypoints as already
        reached and the drone never moves. Pass a lookahead smaller than
        the shortest hop. Values ``<= 0`` fall back to the 3 m default.

        Face the hop with MaxDegreeOfFreedom + clipped absolute yaw so the
        camera turns. The forward-facing drivetrain overshoots heading by
        ~100° on short hops. Vertical-only hops use moveByVelocityAsync:
        a lookahead Z-hold from hover only crawls ~0.14 m, and a forward
        nudge would walk off a rooftop.
        """
        pos = self.get_position()
        qw, qx, qy, qz = self.get_orientation()
        plan = plan_move_to(pos, (x, y, z), yaw_from_quat(qw, qx, qy, qz))
        if plan["kind"] == "noop":
            return
        if plan["kind"] == "vertical":
            vel = plan_vertical_velocity(
                pos[2],
                plan["z"],
                speed=min(float(velocity), 2.0),
                max_duration=min(float(timeout_sec), 10.0),
            )
            self.client.moveByVelocityAsync(
                0.0, 0.0, vel["vz"], vel["duration"],
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self._vehicle_name,
            ).join()
            self.hover()
            logger.debug("moveByVelocity Z complete: vz=%.2f dur=%.2f",
                         vel["vz"], vel["duration"])
            return
        self.client.moveToPositionAsync(
            plan["x"], plan["y"], plan["z"], velocity,
            timeout_sec=timeout_sec,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=plan["yaw_deg"]),
            lookahead=0.5,
            adaptive_lookahead=0,
            vehicle_name=self._vehicle_name,
        ).join()
        logger.debug(f"moveToPosition complete: ({x:.1f}, {y:.1f}, {z:.1f})")

    def land(self) -> None:
        """Land the vehicle at current horizontal position."""
        self.client.landAsync(
            timeout_sec=self._timeout_sec,
            vehicle_name=self._vehicle_name,
        ).join()
        self.client.armDisarm(False, vehicle_name=self._vehicle_name)
        logger.info("Landing complete — disarmed.")

    def hover(self) -> None:
        """Command the vehicle to hold its current position."""
        self.client.hoverAsync(vehicle_name=self._vehicle_name).join()

    def get_position(self) -> Tuple[float, float, float]:
        """Return current position as (x, y, z) in NED."""
        state = self.client.getMultirotorState(vehicle_name=self._vehicle_name)
        pos = state.kinematics_estimated.position
        return (pos.x_val, pos.y_val, pos.z_val)

    def get_velocity(self) -> Tuple[float, float, float]:
        """Return current linear velocity as (vx, vy, vz) in NED."""
        state = self.client.getMultirotorState(vehicle_name=self._vehicle_name)
        vel = state.kinematics_estimated.linear_velocity
        return (vel.x_val, vel.y_val, vel.z_val)

    def get_orientation(self) -> Tuple[float, float, float, float]:
        """Return current orientation as quaternion (w, x, y, z)."""
        state = self.client.getMultirotorState(vehicle_name=self._vehicle_name)
        q = state.kinematics_estimated.orientation
        return (q.w_val, q.x_val, q.y_val, q.z_val)

    def has_collided(self) -> bool:
        """Check whether a collision has occurred since last reset."""
        info = self.client.simGetCollisionInfo(vehicle_name=self._vehicle_name)
        return info.has_collided

    def get_collision_info(self) -> dict:
        """Return detailed collision information."""
        info = self.client.simGetCollisionInfo(vehicle_name=self._vehicle_name)
        return {
            "has_collided": info.has_collided,
            "object_name": info.object_name,
            "position": (info.position.x_val, info.position.y_val, info.position.z_val),
            "time_stamp": info.time_stamp,
        }

    def get_camera_image(self, camera_name: str = "front_center") -> Optional[np.ndarray]:
        """Capture an RGB image from the specified camera.

        Returns an (H, W, 3) uint8 numpy array or None on failure.
        """
        responses = self.client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)],
            vehicle_name=self._vehicle_name,
        )
        if not responses or responses[0].width == 0:
            return None
        img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img.reshape(responses[0].height, responses[0].width, 3)
        return img

    # ── Weather / lighting controls for visual diversity ─────────────

    def set_weather(self, rain: float = 0.0, snow: float = 0.0,
                    dust: float = 0.0, fog: float = 0.0) -> None:
        """Set weather parameters (each 0.0–1.0). Requires AirSim Weather API."""
        try:
            self.client.simEnableWeather(True)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, rain)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, snow)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, dust)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog)
            logger.info(f"Weather set: rain={rain:.1f}, snow={snow:.1f}, "
                        f"dust={dust:.1f}, fog={fog:.1f}")
        except Exception as e:
            logger.warning(f"Weather API not available: {e}")

    def set_time_of_day(self, hour: int = 12, is_enabled: bool = True,
                        speed: float = 1.0) -> None:
        """Set the simulation time of day for lighting variation."""
        try:
            time_str = f"2024-06-15 {hour:02d}:00:00"
            self.client.simSetTimeOfDay(
                is_enabled, time_str,
                start_datetime=time_str,
                celestial_clock_speed=speed,
            )
            logger.info(f"Time of day set to {hour}:00")
        except Exception as e:
            logger.warning(f"TimeOfDay API not available: {e}")
