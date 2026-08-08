"""
telemetry.py — Background telemetry recording thread.

Continuously polls the drone's kinematics and collision state,
storing timestamped records for post-flight analysis.

Uses its own dedicated AirSim client connection to avoid thread-safety
issues with msgpack-rpc-python's tornado IOLoop (the transport layer
cannot handle concurrent calls from multiple threads on one socket).
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import airsim
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    """Single telemetry sample."""
    timestamp: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    qw: float
    qx: float
    qy: float
    qz: float
    phase: str


@dataclass
class CollisionEvent:
    """Record of a collision event."""
    timestamp: float
    object_name: str
    position: Tuple[float, float, float]


class TelemetryThread:
    """Daemon thread that records drone kinematics and collision events.

    Creates its own airsim.MultirotorClient to avoid sharing the transport
    layer with the FSM thread (msgpack-rpc-python is not thread-safe).

    Usage:
        telem = TelemetryThread(vehicle_name="Drone0", rate_hz=10)
        telem.start()
        ...  # fly
        telem.stop()
        trajectory = telem.get_trajectory()
        collisions = telem.get_collisions()
    """

    def __init__(self, vehicle_name: str = "Drone0", rate_hz: float = 10.0):
        self._vehicle_name = vehicle_name
        self._interval = 1.0 / rate_hz
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._trajectory: List[TelemetryRecord] = []
        self._collisions: List[CollisionEvent] = []
        self._current_phase: str = "IDLE"
        self._last_collision_ts: float = 0.0
        self._telem_client: Optional[airsim.MultirotorClient] = None

    @property
    def current_phase(self) -> str:
        return self._current_phase

    @current_phase.setter
    def current_phase(self, phase: str) -> None:
        self._current_phase = phase

    def start(self) -> None:
        """Start the telemetry recording thread."""
        self._stop_event.clear()
        self._trajectory.clear()
        self._collisions.clear()
        self._last_collision_ts = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry")
        self._thread.start()
        logger.debug("Telemetry thread started.")

    def stop(self) -> None:
        """Stop the telemetry recording thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._telem_client = None
        logger.debug(f"Telemetry thread stopped — {len(self._trajectory)} records.")

    def _connect(self) -> None:
        """Create a dedicated read-only client for telemetry polling."""
        self._telem_client = airsim.MultirotorClient()
        self._telem_client.confirmConnection()
        logger.debug("Telemetry client connected.")

    def _run(self) -> None:
        """Main loop: connect own client, then poll at configured rate."""
        try:
            self._connect()
        except Exception as e:
            logger.error(f"Telemetry client connection failed: {e}")
            return

        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                self._record_sample()
            except Exception as e:
                logger.warning(f"Telemetry poll error: {e}")
            elapsed = time.time() - t0
            sleep_time = max(0.0, self._interval - elapsed)
            self._stop_event.wait(timeout=sleep_time)

    def _record_sample(self) -> None:
        """Take one telemetry sample and check for collisions."""
        state = self._telem_client.getMultirotorState(
            vehicle_name=self._vehicle_name
        )
        now = time.time()

        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orient = state.kinematics_estimated.orientation

        record = TelemetryRecord(
            timestamp=now,
            x=pos.x_val, y=pos.y_val, z=pos.z_val,
            vx=vel.x_val, vy=vel.y_val, vz=vel.z_val,
            qw=orient.w_val, qx=orient.x_val,
            qy=orient.y_val, qz=orient.z_val,
            phase=self._current_phase,
        )

        with self._lock:
            self._trajectory.append(record)

        col_info = self._telem_client.simGetCollisionInfo(
            vehicle_name=self._vehicle_name
        )
        if col_info.has_collided and col_info.time_stamp != self._last_collision_ts:
            self._last_collision_ts = col_info.time_stamp
            col_pos = (col_info.position.x_val,
                       col_info.position.y_val,
                       col_info.position.z_val)
            event = CollisionEvent(
                timestamp=now,
                object_name=col_info.object_name,
                position=col_pos,
            )
            with self._lock:
                self._collisions.append(event)
            # Only log once per object to avoid flooding the log
            seen_objects = {c.object_name for c in self._collisions[:-1]}
            if event.object_name not in seen_objects:
                logger.warning(f"Collision with '{event.object_name}' "
                               f"at ({col_pos[0]:.1f}, {col_pos[1]:.1f}, {col_pos[2]:.1f})")

    def get_trajectory(self) -> List[TelemetryRecord]:
        """Return a copy of all telemetry records."""
        with self._lock:
            return list(self._trajectory)

    def get_collisions(self) -> List[CollisionEvent]:
        """Return a copy of all collision events."""
        with self._lock:
            return list(self._collisions)

    def get_trajectory_arrays(self) -> Dict[str, np.ndarray]:
        """Return trajectory as numpy arrays for easy analysis."""
        with self._lock:
            records = list(self._trajectory)
        if not records:
            return {"time": np.array([]), "x": np.array([]),
                    "y": np.array([]), "z": np.array([]), "phases": []}
        return {
            "time": np.array([r.timestamp for r in records]),
            "x": np.array([r.x for r in records]),
            "y": np.array([r.y for r in records]),
            "z": np.array([r.z for r in records]),
            "phases": [r.phase for r in records],
        }
