"""
drone_fsm.py — Finite State Machine for AirSim drone missions.

States: IDLE → TAKEOFF → NAVIGATE → LAND → DONE

The FSM is controller-agnostic: any BaseController implementation can drive
the NAVIGATE phase. The FSM manages lifecycle (arm/disarm, API control) and
delegates navigation decisions to the injected controller.
"""

import logging
import time
from enum import Enum, auto

from ..controllers.base_controller import BaseController, DroneState
from .airsim_client import AirSimClient
from .telemetry import TelemetryThread

logger = logging.getLogger(__name__)


class FlightPhase(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    NAVIGATE = auto()
    LAND = auto()
    DONE = auto()


class DroneFSM:
    """Finite State Machine for executing a single drone navigation task.

    Args:
        client: Connected AirSimClient instance.
        controller: Navigation controller implementing BaseController.
        telemetry: TelemetryThread instance for recording.
        takeoff_altitude: Target altitude for takeoff (NED, negative = up).
        mission_timeout: Maximum mission duration in seconds.
        nav_speed: Default navigation speed in m/s.
    """

    def __init__(
        self,
        client: AirSimClient,
        controller: BaseController,
        telemetry: TelemetryThread,
        takeoff_altitude: float = -5.0,
        mission_timeout: float = 90.0,
        nav_speed: float = 5.0,
        capture_images: bool = False,
    ):
        self._client = client
        self._controller = controller
        self._telemetry = telemetry
        self._takeoff_altitude = takeoff_altitude
        self._mission_timeout = mission_timeout
        self._nav_speed = nav_speed
        self._capture_images = capture_images
        self._phase = FlightPhase.IDLE
        self._start_time: float = 0.0
        self._success: bool = False

    @property
    def phase(self) -> FlightPhase:
        return self._phase

    @property
    def success(self) -> bool:
        return self._success

    def execute(self) -> bool:
        """Run the full FSM from IDLE through DONE.

        Returns:
            True if the mission completed successfully (goal reached),
            False if it timed out or failed.
        """
        self._start_time = time.time()
        self._success = False

        try:
            self._phase_idle()
            logger.info("Starting telemetry thread...")
            self._telemetry.start()
            logger.info("Telemetry thread started (connecting in background)")
            self._phase_takeoff()
            self._phase_navigate()
            self._phase_land()
            self._phase_done()
        except TimeoutError as e:
            logger.error(f"Mission timeout: {e}")
            self._emergency_land()
            self._phase = FlightPhase.DONE
            self._telemetry.current_phase = self._phase.name
        except Exception as e:
            logger.error(f"FSM error: {e}", exc_info=True)
            self._emergency_land()
            self._phase = FlightPhase.DONE
            self._telemetry.current_phase = self._phase.name

        return self._success

    def _check_timeout(self) -> None:
        elapsed = time.time() - self._start_time
        if elapsed > self._mission_timeout:
            raise TimeoutError(
                f"Mission exceeded {self._mission_timeout}s timeout "
                f"(elapsed: {elapsed:.1f}s, phase: {self._phase.name})"
            )

    def _phase_idle(self) -> None:
        """IDLE: Enable API control and arm the vehicle."""
        self._phase = FlightPhase.IDLE
        self._telemetry.current_phase = self._phase.name
        logger.info("FSM → IDLE: Enabling API control")

        # Small delay to let AirSim settle after teleport/reset
        time.sleep(1.0)
        self._client.client.enableApiControl(
            True, vehicle_name=self._client.vehicle_name
        )
        self._client.client.armDisarm(
            True, vehicle_name=self._client.vehicle_name
        )
        time.sleep(1.0)

    def _phase_takeoff(self) -> None:
        """TAKEOFF: Ascend to the configured takeoff altitude."""
        self._check_timeout()
        self._phase = FlightPhase.TAKEOFF
        self._telemetry.current_phase = self._phase.name
        logger.info(f"FSM → TAKEOFF: climbing to z={self._takeoff_altitude:.1f}")

        self._client.takeoff(altitude=self._takeoff_altitude)

    def _phase_navigate(self) -> None:
        """NAVIGATE: Delegate to the controller until goal is reached."""
        self._check_timeout()
        self._phase = FlightPhase.NAVIGATE
        self._telemetry.current_phase = self._phase.name
        logger.info("FSM → NAVIGATE: controller active")

        consecutive_errors = 0
        max_consecutive_errors = 3

        while True:
            self._check_timeout()

            state = self._get_drone_state()

            if self._controller.is_goal_reached(state):
                self._success = True
                logger.info("Goal reached — transitioning to LAND")
                break

            try:
                action = self._controller.get_action(state)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    f"Controller error ({consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                if consecutive_errors >= max_consecutive_errors:
                    raise
                time.sleep(0.5)
                continue

            remaining = self._mission_timeout - (time.time() - self._start_time)
            if self._capture_images:
                move_timeout = min(5.0, max(2.0, remaining))
            else:
                move_timeout = min(30.0, max(5.0, remaining))

            self._client.move_to(
                action.target_position[0],
                action.target_position[1],
                action.target_position[2],
                velocity=action.velocity,
                timeout_sec=move_timeout,
            )

            state = self._get_drone_state()
            if self._controller.is_goal_reached(state):
                self._success = True
                logger.info("Goal reached — transitioning to LAND")
                break

    def _phase_land(self) -> None:
        """LAND: Descend and disarm."""
        self._check_timeout()
        self._phase = FlightPhase.LAND
        self._telemetry.current_phase = self._phase.name
        logger.info("FSM → LAND: descending")

        self._client.land()

    def _phase_done(self) -> None:
        """DONE: Mission complete."""
        self._phase = FlightPhase.DONE
        self._telemetry.current_phase = self._phase.name
        elapsed = time.time() - self._start_time
        status = "SUCCESS" if self._success else "FAILED"
        logger.info(f"FSM → DONE [{status}] — total time: {elapsed:.1f}s")

    def _emergency_land(self) -> None:
        """Attempt to land safely after an error."""
        try:
            self._client.land()
        except Exception as e:
            logger.error(f"Emergency landing failed: {e}")

    def _get_drone_state(self) -> DroneState:
        """Build a DroneState snapshot from current telemetry."""
        image = None
        if self._capture_images:
            image = self._client.get_camera_image()
        return DroneState(
            position=self._client.get_position(),
            velocity=self._client.get_velocity(),
            orientation=self._client.get_orientation(),
            image=image,
        )
