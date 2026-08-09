"""
frame_recorder.py — Captures RGB frames from AirSim during flight.

Runs as a daemon thread alongside telemetry, saving timestamped frames
to disk. Supports dual-camera split-screen recording (front + bottom).
After the mission, frames can be stitched into a video.
"""

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

import airsim
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameRecorder:
    """Records camera frames from AirSim during flight.

    Creates its own AirSim client to avoid thread-safety issues.
    Supports single-camera or dual-camera (split-screen) recording.

    Usage:
        recorder = FrameRecorder(output_dir="output/frames/task_1", fps=5)
        recorder.start()
        ...  # fly
        recorder.stop()
        recorder.make_video()
    """

    def __init__(
        self,
        output_dir: str,
        vehicle_name: str = "Drone0",
        camera_name: str = "front_center",
        fps: float = 5.0,
        save_format: str = "jpg",
        split_screen: bool = True,
        secondary_camera: str = "bottom_center",
    ):
        self._output_dir = Path(output_dir)
        self._vehicle_name = vehicle_name
        self._camera_name = camera_name
        self._secondary_camera = secondary_camera
        self._split_screen = split_screen
        self._interval = 1.0 / fps
        self._fps = fps
        self._save_format = save_format
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._client: Optional[airsim.MultirotorClient] = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def start(self) -> None:
        """Start the frame recording thread."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_previous()
        self._stop_event.clear()
        self._frame_count = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="frame_recorder"
        )
        self._thread.start()
        mode = "split-screen" if self._split_screen else "single"
        logger.info(f"Frame recorder started ({mode}) — saving to {self._output_dir}")

    def _cleanup_previous(self) -> None:
        """Remove old frames and videos from previous runs."""
        old_frames = list(self._output_dir.glob(f"frame_*.{self._save_format}"))
        old_videos = list(self._output_dir.glob("*.mp4"))
        removed = 0
        for f in old_frames + old_videos:
            f.unlink()
            removed += 1
        if removed:
            logger.info(f"Cleaned up {removed} old file(s) from {self._output_dir}")

    def stop(self) -> None:
        """Stop the frame recording thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._client = None
        logger.info(f"Frame recorder stopped — {self._frame_count} frames captured.")

    def _connect(self) -> None:
        """Create a dedicated client for image capture."""
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()

    def _run(self) -> None:
        """Main loop: connect and capture frames at configured rate."""
        try:
            self._connect()
        except Exception as e:
            logger.error(f"Frame recorder client connection failed: {e}")
            return

        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                self._capture_frame()
            except Exception as e:
                logger.warning(f"Frame capture error: {e}")
            elapsed = time.time() - t0
            sleep_time = max(0.0, self._interval - elapsed)
            self._stop_event.wait(timeout=sleep_time)

    def _capture_frame(self) -> None:
        """Capture frame(s) and save to disk. Supports split-screen layout."""
        requests = [
            airsim.ImageRequest(self._camera_name, airsim.ImageType.Scene, False, False)
        ]
        if self._split_screen:
            requests.append(
                airsim.ImageRequest(self._secondary_camera, airsim.ImageType.Scene, False, False)
            )

        responses = self._client.simGetImages(
            requests, vehicle_name=self._vehicle_name
        )

        if not responses or responses[0].width == 0:
            return

        front_img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        front_img = front_img.reshape(responses[0].height, responses[0].width, 3)

        if self._split_screen and len(responses) > 1 and responses[1].width > 0:
            bottom_img = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
            bottom_img = bottom_img.reshape(responses[1].height, responses[1].width, 3)

            # Resize bottom to match front height, then stack side-by-side
            h_front = front_img.shape[0]
            scale = h_front / bottom_img.shape[0]
            new_w = int(bottom_img.shape[1] * scale)
            bottom_resized = cv2.resize(bottom_img, (new_w, h_front))

            combined = np.hstack([front_img, bottom_resized])
        else:
            combined = front_img

        filename = f"frame_{self._frame_count:06d}.{self._save_format}"
        filepath = self._output_dir / filename
        cv2.imwrite(str(filepath), combined)
        self._frame_count += 1

    def make_video(self, output_path: Optional[str] = None, cleanup_frames: bool = False) -> Optional[str]:
        """Stitch saved frames into an MP4 video using OpenCV.

        Args:
            output_path: Path for the output video. Defaults to <output_dir>/flight.mp4.
            cleanup_frames: If True, delete individual frame files after creating video.

        Returns:
            Path to the created video, or None if no frames exist.
        """
        if self._frame_count == 0:
            logger.warning("No frames to stitch into video.")
            return None

        if output_path is None:
            output_path = str(self._output_dir / "flight.mp4")

        frame_files = sorted(self._output_dir.glob(f"frame_*.{self._save_format}"))
        if not frame_files:
            return None

        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self._fps, (w, h))

        for fpath in frame_files:
            frame = cv2.imread(str(fpath))
            if frame is not None:
                writer.write(frame)

        writer.release()
        logger.info(f"Video saved: {output_path} ({len(frame_files)} frames, {self._fps} fps)")

        if cleanup_frames:
            for fpath in frame_files:
                fpath.unlink()
            logger.info("Frame files cleaned up.")

        return output_path
