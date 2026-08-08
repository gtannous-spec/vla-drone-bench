"""
frame_recorder.py — Captures RGB frames from AirSim during flight.

Runs as a daemon thread alongside telemetry, saving timestamped frames
to disk. After the mission, frames can be stitched into a video with ffmpeg.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import airsim
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameRecorder:
    """Records camera frames from AirSim during flight.

    Creates its own AirSim client to avoid thread-safety issues.

    Usage:
        recorder = FrameRecorder(output_dir="output/frames/task_1", fps=5)
        recorder.start()
        ...  # fly
        recorder.stop()
        recorder.make_video()  # optional: stitch into mp4
    """

    def __init__(
        self,
        output_dir: str,
        vehicle_name: str = "Drone0",
        camera_name: str = "front_center",
        fps: float = 5.0,
        save_format: str = "jpg",
    ):
        self._output_dir = Path(output_dir)
        self._vehicle_name = vehicle_name
        self._camera_name = camera_name
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
        self._stop_event.clear()
        self._frame_count = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="frame_recorder"
        )
        self._thread.start()
        logger.info(f"Frame recorder started — saving to {self._output_dir}")

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
        """Capture one frame and save to disk."""
        responses = self._client.simGetImages(
            [airsim.ImageRequest(
                self._camera_name, airsim.ImageType.Scene, False, False
            )],
            vehicle_name=self._vehicle_name,
        )

        if not responses or responses[0].width == 0:
            return

        img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img.reshape(responses[0].height, responses[0].width, 3)

        filename = f"frame_{self._frame_count:06d}.{self._save_format}"
        filepath = self._output_dir / filename
        cv2.imwrite(str(filepath), img)
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
