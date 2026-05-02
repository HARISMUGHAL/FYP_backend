from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    source: str | int = 0
    warmup_frames: int = 3
    retry_delay_sec: float = 1.0


class CameraService:
    """
    Manages camera lifecycle and single-frame capture.

    Supports:
    - Local webcam index (0, 1, ...)
    - Camera URL (for mobile/IP camera streams)
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open camera stream if it is not already open."""
        if self._cap is not None and self._cap.isOpened():
            return

        self._cap = cv2.VideoCapture(self.config.source)
        if not self._cap.isOpened():
            self.close()
            raise RuntimeError(f"Unable to open camera source: {self.config.source}")

        for _ in range(max(0, self.config.warmup_frames)):
            self._cap.read()

    def close(self) -> None:
        """Release camera stream resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def capture_frame(self) -> np.ndarray:
        """
        Capture one frame from camera.

        Returns:
            BGR numpy image from OpenCV.
        """
        self.open()
        if self._cap is None:
            raise RuntimeError("Camera is not initialized.")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame
