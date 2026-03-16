"""
Live Detector
=============
Runs YOLO object detection on camera frames and publishes normalized boxes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .camera_manager import CameraManager

logger = logging.getLogger(__name__)


class LiveDetector:
    """YOLO-based detector that publishes latest detection boxes."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        infer_every_n: int = 2,
        imgsz: int = 640,
    ) -> None:
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._infer_every_n = max(1, infer_every_n)

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._camera_manager: Optional[CameraManager] = None
        self._frame_count = 0
        self._latest_detection: dict = {"ts": 0.0, "boxes": []}

        self._model: Optional[YOLO] = None
        self._enabled = False
        try:
            self._model = YOLO(model_path)
            self._enabled = True
            logger.info("Live detector enabled with model: %s", model_path)
        except Exception as exc:
            logger.error("Could not load YOLO model '%s': %s", model_path, exc)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    def start(self, camera_manager: CameraManager) -> None:
        """Start background detection loop."""
        if not self.enabled:
            return
        self._camera_manager = camera_manager
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Live detector thread started")

    def stop(self) -> None:
        """Stop background detection loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def get_latest(self) -> dict:
        """Return latest detection payload for WebSocket broadcasting."""
        with self._lock:
            return {
                "ts": self._latest_detection["ts"],
                "boxes": [dict(b) for b in self._latest_detection["boxes"]],
            }

    def _loop(self) -> None:
        while self._running and self._camera_manager is not None:
            if not self._camera_manager.wait_for_frame(1.0):
                continue

            jpeg = self._camera_manager.get_frame()
            if not jpeg:
                continue

            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            with self._lock:
                self._frame_count += 1
                run_infer = (self._frame_count % self._infer_every_n) == 0

            if not run_infer:
                continue

            boxes = self._run_inference(frame)
            with self._lock:
                self._latest_detection = {"ts": time.time(), "boxes": boxes}

    def _run_inference(self, frame: np.ndarray) -> list[dict]:
        if not self._model:
            return []
        try:
            results = self._model.predict(
                source=frame,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                verbose=False,
            )
        except Exception as exc:
            logger.debug("Live detection failed: %s", exc)
            return []

        if not results:
            return []

        h, w = frame.shape[:2]
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        boxes = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls_idx = int(b.cls[0]) if b.cls is not None else -1
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                boxes.append(
                    {
                        "x1": max(0.0, min(1.0, x1 / w)),
                        "y1": max(0.0, min(1.0, y1 / h)),
                        "x2": max(0.0, min(1.0, x2 / w)),
                        "y2": max(0.0, min(1.0, y2 / h)),
                        "label": label,
                        "conf": conf,
                    }
                )
        return boxes