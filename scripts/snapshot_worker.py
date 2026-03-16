"""
Snapshot Worker
===============
Copies one frame per second from CameraManager into a rolling deque.
The InferenceScheduler reads from this buffer.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    timestamp: float
    jpeg: bytes = field(repr=False)


class SnapshotWorker:
    """Grabs a frame from CameraManager every `interval_sec` seconds."""

    def __init__(
        self,
        camera_manager,
        interval_sec: float = 1.0,
        buffer_size: int = 10,
        resize_width: int = 960,
        resize_height: int = 720,
    ) -> None:
        self._cam = camera_manager
        self._interval = interval_sec
        self._resize_width = resize_width
        self._resize_height = resize_height
        self._buffer: Deque[Snapshot] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Snapshot worker started (interval=%.1f s)", self._interval)

    def get_recent(self, n: int) -> List[Snapshot]:
        """Return the most recent n snapshots (oldest-first)."""
        with self._lock:
            snaps = list(self._buffer)
        return snaps[-n:] if len(snaps) >= n else snaps

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Snapshot worker stopped")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while self._running:
            t0 = time.monotonic()
            frame = self._cam.get_frame()
            if frame:
                resized = self._resize_jpeg(frame)
                snap = Snapshot(timestamp=time.time(), jpeg=resized)
                with self._lock:
                    self._buffer.append(snap)
                logger.debug("Snapshot @ %.3f", snap.timestamp)
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, self._interval - elapsed)
            time.sleep(sleep_time)

    def _resize_jpeg(self, jpeg: bytes) -> bytes:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jpeg

        resized = cv2.resize(
            frame,
            (self._resize_width, self._resize_height),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return jpeg
        return encoded.tobytes()
