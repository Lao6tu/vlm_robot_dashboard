"""
Camera Manager
==============
Drives the Raspberry Pi camera via rpicam-vid piped MJPEG output.

rpicam-vid encodes JPEG frames on the VideoCore hardware encoder.

A watchdog loop restarts rpicam-vid automatically if the process dies.
"""

import logging
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# JPEG start-of-image / end-of-image markers
_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"

# How long to wait before restarting rpicam-vid after a crash
_RESTART_DELAY_SEC = 1.0
# Chunk size for reading the pipe
_READ_CHUNK = 65536


class CameraManager:
    """Runs rpicam-vid as a subprocess and parses its MJPEG stdout pipe."""

    def __init__(
        self,
        framerate: int = 30,
        output_width: int = 640,
        output_height: int = 480,
        sensor_width: int = 0,
        sensor_height: int = 0,
    ) -> None:
        self.framerate = framerate
        self.output_width = output_width
        self.output_height = output_height
        # When both are non-zero, passed as --mode W:H to force full-sensor
        # capture so the entire FOV is used before downscaling to output size.
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Camera started via rpicam-vid: %dx%d @ %d fps",
            self.output_width, self.output_height, self.framerate,
        )

    def get_frame(self) -> Optional[bytes]:
        """Return the most recent JPEG frame as bytes, or None if not ready."""
        with self._lock:
            return self._latest_jpeg

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        """Block until a new frame is available (or timeout). Returns True if a frame arrived."""
        arrived = self._frame_event.wait(timeout)
        if arrived:
            self._frame_event.clear()
        return arrived

    def stop(self) -> None:
        """Signal the capture thread to stop; it will terminate rpicam-vid."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Camera stopped")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        cmd = [
            "rpicam-vid",
            "--width",          str(self.output_width),
            "--height",         str(self.output_height),
            "--framerate",      str(self.framerate),
            "--codec",          "mjpeg",
            "--timeout",        "0",        # run indefinitely
            "--output",         "-",        # pipe to stdout
            "--nopreview",
            "--autofocus-mode", "continuous",
        ]
        if self.sensor_width and self.sensor_height:
            # Force the full-sensor capture mode so rpicam-vid uses the entire
            # imaging area (full FOV) and then ISP-downscales to output size.
            cmd += ["--mode", f"{self.sensor_width}:{self.sensor_height}"]
        return cmd

    def _watchdog_loop(self) -> None:
        """Outer loop: start rpicam-vid and restart it if it crashes."""
        while self._running:
            cmd = self._build_cmd()
            logger.info("Launching: %s", " ".join(cmd))
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                logger.error("rpicam-vid not found — is the rpicam-apps package installed?")
                self._running = False
                return

            try:
                self._read_pipe(proc)
            finally:
                # Make sure the process is reaped whether _read_pipe returned
                # normally or via exception.
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            if self._running:
                logger.warning(
                    "rpicam-vid exited unexpectedly — restarting in %.1f s",
                    _RESTART_DELAY_SEC,
                )
                time.sleep(_RESTART_DELAY_SEC)

    def _read_pipe(self, proc: subprocess.Popen) -> None:
        """Read MJPEG bytes from the pipe and extract complete JPEG frames."""
        buf = b""
        while self._running:
            chunk = proc.stdout.read(_READ_CHUNK)
            if not chunk:
                # EOF — process exited
                break
            buf += chunk

            # Extract every complete JPEG frame sitting in the buffer.
            while True:
                soi = buf.find(_SOI)
                if soi == -1:
                    buf = b""
                    break
                eoi = buf.find(_EOI, soi + 2)
                if eoi == -1:
                    # Incomplete frame — keep buffered bytes from SOI onwards
                    buf = buf[soi:]
                    break
                frame = buf[soi: eoi + 2]
                buf = buf[eoi + 2:]
                with self._lock:
                    self._latest_jpeg = frame
                self._frame_event.set()
