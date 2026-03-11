"""
Web Application
===============
FastAPI app exposing:
  GET  /                    — Dashboard HTML
  GET  /stream.mjpeg        — Live MJPEG camera stream
  GET  /api/status          — JSON system status
  WS   /ws/results          — Push inference results to browser
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config
from .camera_manager import CameraManager
from .inference_scheduler import InferenceScheduler
from .result_manager import ResultManager
from .snapshot_worker import SnapshotWorker

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def create_app(
    camera_manager: CameraManager,
    result_manager: ResultManager,
    snapshot_worker: Optional[SnapshotWorker] = None,
    inference_scheduler: Optional[InferenceScheduler] = None,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── Startup ──────────────────────────────────────────────────────────
        loop = asyncio.get_running_loop()
        result_manager.set_event_loop(loop)

        camera_manager.start()
        if snapshot_worker:
            snapshot_worker.start()
        if inference_scheduler:
            inference_scheduler.start()

        yield

        # ── Shutdown ─────────────────────────────────────────────────────────
        if inference_scheduler:
            inference_scheduler.stop()
        if snapshot_worker:
            snapshot_worker.stop()
        camera_manager.stop()

    app = FastAPI(title="Robot Camera Inference Dashboard", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    # ── HTML pages ────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    # ── MJPEG stream ──────────────────────────────────────────────────────────

    @app.get("/stream.mjpeg")
    async def mjpeg_stream() -> StreamingResponse:
        loop = asyncio.get_event_loop()

        async def generate():
            try:
                while True:
                    # Block in a thread-pool until a fresh frame arrives (event-driven,
                    # no busy-poll) — eliminates up to 33 ms of sleep-based latency.
                    await loop.run_in_executor(
                        None, lambda: camera_manager.wait_for_frame(1.0)
                    )
                    frame = camera_manager.get_frame()
                    if frame:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                            + frame
                            + b"\r\n"
                        )
            except (GeneratorExit, asyncio.CancelledError):
                logger.info("MJPEG client disconnected")

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ── REST status ───────────────────────────────────────────────────────────

    @app.get("/api/config")
    def api_config() -> JSONResponse:
        snapshot_fps = round(1.0 / config.SNAPSHOT_INTERVAL_SEC, 2)
        return JSONResponse({"snapshot_fps": snapshot_fps})

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        return JSONResponse(
            {
                "camera_ready": camera_manager.get_frame() is not None,
                "latest_result": result_manager.get_latest(),
                "timestamp": time.time(),
            }
        )

    # ── WebSocket results push ────────────────────────────────────────────────

    @app.websocket("/ws/results")
    async def ws_results(websocket: WebSocket) -> None:
        await websocket.accept()
        q = result_manager.subscribe()

        # Immediately send whatever we already have
        latest = result_manager.get_latest()
        if latest:
            await websocket.send_text(json.dumps(latest))

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25)
                    await websocket.send_text(msg)
                except asyncio.TimeoutError:
                    # Keep-alive ping so the browser doesn't drop the connection
                    await websocket.send_text(json.dumps({"_ping": True}))
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as exc:
            logger.error("WebSocket error: %s", exc)
        finally:
            result_manager.unsubscribe(q)

    return app
