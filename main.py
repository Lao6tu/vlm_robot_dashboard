"""
Entry point
===========
Wires all components together and starts the uvicorn server.
"""

import logging

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from scripts import config
from scripts.camera_manager import CameraManager
from scripts.inference_scheduler import InferenceScheduler
from scripts.live_detector import LiveDetector
from scripts.result_manager import ResultManager
from scripts.snapshot_worker import SnapshotWorker
from scripts.web_app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    result_manager = ResultManager()

    camera_manager = CameraManager(
        framerate=config.CAMERA_FRAMERATE,
        output_width=config.OUTPUT_WIDTH,
        output_height=config.OUTPUT_HEIGHT,
        sensor_width=config.SENSOR_WIDTH,
        sensor_height=config.SENSOR_HEIGHT,
    )

    snapshot_worker = SnapshotWorker(
        camera_manager=camera_manager,
        interval_sec=config.SNAPSHOT_INTERVAL_SEC,
        buffer_size=config.SNAPSHOT_BUFFER_SIZE,
    )

    inference_scheduler = InferenceScheduler(
        snapshot_worker=snapshot_worker,
        result_manager=result_manager,
        interval_sec=config.INFERENCE_INTERVAL_SEC,
        frames_per_request=config.INFERENCE_FRAMES,
        base_url=config.INFERENCE_API_URL,
        api_key=config.INFERENCE_API_KEY,
        model=config.INFERENCE_MODEL,
        prompt=config.INFERENCE_PROMPT,
        timeout_sec=config.INFERENCE_TIMEOUT_SEC,
    )

    live_detector = None
    if config.LIVE_DETECTION_ENABLED:
        live_detector = LiveDetector(
            model_path=config.YOLO_MODEL_PATH,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            infer_every_n=config.YOLO_INFER_EVERY_N,
            imgsz=config.YOLO_IMGSZ,
        )

    app = create_app(
        camera_manager=camera_manager,
        result_manager=result_manager,
        snapshot_worker=snapshot_worker,
        inference_scheduler=inference_scheduler,
        live_detector=live_detector,
    )

    logger.info("Starting Robot Camera Inference server on %s:%d", config.HOST, config.PORT)
    uvicorn.run(app, host=config.HOST, port=config.PORT, reload=False)


if __name__ == "__main__":
    main()
