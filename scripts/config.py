import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _decode_env_text(value: str) -> str:
    return value.encode().decode("unicode_escape")


def _load_inference_prompt() -> str:
    default_prompt = (
        "You are a robot vision system. Describe what you observe in the image(s) concisely."
    )
    prompt_file = os.getenv("INFERENCE_PROMPT_FILE", "").strip()
    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = Path(__file__).resolve().parent.parent / prompt_path
        return prompt_path.read_text(encoding="utf-8").strip()

    return _decode_env_text(os.getenv("INFERENCE_PROMPT", default_prompt))

# ── Server ──────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── Camera ───────────────────────────────────────────────────────────────────
# Full sensor resolution — passed to rpicam-vid as --mode W:H so the entire
# sensor area (full FOV) is captured before downscaling to OUTPUT size.
# Set to 0 to let rpicam-vid pick the mode automatically.
SENSOR_WIDTH  = int(os.getenv("SENSOR_WIDTH",  "0"))
SENSOR_HEIGHT = int(os.getenv("SENSOR_HEIGHT", "0"))

CAMERA_FRAMERATE = int(os.getenv("CAMERA_FRAMERATE", "30"))
MJPEG_FPS = int(os.getenv("MJPEG_FPS", "30"))

# Output (downscaled) resolution delivered to the stream and snapshot buffer
OUTPUT_WIDTH  = int(os.getenv("OUTPUT_WIDTH",  "640"))
OUTPUT_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "480"))

# ── Snapshot worker ───────────────────────────────────────────────────────────
SNAPSHOT_INTERVAL_SEC = float(os.getenv("SNAPSHOT_INTERVAL_SEC", "1.0"))
SNAPSHOT_BUFFER_SIZE = int(os.getenv("SNAPSHOT_BUFFER_SIZE", "10"))

# ── Inference scheduler ───────────────────────────────────────────────────────
# Base URL for an OpenAI-compatible server, e.g. http://192.168.1.100:8080
# The scheduler will call  {INFERENCE_API_URL}/v1/chat/completions
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://192.168.1.100:8080")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "none")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "llava")
INFERENCE_PROMPT = _load_inference_prompt()
INFERENCE_INTERVAL_SEC = float(os.getenv("INFERENCE_INTERVAL_SEC", "3"))
INFERENCE_FRAMES = int(os.getenv("INFERENCE_FRAMES", "2"))   # frames per request
INFERENCE_TIMEOUT_SEC = int(os.getenv("INFERENCE_TIMEOUT_SEC", "30"))

# ── Live YOLO detection on MJPEG feed ───────────────────────────────────────
LIVE_DETECTION_ENABLED = os.getenv("LIVE_DETECTION_ENABLED", "1") == "1"
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo26n.pt")
YOLO_PREFER_NCNN = os.getenv("YOLO_PREFER_NCNN", "1") == "1"
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_INFER_EVERY_N = int(os.getenv("YOLO_INFER_EVERY_N", "2"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
