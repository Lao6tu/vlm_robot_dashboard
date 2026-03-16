# Robot Camera Inference Dashboard

RPi5 robot camera system with a real-time VLM inference pipeline.

## Architecture

```
Browser ──► RPi5 Web UI (FastAPI)
               │
               ├─ CameraManager   ← rpicam-vid subprocess (MJPEG pipe, ~28 FPS)
               ├─ SnapshotWorker  ← 1 frame/s from latest_frame, resized to 960x720 → deque
               ├─ InferenceScheduler  ← every 3s, 2 frames → POST to VLM API
               ├─ LiveDetector    ← YOLO detects boxes in background thread
               ├─ ResultManager   ← stores result, fans out via WebSocket
               ├─ stream.mjpeg    ← raw live MJPEG from latest_frame (no box drawing)
               └─ ws/detections   ← latest normalized boxes for canvas overlay

                      ▼
             Inference API Server
               └─ POST /v1/chat/completions  (OpenAI-compatible vision API)
               └─ returns JSON
```

## File layout

| File | Role |
|---|---|
| `config.py` | All tuneable constants / env vars |
| `camera_manager.py` | Runs rpicam-vid as a subprocess, parses MJPEG pipe, keeps `latest_jpeg` fresh |
| `snapshot_worker.py` | Grabs 1 frame/s, resizes to 960x720, stores in rolling `deque` |
| `inference_scheduler.py` | Picks 2 frames every 3 s, POSTs to VLM API |
| `live_detector.py` | Runs YOLO in background and publishes normalized detection boxes |
| `result_manager.py` | Thread-safe result store + asyncio WebSocket broadcast |
| `web_app.py` | FastAPI routes: `/`, `/stream.mjpeg`, `/api/status`, `/api/config`, `/ws/results`, `/ws/detections` |
| `main.py` | Wires everything, starts uvicorn |

## Setup

### 1 — System dependencies (Raspberry Pi OS Bookworm)

```bash
sudo apt update
sudo apt install rpicam-apps
```

### 2 — Python environment

```bash
cd /home/drone/robot_inference

# Create a venv that can see the system picamera2
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

pip install -r requirements.txt
```

### 3 — Configure

```bash
nano .env          # set INFERENCE_API_URL, SENSOR_WIDTH/HEIGHT if needed
```

### 4 — Run

```bash
source .venv/bin/activate
python main.py
```

Open `http://<rpi-ip>:8000` in a browser.

### Live detection overlay behavior

- MJPEG stream stays raw (camera frames are not modified by YOLO).
- YOLO runs in a background thread and only emits box coordinates.
- Browser draws boxes in a canvas overlay via `/ws/detections`.

## Install as systemd service

```bash
sudo cp robot-inference.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable robot-inference
sudo systemctl start robot-inference
sudo journalctl -u robot-inference -f   # follow logs
```

## Inference API contract

Uses the **OpenAI-compatible Chat Completions API** with vision:

**Request** (POST JSON to `{INFERENCE_API_URL}/v1/chat/completions`):
```json
{
  "model": "<INFERENCE_MODEL>",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "<INFERENCE_PROMPT>"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}}
    ]
  }]
}
```

**Response** — the assistant reply must be a JSON object.  
Internal keys prefixed with `_` are stripped before display.  
On error, include an `"error"` key for the UI to show a warning.

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_API_URL` | *(required)* | Remote VLM endpoint base URL |
| `INFERENCE_API_KEY` | `none` | Bearer token (use any placeholder if no auth) |
| `INFERENCE_MODEL` | `llava` | Model name as recognised by the server |
| `INFERENCE_PROMPT` | *(built-in)* | System/user prompt sent with every request |
| `INFERENCE_INTERVAL_SEC` | `3` | Seconds between inference calls |
| `INFERENCE_FRAMES` | `2` | Frames per inference request |
| `INFERENCE_TIMEOUT_SEC` | `30` | HTTP timeout |
| `SNAPSHOT_INTERVAL_SEC` | `1.0` | Snapshot cadence (FPS shown in UI badge) |
| `SNAPSHOT_BUFFER_SIZE` | `10` | Rolling buffer depth |
| `SENSOR_WIDTH` / `SENSOR_HEIGHT` | `0` / `0` | Full sensor resolution passed as `--mode` to rpicam-vid for full-FOV capture; `0` = auto |
| `OUTPUT_WIDTH` / `OUTPUT_HEIGHT` | `640` / `480` | Output resolution (rpicam-vid downscales from sensor mode) |
| `CAMERA_FRAMERATE` | `30` | Camera framerate |
| `MJPEG_FPS` | `30` | MJPEG stream rate |
| `LIVE_DETECTION_ENABLED` | `1` | Enable YOLO live detection worker |
| `YOLO_MODEL_PATH` | `yolo26n.pt` | YOLO model path |
| `YOLO_CONF` | `0.35` | Detection confidence threshold |
| `YOLO_IOU` | `0.45` | NMS IoU threshold |
| `YOLO_INFER_EVERY_N` | `2` | Run YOLO once per N frames (lower load with higher N) |
| `YOLO_IMGSZ` | `640` | YOLO inference image size |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind |

Note: snapshots are currently resized to `960x720` in code before being sent for inference.
