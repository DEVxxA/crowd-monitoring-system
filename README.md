Website : https://crowd-monitoring-system.onrender.com/
# CrisisSense

CrisisSense is a Python-based crowd monitoring system for estimating crowd density, visualizing risk, and sending email alerts when a dangerous crowd condition is detected. It combines a CSRNet crowd-counting model, ONNX Runtime inference, OpenCV video processing, and a FastAPI dashboard.

## Features

- Real-time crowd count and risk score dashboard
- Video upload analysis with heatmap overlay
- Live webcam/CCTV feed mode
- Crowd-density inference using `csrnet_shanghaiA.onnx`
- Motion-based risk signal using optical flow
- Event log for video, CCTV, detection, risk, and alert activity
- OTP-verified authority contact setup for email alerts
- Training, ONNX export, and evaluation scripts for CSRNet

## Project Structure

```text
.
backend/
  api.py                       # FastAPI app and dashboard routes
  templates/index.html         # Web dashboard
  static/style.css             # Dashboard styling
data/                          # ShanghaiTech dataset folders
datasets/                      # Dataset loader
inference/
  video_infer.py               # Video/webcam inference pipeline
  event_log.py                 # In-memory event log
models/
  csrnet.py                    # CSRNet model definition
utils/
  density_map.py               # Density map helpers
videos/                        # Sample or local test videos
alert_system.py                # OTP and email alert handling
train.py                       # CSRNet training script
export_onnx.py                 # PyTorch to ONNX export
evaluate.py                    # Model evaluation script
requirements.txt               # Minimal runtime dependencies
requirement.txt                # Pinned dependency snapshot
```

## Requirements

- Python 3.10 or newer
- A webcam for live CCTV mode, or a video file for upload mode
- The model files in the project root:
  - `csrnet_shanghaiA.onnx`
  - `csrnet_shanghaiA.onnx.data`
  - `csrnet_shanghaiA.pth` if training/exporting is needed

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

For training or evaluation, install the extra ML dependencies:

```bash
pip install torch torchvision scikit-image
```

## Environment Variables

Create a `.env` file in the project root for email alerts:

```env
GMAIL_ADDRESS=your_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
CAMERA_LOCATION=Main Entrance Camera
```

Use a Gmail App Password instead of your normal Gmail password. Keep `.env` private and do not commit real credentials.

## Run the Dashboard

Start the FastAPI server from the project root:

```bash
uvicorn backend.api:app --reload
```

Open the dashboard:

```text
http://127.0.0.1:8000
```

From the dashboard, you can:

- Upload a crowd video and analyze it
- Start a live webcam feed
- Enable detection on the live feed
- Reset the current stream
- View live crowd metrics, risk score, risk chart, and event logs
- Configure an authority email contact for danger alerts

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Dashboard page |
| `GET` | `/metrics` | Latest crowd count, risk score, and risk label |
| `GET` | `/events` | Recent event log entries |
| `POST` | `/upload` | Upload a video for analysis |
| `POST` | `/start_webcam` | Start raw webcam feed |
| `POST` | `/start_detection` | Start webcam detection mode |
| `POST` | `/stop_webcam` | Stop CCTV mode |
| `GET` | `/video_feed` | MJPEG stream for the dashboard |
| `POST` | `/reset` | Clear current stream and metrics |
| `GET` | `/api/alert/contact` | Get configured alert contact |
| `POST` | `/api/alert/send-otp` | Send OTP email |
| `POST` | `/api/alert/verify-otp` | Verify OTP and save contact |
| `DELETE` | `/api/alert/contact` | Remove alert contact |
| `POST` | `/api/alert/test` | Send a test alert |

## Training

The training script uses ShanghaiTech Part A by default:

```bash
python train.py
```

Expected dataset layout:

```text
data/
  shanghaiA/
    images/
    ground_truth/
```

The trained model is saved as:

```text
csrnet_shanghaiA.pth
```

## Export to ONNX

After training, export the PyTorch model to ONNX:

```bash
python export_onnx.py
```

The inference pipeline uses `csrnet_shanghaiA.onnx` through ONNX Runtime.

## Evaluation

Evaluate the default ShanghaiTech Part A split:

```bash
python evaluate.py
```

Evaluate another split:

```bash
python evaluate.py shanghaiB
```

The evaluation reports crowd-count accuracy, heatmap similarity, and danger/risk classification metrics.

## Notes

- `backend/api.py` is the main web application entry point.
- `inference/video_infer.py` loads the ONNX model at import time, so the ONNX files must exist before starting the server.
- Webcam access can fail if another app is already using the camera.
- Large model, dataset, video, and credential files should generally stay out of version control.
