import os
import sys
import tempfile
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load environment variables from .env file when python-dotenv is available.
if load_dotenv is not None:
    load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from alert_system import (
        delete_contact,
        load_contact,
        send_email_alert,
        send_otp_email,
        send_whatsapp_alert,
        verify_otp,
    )
    ALERTS_IMPORT_ERROR = None
except ImportError as exc:
    ALERTS_IMPORT_ERROR = str(exc)

    def _alert_dependency_error():
        return {
            "success": False,
            "error": (
                "Alert features are unavailable because an optional dependency "
                f"could not be imported: {ALERTS_IMPORT_ERROR}"
            ),
        }

    def send_otp_email(*args, **kwargs):
        return _alert_dependency_error()

    def send_email_alert(*args, **kwargs):
        return _alert_dependency_error()

    def verify_otp(*args, **kwargs):
        return _alert_dependency_error()

    def load_contact():
        return None

    def delete_contact():
        return None

    def send_whatsapp_alert(*args, **kwargs):
        return _alert_dependency_error()


# Import inference modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from inference.event_log import EVENT_LOGS, add_event
from inference.video_infer import process_video, process_webcam, process_webcam_raw
from shared_state import latest_metrics


app = FastAPI(title="Crisisense Backend")

# Setup templates and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# App state
app.state.video_path = None
app.state.mode = "video"  # video | webcam_raw | webcam_detect
app.state.stream_error = None

# JPEG encode params
ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 70]

# Target stream frame rate
STREAM_FPS = 25
FRAME_DELAY = 1.0 / STREAM_FPS


def error_frame_bytes(message: str) -> bytes:
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    frame[:] = (20, 20, 28)

    lines = [
        "Live camera unavailable",
        message,
        "Close other apps using the webcam and try again.", ]
    y = 120
    for i, line in enumerate(lines):
        scale = 0.9 if i == 0 else 0.6
        color = (80, 180, 255) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += 60

    _, buf = cv2.imencode(".jpg", frame, ENCODE_PARAMS)
    return buf.tobytes()


class OTPRequest(BaseModel):
    email: str
    phone: str = ""


class OTPVerify(BaseModel):
    email: str
    otp: str


class DeleteContact(BaseModel):
    confirm: bool = True


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
    

@app.get("/metrics")
async def metrics():
    return latest_metrics



@app.get("/events")
async def events():
    return list(EVENT_LOGS)


@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(await video.read())
    temp.close()

    app.state.video_path = temp.name
    app.state.mode = "video"

    add_event("VIDEO", "New video uploaded", "normal")
    return RedirectResponse("/", status_code=303)


@app.post("/start_webcam")
async def start_webcam():
    app.state.mode = "webcam_raw"
    app.state.stream_error = None
    add_event("CCTV", "Live CCTV started (raw feed)", "warning")
    return RedirectResponse("/", status_code=303)


@app.post("/start_detection")
async def start_detection():
    app.state.mode = "webcam_detect"
    app.state.stream_error = None
    add_event("DETECTION", "Live detection enabled", "normal")
    return RedirectResponse("/", status_code=303)


@app.post("/stop_webcam")
async def stop_webcam():
    app.state.mode = "video"
    app.state.stream_error = None
    add_event("CCTV", "CCTV stopped", "normal")
    return RedirectResponse("/", status_code=303)


@app.get("/video_feed")
def video_feed():
    def generate():
        last_time = time.time()

        while True:
            if app.state.mode == "webcam_raw":
                try:
                    for frame in process_webcam_raw():
                        app.state.stream_error = None
                        now = time.time()
                        elapsed = now - last_time
                        if elapsed < FRAME_DELAY:
                            time.sleep(FRAME_DELAY - elapsed)
                        last_time = time.time()

                        frame = cv2.resize(frame, (854, 480))
                        _, buf = cv2.imencode(".jpg", frame, ENCODE_PARAMS)
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buf.tobytes()
                            + b"\r\n"
                        )
                except Exception as exc:
                    app.state.stream_error = str(exc)
                    add_event("CCTV", f"Webcam failed: {exc}", "danger")
                    while app.state.mode == "webcam_raw":
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + error_frame_bytes(str(exc))
                            + b"\r\n"
                        )
                        time.sleep(1)

            elif app.state.mode == "webcam_detect":
                try:
                    for frame in process_webcam():
                        app.state.stream_error = None
                        now = time.time()
                        elapsed = now - last_time
                        if elapsed < FRAME_DELAY:
                            time.sleep(FRAME_DELAY - elapsed)
                        last_time = time.time()

                        frame = cv2.resize(frame, (854, 480))
                        _, buf = cv2.imencode(".jpg", frame, ENCODE_PARAMS)
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buf.tobytes()
                            + b"\r\n"
                        )
                except Exception as exc:
                    app.state.stream_error = str(exc)
                    add_event("DETECTION", f"Live detection failed: {exc}", "danger")
                    while app.state.mode == "webcam_detect":
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + error_frame_bytes(str(exc))
                            + b"\r\n"
                        )
                        time.sleep(1)

            else:
                if app.state.video_path is None:
                    time.sleep(0.2)
                    continue

                for frame in process_video(app.state.video_path):
                    now = time.time()
                    elapsed = now - last_time
                    if elapsed < FRAME_DELAY:
                        time.sleep(FRAME_DELAY - elapsed)
                    last_time = time.time()

                    frame = cv2.resize(frame, (854, 480))
                    _, buf = cv2.imencode(".jpg", frame, ENCODE_PARAMS)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )

                app.state.video_path = None

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/reset")
async def reset():
    app.state.video_path = None
    app.state.mode = "video"
    app.state.stream_error = None
    latest_metrics.update({
        "count": 0,
        "risk": 0,
        "risk_label": "NORMAL",
    })
    add_event("VIDEO", "Video feed reset", "normal")
    return {"success": True}


@app.get("/api/alert/contact")
async def get_alert_contact():
    contact = load_contact()
    if not contact:
        return {"configured": False}

    return {
        "configured": True,
        "email": contact.get("email", ""),
        "phone": contact.get("phone", ""),
        "verified_at": contact.get("verified_at"),
    }


@app.post("/api/alert/send-otp")
async def send_alert_otp(request: OTPRequest):
    email = request.email.strip()
    if not email or "@" not in email:
        return {"success": False, "error": "Please enter a valid email address."}

    result = send_otp_email(email, request.phone.strip())
    if result.get("success"):
        add_event("ALERT", f"Verification OTP sent to {email}", "normal")
    else:
        add_event("ALERT", f"OTP email failed: {result.get('error', 'Unknown error')}", "danger")
    return result


@app.post("/api/alert/verify-otp")
async def verify_alert_otp(request: OTPVerify):
    result = verify_otp(request.email.strip(), request.otp.strip())
    if result.get("success"):
        add_event("ALERT", f"Email alerts enabled for {request.email.strip()}", "normal")
    else:
        add_event("ALERT", f"OTP verification failed: {result.get('error', 'Unknown error')}", "warning")
    return result


@app.delete("/api/alert/contact")
async def remove_alert_contact():
    delete_contact()
    add_event("ALERT", "Alert contact removed", "warning")
    return {"success": True}


@app.post("/api/alert/test")
async def test_alert():
    result = send_email_alert(
        crowd_count=max(int(latest_metrics.get("count", 0)), 1),
        risk_label="DANGER",
        risk_score=max(int(latest_metrics.get("risk", 0)), 75),
        location=os.getenv("CAMERA_LOCATION", "Main Entrance Camera"),
    )
    if result.get("success"):
        add_event("ALERT", "Test email alert sent", "normal")
    else:
        add_event("ALERT", f"Test email alert failed: {result.get('error', 'Unknown error')}", "danger")
    return result
