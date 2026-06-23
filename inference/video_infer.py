import os
import queue
import sys
import threading

import cv2
import numpy as np
import onnxruntime as ort

from alert_system import mark_alert_sent, send_whatsapp_alert, should_send_alert
from inference.event_log import add_event
from shared_state import latest_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# ================= CONFIG =================
ONNX_PATH = os.path.join(PROJECT_ROOT, "csrnet_shanghaiA.onnx")

FRAME_SKIP = 3
FLOW_INTERVAL = 6
HEATMAP_INTERVAL = 4
FLOW_RESIZE = (320, 240)
HEATMAP_ALPHA = 0.6
SMOOTHING_ALPHA = 0.75

WEBCAM_FLOW_INTERVAL = 12
WEBCAM_SMOOTHING = 0.90
WEBCAM_CHAOS_THRESHOLD = 1.6
WEBCAM_MIN_PEOPLE = 3

HISTORY_LENGTH = 20
CHAOS_SCORE_THRESHOLD = 0.9
HIGH_DENSITY_COUNT = 500

# Inference resolution must be divisible by 8.
INFER_W = 640
INFER_H = 480

# ImageNet normalization as numpy (no torch needed at runtime)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# ================= LOAD ONNX MODEL =================
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = os.cpu_count()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    ONNX_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

print(f"ONNX model loaded - running on CPU with {os.cpu_count()} threads")


def _open_camera(preferred_index=0):
    candidates = []

    if os.name == "nt":
        candidates.extend(
            [
                (preferred_index, cv2.CAP_DSHOW),
                (preferred_index, cv2.CAP_MSMF),
                (1, cv2.CAP_DSHOW),
                (1, cv2.CAP_MSMF),
            ]
        )

    candidates.extend(
        [
            (preferred_index, cv2.CAP_ANY),
            (1, cv2.CAP_ANY),
        ]
    )

    seen = set()
    for index, backend in candidates:
        key = (index, backend)
        if key in seen:
            continue
        seen.add(key)

        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            return cap
        cap.release()

    raise RuntimeError("Cannot access webcam. Check that the camera is connected and not in use by another app.")


def preprocess_frame(frame):
    new_w = (INFER_W // 8) * 8
    new_h = (INFER_H // 8) * 8
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (new_w, new_h))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = (img - MEAN) / STD
    img = np.expand_dims(img, axis=0)
    return img


def run_inference(frame):
    inp = preprocess_frame(frame)
    out = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
    dmap = out.squeeze()
    return dmap


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    raw_q = queue.Queue(maxsize=8)
    out_q = queue.Queue(maxsize=8)

    prev_label = None
    smoothed_count = None
    prev_gray = None
    chaos_history = []
    cached_heatmap = None

    def reader():
        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                raw_q.put(None)
                break
            fid += 1
            raw_q.put((fid, frame))

    def inferencer():
        nonlocal smoothed_count, prev_gray, chaos_history
        nonlocal cached_heatmap, prev_label

        while True:
            item = raw_q.get()
            if item is None:
                out_q.put(None)
                break

            frame_id, frame = item
            h, w, _ = frame.shape

            if frame_id % FRAME_SKIP == 0:
                dmap = run_inference(frame)

                scale = (h * w) / (dmap.shape[0] * dmap.shape[1])
                raw = float(dmap.sum()) * scale

                smoothed_count = raw if smoothed_count is None else (
                    SMOOTHING_ALPHA * smoothed_count + (1 - SMOOTHING_ALPHA) * raw
                )
                crowd_count = int(smoothed_count)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame_id % FLOW_INTERVAL == 0 and prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        cv2.resize(prev_gray, FLOW_RESIZE),
                        cv2.resize(gray, FLOW_RESIZE),
                        None,
                        0.5,
                        2,
                        15,
                        2,
                        5,
                        1.1,
                        0,
                    )
                    chaos_history.append(float(np.var(flow)))
                    if len(chaos_history) > HISTORY_LENGTH:
                        chaos_history.pop(0)
                prev_gray = gray

                risk = 60 if (len(chaos_history) > 0 and np.mean(chaos_history) > CHAOS_SCORE_THRESHOLD) else 0
                if crowd_count > HIGH_DENSITY_COUNT:
                    risk += 40
                risk = min(100, risk)
                label = "NORMAL" if risk < 30 else "WARNING" if risk < 60 else "DANGER"

                if prev_label != label:
                    add_event("RISK", f"Risk changed to {label}", "danger" if label == "DANGER" else "warning")
                    prev_label = label

                latest_metrics.update({
                    "count": crowd_count,
                    "risk": risk,
                    "risk_label": label,
                })

                if label == "DANGER" and should_send_alert():
                    send_whatsapp_alert(
                        crowd_count=crowd_count,
                        risk_label=label,
                        risk_score=risk,
                        location="Video Feed",
                    )
                    mark_alert_sent()

                if cached_heatmap is None or frame_id % HEATMAP_INTERVAL == 0:
                    norm = dmap / (dmap.max() + 1e-8)
                    heatmap_resized = cv2.resize(norm, (w, h))
                    cached_heatmap = cv2.applyColorMap(
                        (heatmap_resized * 255).astype(np.uint8),
                        cv2.COLORMAP_JET,
                    )

            if cached_heatmap is not None:
                out_frame = cv2.addWeighted(frame, 1 - HEATMAP_ALPHA, cached_heatmap, HEATMAP_ALPHA, 0)
            else:
                out_frame = frame

            out_q.put(out_frame)

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=inferencer, daemon=True).start()

    while True:
        frame = out_q.get()
        if frame is None:
            break
        yield frame

    cap.release()


def process_webcam_raw(camera_index=0):
    cap = _open_camera(camera_index)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

    cap.release()


def process_webcam(camera_index=0):
    cap = _open_camera(camera_index)

    raw_q = queue.Queue(maxsize=8)
    out_q = queue.Queue(maxsize=8)

    smoothed_count = None
    prev_gray = None
    chaos_history = []
    cached_heatmap = None
    prev_label = None

    def reader():
        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                raw_q.put(None)
                break
            fid += 1
            raw_q.put((fid, frame))

    def inferencer():
        nonlocal smoothed_count, prev_gray, chaos_history
        nonlocal cached_heatmap, prev_label

        while True:
            item = raw_q.get()
            if item is None:
                out_q.put(None)
                break

            frame_id, frame = item
            h, w, _ = frame.shape

            dmap = run_inference(frame)
            scale = (h * w) / (dmap.shape[0] * dmap.shape[1])
            raw = float(dmap.sum()) * scale

            smoothed_count = raw if smoothed_count is None else (
                WEBCAM_SMOOTHING * smoothed_count + (1 - WEBCAM_SMOOTHING) * raw
            )
            crowd_count = int(smoothed_count)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_id % WEBCAM_FLOW_INTERVAL == 0 and prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.resize(prev_gray, FLOW_RESIZE),
                    cv2.resize(gray, FLOW_RESIZE),
                    None,
                    0.5,
                    2,
                    15,
                    2,
                    5,
                    1.1,
                    0,
                )
                chaos_history.append(float(np.var(flow)))
                if len(chaos_history) > HISTORY_LENGTH:
                    chaos_history.pop(0)
            prev_gray = gray

            risk = 60 if (
                crowd_count >= WEBCAM_MIN_PEOPLE
                and len(chaos_history) > 0
                and np.mean(chaos_history) > WEBCAM_CHAOS_THRESHOLD
            ) else 0
            label = "DANGER" if risk >= 60 else "NORMAL"

            if prev_label != label:
                add_event("CCTV", f"Live risk changed to {label}", "danger")
                prev_label = label

            latest_metrics.update({
                "count": crowd_count,
                "risk": risk,
                "risk_label": label,
            })

            if label == "DANGER" and should_send_alert():
                send_whatsapp_alert(
                    crowd_count=crowd_count,
                    risk_label=label,
                    risk_score=risk,
                    location="Live CCTV Feed",
                )
                mark_alert_sent()

            if cached_heatmap is None or frame_id % HEATMAP_INTERVAL == 0:
                norm = dmap / (dmap.max() + 1e-8)
                heatmap_resized = cv2.resize(norm, (w, h))
                cached_heatmap = cv2.applyColorMap(
                    (heatmap_resized * 255).astype(np.uint8),
                    cv2.COLORMAP_JET,
                )

            if cached_heatmap is not None:
                out_frame = cv2.addWeighted(frame, 1 - HEATMAP_ALPHA, cached_heatmap, HEATMAP_ALPHA, 0)
            else:
                out_frame = frame

            out_q.put(out_frame)

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=inferencer, daemon=True).start()

    while True:
        frame = out_q.get()
        if frame is None:
            break
        yield frame

    cap.release()
