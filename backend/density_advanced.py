import cv2
import numpy as np
import os
from uuid import uuid4
from lwcc import LWCC

def frame_to_density_overlay(frame, count_estimate):
    """
    Create a simple overlay showing crowd density estimate.
    """
    h, w, _ = frame.shape

    # Normalize estimate to a simple colored bar
    max_count = 200  # scale for normalization
    intensity = min(count_estimate / max_count, 1.0)

    # Red bar for high crowd
    overlay = frame.copy()
    color = (0, 0, 255) if intensity > 0.5 else (0, 255, 0)
    cv2.rectangle(overlay, (0, 0), (int(intensity * w), 30), color, -1)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def run_video_advanced_density(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open video")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame to temporary file
        temp_filename = f"temp_{uuid4().hex}.jpg"
        cv2.imwrite(temp_filename, frame)

        # Get count from LWCC using file path
        count_estimate = LWCC.get_count(
            temp_filename,
            model_name="CSRNet",
            model_weights="SHB"
        )

        # Remove temp file
        try:
            os.remove(temp_filename)
        except Exception:
            pass

        # Create an overlay
        result_frame = frame_to_density_overlay(frame, count_estimate)

        # Put text of count
        cv2.putText(
            result_frame,
            f"Est Crowd: {int(count_estimate)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Advanced Density (CNN + LWCC)", result_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
