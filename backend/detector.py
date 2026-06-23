import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

GRID_ROWS = 4
GRID_COLS = 4

# Density threshold as % of grid area occupied
DENSITY_THRESHOLD = 0.25   # 25% occupied = high risk


def detect_people(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        cell_h = h // GRID_ROWS
        cell_w = w // GRID_COLS

        # Area-based density grid
        grid_area = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    box_area = (x2 - x1) * (y2 - y1)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    row = min(cy // cell_h, GRID_ROWS - 1)
                    col = min(cx // cell_w, GRID_COLS - 1)

                    grid_area[row][col] += box_area

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw grid and density ratio
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                x = j * cell_w
                y = i * cell_h

                cell_area = cell_h * cell_w
                density_ratio = grid_area[i][j] / cell_area

                color = (0, 255, 0)
                if density_ratio >= DENSITY_THRESHOLD:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + cell_w, y + cell_h), color, 2)

                cv2.putText(
                    frame,
                    f"{density_ratio:.2f}",
                    (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        overall_density = grid_area.sum() / (h * w)

        cv2.putText(
            frame,
            f"Overall Density: {overall_density:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow("CrisisSense - Area-Based Density", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
