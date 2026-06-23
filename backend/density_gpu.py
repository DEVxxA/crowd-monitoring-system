import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from csrnet_model import CSRNet
import torchvision.transforms as transforms

# Make sure torch.cuda.is_available() is true
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CSRNet and load pretrained weights
model = CSRNet()
model = model.to(device)
model.eval()

# Replace this path with your CSRNet .pth file
checkpoint_path = r"C:\Users\Ajay\Downloads\crisisense\backend\csrnet_shanghai.pth"
model.load_state_dict(torch.load(checkpoint_path))
print("Loaded CSRNet weights")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_density_map(frame):
    # Resize for model (must be divisible by 8 because CSRNet architecture)
    h, w, _ = frame.shape
    new_h, new_w = (h // 8) * 8, (w // 8) * 8
    frame_resized = cv2.resize(frame, (new_w, new_h))

    img = transform(frame_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        density_map = model(img)
    density_map = density_map.squeeze().cpu().numpy()
    return density_map

def heatmap_overlay(frame, density_map):
    heatmap = density_map / (density_map.max() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay

def run_gpu_density(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        density_map = get_density_map(frame)
        overlay = heatmap_overlay(frame, density_map)

        total_count = int(density_map.sum())
        cv2.putText(overlay, f"Count: {total_count}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("GPU Crowd Density", overlay)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()