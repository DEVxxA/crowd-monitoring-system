import sys
import os

# Make project root visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
from utils.density_map import generate_density_map, load_gt_points

# CHANGE IMAGE INDEX IF NEEDED
img_path = "data/shanghaiA/images/IMG_1.jpg"
gt_path = "data/shanghaiA/ground_truth/GT_IMG_1.mat"

# Load image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load GT points
points = load_gt_points(gt_path)

# Generate density map
density = generate_density_map(img, points)

print("GT Count:", points.shape[0])
print("Density Sum:", int(density.sum()))

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Density Map")
plt.imshow(density, cmap="jet")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
