import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter


def load_gt_points(mat_path):
    mat = sio.loadmat(mat_path)
    image_info = mat["image_info"][0][0]
    points = image_info["location"][0][0]
    return np.array(points, dtype=np.float32)


def generate_density_map(img, points, sigma=4):
    h, w, _ = img.shape
    density = np.zeros((h, w), dtype=np.float32)

    if points is None or len(points) == 0:
        return density

    for i in range(points.shape[0]):
        x = min(w - 1, max(0, int(points[i, 0])))
        y = min(h - 1, max(0, int(points[i, 1])))
        density[y, x] += 1

    density = gaussian_filter(density, sigma=sigma)
    return density
