import os
import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.density_map import load_gt_points, generate_density_map


class ShanghaiTechDataset(Dataset):
    def __init__(self, img_dir, gt_dir, crop_size=512):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.crop_size = crop_size
        self.img_files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, "GT_" + img_name.replace(".jpg", ".mat"))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        points = load_gt_points(gt_path)

        # -------- Ensure fixed 512×512 --------
        if h < self.crop_size or w < self.crop_size:
            scale_x = self.crop_size / w
            scale_y = self.crop_size / h
            img = cv2.resize(img, (self.crop_size, self.crop_size))
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
        else:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            img = img[y:y+self.crop_size, x:x+self.crop_size]

            mask = (
                (points[:, 0] >= x) & (points[:, 0] <= x + self.crop_size) &
                (points[:, 1] >= y) & (points[:, 1] <= y + self.crop_size)
            )
            points = points[mask]
            points[:, 0] -= x
            points[:, 1] -= y

        # -------- Density map --------
        density = generate_density_map(img, points)
        density = torch.from_numpy(density).unsqueeze(0)  # [1,512,512]

        # 🔥 Downsample GT to match CSRNet output (÷32)
        density = F.interpolate(
            density.unsqueeze(0),
            scale_factor=1/32,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)  # [1,16,16]

        # -------- Image tensor --------
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, density
