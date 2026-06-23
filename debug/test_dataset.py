import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.shanghai_dataset import ShanghaiTechDataset

dataset = ShanghaiTechDataset(
    img_dir="data/shanghaiA/images",
    gt_dir="data/shanghaiA/ground_truth"
)

img, density = dataset[0]

print("Image shape:", img.shape)
print("Density shape:", density.shape)
print("Density sum:", int(density.sum()))
