import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.shanghai_dataset import ShanghaiTechDataset
from models.csrnet import CSRNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
BATCH_SIZE = 4
LR = 1e-5
SAVE_PATH = "csrnet_shanghaiA.pth"

dataset = ShanghaiTechDataset(
    img_dir="data/shanghaiA/images",
    gt_dir="data/shanghaiA/ground_truth"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,   # 🔥 Windows fix
    pin_memory=True
)

model = CSRNet().to(DEVICE)
criterion = nn.MSELoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Training started on", DEVICE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, densities in loader:
        imgs = imgs.to(DEVICE)
        densities = densities.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, densities)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss:.2f}")

torch.save(model.state_dict(), SAVE_PATH)
print("Model saved:", SAVE_PATH)
