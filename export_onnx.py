"""
Run this ONCE from your project root to convert CSRNet to ONNX:
    python export_onnx.py

This produces csrnet_shanghaiA.onnx in your project root.
After that, video_infer.py will use ONNX Runtime instead of PyTorch —
3-5x faster on CPU, no GPU needed.
"""

import torch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from models.csrnet import CSRNet

# ── paths ──────────────────────────────────────────────────────────────────
PTH_PATH  = os.path.join(PROJECT_ROOT, "csrnet_shanghaiA.pth")
ONNX_PATH = os.path.join(PROJECT_ROOT, "csrnet_shanghaiA.onnx")

# ── load model ─────────────────────────────────────────────────────────────
print("Loading PyTorch model...")
model = CSRNet()
model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
model.eval()

# ── dummy input (batch=1, 3 channels, 480x640) ─────────────────────────────
dummy = torch.randn(1, 3, 480, 640)

# ── export ─────────────────────────────────────────────────────────────────
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }
)

print(f"✅ ONNX model saved to: {ONNX_PATH}")
print("Now run the server normally — video_infer.py will use ONNX automatically.")