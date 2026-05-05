"""
Fig 5.1 — Sample YOLO detection inference on a 640×640 validation tile.
Shows predicted bounding boxes with confidence scores.

Run:  python gen_fig_5_1_detection.py
Output: fig_5_1_detection.png  (200 dpi, white background)
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── paths ─────────────────────────────────────────────────────────────────────
VAL_IMG_DIR = Path("D:/Projects/Masters/Data/DET_data_sliced_split/val/images")
VAL_LBL_DIR = Path("D:/Projects/Masters/Data/DET_data_sliced_split/val/labels")
MODEL_PATH  = Path("D:/Projects/Masters/Trained_Models/DET_medium-best.pt")
OUT         = Path(__file__).parent / "fig_5_1_detection.png"

CONF_THR = 0.45

# ── find the tile with the most annotated bees ───────────────────────────────
def count_labels(lbl_path):
    if not lbl_path.exists():
        return 0
    with open(lbl_path) as f:
        return sum(1 for line in f if line.strip())

print("Scanning validation tiles for bee density ...")
candidates = []
for img_path in sorted(VAL_IMG_DIR.glob("*.jpg")):
    lbl_path = VAL_LBL_DIR / (img_path.stem + ".txt")
    n = count_labels(lbl_path)
    if n >= 5:
        candidates.append((n, img_path, lbl_path))

if not candidates:
    # fall back to tile with most bees if none has 5+
    for img_path in sorted(VAL_IMG_DIR.glob("*.jpg")):
        lbl_path = VAL_LBL_DIR / (img_path.stem + ".txt")
        n = count_labels(lbl_path)
        candidates.append((n, img_path, lbl_path))

candidates.sort(reverse=True)
_, img_path, lbl_path = candidates[0]
print(f"Selected tile: {img_path.name}  ({candidates[0][0]} annotated bees)")

# ── run YOLO inference ────────────────────────────────────────────────────────
print("Loading YOLO model ...")
from ultralytics import YOLO
model = YOLO(str(MODEL_PATH))

results = model.predict(
    str(img_path),
    conf=CONF_THR,
    iou=0.45,
    imgsz=640,
    verbose=False,
)[0]

boxes  = results.boxes.xyxy.cpu().numpy()    # [N, 4]  x1 y1 x2 y2
confs  = results.boxes.conf.cpu().numpy()    # [N]
print(f"Detections: {len(boxes)}  (conf ≥ {CONF_THR})")

# ── build the figure ──────────────────────────────────────────────────────────
img = Image.open(img_path).convert("RGB")
W, H = img.size   # should be 640×640

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("white")
ax.imshow(img)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)   # image coords (y down)
ax.axis("off")

# colour + style
BOX_COLOR  = "#00E676"   # vivid green
TEXT_COLOR = "#FFFFFF"
TEXT_BG    = "#00C853"

for (x1, y1, x2, y2), conf in sorted(zip(boxes, confs), key=lambda t: -t[1]):
    bw, bh = x2 - x1, y2 - y1
    rect = patches.FancyBboxPatch(
        (x1, y1), bw, bh,
        boxstyle="square,pad=0",
        linewidth=1.8,
        edgecolor=BOX_COLOR,
        facecolor="none",
        zorder=3,
    )
    ax.add_patch(rect)

    label = f"{conf:.2f}"
    ax.text(
        x1 + 2, y1 - 3,
        label,
        fontsize=7, fontweight="bold",
        color=TEXT_COLOR,
        bbox=dict(facecolor=TEXT_BG, edgecolor="none",
                  pad=1.5, boxstyle="round,pad=0.2"),
        va="bottom", zorder=4,
    )

ax.set_title(
    f"YOLOv11-medium inference  —  {len(boxes)} bees detected  "
    f"(conf ≥ {CONF_THR})",
    fontsize=10, fontweight="bold", color="#1B2631", pad=8,
)

plt.tight_layout(pad=0.4)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
