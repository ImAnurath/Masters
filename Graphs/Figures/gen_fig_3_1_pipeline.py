"""
Fig 3.1 — End-to-end pipeline flow diagram.
Run with:  python gen_fig_3_1_pipeline.py
Output:    fig_3_1_pipeline.png  (200 dpi, white background)
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "fig_3_1_pipeline.png")

# ── colour palette ──────────────────────────────────────────────────────────
BLU = "#D6EAF8"   # input frame
GRN = "#A9DFBF"   # detection
ORG = "#FDEBD0"   # tracking
PRP = "#E8DAEF"   # feature extraction / buffer
RED = "#FADBD8"   # classifier
OUT_C = "#D5F5E3" # output label

# ── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 5)
ax.axis("off")
fig.patch.set_facecolor("white")

BW   = 2.0    # box width
BH   = 1.6    # box height
Y    = 2.7    # vertical centre of all boxes
STEP = 2.5    # centre-to-centre distance

xs = [1.1 + i * STEP for i in range(8)]

stages = [
    ("Video\nFrame",                          BLU),
    ("SAHI Tiling\n640×640 tiles\n25% overlap", GRN),
    ("YOLOv11-medium\nDetection\nconf ≥ 0.45",  GRN),
    ("ByteTracker\nIdentity\nPersistence",     ORG),
    ("DINOv2-small\nFeature\nExtraction",      PRP),
    ("Rolling Buffer\nT = 16 frames",          PRP),
    ("Temporal\nTransformer\nClassifier",      RED),
    ("Behavior\nLabel",                        OUT_C),
]

# labels placed BELOW selected arrows (index of the left-side box)
below_labels = {
    2: "Bounding boxes",
    3: "Per-bee crops",
    5: "16×384 tensor",
}

# ── draw boxes ───────────────────────────────────────────────────────────────
for x, (label, color) in zip(xs, stages):
    p = FancyBboxPatch(
        (x - BW / 2, Y - BH / 2), BW, BH,
        boxstyle="round,pad=0.13",
        facecolor=color, edgecolor="#2C3E50", linewidth=1.9, zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x, Y, label,
        ha="center", va="center", fontsize=9.2,
        fontweight="bold", color="#1B2631",
        multialignment="center", zorder=3,
    )

# ── draw arrows ──────────────────────────────────────────────────────────────
for i in range(len(xs) - 1):
    x1 = xs[i] + BW / 2
    x2 = xs[i + 1] - BW / 2
    ax.annotate(
        "", xy=(x2, Y), xytext=(x1, Y),
        arrowprops=dict(
            arrowstyle="->", color="#2C3E50", lw=1.6, mutation_scale=18
        ),
        zorder=1,
    )
    if i in below_labels:
        mx = (x1 + x2) / 2
        ax.text(
            mx, Y - BH / 2 - 0.35,
            below_labels[i],
            ha="center", va="top", fontsize=7.8,
            color="#555555", style="italic",
        )

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
