"""
Figure 3.5 — Temporal Sequence Classifier: clean block diagram with real bee crops.

Run:    python gen_fig_3_5_classifier.py
Output: fig_3_5_classifier.png
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT         = Path(__file__).parent / "fig_3_5_classifier.png"
DATASET_DIR = Path("D:/Projects/Masters/Data/AR_v2_dataset")
TARGET_CLS  = "fanning"
BUFFER_SIZE = 16
N_SHOW      = 8

# ── colour scheme ─────────────────────────────────────────────────────────────
C_INPUT  = ("#D6EAF8", "#2471A3")
C_EMBED  = ("#D5F5E3", "#1A8754")
C_TRANS  = ("#F5EEF8", "#7D3C98")
C_HEAD   = ("#FDEDEC", "#C0392B")
C_OUTPUT = ("#FEF9E7", "#D4AC0D")
C_DARK   = "#2C3E50"
C_ARROW  = "#555555"

# ── find a fanning tracklet and load N_SHOW crops ─────────────────────────────
_RE = re.compile(r"^(fanning|trophallaxis)_(.+?)_(\d{5})_(\d+)$")

def find_crops(cls_dir, n_show=N_SHOW, buf=BUFFER_SIZE):
    groups = defaultdict(list)
    for fp in sorted(cls_dir.iterdir()):
        if fp.suffix.lower() != ".jpg":
            continue
        m = _RE.match(fp.stem)
        if m:
            key = (m.group(2), m.group(4))
            groups[key].append((int(m.group(3)), fp))
    candidates = [sorted(v) for v in groups.values() if len(v) >= buf]
    if not candidates:
        return None
    best = max(candidates, key=len)
    mid  = max(0, len(best) // 2 - buf // 2)
    seq  = best[mid: mid + buf]
    step = max(1, buf // n_show)
    return [seq[i][1] for i in range(0, buf, step)][:n_show]

print("Loading bee crops …")
crop_paths = find_crops(DATASET_DIR / "val" / TARGET_CLS)
if crop_paths is None:
    crop_paths = list((DATASET_DIR / "val" / TARGET_CLS).glob("*.jpg"))[:N_SHOW]

def load_crop(path, size=96):
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS))

crops = [load_crop(p) for p in crop_paths]
print(f"  Loaded {len(crops)} crops from {TARGET_CLS}")

# ── figure setup ──────────────────────────────────────────────────────────────
XMAX, YMAX = 10.0, 19.5
fig, ax = plt.subplots(figsize=(6.5, 11))
fig.patch.set_facecolor("white")
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────
def to_frac(x, y, w, h):
    """Convert data coords to axes fraction for inset_axes."""
    return [x / XMAX, y / YMAX, w / XMAX, h / YMAX]

def box(cx, cy, w, h, label, sublabel="",
        face="#FFF", edge="#333", fontsize=9.5, subfontsize=7.8):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=face, edgecolor=edge, linewidth=1.8, zorder=3,
    ))
    y_t = cy + (0.18 if sublabel else 0)
    ax.text(cx, y_t, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C_DARK, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=subfontsize, color="#555", style="italic", zorder=4)

def arrow(x, y_top, y_bot, label=""):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.6, mutation_scale=14), zorder=2)
    if label:
        ax.text(x + 0.18, (y_top + y_bot) / 2, label,
                ha="left", va="center", fontsize=7.2, color="#555", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.8, alpha=0.9))

CX = 5.0

# ── 1. Bee crop strip ─────────────────────────────────────────────────────────
STRIP_BOT  = 16.8
STRIP_H    = 2.4
STRIP_L    = 0.6
STRIP_W    = 8.8

# light background behind strip
ax.add_patch(FancyBboxPatch(
    (STRIP_L, STRIP_BOT), STRIP_W, STRIP_H,
    boxstyle="round,pad=0.1",
    facecolor=C_INPUT[0], edgecolor=C_INPUT[1], linewidth=1.8, zorder=2,
))
ax.text(CX, STRIP_BOT + STRIP_H + 0.05,
        "Input — Bee Tracklet   (T = 16 frames, 8 shown)",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        color=C_INPUT[1], zorder=4)

# place crop images
n    = len(crops)
gap  = 0.06
img_w = (STRIP_W - gap * (n + 1)) / n
img_h = STRIP_H - 0.30

for i, img in enumerate(crops):
    x0 = STRIP_L + gap + i * (img_w + gap)
    y0 = STRIP_BOT + 0.15
    inset = ax.inset_axes(to_frac(x0, y0, img_w, img_h))
    inset.imshow(img)
    inset.set_xticks([]); inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_edgecolor(C_INPUT[1]); sp.set_linewidth(1.2)
    ax.text((x0 + img_w/2) / XMAX, (y0 - 0.08) / YMAX,
            f"t={i*2+1}", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5, color="#666")

# ── input label box ───────────────────────────────────────────────────────────
Y_LBL = 15.75
box(CX, Y_LBL, 8.8, 0.72,
    "X  ∈  ℝ¹⁶ ˣ ⁷⁶⁸   — 768-dim DINOv2 feature vector per frame",
    "CLS token (384-d) ∥ mean of 256 patch tokens (384-d)",
    face=C_INPUT[0], edge=C_INPUT[1], fontsize=9, subfontsize=7.5)

# ── arrow ─────────────────────────────────────────────────────────────────────
arrow(CX, Y_LBL - 0.37, Y_LBL - 0.92,
      "prepend learnable [CLS] token  →  17 tokens")

# ── 2. Positional embedding ───────────────────────────────────────────────────
Y_PE = 14.52
box(CX, Y_PE, 8.8, 0.72,
    "Add Learnable Positional Embeddings",
    "X'  ∈  ℝ¹⁷ ˣ ⁷⁶⁸   (temporal order encoded in each token)",
    face=C_EMBED[0], edge=C_EMBED[1], fontsize=9, subfontsize=7.5)

# ── arrow ─────────────────────────────────────────────────────────────────────
arrow(CX, Y_PE - 0.37, Y_PE - 0.80)

# ── 3. Transformer encoder ×3 ────────────────────────────────────────────────
Y_TB_TOP = 13.58
TB_H     = 3.80
Y_TB_CX  = Y_TB_TOP - TB_H / 2

ax.add_patch(FancyBboxPatch(
    (CX - 3.8, Y_TB_TOP - TB_H), 7.6, TB_H,
    boxstyle="round,pad=0.15",
    facecolor=C_TRANS[0], edgecolor=C_TRANS[1],
    linewidth=2.2, linestyle="--", zorder=2,
))
ax.text(CX + 3.8, Y_TB_TOP - 0.16, "× 3",
        ha="right", va="top", fontsize=11, fontweight="bold",
        color=C_TRANS[1], zorder=5)
ax.text(CX - 3.75, Y_TB_TOP - 0.16, "Transformer Encoder Block",
        ha="left", va="top", fontsize=9, fontweight="bold",
        color=C_TRANS[1], zorder=5)

sub_items = [
    ("Pre-LayerNorm",               "",                       False),
    ("Multi-Head Self-Attention",   "6 heads  ·  head dim = 128  ·  + residual",  True),
    ("Pre-LayerNorm",               "",                       False),
    ("Feed-Forward Network (GELU)", "hidden dim = 1,536  ·  + residual",          True),
    ("Dropout  (p = 0.35)",         "",                       False),
]
sub_h, sub_gap = 0.54, 0.06
total_h = len(sub_items) * sub_h + (len(sub_items) - 1) * sub_gap
y_s = Y_TB_CX + total_h / 2 - sub_h / 2

for i, (lbl, sub, hi) in enumerate(sub_items):
    yc   = y_s - i * (sub_h + sub_gap)
    face = "#E8DAEF" if hi else "#F5F5F5"
    edge = C_TRANS[1] if hi else "#BBBBBB"
    ax.add_patch(FancyBboxPatch(
        (CX - 3.3, yc - sub_h/2), 6.6, sub_h,
        boxstyle="round,pad=0.07",
        facecolor=face, edgecolor=edge, linewidth=1.4 if hi else 0.9, zorder=3,
    ))
    y_t = yc + (0.10 if sub else 0)
    ax.text(CX, y_t, lbl, ha="center", va="center",
            fontsize=8.5, fontweight="bold" if hi else "normal",
            color=C_DARK, zorder=4)
    if sub:
        ax.text(CX, yc - 0.14, sub, ha="center", va="center",
                fontsize=7.2, color="#666", style="italic", zorder=4)

# ── arrow ─────────────────────────────────────────────────────────────────────
Y_AFTER = Y_TB_TOP - TB_H
arrow(CX, Y_AFTER, Y_AFTER - 0.52, "extract [CLS] token  z₀  ∈  ℝ⁷⁶⁸")

# ── 4. Classification head ────────────────────────────────────────────────────
HEAD_ROWS = [
    ("LayerNorm",                          False),
    ("Linear  768 → 256  +  GELU + Dropout (0.35)", True),
    ("Linear  256 → 3",                    True),
    ("Softmax",                            False),
]
head_h, head_gap = 0.54, 0.05
total_hh  = len(HEAD_ROWS) * head_h + (len(HEAD_ROWS) - 1) * head_gap
Y_HEAD_CX = Y_AFTER - 0.52 - total_hh / 2 - 0.38

ax.add_patch(FancyBboxPatch(
    (CX - 3.3, Y_HEAD_CX - total_hh/2 - 0.18), 6.6, total_hh + 0.36,
    boxstyle="round,pad=0.12",
    facecolor=C_HEAD[0], edgecolor=C_HEAD[1], linewidth=2.0, zorder=2,
))
ax.text(CX - 3.15, Y_HEAD_CX + total_hh/2 + 0.12,
        "Classification Head",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=C_HEAD[1], zorder=5)

y_hs = Y_HEAD_CX + total_hh/2 - head_h/2
for i, (lbl, hi) in enumerate(HEAD_ROWS):
    yc   = y_hs - i * (head_h + head_gap)
    face = "#FADBD8" if hi else "#F5F5F5"
    edge = C_HEAD[1] if hi else "#BBBBBB"
    ax.add_patch(FancyBboxPatch(
        (CX - 2.85, yc - head_h/2), 5.7, head_h,
        boxstyle="round,pad=0.07",
        facecolor=face, edgecolor=edge, linewidth=1.4 if hi else 0.9, zorder=3,
    ))
    ax.text(CX, yc, lbl, ha="center", va="center",
            fontsize=8.5, fontweight="bold" if hi else "normal",
            color=C_DARK, zorder=4)

# ── arrow ─────────────────────────────────────────────────────────────────────
Y_OUT_TOP = Y_HEAD_CX - total_hh/2 - 0.18
arrow(CX, Y_OUT_TOP, Y_OUT_TOP - 0.42)

# ── 5. Output ─────────────────────────────────────────────────────────────────
Y_OUT = Y_OUT_TOP - 0.42 - 0.40
box(CX, Y_OUT, 8.8, 0.75,
    "Output:  P(Fanning)     P(Neutral)     P(Trophallaxis)",
    "Softmax probability distribution over 3 behavior classes",
    face=C_OUTPUT[0], edge=C_OUTPUT[1], fontsize=9.5, subfontsize=7.8)

# ── trim whitespace by adjusting ylim ────────────────────────────────────────
ax.set_ylim(Y_OUT - 0.55, YMAX)

plt.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.10, facecolor="white")
plt.close()
print(f"Saved: {OUT}")
