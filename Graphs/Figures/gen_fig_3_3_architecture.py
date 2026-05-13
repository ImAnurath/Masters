"""
Fig 3.3 — Temporal Sequence Classifier architecture diagram.
Professional vertical-stack style, conference-paper aesthetic.

Run:    python gen_fig_3_3_architecture.py
Output: fig_3_3_architecture.png  (200 dpi, white background)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = "D:/Projects/Masters/Graphs/Figures/fig_3_3_architecture.png"
DPI = 200

# ── palette — two-tone, inspired by Transformer / BERT paper figures ──────────
BLUE_F  = "#EBF5FB";  BLUE_E  = "#2471A3"   # input / token blocks
PURP_F  = "#F4ECF7";  PURP_E  = "#7D3C98"   # transformer encoder
AMBE_F  = "#FEF5E7";  AMBE_E  = "#B9770E"   # classification head
FAN_F   = "#EAFAF1";  FAN_E   = "#1E8449"   # fanning output
NEU_F   = "#F2F3F4";  NEU_E   = "#717D7E"   # neutral output
TRO_F   = "#FEF5E7";  TRO_E   = "#B9770E"   # trophallaxis output
DBOX_F  = "#FDFEFE";  DBOX_E  = "#AAB7B8"   # detail panel background
C_DARK  = "#1C2833"
C_SUB   = "#5D6D7E"
C_DIM   = "#2471A3"
C_ARR   = "#2C3E50"

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 12.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── layout constants ──────────────────────────────────────────────────────────
LX   = 0.55   # left edge of main column
LW   = 5.10   # main block width
LMX  = LX + LW / 2   # main column centre-x

DX   = 6.25   # detail panel left edge
DW   = 3.30   # detail panel width
DMX  = DX + DW / 2


# ── helpers ───────────────────────────────────────────────────────────────────

def block(y, h, face, edge, title, sub="", dim=""):
    """Draw one main-column block."""
    p = FancyBboxPatch((LX, y), LW, h,
                        boxstyle="round,pad=0.05,rounding_size=0.12",
                        facecolor=face, edgecolor=edge,
                        linewidth=1.6, zorder=3)
    ax.add_patch(p)
    ty = y + h / 2 + (0.11 if sub else 0)
    ax.text(LMX, ty, title,
            ha="center", va="center", fontsize=10.8,
            fontweight="bold", color=C_DARK, zorder=4,
            multialignment="center")
    if sub:
        ax.text(LMX, y + h / 2 - 0.15, sub,
                ha="center", va="center", fontsize=8.4,
                color=C_SUB, zorder=4, multialignment="center")
    if dim:
        ax.text(LX + LW + 0.14, y + h / 2, dim,
                ha="left", va="center", fontsize=8.2,
                color=C_DIM, fontstyle="italic", fontweight="bold", zorder=4)


def vert_arrow(y_from, y_to, label=""):
    ax.annotate("", xy=(LMX, y_to), xytext=(LMX, y_from),
                arrowprops=dict(arrowstyle="-|>", color=C_ARR,
                                lw=1.7, mutation_scale=16), zorder=2)
    if label:
        ax.text(LMX + 0.18, (y_from + y_to) / 2, label,
                ha="left", va="center", fontsize=7.8,
                color=C_SUB, fontstyle="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="#D5D8DC", linewidth=0.8, alpha=0.95))


def detail_row(cx, y, w, h, text, filled=False):
    face = "#D4E6F1" if filled else "#FDFEFE"
    edge = "#2471A3" if filled else "#BDC3C7"
    lw   = 1.2 if filled else 0.9
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.07",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=5))
    ax.text(cx, y, text,
            ha="center", va="center", fontsize=7.8,
            color=C_DARK, fontweight="bold" if filled else "normal",
            zorder=6, multialignment="center")


def detail_arrow(cx, y_from, y_to):
    ax.annotate("", xy=(cx, y_to), xytext=(cx, y_from),
                arrowprops=dict(arrowstyle="-|>", color="#AAB7B8",
                                lw=1.0, mutation_scale=9), zorder=4)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN COLUMN
# ══════════════════════════════════════════════════════════════════════════════

# ── Block 1 — Input ───────────────────────────────────────────────────────────
B1_Y, B1_H = 11.20, 0.75
block(B1_Y, B1_H, BLUE_F, BLUE_E,
      "Input Feature Sequence",
      sub="16 DINOv2-small CLS tokens  (one per buffered frame)",
      dim="16 × 384")

vert_arrow(B1_Y, B1_Y - 0.38, "prepend learnable [CLS] token")

# ── Block 2 — Token sequence ──────────────────────────────────────────────────
B2_Y, B2_H = 10.35, 0.72
block(B2_Y, B2_H, BLUE_F, BLUE_E,
      "Token Sequence",
      sub="[CLS]  |  t₁  |  t₂  |  …  |  t₁₆   (+ positional embeddings)",
      dim="17 × 384")

vert_arrow(B2_Y, B2_Y - 0.40)

# ── Block 3 — Transformer encoder ─────────────────────────────────────────────
B3_Y, B3_H = 8.95, 1.30
block(B3_Y, B3_H, PURP_F, PURP_E,
      "Transformer Encoder",
      sub="2 identical pre-norm encoder layers\n"
          "4-head self-attention  ·  FFN dim = 768  ·  Dropout 0.3",
      dim="17 × 384")

# ×2 badge
bx = LX + LW - 0.52
by_badge = B3_Y + B3_H - 0.38
ax.add_patch(FancyBboxPatch((bx, by_badge), 0.44, 0.28,
                             boxstyle="round,pad=0.04,rounding_size=0.06",
                             facecolor=PURP_E, edgecolor="none", zorder=5))
ax.text(bx + 0.22, by_badge + 0.14, "× 2",
        ha="center", va="center", fontsize=8.5,
        fontweight="bold", color="white", zorder=6)

vert_arrow(B3_Y, B3_Y - 0.42, "z₀  [CLS] output  (384-d)")

# ── Block 4 — Classification head ─────────────────────────────────────────────
B4_Y, B4_H = 7.50, 1.30
block(B4_Y, B4_H, AMBE_F, AMBE_E,
      "Classification Head",
      sub="LayerNorm  →  Linear(384→128)  →  GELU\n"
          "→  Dropout(0.3)  →  Linear(128→3)  →  Softmax",
      dim="3-d")

vert_arrow(B4_Y, B4_Y - 0.40, "softmax  (p₀, p₁, p₂)")

# ── Block 5 — Output badges ───────────────────────────────────────────────────
OUT_Y, OUT_H = 6.70, 0.70
OW = LW / 3 - 0.07
GAP = 0.105

for k, (ox, fc, ec, lbl, pl) in enumerate([
    (LX,                      FAN_F, FAN_E, "Fanning",      "p₀"),
    (LX + OW + GAP,           NEU_F, NEU_E, "Neutral",      "p₁"),
    (LX + 2 * (OW + GAP),     TRO_F, TRO_E, "Trophallaxis", "p₂"),
]):
    ax.add_patch(FancyBboxPatch(
        (ox, OUT_Y), OW, OUT_H,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3))
    ax.text(ox + OW / 2, OUT_Y + OUT_H * 0.60, lbl,
            ha="center", va="center", fontsize=9.8,
            fontweight="bold", color=C_DARK, zorder=4)
    ax.text(ox + OW / 2, OUT_Y + OUT_H * 0.22, pl,
            ha="center", va="center", fontsize=8.5,
            color=C_SUB, fontstyle="italic", zorder=4)

# fan-out lines from arrow tip to three boxes
tip_y = OUT_Y + OUT_H
spread_y = tip_y + 0.28
centres = [LX + OW / 2, LX + OW + GAP + OW / 2, LX + 2 * (OW + GAP) + OW / 2]
for cx in centres:
    ax.plot([LMX, cx], [spread_y, tip_y + 0.01],
            color=C_ARR, lw=1.3, zorder=2)
ax.plot([LMX, LMX], [B4_Y, spread_y],
        color=C_ARR, lw=1.7, zorder=2)


# ══════════════════════════════════════════════════════════════════════════════
#  DETAIL PANEL — Encoder block internals
# ══════════════════════════════════════════════════════════════════════════════
DP_Y  = B3_Y - 0.05
DP_H  = B3_H + 0.12
DP_MX = DMX
RW    = DW * 0.76
ROW_H = 0.38

# background
ax.add_patch(FancyBboxPatch(
    (DX, DP_Y), DW, DP_H,
    boxstyle="round,pad=0.05,rounding_size=0.14",
    facecolor=DBOX_F, edgecolor=DBOX_E,
    linewidth=1.2, zorder=2, linestyle=(0, (6, 3))))

ax.text(DP_MX, DP_Y + DP_H - 0.14, "Encoder Block  (×2)",
        ha="center", va="top", fontsize=9.2,
        fontweight="bold", color=C_DARK, zorder=4)

# rows inside detail panel (text, filled=highlight)
rows = [
    ("LayerNorm",                        False),
    ("Multi-Head Self-Attention\n(4 heads,  dₖ = 96)", True),
    ("+ Residual",                       False),
    ("LayerNorm",                        False),
    ("Feed-Forward  (768-d,  GELU)",     True),
    ("+ Residual",                       False),
    ("Dropout  (p = 0.3)",               False),
]

n = len(rows)
usable_h = DP_H - 0.35
step = usable_h / n
y_start = DP_Y + DP_H - 0.32

for i, (txt, filled) in enumerate(rows):
    ry = y_start - i * step - step / 2
    rh = step * 0.60
    detail_row(DP_MX, ry, RW, rh, txt, filled=filled)
    if i < n - 1:
        detail_arrow(DP_MX, ry - rh / 2 - 0.01, ry - step * 0.42)


# ── connector: bracket from transformer block to detail panel ─────────────────
main_right  = LX + LW
b3_mid_y    = B3_Y + B3_H / 2
conn_x      = main_right + 0.04

# horizontal jog
ax.annotate("", xy=(DX - 0.02, b3_mid_y),
            xytext=(conn_x, b3_mid_y),
            arrowprops=dict(arrowstyle="-|>", color=PURP_E,
                            lw=1.3, mutation_scale=11,
                            linestyle="dashed"), zorder=3)
ax.text((conn_x + DX) / 2, b3_mid_y + 0.12, "detail",
        ha="center", va="bottom", fontsize=7.2,
        color=PURP_E, fontstyle="italic")


# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION LABELS (left margin)
# ══════════════════════════════════════════════════════════════════════════════
for (by, bh), lbl in [
    ((B1_Y, B1_H), "T = 16"),
    ((B2_Y, B2_H), "T = 17"),
    ((B3_Y, B3_H), "T = 17"),
]:
    ax.text(LX - 0.10, by + bh / 2, lbl,
            ha="right", va="center", fontsize=7.5,
            color=C_DIM, fontstyle="italic")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════════════
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig(OUT, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.close()
print(f"Saved: {OUT}")
