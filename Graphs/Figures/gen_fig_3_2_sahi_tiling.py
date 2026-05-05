"""
Fig 3.2 — SAHI tiling illustration.

Left panel : full 1080p hive-entrance frame with overlapping 640×640 tile grid
             drawn as a coloured overlay and one small bee circled to show how
             tiny it is when the frame is resized directly to 640×640.
Right panel: the 640×640 tile that contains that bee, shown at native scale with
             the same bee circled to show it is now clearly visible.

Run:  python gen_fig_3_2_sahi_tiling.py
Output: fig_3_2_sahi_tiling.png
"""

import os, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── paths ─────────────────────────────────────────────────────────────────────
IMG_DIR   = Path("D:/Projects/Masters/Data/DET_data_OG/images")
LBL_DIR   = Path("D:/Projects/Masters/Data/DET_data_OG/labels")
OUT       = Path(__file__).parent / "fig_3_2_sahi_tiling.png"

TILE_SIZE = 640
OVERLAP   = 0.25          # 25 % overlap → stride = 480 px
CONF_THR  = 0.45          # display only (visual)

# ── pick a frame that has labels ─────────────────────────────────────────────
def pick_frame():
    for img_path in sorted(IMG_DIR.glob("*.jpg")):
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            return img_path, lbl_path
    raise FileNotFoundError("No labelled images found.")

img_path, lbl_path = pick_frame()
print(f"Using: {img_path.name}")

img = Image.open(img_path).convert("RGB")
W, H = img.size
print(f"Frame size: {W}×{H}")

# ── load annotations (YOLO format: class cx cy w h, normalised) ───────────────
bees = []
with open(lbl_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        # absolute pixel coords
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        area = (x2 - x1) * (y2 - y1)
        bees.append((area, x1, y1, x2, y2, cx * W, cy * H))

if not bees:
    raise ValueError("No bee annotations found in label file.")

# pick the smallest bee that sits in the hive-board area (avoid bottom vegetation strip)
bees_on_board = [b for b in bees if b[6] < H * 0.82]   # exclude bottom 18 %
pool = bees_on_board if bees_on_board else bees
pool.sort(key=lambda b: b[0])
_, bx1, by1, bx2, by2, bcx, bcy = pool[0]
bee_w = bx2 - bx1
bee_h = by2 - by1
print(f"Selected bee: size={bee_w}×{bee_h}px at ({int(bcx)},{int(bcy)})")

# ── compute SAHI tile positions ───────────────────────────────────────────────
stride = int(TILE_SIZE * (1 - OVERLAP))   # 480

def tile_positions(dim, tile, st):
    """Start positions along one axis, last tile clipped to dim boundary."""
    n = math.ceil(max(dim - tile, 0) / st) + 1
    positions = [min(i * st, max(dim - tile, 0)) for i in range(n)]
    return sorted(set(positions))

xs = tile_positions(W, TILE_SIZE, stride)
ys = tile_positions(H, TILE_SIZE, stride)
print(f"Tile grid: {len(xs)} columns × {len(ys)} rows  ({len(xs)*len(ys)} tiles total)")

# find which tile the selected bee falls in (pick the tile whose centre is closest)
def best_tile(bcx, bcy):
    best, best_d = None, float("inf")
    for tx in xs:
        for ty in ys:
            if tx <= bcx <= tx + TILE_SIZE and ty <= bcy <= ty + TILE_SIZE:
                d = abs(bcx - (tx + TILE_SIZE / 2)) + abs(bcy - (ty + TILE_SIZE / 2))
                if d < best_d:
                    best_d = d
                    best = (tx, ty)
    return best or (xs[0], ys[0])

tile_x, tile_y = best_tile(bcx, bcy)
print(f"Tile origin: ({tile_x}, {tile_y})")

# ── build LEFT panel — full frame with grid overlay ───────────────────────────
DISPLAY_H = 560          # height to display the full frame at (keeps it compact)
scale     = DISPLAY_H / H
dw        = int(W * scale)

frame_disp = img.resize((dw, DISPLAY_H), Image.LANCZOS)
draw       = ImageDraw.Draw(frame_disp, "RGBA")

# distinct colours for each tile (RGBA fill + solid border)
TILE_PALETTE = [
    (220,  80,  80),   # red
    ( 60, 160, 220),   # blue
    ( 80, 190,  90),   # green
    (230, 160,  40),   # amber
    (160,  80, 200),   # purple
    ( 50, 200, 180),   # teal
    (230, 100, 180),   # pink
    (140, 200,  60),   # lime
]

tile_list = [(tx, ty) for ty in ys for tx in xs]   # row-major order

for idx, (tx, ty) in enumerate(tile_list):
    is_selected = (tx == tile_x and ty == tile_y)
    r_c, g_c, b_c = TILE_PALETTE[idx % len(TILE_PALETTE)]
    rx1 = int(tx * scale)
    ry1 = int(ty * scale)
    rx2 = int((tx + TILE_SIZE) * scale)
    ry2 = int((ty + TILE_SIZE) * scale)

    draw.rectangle([rx1, ry1, rx2, ry2],
                   fill=(r_c, g_c, b_c, 45),
                   outline=(r_c, g_c, b_c, 210), width=2)

    # selected tile: extra white dashed border to make it pop
    if is_selected:
        draw.rectangle([rx1 + 3, ry1 + 3, rx2 - 3, ry2 - 3],
                       fill=None, outline=(255, 255, 255, 230), width=2)

# circle the bee — make it visually prominent even though the bee is tiny
cx_d = int(bcx * scale)
cy_d = int(bcy * scale)
r    = max(24, int(max(bee_w, bee_h) * scale * 3.5))
# outer ring
draw.ellipse([cx_d - r - 3, cy_d - r - 3, cx_d + r + 3, cy_d + r + 3],
             outline=(255, 255, 0, 180), width=3)
# inner ring
draw.ellipse([cx_d - r, cy_d - r, cx_d + r, cy_d + r],
             outline=(255, 30, 30, 255), width=3)

# leader line + label positioned to avoid tile borders
lx, ly = cx_d + r + 8, cy_d - 30
draw.line([cx_d + r + 2, cy_d, lx, ly], fill=(255, 30, 30, 255), width=2)
draw.text((lx + 2, ly - 12), "bee (tiny!)", fill=(255, 30, 30, 255))

# ── build RIGHT panel — extracted tile at native 640×640 ──────────────────────
tile_crop = img.crop((tile_x, tile_y, tile_x + TILE_SIZE, tile_y + TILE_SIZE))

# circle the bee inside the tile
draw_t = ImageDraw.Draw(tile_crop, "RGBA")
tbx = int(bcx - tile_x)
tby = int(bcy - tile_y)
rt  = max(22, int(max(bee_w, bee_h) * 2.5))
draw_t.ellipse([tbx - rt, tby - rt, tbx + rt, tby + rt],
               outline=(255, 30, 30, 255), width=4)
draw_t.line([tbx + rt + 2, tby, tbx + rt + 32, tby - 24],
            fill=(255, 30, 30, 255), width=2)
draw_t.text((tbx + rt + 34, tby - 34), "bee", fill=(255, 30, 30, 255))

# resize tile to same display height as left panel
tile_disp = tile_crop.resize((DISPLAY_H, DISPLAY_H), Image.LANCZOS)

# ── compose side-by-side with matplotlib ─────────────────────────────────────
fig, (ax_full, ax_tile) = plt.subplots(
    1, 2,
    figsize=(14, 5),
    gridspec_kw={"width_ratios": [dw / DISPLAY_H, 1]}
)
fig.patch.set_facecolor("white")

ax_full.imshow(frame_disp)
ax_full.set_title(
    f"Full frame  ({W}×{H} px)\n"
    f"Direct resize to 640×640 → bee ≈ {max(1, round(bee_w * 640 / W))}×"
    f"{max(1, round(bee_h * 640 / H))} px",
    fontsize=10, fontweight="bold", color="#1B2631", pad=8
)
# draw a faint dashed rectangle around the selected tile in matplotlib too
rect = patches.Rectangle(
    (int(tile_x * scale), int(tile_y * scale)),
    int(TILE_SIZE * scale), int(TILE_SIZE * scale),
    linewidth=2, edgecolor="orange", facecolor="none",
    linestyle="--"
)
ax_full.add_patch(rect)
ax_full.axis("off")

ax_tile.imshow(tile_disp)
ax_tile.set_title(
    f"Extracted 640×640 tile  (native scale)\n"
    f"Bee occupies {bee_w}×{bee_h} px — clearly detectable",
    fontsize=10, fontweight="bold", color="#1B2631", pad=8
)
ax_tile.axis("off")

plt.tight_layout(pad=0.8)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
