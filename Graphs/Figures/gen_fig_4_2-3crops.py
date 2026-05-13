"""
Generates Fig 4.2 and Fig 4.3 — representative bee crop figures for Chapter 4.

Fig 4.2 : best single crop per class (sharpest + well-exposed), side by side.
Fig 4.3 : 3×3 grid, one row per class, crops sampled from three different
          temporal segments to show within-class visual variation.

Run:  python gen_fig_4_crops.py
Outputs:
  fig_4_2_best_crops.png
  fig_4_3_crop_grid.png
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCRIPT_DIR  = Path(__file__).parent
DATA_ROOT   = Path("D:/Projects/Masters/Data/AR_v2_dataset/train")
CLASSES     = ["fanning", "neutral", "trophallaxis"]
LABELS      = ["Fanning", "Neutral", "Trophallaxis"]
COLORS      = ["#AED6F1", "#A9DFBF", "#F9E79F"]   # blue / green / yellow

SAMPLE_N    = 1000    # images sampled per class / segment for scoring
RANDOM_SEED = 69

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── scoring helpers ───────────────────────────────────────────────────────────

def laplacian_var(img: Image.Image) -> float:
    """Sharpness proxy — variance of Laplacian (higher = sharper)."""
    g   = np.array(img.convert("L"), dtype=float)
    lap = (np.roll(g, 1, 0) + np.roll(g, -1, 0) +
           np.roll(g, 1, 1) + np.roll(g, -1, 1) - 4 * g)
    return float(lap.var())


def has_reflect_padding(img: Image.Image, strip: int = 8,
                        threshold: float = 0.97) -> bool:
    """
    Detect reflect-padded crops by checking whether any edge strip is a
    near-perfect mirror of the immediately adjacent strip.
    """
    arr = np.array(img.convert("L"), dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.flatten(), b.flatten()
        if a.std() < 1e-3 or b.std() < 1e-3:
            return 1.0
        return float(np.corrcoef(a, b)[0, 1])

    checks = [
        corr(arr[:, :strip],   arr[:, strip:strip * 2][:, ::-1]),    # left
        corr(arr[:, -strip:],  arr[:, -strip * 2:-strip][:, ::-1]),  # right
        corr(arr[:strip, :],   arr[strip:strip * 2, :][::-1, :]),    # top
        corr(arr[-strip:, :],  arr[-strip * 2:-strip, :][::-1, :]),  # bottom
    ]
    return any(c > threshold for c in checks)


def score_paths(paths: list[str], max_n: int = SAMPLE_N,
                strategy: str = "sharp") -> list[tuple[float, str]]:
    """
    Sample up to max_n images, score by sharpness + brightness filter.
    Reflect-padded crops (border artefacts from data_prep.py) are excluded.

    strategy="sharp"    → return highest-sharpness images first (neutral, trophallaxis)
    strategy="moderate" → return images in the mid sharpness range first (fanning —
                          moderate blur captures wing motion better than top-sharp frames
                          which tend to be cluttered multi-bee scenes with many edges)
    """
    sample = random.sample(paths, min(max_n, len(paths)))
    scored = []
    for p in sample:
        try:
            img = Image.open(p)
            brightness = float(np.array(img.convert("L")).mean())
            if brightness < 35 or brightness > 220:
                continue
            if has_reflect_padding(img):
                continue
            scored.append((laplacian_var(img), p))
        except Exception:
            continue
    scored.sort(reverse=True)

    if strategy == "moderate" and len(scored) >= 6:
        # Pick from the 25th–60th percentile of sharpness scores
        lo = len(scored) // 4
        hi = int(len(scored) * 0.60)
        scored = scored[lo:hi]

    return scored


def load_rgb(path: str, size: int = 224) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size))


# ── collect all paths per class ───────────────────────────────────────────────

print("Scanning dataset ...")
all_paths: dict[str, list[str]] = {}
for cls in CLASSES:
    paths = sorted((DATA_ROOT / cls).glob("*.jpg"))
    all_paths[cls] = [str(p) for p in paths]
    print(f"  {cls}: {len(all_paths[cls]):,} images")


# ── Fig 4.2 — best single crop per class ─────────────────────────────────────

print("\nScoring for Fig 4.2 (best single crop per class) ...")
# fanning: prefer moderate sharpness → more likely to show motion-blurred wings
STRATEGIES = {"fanning": "moderate", "neutral": "sharp", "trophallaxis": "sharp"}
best: dict[str, str] = {}
for cls in CLASSES:
    scored = score_paths(all_paths[cls], strategy=STRATEGIES[cls])
    best[cls] = scored[0][1]
    print(f"  {cls}: {Path(best[cls]).name}  [sharpness = {scored[0][0]:.1f}]")

fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
fig.patch.set_facecolor("white")

for ax, cls, label, color in zip(axes, CLASSES, LABELS, COLORS):
    ax.imshow(load_rgb(best[cls]), interpolation='lanczos')
    ax.set_title(label, fontsize=13, fontweight="bold", pad=9, color="#1B2631")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(pad=1.0)
out42 = str(SCRIPT_DIR / "fig_4_2_best_crops.png")
plt.savefig(out42, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out42}")


# ── Fig 4.3 — 3×3 grid (3 temporal segments per class) ───────────────────────

print("\nBuilding Fig 4.3 (3×3 grid) ...")
COLS = 3   # one image per temporal segment

grid: dict[str, list[str]] = {}
for cls in CLASSES:
    paths = all_paths[cls]          # already sorted alphabetically / by frame
    n     = len(paths)
    segs  = [paths[i * n // COLS : (i + 1) * n // COLS] for i in range(COLS)]
    picks = []
    for seg in segs:
        scored = score_paths(seg, max_n=min(200, len(seg)),
                             strategy=STRATEGIES[cls])
        picks.append(scored[0][1] if scored else random.choice(seg))
    grid[cls] = picks
    print(f"  {cls}: {[Path(p).name for p in picks]}")

fig, axes = plt.subplots(3, COLS, figsize=(COLS * 3.1, 3 * 3.2))
fig.patch.set_facecolor("white")

for row, (cls, label, color) in enumerate(zip(CLASSES, LABELS, COLORS)):
    for col in range(COLS):
        ax = axes[row][col]
        ax.imshow(load_rgb(grid[cls][col]), interpolation='lanczos')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(label, fontsize=12, fontweight="bold",
                          color="#1B2631", labelpad=8)
    # segment labels across the top row only
    if row == 0:
        for col, seg_lbl in enumerate(["Early segment", "Mid segment", "Late segment"]):
            axes[0][col].set_title(seg_lbl, fontsize=9, color="#555555", pad=5)

plt.tight_layout(pad=0.6)
out43 = str(SCRIPT_DIR / "fig_4_3_crop_grid.png")
plt.savefig(out43, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out43}")
