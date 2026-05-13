"""
Fig 4.4 — Grouped bar chart of sequence counts per class (train vs. val).
Log-scale y-axis to show the extreme neutral-class imbalance without
completely flattening the minority bars.

Run:  python gen_fig_4_4_sequence_counts.py
Output: fig_4_4_sequence_counts.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT = Path(__file__).parent / "fig_4_4_sequence_counts.png"

# ── data (AR_v2_dataset — from train.ipynb "Building sequences..." output) ───
classes = ["Fanning", "Trophallaxis", "Neutral"]
train   = [442,       76,             1_893]
val     = [153,       34,               371]

# Approximate class weights from the trained model.
# Weights are computed from training *window* counts after proportional
# multi-window expansion (not raw sequence counts), so they are hardcoded
# here to match the values stated in the thesis text (Section 3.5.2).
weights = [0.93, 2.40, 0.66]  # w_fan, w_tro, w_neu

# ── layout ────────────────────────────────────────────────────────────────────
x      = np.arange(len(classes))
width  = 0.34
gap    = 0.06

TRAIN_C = "#4A90D9"   # blue
VAL_C   = "#E8873A"   # orange

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("white")
ax2 = ax.twinx()   # secondary axis for class weights

bars_train = ax.bar(x - width / 2 - gap / 2, train, width,
                    label="Train", color=TRAIN_C, edgecolor="white", linewidth=0.6)
bars_val   = ax.bar(x + width / 2 + gap / 2, val,   width,
                    label="Validation", color=VAL_C, edgecolor="white", linewidth=0.6)

# ── log scale ─────────────────────────────────────────────────────────────────
ax.set_yscale("log")
ax.set_ylim(8, 5_000)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f"{int(v):,}" if v >= 1 else ""
))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))

# ── value labels on top of each bar ──────────────────────────────────────────
def label_bars(bars, values):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.12,
            f"{val:,}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#1B2631",
        )

label_bars(bars_train, train)
label_bars(bars_val,   val)

# ── secondary axis — class weights ────────────────────────────────────────────
WEIGHT_C = "#C0392B"   # dark red

ax2.plot(x, weights, color=WEIGHT_C, linewidth=1.8,
         linestyle="--", marker="D", markersize=9,
         markerfacecolor=WEIGHT_C, markeredgecolor="white",
         markeredgewidth=1.2, zorder=5, label="Class weight")

# weight labels with leader lines — (text_x, text_y, ha, va)
label_specs = [
    (0.0,  1.20, "center", "bottom"),  # Fanning     — above diamond
    (1.0,  2.70, "center", "bottom"),  # Trophallaxis — above diamond
    (2.0,  0.85, "center", "bottom"),  # Neutral      — above diamond
]
for (xi, w), (tx, ty, ha, va) in zip(zip(x, weights), label_specs):
    ax2.annotate(
        f"w = {w:.2f}",
        xy=(xi, w), xytext=(tx, ty),
        ha=ha, va=va,
        fontsize=8.5, fontweight="bold", color=WEIGHT_C,
        arrowprops=dict(arrowstyle="-", color=WEIGHT_C, lw=1.1),
        annotation_clip=False,
    )

ax2.set_ylabel("Class weight  (loss weighting)", fontsize=10, color=WEIGHT_C)
ax2.tick_params(axis="y", labelcolor=WEIGHT_C)
ax2.set_ylim(0, 3.5)
ax2.spines["right"].set_color(WEIGHT_C)
ax2.spines["right"].set_linewidth(1.2)

# ── axes styling ──────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
ax.set_ylabel("Number of sequences  (log scale)", fontsize=10, color="#333")
ax.set_title("Temporal sequences per class — train vs. validation",
             fontsize=11, fontweight="bold", color="#1B2631", pad=10)
# combined legend for both axes
lines_bars = [plt.Rectangle((0,0),1,1, color=TRAIN_C),
              plt.Rectangle((0,0),1,1, color=VAL_C)]
lines_wt,  = ax2.plot([], [], color=WEIGHT_C, linestyle="--",
                       marker="D", markersize=7)
ax.legend(lines_bars + [lines_wt],
          ["Train sequences", "Validation sequences", "Class weight (right axis)"],
          fontsize=9, framealpha=0.9, loc="upper left")
ax.spines[["top"]].set_visible(False)
ax.grid(axis="y", which="major", linestyle="--", alpha=0.4)
ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2)

# ── imbalance annotation ──────────────────────────────────────────────────────
ratio = train[2] / (train[0] + train[1])
ax.text(
    x[2], 3_400,
    f"Neutral is ×{ratio:.1f} more frequent than\n fanning + trophallaxis combined",
    ha="center", va="bottom",
    fontsize=6.5, fontweight="bold", color="#1B2631",
)

plt.tight_layout(pad=1.0)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
