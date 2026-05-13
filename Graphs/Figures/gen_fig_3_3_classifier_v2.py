"""
Fig 3.3 v2 — Temporal Sequence Classifier: architecture walk-through.
Redesigned for readability: larger fonts, clear step badges, simplified Step 4.

Run:    python gen_fig_3_3_classifier_v2.py
Output: fig_3_3_classifier_v2.png  (200 dpi, white background)
"""

import re, pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("D:/Projects/Masters")
FEATURES_PATH = BASE_DIR / "ActionRecognition/features.pkl"
DATASET_DIR   = BASE_DIR / "Data/AR_merged_dataset"
CKPT_PATH     = BASE_DIR / "Trained_Models/best_temporal_buf16.pt"
OUT           = Path(__file__).parent / "fig_3_3_classifier_v2.png"

BUFFER_SIZE  = 16
TARGET_CLASS = "fanning"

# ── model (unchanged from training) ───────────────────────────────────────────
class TemporalSequenceClassifier(nn.Module):
    def __init__(self, feature_dim=384, num_classes=3, num_heads=4,
                 num_layers=2, dropout=0.3, max_seq_len=33):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.pos_embed = nn.Embedding(max_seq_len, feature_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=feature_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(feature_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward_stages(self, x):
        B, T, _ = x.shape
        cls   = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat([cls, x], dim=1)
        pos   = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x_emb = x_emb + self.pos_embed(pos)
        x_out = self.transformer(x_emb)
        logits = self.classifier(self.norm(x_out[:, 0]))
        return F.softmax(logits, dim=-1), x_emb, x_out


# ── stem parsing ──────────────────────────────────────────────────────────────
_RE_ACTION  = re.compile(r"^(fanning|trophallaxis)_(.+?)_(\d{5})_(\d+)$")
_RE_NEUTRAL = re.compile(r"^neutral_(.+?)_tile(\d+)_(\d+)_(\d+)$")
CLASS_TO_IDX = {"fanning": 0, "neutral": 1, "trophallaxis": 2}

def parse_stem(stem):
    m = _RE_ACTION.match(stem)
    if m:
        return CLASS_TO_IDX[m.group(1)], (m.group(2), m.group(4)), int(m.group(3))
    m = _RE_NEUTRAL.match(stem)
    if m:
        return CLASS_TO_IDX["neutral"], (m.group(1), m.group(2)), int(m.group(4))
    return None, None, None

def find_best_tracklet(feat_dict, cls_dir, min_len=BUFFER_SIZE):
    groups = defaultdict(list)
    for fpath in sorted(cls_dir.iterdir()):
        if not fpath.is_file():
            continue
        stem = fpath.stem
        if stem not in feat_dict:
            continue
        _, key, sort_key = parse_stem(stem)
        if key is not None:
            groups[key].append((sort_key, stem, fpath))
    candidates = [v for v in groups.values() if len(v) >= min_len]
    if not candidates:
        return None
    return max(candidates, key=len)


# ── load data + model ─────────────────────────────────────────────────────────
print("Loading features.pkl …")
with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)
feat_val = features["val"]

print(f"Finding a {TARGET_CLASS} tracklet …")
cls_dir  = DATASET_DIR / "val" / TARGET_CLASS
tracklet = find_best_tracklet(feat_val, cls_dir)
if tracklet is None:
    raise ValueError(f"No {TARGET_CLASS} tracklet with ≥{BUFFER_SIZE} frames found.")
tracklet.sort(key=lambda item: item[0])
mid      = max(0, len(tracklet) // 2 - BUFFER_SIZE // 2)
seq      = tracklet[mid : mid + BUFFER_SIZE]
print(f"  Tracklet length: {len(tracklet)}  →  using frames {mid}–{mid + BUFFER_SIZE - 1}")

print("Loading model …")
device = torch.device("cpu")
ckpt   = torch.load(CKPT_PATH, map_location=device)
cfg    = ckpt["config"]
model  = TemporalSequenceClassifier(
    feature_dim=cfg["feature_dim"], num_classes=cfg["num_classes"],
    num_heads=cfg["num_heads"],     num_layers=cfg["num_layers"],
    dropout=cfg["dropout"],         max_seq_len=cfg["buffer_size"] + 1,
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

feat_tensor = torch.stack([
    torch.from_numpy(feat_val[stem].copy()) for _, stem, _ in seq
]).unsqueeze(0)

with torch.no_grad():
    probs, x_emb, x_out = model.forward_stages(feat_tensor)

probs_np = probs[0].numpy()
feat_np  = feat_tensor[0].numpy()
emb_np   = x_emb[0].numpy()
out_np   = x_out[0].numpy()

nrm     = out_np / (np.linalg.norm(out_np, axis=-1, keepdims=True) + 1e-8)
sim_map = nrm @ nrm.T

pred_idx = int(probs_np.argmax())
CLASSES  = ["Fanning", "Neutral", "Trophallaxis"]
print(f"  Predicted: {CLASSES[pred_idx]}  ({probs_np[pred_idx]:.1%})")

def load_crop(path, size=88):
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS))

show_idx  = list(range(0, 16, 2))
crop_imgs = [load_crop(seq[i][2]) for i in show_idx]


# ── colour palette ────────────────────────────────────────────────────────────
C_DARK  = "#2C3E50"
C_BG    = "#F8F9FA"

STEP_COLORS = ["#2471A3", "#1A8754", "#7D3C98", "#C0392B"]

CLS_FACE  = ["#AED6F1", "#A9DFBF", "#F9E79F"]
CLS_EDGE  = ["#2471A3", "#1A8754", "#D4AC0D"]

def pct_range(arr):
    return np.percentile(arr, 2), np.percentile(arr, 98)

feat_vmin, feat_vmax = pct_range(feat_np)
emb_vmin,  emb_vmax  = pct_range(emb_np)


# ── layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 21), facecolor="white")
gs  = gridspec.GridSpec(
    9, 1,
    height_ratios=[0.22, 2.2, 0.52, 1.65, 0.52, 1.65, 0.52, 2.3, 0.22],
    hspace=0.02,
    left=0.03, right=0.97, top=0.990, bottom=0.006,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def panel_bg(ax, step_idx):
    """Light tinted background per step."""
    tints = ["#EBF5FB", "#EAFAF1", "#F5EEF8", "#FDEDEC"]
    ax.set_facecolor(tints[step_idx])
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
        sp.set_linewidth(1.4)
    ax.set_xticks([])
    ax.set_yticks([])


def step_badge(ax, number, title, subtitle="", step_idx=0):
    """Draw numbered badge + title at the top of a panel."""
    color = STEP_COLORS[step_idx]
    # circle badge
    ax.text(0.025, 0.965, f" {number} ",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.28", facecolor=color, edgecolor="none"))
    # title
    ax.text(0.058, 0.965, title,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12.5, fontweight="bold", color=C_DARK)
    if subtitle:
        ax.text(0.058, 0.905, subtitle,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.8, color="#666666", style="italic")


def annotation_box(ax, text, x=0.98, y=0.04, ha="right"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="bottom",
            fontsize=8.5, color="#444444", style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.88, edgecolor="#CCCCCC", linewidth=0.9))


def arrow_row(ax, label=""):
    ax.axis("off")
    ax.annotate("", xy=(0.5, 0.10), xytext=(0.5, 0.90),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C_DARK,
                                lw=2.0, mutation_scale=20))
    if label:
        ax.text(0.53, 0.50, label,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=9.5, color="#444",
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec="#cccccc", lw=0.9))


def heatmap_panel(parent_ax, bounds, data, cmap, vmin, vmax,
                  ytick_labels, xtick_vals, xlabel, cbar_label, cbar_bounds):
    inset = parent_ax.inset_axes(tuple(bounds))
    im    = inset.imshow(data, aspect="auto", cmap=cmap,
                         vmin=vmin, vmax=vmax, interpolation="nearest")
    inset.set_yticks(range(len(ytick_labels)))
    inset.set_yticklabels(ytick_labels, fontsize=7.5)
    inset.set_xticks(xtick_vals)
    inset.set_xticklabels([str(v) for v in xtick_vals], fontsize=7.5)
    inset.set_xlabel(xlabel, fontsize=9, labelpad=3)
    inset.tick_params(length=2, pad=2)
    cax = parent_ax.inset_axes(tuple(cbar_bounds))
    cb  = fig.colorbar(im, cax=cax, label=cbar_label)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(cbar_label, fontsize=7.5)
    return inset


# ── ROW 0 — figure title ──────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.text(0.5, 0.5,
         f"Fig 3.3  ·  Temporal Sequence Classifier — Forward Pass  "
         f"({TARGET_CLASS.capitalize()} bee,  T = 16 frames)",
         transform=ax0.transAxes, ha="center", va="center",
         fontsize=13.5, fontweight="bold", color=C_DARK)


# ── ROW 1 — Step 1: bee crop strip ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
panel_bg(ax1, 0)
step_badge(ax1, "1", "Input — Bee Tracklet",
           r"16 consecutive crops from one tracked bee  (8 shown; every other frame)",
           step_idx=0)

n_show = len(crop_imgs)
pad_l  = 0.025
pad_r  = 0.025
cell_w = (1.0 - pad_l - pad_r) / n_show
th     = 0.60
y_bot  = 0.12

for i, img in enumerate(crop_imgs):
    x0 = pad_l + i * cell_w
    ax_in = ax1.inset_axes((x0 + 0.003, y_bot, cell_w - 0.006, th))
    ax_in.imshow(img)
    ax_in.axis("off")
    ax1.add_patch(FancyBboxPatch(
        (x0 + 0.002, y_bot - 0.004), cell_w - 0.004, th + 0.008,
        boxstyle="square,pad=0", linewidth=1.3,
        edgecolor="#999999", facecolor="none",
        transform=ax1.transAxes, zorder=5,
    ))
    ax1.text(x0 + cell_w / 2, y_bot - 0.065,
             f"t = {show_idx[i] + 1}",
             ha="center", va="top", fontsize=8, color="#555555",
             transform=ax1.transAxes)

annotation_box(ax1,
    "Each crop: 224×224 px padded region from the hive frame,\n"
    "centered on one ByteTracker-assigned bee identity")


# ── ROW 2 — arrow ─────────────────────────────────────────────────────────────
arrow_row(fig.add_subplot(gs[2]),
          "DINOv2-small (frozen)  →  384-d CLS token per crop")


# ── ROW 3 — Step 2: feature matrix ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
panel_bg(ax3, 1)
step_badge(ax3, "2",
           r"DINOv2 Feature Sequence   $\mathbf{X} \in \mathbb{R}^{16 \times 384}$",
           "Each row is one frame's 384-d CLS token — the backbone's semantic summary of that crop",
           step_idx=1)

feat_inset = heatmap_panel(
    ax3,
    bounds       = [0.020, 0.07, 0.940, 0.72],
    data         = feat_np,
    cmap         = "RdBu_r",
    vmin         = feat_vmin, vmax=feat_vmax,
    ytick_labels = [f"t = {i+1}" for i in range(16)],
    xtick_vals   = [0, 64, 128, 192, 256, 320, 383],
    xlabel       = "Feature dimension (0 – 383)",
    cbar_label   = "activation",
    cbar_bounds  = [0.963, 0.07, 0.013, 0.72],
)
annotation_box(ax3,
    "Columns = learned visual dimensions from DINOv2 self-supervised pretraining.\n"
    "Similar rows = visually similar frames within the fanning sequence.")


# ── ROW 4 — arrow ─────────────────────────────────────────────────────────────
arrow_row(fig.add_subplot(gs[4]),
          "Prepend learnable [CLS] token  +  add learnable positional embeddings  →  17 tokens")


# ── ROW 5 — Step 3: token matrix ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[5])
panel_bg(ax5, 2)
step_badge(ax5, "3",
           r"Token Sequence   $\mathbf{X'} \in \mathbb{R}^{17 \times 384}$",
           "[CLS] token prepended (row 0) + positional embeddings added to all rows",
           step_idx=2)

yticks_emb = ["[CLS]"] + [f"t = {i+1}" for i in range(16)]
emb_inset = heatmap_panel(
    ax5,
    bounds       = [0.020, 0.07, 0.940, 0.72],
    data         = emb_np,
    cmap         = "RdBu_r",
    vmin         = emb_vmin, vmax=emb_vmax,
    ytick_labels = yticks_emb,
    xtick_vals   = [0, 64, 128, 192, 256, 320, 383],
    xlabel       = "Feature dimension (0 – 383)",
    cbar_label   = "activation",
    cbar_bounds  = [0.963, 0.07, 0.013, 0.72],
)
# highlight CLS row
emb_inset.add_patch(Rectangle(
    (-0.5, -0.5), 384, 1.0,
    linewidth=2.5, edgecolor="#C0392B", facecolor="#FADBD840", zorder=5,
))
emb_inset.get_yticklabels()[0].set_color("#C0392B")
emb_inset.get_yticklabels()[0].set_fontweight("bold")
emb_inset.get_yticklabels()[0].set_fontsize(8.5)

annotation_box(ax5,
    "[CLS] (row 0, red) will aggregate context across all 16 frames\n"
    "via self-attention and carry the final classification signal.")


# ── ROW 6 — arrow ─────────────────────────────────────────────────────────────
arrow_row(fig.add_subplot(gs[6]),
          "2 × Transformer Encoder Block  "
          "(4-head self-attention  ·  FFN dim = 768  ·  Pre-LayerNorm  ·  Dropout 0.3)")


# ── ROW 7 — Step 4: similarity map + probability bars ─────────────────────────
ax7 = fig.add_subplot(gs[7])
panel_bg(ax7, 3)
step_badge(ax7, "4",
           "Transformer Output  →  Classification",
           "Left: token cosine-similarity after 2 encoder blocks  ·  "
           "Right: MLP head softmax output",
           step_idx=3)

# ── 7a: similarity map (left 36%) ─────────────────────────────────────────────
sim_inset = ax7.inset_axes((0.025, 0.09, 0.320, 0.78))
im_sim = sim_inset.imshow(sim_map, aspect="auto", cmap="Blues",
                           vmin=0.3, vmax=1.0, interpolation="nearest")
tick_pos    = list(range(0, 17, 4))
tick_labels = ["CLS"] + [str(i) for i in range(4, 17, 4)]
sim_inset.set_xticks(tick_pos)
sim_inset.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha="right")
sim_inset.set_yticks(tick_pos)
sim_inset.set_yticklabels(tick_labels, fontsize=8)
sim_inset.set_title("Token cosine-similarity  (17 × 17)\nProxy for self-attention structure",
                    fontsize=8.5, pad=4, color=C_DARK)
sim_inset.tick_params(length=2, pad=1)
# CLS row/column highlight
for spine in sim_inset.spines.values():
    spine.set_edgecolor("#AAAAAA")
cax_sim = ax7.inset_axes((0.350, 0.09, 0.012, 0.78))
cb_sim  = fig.colorbar(im_sim, cax=cax_sim)
cb_sim.ax.tick_params(labelsize=7)

# ── 7b: probability bars (right 57%) ──────────────────────────────────────────
prob_ax = ax7.inset_axes((0.390, 0.10, 0.590, 0.78))

bar_colors = [CLS_FACE[i] for i in range(3)]
bar_edges  = [CLS_EDGE[i] for i in range(3)]
y_pos      = [2, 1, 0]   # trophallaxis=2, neutral=1, fanning=0 (visual order top-to-bottom)
bar_vals   = [probs_np[2], probs_np[1], probs_np[0]]
b_colors   = [CLS_FACE[2], CLS_FACE[1], CLS_FACE[0]]
b_edges    = [CLS_EDGE[2], CLS_EDGE[1], CLS_EDGE[0]]
b_labels   = ["Trophallaxis", "Neutral", "Fanning"]

bars = prob_ax.barh(y_pos, bar_vals,
                    color=b_colors, edgecolor=b_edges,
                    linewidth=1.8, height=0.62)

# bold label + percentage on each bar
for bar, prob, lbl, ec in zip(bars, bar_vals, b_labels, b_edges):
    # percentage inside/after bar
    prob_ax.text(
        min(prob + 0.03, 1.05),
        bar.get_y() + bar.get_height() / 2,
        f"{prob:.1%}",
        va="center", ha="left",
        fontsize=13, fontweight="bold", color=ec,
    )
    # class name on the left
    prob_ax.text(
        -0.02, bar.get_y() + bar.get_height() / 2,
        lbl, va="center", ha="right",
        fontsize=12, fontweight="bold", color=C_DARK,
    )

# highlight predicted bar with a thick border
pred_bar_y = 2 - pred_idx   # convert class index to bar y position
prob_ax.add_patch(Rectangle(
    (0, pred_bar_y - 0.31), probs_np[pred_idx], 0.62,
    linewidth=3.0, edgecolor=CLS_EDGE[pred_idx],
    facecolor="none", zorder=5,
))

prob_ax.set_xlim(-0.25, 1.22)
prob_ax.set_ylim(-0.5, 2.5)
prob_ax.axis("off")
prob_ax.set_title(
    f"Predicted:  {CLASSES[pred_idx]}  ({probs_np[pred_idx]:.1%})",
    fontsize=12.5, fontweight="bold",
    color=CLS_EDGE[pred_idx], pad=8,
)


# ── ROW 8 — footnote ──────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[8])
ax8.axis("off")
ax8.text(
    0.5, 0.55,
    "The [CLS] token output  z₀ ∈ ℝ³⁸⁴  is passed through "
    "LayerNorm → Linear(384→128) → GELU → Dropout(0.3) → Linear(128→3) → Softmax "
    "to produce the final 3-class behavior prediction.",
    transform=ax8.transAxes, ha="center", va="center",
    fontsize=8.5, color="#666666", style="italic",
)

# ── save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close()
print(f"Saved: {OUT}")
