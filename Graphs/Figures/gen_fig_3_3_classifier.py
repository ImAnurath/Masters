"""
Fig 3.3 — Temporal Sequence Classifier: step-by-step visual walkthrough.
Shows a real fanning bee tracklet flowing through each pipeline stage with
actual images and tensor visualisations at every step.

Run:    python gen_fig_3_3_classifier.py
Output: fig_3_3_classifier.png  (200 dpi, white background)
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
OUT           = Path(__file__).parent / "fig_3_3_classifier.png"

BUFFER_SIZE  = 16
TARGET_CLASS = "fanning"

# ── model ─────────────────────────────────────────────────────────────────────
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

    def forward(self, x, padding_mask=None):
        B, T, _ = x.shape
        cls  = self.cls_token.expand(B, -1, -1)
        x    = torch.cat([cls, x], dim=1)
        pos  = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x    = x + self.pos_embed(pos)
        if padding_mask is not None:
            full_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=x.device), padding_mask
            ], dim=1)
        else:
            full_mask = None
        x = self.transformer(x, src_key_padding_mask=full_mask)
        return self.classifier(self.norm(x[:, 0]))

    def forward_stages(self, x):
        """Returns (probs, embedded_tokens, transformer_output)."""
        B, T, _ = x.shape
        cls   = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat([cls, x], dim=1)             # (B, T+1, D)
        pos   = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x_emb = x_emb + self.pos_embed(pos)            # (B, T+1, D) after pos-embed
        x_out = self.transformer(x_emb)                 # (B, T+1, D) after transformer
        logits = self.classifier(self.norm(x_out[:, 0]))
        return F.softmax(logits, dim=-1), x_emb, x_out


# ── stem parsing (matches eval_best_model.py) ─────────────────────────────────
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
    """Return the longest tracklet (list of (sort_key, stem, path)) that has ≥ min_len frames."""
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


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading features.pkl …")
with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)
feat_val = features["val"]

print(f"Finding a {TARGET_CLASS} tracklet in the val split …")
cls_dir  = DATASET_DIR / "val" / TARGET_CLASS
tracklet = find_best_tracklet(feat_val, cls_dir)
if tracklet is None:
    raise ValueError(f"No {TARGET_CLASS} tracklet with ≥{BUFFER_SIZE} frames found in val set.")
tracklet.sort(key=lambda item: item[0])

mid = max(0, len(tracklet) // 2 - BUFFER_SIZE // 2)
seq = tracklet[mid : mid + BUFFER_SIZE]
print(f"  Total tracklet length: {len(tracklet)} frames  →  using frames {mid}–{mid + BUFFER_SIZE - 1}")

# ── load model ────────────────────────────────────────────────────────────────
print("Loading model …")
device = torch.device("cpu")
ckpt   = torch.load(CKPT_PATH, map_location=device)
cfg    = ckpt["config"]
model  = TemporalSequenceClassifier(
    feature_dim=cfg["feature_dim"],
    num_classes=cfg["num_classes"],
    num_heads=cfg["num_heads"],
    num_layers=cfg["num_layers"],
    dropout=cfg["dropout"],
    max_seq_len=cfg["buffer_size"] + 1,
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── run inference and capture intermediate tensors ────────────────────────────
feat_tensor = torch.stack([
    torch.from_numpy(feat_val[stem].copy()) for _, stem, _ in seq
]).unsqueeze(0)   # (1, 16, 384)

with torch.no_grad():
    probs, x_emb, x_out = model.forward_stages(feat_tensor)

probs_np = probs[0].numpy()          # (3,)
feat_np  = feat_tensor[0].numpy()    # (16, 384)  raw DINOv2 features
emb_np   = x_emb[0].numpy()         # (17, 384)  after CLS prepend + pos-embed
out_np   = x_out[0].numpy()         # (17, 384)  after 2× transformer blocks

# cosine-similarity map used as a proxy for self-attention
nrm      = out_np / (np.linalg.norm(out_np, axis=-1, keepdims=True) + 1e-8)
sim_map  = nrm @ nrm.T              # (17, 17)

pred_idx = int(probs_np.argmax())
print(f"  fanning={probs_np[0]:.3f}  neutral={probs_np[1]:.3f}  trophallaxis={probs_np[2]:.3f}")
print(f"  → Predicted: {['fanning','neutral','trophallaxis'][pred_idx]}")

# ── load bee crop thumbnails (show 8 = every other frame) ─────────────────────
def load_crop(path, size=80):
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS))

show_idx  = list(range(0, 16, 2))   # t=1,3,5,7,9,11,13,15
crop_imgs = [load_crop(seq[i][2]) for i in show_idx]

# ── colour palette ────────────────────────────────────────────────────────────
C_DARK   = "#2C3E50"
C_BG     = ["#EBF5FB", "#EBF5FB", "#EDE7F6", "#EDE7F6"]   # step panel backgrounds
C_FAN_F  = "#AED6F1";  C_FAN_E  = "#2980B9"
C_NEU_F  = "#A9DFBF";  C_NEU_E  = "#27AE60"
C_TRO_F  = "#F9E79F";  C_TRO_E  = "#F39C12"
CLASSES  = ["Fanning", "Neutral", "Trophallaxis"]
CLS_FC   = [C_FAN_F, C_NEU_F, C_TRO_F]
CLS_EC   = [C_FAN_E, C_NEU_E, C_TRO_E]

# colourmap range: clip at 2nd / 98th percentile for nicer heatmaps
def pct_range(arr):
    return np.percentile(arr, 2), np.percentile(arr, 98)

feat_vmin, feat_vmax = pct_range(feat_np)
emb_vmin,  emb_vmax  = pct_range(emb_np)

# ── figure layout ─────────────────────────────────────────────────────────────
# rows:
#   0  figure title
#   1  Step 1 — bee crop strip
#   2  arrow
#   3  Step 2 — 16×384 feature matrix
#   4  arrow
#   5  Step 3 — 17×384 token matrix (CLS highlighted)
#   6  arrow
#   7  Step 4 — similarity map | CLS vector profile | probability bars
#   8  footnote

fig = plt.figure(figsize=(13, 20), facecolor="white")
gs  = gridspec.GridSpec(
    9, 1,
    height_ratios=[0.30, 2.3, 0.50, 1.55, 0.50, 1.55, 0.50, 2.2, 0.28],
    hspace=0.02,
    left=0.03, right=0.97, top=0.985, bottom=0.008,
)


def panel_style(ax, bg):
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_DARK)
        sp.set_linewidth(1.7)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_title(ax, title, sub=""):
    ax.text(0.5, 0.975, title,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=C_DARK)
    if sub:
        ax.text(0.5, 0.900, sub,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.2, color="#555555", style="italic")


def arrow_row(ax, label=""):
    ax.axis("off")
    ax.annotate("", xy=(0.5, 0.08), xytext=(0.5, 0.92),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C_DARK,
                                lw=1.9, mutation_scale=19))
    if label:
        ax.text(0.52, 0.50, label,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8.5, color="#555",  style="italic",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.8))


def heatmap_inset(parent_ax, bounds, data, cmap, vmin, vmax,
                  ytick_labels, xtick_vals, xlabel,
                  cbar_label, cbar_bounds, title=""):
    inset = parent_ax.inset_axes(tuple(bounds))
    im    = inset.imshow(data, aspect="auto", cmap=cmap,
                         vmin=vmin, vmax=vmax, interpolation="nearest")
    inset.set_yticks(range(len(ytick_labels)))
    inset.set_yticklabels(ytick_labels, fontsize=6.5)
    inset.set_xticks(xtick_vals)
    inset.set_xticklabels([str(v) for v in xtick_vals], fontsize=6.5)
    inset.set_xlabel(xlabel, fontsize=7.5, labelpad=2)
    inset.tick_params(length=2, pad=2)
    if title:
        inset.set_title(title, fontsize=7.5, pad=3, color=C_DARK)
    cax = parent_ax.inset_axes(tuple(cbar_bounds))
    fig.colorbar(im, cax=cax, label=cbar_label)
    cax.tick_params(labelsize=6)
    return inset, im


# ── ROW 0 — figure title ──────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.text(0.5, 0.5,
    f"Fig 3.3  ·  Temporal Sequence Classifier  —  {TARGET_CLASS.capitalize()} Bee  (T = 16 frames)",
    transform=ax0.transAxes, ha="center", va="center",
    fontsize=13, fontweight="bold", color=C_DARK)

# ── ROW 1 — Step 1: bee crop strip ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
panel_style(ax1, C_BG[0])
panel_title(ax1, "Step 1   Bee Tracklet Input",
    r"$\mathbf{X}_{raw}$: 16 consecutive bee-crop frames  (every other frame shown; "
    r"$T = 16$, total sequence duration ≈ 0.5 s @ 30 fps)")

n_show  = len(crop_imgs)      # 8
pad_l   = 0.025
pad_r   = 0.025
usable  = 1.0 - pad_l - pad_r
cell_w  = usable / n_show
thumb_h = 0.66
y_bot   = 0.09

for i, img in enumerate(crop_imgs):
    x_cell = pad_l + i * cell_w
    # thumbnail
    ax_in = ax1.inset_axes((x_cell + 0.003, y_bot, cell_w - 0.006, thumb_h))
    ax_in.imshow(img)
    ax_in.axis("off")
    # border
    ax1.add_patch(FancyBboxPatch(
        (x_cell + 0.002, y_bot - 0.005),
        cell_w - 0.004, thumb_h + 0.01,
        boxstyle="square,pad=0", linewidth=1.2,
        edgecolor="#888", facecolor="none",
        transform=ax1.transAxes, zorder=5,
    ))
    # frame label below
    ax1.text(x_cell + cell_w / 2, y_bot - 0.06,
             f"t = {show_idx[i] + 1}",
             ha="center", va="top", fontsize=7, color="#444",
             transform=ax1.transAxes)

# ── ROW 2 — arrow ─────────────────────────────────────────────────────────────
arrow_row(fig.add_subplot(gs[2]),
          "DINOv2 ViT-S/14  (frozen)  →  CLS embedding per frame  (dim = 384)")

# ── ROW 3 — Step 2: 16×384 feature matrix ─────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
panel_style(ax3, C_BG[1])
panel_title(ax3, "Step 2   DINOv2 Feature Sequence",
    r"$\mathbf{X} \in \mathbb{R}^{16 \times 384}$  — each row is one frame's patch-level CLS embedding")

feat_inset, _ = heatmap_inset(
    ax3,
    bounds     = [0.020, 0.07, 0.935, 0.74],
    data       = feat_np,
    cmap       = "RdBu_r",
    vmin       = feat_vmin, vmax=feat_vmax,
    ytick_labels = [f"t = {i+1}" for i in range(16)],
    xtick_vals   = [0, 64, 128, 192, 256, 320, 383],
    xlabel       = "Feature dimension",
    cbar_label   = "value",
    cbar_bounds  = [0.960, 0.07, 0.014, 0.74],
)

# ── ROW 4 — arrow ─────────────────────────────────────────────────────────────
arrow_row(fig.add_subplot(gs[4]),
          "prepend learnable [CLS] token  +  add positional embedding  (17 tokens total)")

# ── ROW 5 — Step 3: 17×384 token matrix ──────────────────────────────────────
ax5 = fig.add_subplot(gs[5])
panel_style(ax5, C_BG[2])
panel_title(ax5, "Step 3   Token Sequence with [CLS]",
    r"$\mathbf{X}' \in \mathbb{R}^{17 \times 384}$  — row 0 = [CLS] token  ·  rows 1–16 = frame tokens + positional embedding")

yticks_emb = ["[CLS]"] + [f"t = {i+1}" for i in range(16)]
emb_inset, _ = heatmap_inset(
    ax5,
    bounds     = [0.020, 0.07, 0.935, 0.74],
    data       = emb_np,
    cmap       = "RdBu_r",
    vmin       = emb_vmin, vmax=emb_vmax,
    ytick_labels = yticks_emb,
    xtick_vals   = [0, 64, 128, 192, 256, 320, 383],
    xlabel       = "Feature dimension",
    cbar_label   = "value",
    cbar_bounds  = [0.960, 0.07, 0.014, 0.74],
)

# highlight CLS row (row 0) with a red rectangle in data coordinates
emb_inset.add_patch(Rectangle(
    (-0.5, -0.5), 384, 1.0,
    linewidth=2.5, edgecolor="#E74C3C", facecolor="#FADBD820", zorder=5,
))
# colour [CLS] y-tick label red
emb_inset.get_yticklabels()[0].set_color("#E74C3C")
emb_inset.get_yticklabels()[0].set_fontweight("bold")

# ── ROW 6 — arrow ─────────────────────────────────────────────────────────────
arrow_row(fig.add_subplot(gs[6]),
          "2 × Transformer Encoder Block  (4-head self-attention · FFN dim = 768 · LayerNorm-first · Dropout 0.3)")

# ── ROW 7 — Step 4: similarity map | CLS profile | probability bars ───────────
ax7 = fig.add_subplot(gs[7])
panel_style(ax7, C_BG[3])
panel_title(ax7, "Step 4   Transformer Output  →  Classification",
    "Left: token cosine-similarity  (proxy for self-attention, 17 × 17)   ·   "
    "Centre: z₀  [CLS] output vector (dim = 384)   ·   Right: softmax class probabilities")

# 7a — cosine-similarity map
sim_inset = ax7.inset_axes((0.02, 0.08, 0.30, 0.82))
im_sim = sim_inset.imshow(sim_map, aspect="auto", cmap="Blues",
                           vmin=0.3, vmax=1.0, interpolation="nearest")
sim_inset.set_xticks(list(range(0, 17, 4)))
sim_inset.set_xticklabels(["CLS"] + [str(i) for i in range(4, 17, 4)],
                           fontsize=6.5, rotation=45, ha="right")
sim_inset.set_yticks(list(range(0, 17, 4)))
sim_inset.set_yticklabels(["CLS"] + [str(i) for i in range(4, 17, 4)], fontsize=6.5)
sim_inset.set_title("Token cosine-similarity  (17 × 17)", fontsize=7.5, pad=3, color=C_DARK)
sim_inset.tick_params(length=2, pad=1)
cax_sim = ax7.inset_axes((0.325, 0.08, 0.012, 0.82))
fig.colorbar(im_sim, cax=cax_sim)
cax_sim.tick_params(labelsize=6)

# 7b — CLS token output vector (horizontal profile)
cls_vec   = out_np[0]    # (384,)
cls_inset = ax7.inset_axes((0.37, 0.08, 0.23, 0.82))
cls_inset.fill_betweenx(np.arange(384), 0, cls_vec, alpha=0.3, color="#2980B9")
cls_inset.plot(cls_vec, np.arange(384), color="#1A5276", linewidth=0.55, alpha=0.9)
cls_inset.axvline(0, color="#bbb", linewidth=0.8, linestyle="--")
cls_inset.set_ylim(383, 0)
cls_inset.set_title(r"$\mathbf{z}_0 \in \mathbb{R}^{384}$  (CLS output)",
                    fontsize=7.5, pad=3, color=C_DARK)
cls_inset.set_xlabel("Value", fontsize=7.5, labelpad=2)
cls_inset.set_ylabel("Dimension", fontsize=7.5, labelpad=2)
cls_inset.tick_params(labelsize=6.5, length=2)
for sp in cls_inset.spines.values():
    sp.set_edgecolor("#aaa")

# 7c — probability bars
prob_inset = ax7.inset_axes((0.64, 0.14, 0.35, 0.72))
bars = prob_inset.barh(
    [2, 1, 0], probs_np,
    color=CLS_FC, edgecolor=CLS_EC, linewidth=1.6, height=0.55,
)
prob_inset.set_xlim(0, 1.08)
prob_inset.set_yticks([2, 1, 0])
prob_inset.set_yticklabels(CLASSES, fontsize=10, fontweight="bold")
prob_inset.set_xlabel("Probability", fontsize=9, labelpad=3)
prob_inset.set_title(
    f"Predicted:  {CLASSES[pred_idx]}  ({probs_np[pred_idx]:.1%})",
    fontsize=9, fontweight="bold", color=CLS_EC[pred_idx], pad=7,
)
for bar, prob in zip(bars, probs_np):
    prob_inset.text(
        bar.get_width() + 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"{prob:.1%}", va="center", fontsize=9, fontweight="bold",
    )
prob_inset.spines["top"].set_visible(False)
prob_inset.spines["right"].set_visible(False)
prob_inset.tick_params(labelsize=8.5)
prob_inset.grid(axis="x", alpha=0.25, linestyle="--")

# ── ROW 8 — footnote ──────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[8])
ax8.axis("off")
ax8.text(
    0.5, 0.55,
    "The [CLS] token aggregates temporal context across all 16 frames and is "
    "passed to the MLP head  (LayerNorm → Linear(384→128) → GELU → Dropout → Linear(128→3))  "
    "for final 3-class prediction.",
    transform=ax8.transAxes, ha="center", va="center",
    fontsize=7.5, color="#666", style="italic",
)

# ── save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
