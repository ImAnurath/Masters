"""
Evaluates best_temporal_buf16.pt on the validation set and saves:
  - confusion_buf16.png  (replaces the one from the mistaken second run)
  - prints the full classification report

Run:  python eval_best_model.py
"""

import os
import re
import pickle
import random
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = "D:/Projects/Masters"
FEATURES_PATH = os.path.join(BASE_DIR, "ActionRecognition/features.pkl")
DATASET_DIR   = os.path.join(BASE_DIR, "Data/AR_merged_dataset")
CKPT_PATH     = os.path.join(BASE_DIR, "Trained_Models/best_temporal_buf16.pt")
OUT_DIR       = os.path.join(BASE_DIR, "ActionRecognition")

BUFFER_SIZE = 16
VAL_STRIDE  = 8
BATCH_SIZE  = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── model (must match training config exactly) ────────────────────────────────
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


# ── dataset ───────────────────────────────────────────────────────────────────
CLASS_TO_IDX = {"fanning": 0, "neutral": 1, "trophallaxis": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
_RE_ACTION   = re.compile(r"^(fanning|trophallaxis)_(.+?)_(\d{5})_(\d+)$")
_RE_NEUTRAL  = re.compile(r"^neutral_(.+?)_tile(\d+)_(\d+)_(\d+)$")


def parse_stem(stem):
    m = _RE_ACTION.match(stem)
    if m:
        return CLASS_TO_IDX[m.group(1)], (m.group(2), m.group(4)), int(m.group(3))
    m = _RE_NEUTRAL.match(stem)
    if m:
        return CLASS_TO_IDX["neutral"], (m.group(1), m.group(2)), int(m.group(4))
    return None, None, None


def build_sequences(features_split, split_dir, min_len=2):
    groups = defaultdict(list)
    for label in sorted(os.listdir(split_dir)):
        if label not in CLASS_TO_IDX:
            continue
        for fname in os.listdir(os.path.join(split_dir, label)):
            stem = os.path.splitext(fname)[0]
            if stem not in features_split:
                continue
            lbl, key, sort_key = parse_stem(stem)
            if key is not None:
                groups[(lbl, key)].append((sort_key, stem))
    seqs = []
    for (lbl, _), items in groups.items():
        stems = [s for _, s in sorted(items)]
        if len(stems) >= min_len:
            seqs.append((lbl, stems))
    return seqs


class ValDataset(Dataset):
    def __init__(self, features, sequences, buffer_size, stride):
        self.features = features
        self.samples  = []
        for lbl, stems in sequences:
            N = len(stems)
            if N < buffer_size:
                self.samples.append((lbl, stems, 0))
            else:
                for start in range(0, N - buffer_size + 1, stride):
                    self.samples.append((lbl, stems, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lbl, stems, start = self.samples[idx]
        T = BUFFER_SIZE
        N = len(stems)
        feats, mask = [], []
        for i in range(T):
            j = start + i
            if j < N:
                feats.append(torch.from_numpy(self.features[stems[j]].copy()))
                mask.append(False)
            else:
                feats.append(torch.from_numpy(self.features[stems[N - 1]].copy()))
                mask.append(True)
        return (torch.stack(feats),
                torch.tensor(lbl, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool))


# ── load everything ───────────────────────────────────────────────────────────
print("Loading features ...")
with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)
print(f"  val features: {len(features['val']):,}")

print("Building val sequences ...")
val_seqs = build_sequences(features["val"], os.path.join(DATASET_DIR, "val"))
counts   = Counter(IDX_TO_CLASS[l] for l, _ in val_seqs)
print(f"  {dict(counts)}")

val_ds     = ValDataset(features["val"], val_seqs, BUFFER_SIZE, VAL_STRIDE)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"  val windows: {len(val_ds):,}")

print(f"\nLoading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)
print(f"  epoch={ckpt['epoch']}  val_f1={ckpt['val_f1']:.4f}")

cfg   = ckpt["config"]
model = TemporalSequenceClassifier(
    feature_dim = cfg["feature_dim"],
    num_classes  = cfg["num_classes"],
    num_heads    = cfg["num_heads"],
    num_layers   = cfg["num_layers"],
    dropout      = cfg["dropout"],
    max_seq_len  = cfg["buffer_size"] + 1,
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── evaluate ──────────────────────────────────────────────────────────────────
print("\nRunning evaluation ...")
all_preds, all_labels = [], []
with torch.no_grad():
    for feats, labels, masks in val_loader:
        logits = model(feats.to(device), padding_mask=masks.to(device))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

classes = ["fanning", "neutral", "trophallaxis"]
print("\n" + "="*55)
print("CLASSIFICATION REPORT — best_temporal_buf16")
print("="*55)
print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

# ── confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True", fontsize=11)
ax.set_title(f"Confusion Matrix  —  Buffer = {BUFFER_SIZE} frames", fontsize=12)
plt.tight_layout()

out = os.path.join(OUT_DIR, f"confusion_buf{BUFFER_SIZE}.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")
