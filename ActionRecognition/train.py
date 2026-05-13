"""
Action Recognition Training — v3 (hardened)

Overfitting / overconfidence countermeasures vs v2:
  1. MixUp (α=0.4, 30% of batches)  — forces smooth decision boundaries,
     prevents the model from assigning near-1.0 confidence to memorised features
  2. Feature dropout (0-3 random frames zeroed)  — robustness to missing detections
  3. Stronger Gaussian noise (σ=0.05, was 0.02)
  4. Label smoothing (ε=0.1)  — directly caps maximum softmax output
  5. 3 Transformer layers + dropout=0.35  — more capacity, more regularisation
  6. LR = 1e-4 with 10-epoch linear warmup  — slower, more stable convergence
  7. Weight decay = 0.1  — stronger L2 regularisation
  8. Confidence calibration report at end  — reveals residual overconfidence

Each run saves to its own timestamped folder:
  runs/YYYYMMDD_HHMMSS/
    config.json          hyperparameters
    train_log.csv        per-epoch loss/F1
    best_model.pt        best val-F1 checkpoint
    confusion.png        confusion matrix
    curves.png           loss + F1 curves
    confidence_hist.png  softmax confidence distributions
    report.txt           sklearn classification report

Update TEMPORAL_PATH in graph_generator.ipynb to:
  D:/Projects/Masters/Tests/action/runs/<timestamp>/best_model.pt
"""

import os
import re
import json
import random
import pickle
import csv
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_DIR   = "D:/Projects/Masters/Data/AR_v2_dataset"
ACTION_DIR    = "D:/Projects/Masters/Tests/action"
FEATURES_PATH = os.path.join(ACTION_DIR, "features_v3.pkl")

BUFFER_SIZE = 16
MIN_SEQ_LEN = 4
VAL_STRIDE  = 8

FEATURE_DIM      = 768
NUM_CLASSES      = 3
NUM_HEADS        = 6
NUM_LAYERS       = 3       # 4 was underfitting trophallaxis (precision 0.88 vs 0.93)
DROPOUT          = 0.35    # 0.50 too aggressive for 66+21 tracked sequences

BATCH_SIZE       = 64
NUM_WORKERS      = 0
MAX_EPOCHS       = 150
PATIENCE         = 25
LR               = 1e-4    # lower than v2 (was 3e-4)
WEIGHT_DECAY     = 0.1     # stronger than v2 (was 0.05)
WARMUP_EPOCHS    = 10
LABEL_SMOOTHING  = 0.1

# Augmentation
NOISE_STD        = 0.05    # was 0.02
FRAME_DROP_MAX   = 3       # zero out up to N random frames
JITTER_MAX       = 3       # temporal jitter up to N frames
MIXUP_PROB       = 0.50    # fraction of batches with MixUp
MIXUP_ALPHA      = 0.60    # Beta distribution parameter
ENTROPY_WEIGHT   = 0.05    # penalise confident predictions (entropy regularisation)
RDROP_WEIGHT     = 0.20    # KL consistency penalty between two dropout passes

# Per-run output
RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR  = os.path.join(ACTION_DIR, "runs", RUN_ID)
Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Run ID      : {RUN_ID}")
print(f"Run dir     : {RUN_DIR}")
print(f"Device      : {device}")
print(f"Buffer size : {BUFFER_SIZE}")
print(f"Feature dim : {FEATURE_DIM}  (CLS + mean-patch)")


# ── Save config ────────────────────────────────────────────────────────────────

CONFIG = dict(
    dataset_dir=DATASET_DIR, buffer_size=BUFFER_SIZE, feature_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
    dropout=DROPOUT, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
    patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY,
    warmup_epochs=WARMUP_EPOCHS, label_smoothing=LABEL_SMOOTHING,
    noise_std=NOISE_STD, frame_drop_max=FRAME_DROP_MAX, jitter_max=JITTER_MAX,
    mixup_prob=MIXUP_PROB, mixup_alpha=MIXUP_ALPHA,
    entropy_weight=ENTROPY_WEIGHT, rdrop_weight=RDROP_WEIGHT,
)
with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)


# ── Feature Extraction ─────────────────────────────────────────────────────────

def _flush(batch_imgs, batch_stems, backbone, dest):
    if not batch_imgs:
        return
    with torch.no_grad(), autocast("cuda"):
        imgs_t  = torch.stack(batch_imgs).to(device)
        out     = backbone(imgs_t)
        cls     = out.last_hidden_state[:, 0]
        patches = out.last_hidden_state[:, 1:].mean(1)
        feat    = torch.cat([cls, patches], dim=1).cpu().float().numpy()
    for stem, f in zip(batch_stems, feat):
        dest[stem] = f
    batch_imgs.clear()
    batch_stems.clear()


def extract_features(dataset_dir, save_path, batch_size=128):
    if os.path.exists(save_path):
        print(f"Loading cached features: {save_path}")
        with open(save_path, "rb") as fh:
            return pickle.load(fh)

    print("Extracting DINOv2 features (CLS + mean-patch, ~20 min)...")
    backbone = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    backbone.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    features = {"train": {}, "val": {}}
    for split in ["train", "val"]:
        split_dir = os.path.join(dataset_dir, split)
        for label in sorted(os.listdir(split_dir)):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            batch_imgs, batch_stems = [], []
            for fname in tqdm(sorted(os.listdir(label_dir)), desc=f"{split}/{label}"):
                stem = os.path.splitext(fname)[0]
                img  = Image.open(os.path.join(label_dir, fname)).convert("RGB")
                batch_imgs.append(transform(img))
                batch_stems.append(stem)
                if len(batch_imgs) == batch_size:
                    _flush(batch_imgs, batch_stems, backbone, features[split])
            _flush(batch_imgs, batch_stems, backbone, features[split])
    del backbone
    torch.cuda.empty_cache()
    with open(save_path, "wb") as fh:
        pickle.dump(features, fh, protocol=4)
    print(f"Saved → {save_path}")
    return features


features = extract_features(DATASET_DIR, FEATURES_PATH)
print(f"Train features: {len(features['train']):,}")
print(f"Val   features: {len(features['val']):,}")


# ── Sequence Building ──────────────────────────────────────────────────────────

CLASS_TO_IDX = {"fanning": 0, "neutral": 1, "trophallaxis": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

_RE_ACTION  = re.compile(r"^(fanning|trophallaxis)_(.+?)_(\d{5})_(\d+)$")
_RE_NEUTRAL = re.compile(r"^neutral_(.+?)_tile(\d+)_(\d+)_(\d+)$")


def parse_stem(stem):
    m = _RE_ACTION.match(stem)
    if m:
        return CLASS_TO_IDX[m.group(1)], (m.group(2), m.group(4)), int(m.group(3))
    m = _RE_NEUTRAL.match(stem)
    if m:
        return CLASS_TO_IDX["neutral"], (m.group(1), m.group(2)), int(m.group(4))
    return None, None, None


def build_sequences(features_split, dataset_dir, split, min_seq_len):
    groups = defaultdict(list)
    split_dir = os.path.join(dataset_dir, split)
    for label in sorted(os.listdir(split_dir)):
        if label not in CLASS_TO_IDX:
            continue
        label_dir = os.path.join(split_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            stem = os.path.splitext(fname)[0]
            if stem not in features_split:
                continue
            label_idx, group_key, sort_key = parse_stem(stem)
            if group_key is None:
                continue
            groups[(label_idx, group_key)].append((sort_key, stem))
    sequences = []
    for (label_idx, _), items in groups.items():
        items.sort(key=lambda x: x[0])
        stems = [s for _, s in items]
        if len(stems) >= min_seq_len:
            sequences.append((label_idx, stems))
    return sequences


print("\nBuilding sequences...")
train_sequences = build_sequences(features["train"], DATASET_DIR, "train", MIN_SEQ_LEN)
val_sequences   = build_sequences(features["val"],   DATASET_DIR, "val",   MIN_SEQ_LEN)

for name, seqs in [("Train", train_sequences), ("Val", val_sequences)]:
    counts  = Counter(IDX_TO_CLASS[lbl] for lbl, _ in seqs)
    lengths = [len(s) for _, s in seqs]
    print(f"\n{name} ({len(seqs)} sequences):")
    for cls in ["fanning", "neutral", "trophallaxis"]:
        print(f"  {cls:15s}: {counts.get(cls, 0):>5} sequences")
    print(f"  Length  min:{min(lengths)}  median:{int(np.median(lengths))}  max:{max(lengths)}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, features, sequences, buffer_size, split="train", stride=None):
        self.features    = features
        self.buffer_size = buffer_size
        self.split       = split

        if split == "train":
            self.samples = []
            for label_idx, stems in sequences:
                n = max(1, len(stems) // buffer_size)
                self.samples.extend([(label_idx, stems)] * n)
        else:
            stride = stride or max(1, buffer_size // 2)
            self.samples = []
            for label_idx, stems in sequences:
                N = len(stems)
                if N < buffer_size:
                    self.samples.append((label_idx, stems, 0))
                else:
                    for start in range(0, N - buffer_size + 1, stride):
                        self.samples.append((label_idx, stems, start))

    def __len__(self):
        return len(self.samples)

    def _get_window(self, stems, start):
        T, N = self.buffer_size, len(stems)
        feats, mask = [], []
        for i in range(T):
            idx = start + i
            if idx < N:
                feats.append(torch.from_numpy(self.features[stems[idx]].copy()))
                mask.append(False)
            else:
                feats.append(torch.from_numpy(self.features[stems[N - 1]].copy()))
                mask.append(True)
        return torch.stack(feats), torch.tensor(mask, dtype=torch.bool)

    def _augment(self, x, mask):
        n_real = int((~mask).sum())

        # 1. Gaussian noise
        x = x + torch.randn_like(x) * NOISE_STD

        # 2. Temporal reversal
        if random.random() < 0.5 and n_real > 1:
            x[:n_real] = x[:n_real].flip(0)

        # 3. Temporal jitter
        if n_real > 2:
            for _ in range(random.randint(0, min(JITTER_MAX, n_real - 1))):
                i = random.randint(0, n_real - 2)
                x[i] = x[i + 1]

        # 4. Feature dropout — zero out entire frame vectors
        if n_real > FRAME_DROP_MAX and FRAME_DROP_MAX > 0:
            n_drop = random.randint(0, FRAME_DROP_MAX)
            if n_drop > 0:
                drop_idx = random.sample(range(n_real), n_drop)
                for i in drop_idx:
                    x[i] = 0.0

        return x

    def __getitem__(self, idx):
        if self.split == "train":
            label_idx, stems = self.samples[idx]
            N     = len(stems)
            start = random.randint(0, max(0, N - self.buffer_size))
        else:
            label_idx, stems, start = self.samples[idx]

        x, mask = self._get_window(stems, start)
        if self.split == "train":
            x = self._augment(x, mask)
        return x, torch.tensor(label_idx, dtype=torch.long), mask


# ── Model ──────────────────────────────────────────────────────────────────────

class TemporalSequenceClassifier(nn.Module):
    def __init__(self, feature_dim=768, num_classes=3, num_heads=6,
                 num_layers=3, dropout=0.35, max_seq_len=17):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.pos_embed = nn.Embedding(max_seq_len, feature_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(feature_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, padding_mask=None):
        B, T, _ = x.shape
        cls  = self.cls_token.expand(B, -1, -1)
        x    = torch.cat([cls, x], dim=1)
        pos  = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x    = x + self.pos_embed(pos)
        if padding_mask is not None:
            cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            full_mask = None
        x = self.transformer(x, src_key_padding_mask=full_mask)
        return self.classifier(self.norm(x[:, 0]))


# ── Training Utilities ─────────────────────────────────────────────────────────

def mixup_batch(feats, labels, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(feats.size(0), device=feats.device)
    return lam * feats + (1 - lam) * feats[idx], labels, labels[idx], lam


def entropy_penalty(logits):
    """Reward uncertainty — maximise entropy over the softmax distribution."""
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()


def rdrop_loss(logits1, logits2):
    """Symmetric KL divergence between two forward passes under different dropout."""
    p1 = torch.log_softmax(logits1, dim=1)
    p2 = torch.log_softmax(logits2, dim=1)
    kl1 = F.kl_div(p1, p2.detach().exp(), reduction="batchmean")
    kl2 = F.kl_div(p2, p1.detach().exp(), reduction="batchmean")
    return 0.5 * (kl1 + kl2)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for feats, labels, masks in tqdm(loader, desc="Train", leave=False):
        feats  = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        optimizer.zero_grad()

        use_mixup = random.random() < MIXUP_PROB
        if use_mixup:
            # MixUp: interpolate inputs, split CE loss across both labels
            feats_m, labels_a, labels_b, lam = mixup_batch(feats, labels)
            with autocast("cuda"):
                logits = model(feats_m, padding_mask=masks)
                ce     = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                loss   = ce - ENTROPY_WEIGHT * entropy_penalty(logits)
        else:
            # R-Drop: two forward passes, penalise inconsistency between them
            with autocast("cuda"):
                logits1 = model(feats, padding_mask=masks)
                logits2 = model(feats, padding_mask=masks)
                ce      = 0.5 * (criterion(logits1, labels) + criterion(logits2, labels))
                ent     = 0.5 * (entropy_penalty(logits1) + entropy_penalty(logits2))
                loss    = ce - ENTROPY_WEIGHT * ent + RDROP_WEIGHT * rdrop_loss(logits1, logits2)
            logits = logits1  # use first pass for metrics

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), f1_score(all_labels, all_preds, average="macro")


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for feats, labels, masks in tqdm(loader, desc="Val  ", leave=False):
            feats  = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            with autocast("cuda"):
                logits = model(feats, padding_mask=masks)
                loss   = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(conf.cpu().numpy())

    return (total_loss / len(loader),
            f1_score(all_labels, all_preds, average="macro"),
            np.array(all_preds), np.array(all_labels), np.array(all_confs))


# ── DataLoaders ────────────────────────────────────────────────────────────────

train_dataset = SequenceDataset(features["train"], train_sequences, BUFFER_SIZE, "train")
val_dataset   = SequenceDataset(features["val"],   val_sequences,   BUFFER_SIZE, "val", VAL_STRIDE)

sample_counts = Counter(lbl for lbl, _ in train_dataset.samples)
total_samples = sum(sample_counts.values())
class_weights = torch.tensor(
    [total_samples / (NUM_CLASSES * max(sample_counts[i], 1)) for i in range(NUM_CLASSES)],
    dtype=torch.float32,
)

print("\nTraining samples per class:")
for i in range(NUM_CLASSES):
    print(f"  {IDX_TO_CLASS[i]:15s}: {sample_counts[i]:>6}  weight={class_weights[i]:.3f}")

per_sample_w = [1.0 / sample_counts[lbl] for lbl, _ in train_dataset.samples]
sampler = WeightedRandomSampler(per_sample_w, len(train_dataset), replacement=True)

train_loader = DataLoader(train_dataset, BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Val samples: {len(val_dataset):,}")


# ── Model + Optimizer ──────────────────────────────────────────────────────────

model = TemporalSequenceClassifier(
    feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES, num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS, dropout=DROPOUT, max_seq_len=BUFFER_SIZE + 1,
).to(device)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
num_steps  = MAX_EPOCHS * len(train_loader)
warmup_steps = WARMUP_EPOCHS * len(train_loader)
scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)
criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                  label_smoothing=LABEL_SMOOTHING)
scaler     = GradScaler()

SAVE_PATH  = os.path.join(RUN_DIR, "best_model.pt")
log_path   = os.path.join(RUN_DIR, "train_log.csv")

with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_f1", "lr"])


# ── Training Loop ──────────────────────────────────────────────────────────────

best_val_f1, epochs_no_improve = 0.0, 0
history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

print("\n── Training ──────────────────────────────────────────────────────────────")
for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_loss, val_f1, _, _, _ = validate(model, val_loader, criterion, device)
    scheduler.step()
    torch.cuda.empty_cache()

    lr_now = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_f1:.4f}",
                                 f"{val_loss:.4f}", f"{val_f1:.4f}", f"{lr_now:.2e}"])

    print(f"Ep {epoch:3d} | Train {train_loss:.4f}/{train_f1:.4f} "
          f"| Val {val_loss:.4f}/{val_f1:.4f} | LR {lr_now:.2e}")

    if val_f1 > best_val_f1:
        best_val_f1, epochs_no_improve = val_f1, 0
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(), "val_f1": val_f1,
            "config": {
                "feature_dim": FEATURE_DIM, "num_classes": NUM_CLASSES,
                "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
                "dropout": DROPOUT, "buffer_size": BUFFER_SIZE,
            },
        }, SAVE_PATH)
        print(f"        ✓ Saved  (Val F1: {best_val_f1:.4f})")
    else:
        epochs_no_improve += 1
        print(f"        — No improvement ({epochs_no_improve}/{PATIENCE})")
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

print(f"\nBest Val F1: {best_val_f1:.4f}")


# ── Evaluation + Confidence Calibration ───────────────────────────────────────

ckpt = torch.load(SAVE_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
print(f"\nLoaded epoch {ckpt['epoch']}  (Val F1: {ckpt['val_f1']:.4f})")

_, _, all_preds, all_labels, all_confs = validate(model, val_loader, criterion, device)

classes = ["fanning", "neutral", "trophallaxis"]
report  = classification_report(all_labels, all_preds, target_names=classes, output_dict=False)
print("\n" + str(report))
with open(os.path.join(RUN_DIR, "report.txt"), "w") as f:
    f.write(f"Run: {RUN_ID}\nEpoch: {ckpt['epoch']}\nVal F1: {ckpt['val_f1']:.4f}\n\n")
    f.write(str(report))

# Confidence calibration — the key overconfidence diagnostic
print("── Confidence calibration (before temperature scaling) ───────────────────")
correct_mask = (all_preds == all_labels)
print(f"  Mean confidence — correct   : {all_confs[correct_mask].mean():.4f}")
print(f"  Mean confidence — incorrect : {all_confs[~correct_mask].mean():.4f}")
print(f"  Fraction confident & wrong  : {((all_confs > 0.9) & ~correct_mask).mean():.4f}")

# Temperature scaling — post-hoc calibration, no retraining needed.
# Finds scalar T that minimises NLL on val set by dividing logits by T.
# T > 1 softens all predictions; T < 1 sharpens them.
def find_temperature(mdl, loader):
    """
    Find T > 1 that minimises NLL on the val set using hard (unsmoothed) CE.
    T > 1 softens predictions; T < 1 would sharpen them.
    We clamp the search to [1.0, 10.0] so it can only ever soften.
    """
    mdl.eval()
    logit_list, label_list = [], []
    with torch.no_grad():
        for feats, labels, masks in loader:
            logit_list.append(mdl(feats.to(device), padding_mask=masks.to(device)).cpu())
            label_list.append(labels)
    all_logits_ts = torch.cat(logit_list)
    all_labels_ts = torch.cat(label_list)
    hard_nll = nn.CrossEntropyLoss()          # no smoothing, no class weights
    T = nn.Parameter(torch.tensor([2.0]))     # start in the softening region
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=500)
    def step():
        opt.zero_grad()
        loss = hard_nll(all_logits_ts / T.clamp(min=1.0), all_labels_ts)
        loss.backward()
        return loss
    opt.step(step)
    return float(T.clamp(min=1.0).item()), all_logits_ts, all_labels_ts

temperature, ts_logits, ts_labels = find_temperature(model, val_loader)
ts_probs   = torch.softmax(ts_logits / temperature, dim=1)
ts_conf, ts_pred = ts_probs.max(dim=1)
ts_conf    = ts_conf.numpy()
ts_pred    = ts_pred.numpy()
ts_labels  = ts_labels.numpy()
ts_correct = (ts_pred == ts_labels)

print(f"\n── Temperature scaling ───────────────────────────────────────────────────")
print(f"  Optimal temperature T       : {temperature:.4f}")
print(f"  Mean confidence — correct   : {ts_conf[ts_correct].mean():.4f}")
print(f"  Mean confidence — incorrect : {ts_conf[~ts_correct].mean():.4f}")
print(f"  Fraction confident & wrong  : {((ts_conf > 0.9) & ~ts_correct).mean():.4f}")
print(f"  Val F1 (unchanged)          : {f1_score(ts_labels, ts_pred, average='macro'):.4f}")

# Save temperature into the checkpoint so graph_generator can use it
ckpt_with_temp = torch.load(SAVE_PATH, map_location="cpu")
ckpt_with_temp["temperature"] = temperature
torch.save(ckpt_with_temp, SAVE_PATH)
print(f"  Temperature saved to checkpoint.")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion Matrix  —  F1={best_val_f1:.4f}  Ep={ckpt['epoch']}")
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "confusion.png"), dpi=150)

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(history["train_f1"],  label="Train"); ax2.plot(history["val_f1"],  label="Val")
ax2.set_title(f"Macro F1"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "curves.png"), dpi=150)

# Confidence histogram — correct vs incorrect
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(all_confs[correct_mask],  bins=30, alpha=0.6, color="#2ecc71", label="Correct")
ax.hist(all_confs[~correct_mask], bins=30, alpha=0.6, color="#e74c3c", label="Incorrect")
ax.axvline(0.7, color="black", linestyle="--", label="Threshold 0.7")
ax.set_xlabel("Softmax confidence"); ax.set_ylabel("Count")
ax.set_title("Confidence distribution: correct vs incorrect predictions")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "confidence_hist.png"), dpi=150)

print(f"\nAll outputs saved to: {RUN_DIR}")
print(f"\nTo use in graph_generator.ipynb, set:")
print(f'  TEMPORAL_PATH = "{SAVE_PATH.replace(chr(92), "/")}"')
