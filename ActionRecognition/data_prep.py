"""
data_prep.py — Rebuild action recognition dataset with real ByteTrack sequences.

Problem with the original data_merge.ipynb:
  bee_id in filenames = ann_idx (line number in YOLO label file), NOT a tracked
  identity. "Bee 0 across 3000 frames" is whichever bee happened to be listed
  first in each label file — a different physical bee every frame.

Fix:
  Read the existing GT bounding boxes frame-by-frame in temporal order, feed
  them to ByteTracker, and use the resulting persistent track_id as the bee ID.
  Same 224x224 crop logic. Same train/val split structure.

Output: D:/Projects/Masters/Data/AR_v2_dataset/
  train/ fanning/ trophallaxis/ neutral/
  val/   fanning/ trophallaxis/ neutral/

Neutral is copied unchanged — it has no temporal structure to fix.

Requires: supervision (already installed at 0.27.0)
Run:      python data_prep.py   (~5-10 min)
Then:     update DATASET_DIR in train.py to AR_v2_dataset
"""

import os
import re
import cv2
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import supervision as sv

AR_DATASET_DIR    = Path("D:/Projects/Masters/Data/AR_dataset")
MERGED_DATASET_DIR = Path("D:/Projects/Masters/Data/AR_merged_dataset")
OUTPUT_DIR        = Path("D:/Projects/Masters/Data/AR_v2_dataset")
CONTEXT_SIZE      = 224

ACTION_CLASSES = ["fanning", "trophallaxis"]
SPLITS         = ["train", "val"]

_RE_STEM = re.compile(r"^(.+)_(\d{5})$")


def parse_stem(stem):
    """'20230609c_00228' → (video_id='20230609c', frame=228)"""
    m = _RE_STEM.match(stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, 0


def get_crop(frame, cx_norm, cy_norm, size=CONTEXT_SIZE):
    h, w = frame.shape[:2]
    half = size // 2
    cx = int(cx_norm * w)
    cy = int(cy_norm * h)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.shape[0] < size or crop.shape[1] < size:
        pb = size - crop.shape[0]
        pr = size - crop.shape[1]
        crop = cv2.copyMakeBorder(crop, 0, pb, 0, pr, cv2.BORDER_REFLECT)
    return crop


def read_labels(label_path, img_w, img_h):
    """
    Returns list of (cx_norm, cy_norm, xyxy) for each annotation.
    xyxy is in pixel coords for ByteTrack.
    """
    if not label_path.exists():
        return []
    results = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = max(0.0, (cx - bw / 2)) * img_w
            y1 = max(0.0, (cy - bh / 2)) * img_h
            x2 = min(1.0, (cx + bw / 2)) * img_w
            y2 = min(1.0, (cy + bh / 2)) * img_h
            results.append((cx, cy, [x1, y1, x2, y2]))
    return results


def process_video(video_id, frames, images_dir, labels_dir,
                  action_class, output_dir):
    """
    frames: list of (frame_number, filename) sorted chronologically.
    Returns number of crops saved.
    """
    tracker = sv.ByteTrack(
        track_activation_threshold=0.1,   # low threshold — GT boxes are "confident"
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    saved = 0

    for frame_num, fname in frames:
        stem  = os.path.splitext(fname)[0]
        frame = cv2.imread(str(images_dir / fname))
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]

        annotations = read_labels(labels_dir / (stem + ".txt"), img_w, img_h)

        if annotations:
            boxes  = np.array([a[2] for a in annotations], dtype=np.float32)
            confs  = np.ones(len(annotations),             dtype=np.float32)
            cls_ids = np.zeros(len(annotations),           dtype=int)
            dets   = sv.Detections(xyxy=boxes, confidence=confs, class_id=cls_ids)
        else:
            dets = sv.Detections.empty()

        tracked = tracker.update_with_detections(dets)

        if tracked.tracker_id is None or len(tracked) == 0:
            continue

        for i, track_id in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx_norm = ((x1 + x2) / 2) / img_w
            cy_norm = ((y1 + y2) / 2) / img_h
            crop = get_crop(frame, cx_norm, cy_norm)
            if crop is None:
                continue
            out_name = f"{action_class}_{video_id}_{frame_num:05d}_{track_id}.jpg"
            cv2.imwrite(str(output_dir / out_name), crop)
            saved += 1

    return saved


def build_split(action_class, split):
    images_dir = AR_DATASET_DIR / action_class / split / "images"
    labels_dir = AR_DATASET_DIR / action_class / split / "labels"
    output_dir = OUTPUT_DIR / split / action_class
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group frames by video, sort chronologically within each video
    video_frames = defaultdict(list)
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        video_id, frame_num = parse_stem(os.path.splitext(fname)[0])
        video_frames[video_id].append((frame_num, fname))

    total = 0
    for video_id, frames in tqdm(sorted(video_frames.items()),
                                  desc=f"{action_class}/{split}"):
        frames.sort(key=lambda x: x[0])
        total += process_video(
            video_id, frames, images_dir, labels_dir, action_class, output_dir
        )
    return total


# ── Main ───────────────────────────────────────────────────────────────────────

print(f"supervision {sv.__version__}")
print(f"Output: {OUTPUT_DIR}\n")

stats = {}
for cls in ACTION_CLASSES:
    stats[cls] = {}
    for split in SPLITS:
        n = build_split(cls, split)
        stats[cls][split] = n
        print(f"  {cls}/{split}: {n} crops")

# Neutral: copy unchanged from existing merged dataset
print("\nCopying neutral...")
for split in SPLITS:
    src = MERGED_DATASET_DIR / split / "neutral"
    dst = OUTPUT_DIR / split / "neutral"
    if dst.exists():
        print(f"  neutral/{split}: already exists, skipping")
    else:
        shutil.copytree(str(src), str(dst))
        print(f"  neutral/{split}: {len(list(dst.iterdir()))} crops copied")

# Summary
print("\n── Final counts ──────────────────────────────────────────────────────────")
for split in SPLITS:
    print(f"\n{split}/")
    total = 0
    for cls in ["fanning", "neutral", "trophallaxis"]:
        d = OUTPUT_DIR / split / cls
        n = len(list(d.iterdir())) if d.exists() else 0
        print(f"  {cls:20s}: {n:>6}")
        total += n
    print(f"  {'TOTAL':20s}: {total:>6}")

print(f"""
Done.
In train.py, change:
  DATASET_DIR   = "D:/Projects/Masters/Data/AR_v2_dataset"
  FEATURES_PATH = os.path.join(OUTPUT_DIR, "features_v3.pkl")
""")
