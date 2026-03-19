import os
import random
import shutil
from pathlib import Path
# This script splits the dataset into train/val/test sets based on specified ratios.
# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6


# workspace root (this script's directory)
root = os.path.dirname(__file__)

# Original data folder
data_dir = Path(root) / "data_sliced"
images_dir = data_dir / "images"
labels_dir = data_dir / "labels"

# new split output folder
split_dir = Path(root) / "data_sliced_split"
splits = ["train", "val", "test"]

# create split directories
for split in splits:
    (split_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / split / "labels").mkdir(parents=True, exist_ok=True)



# collect images (only those with a matching label)
images = list(images_dir.glob("*.*"))
images = [img for img in images if (labels_dir / f"{img.stem}.txt").exists()]

# shuffle for randomness
random.seed(69)
random.shuffle(images)

# split
n = len(images)
train_end = int(train_ratio * n)
val_end = train_end + int(val_ratio * n)

train_files = images[:train_end]
val_files = images[train_end:val_end]
test_files = images[val_end:]

# helper function to copy image + label
def copy_files(files, split):
    for img in files:
        label = labels_dir / f"{img.stem}.txt"
        shutil.copy(img, split_dir / split / "images" / img.name)
        shutil.copy(label, split_dir / split / "labels" / label.name)

# copy files to split dirs
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print(f"Done! Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
print(f"Splits saved in: {split_dir}")