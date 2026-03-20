import os
import shutil
from pathlib import Path
from PIL import Image
from sahi.slicing import slice_image

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR        = "Masters/Data/DET_data_OG"          # folder with images/ and labels/ subdirs
OUTPUT_DIR      = "Masters/Data/data_sliced"   # will be created automatically
SLICE_SIZE      = 640               # slice width and height in pixels
OVERLAP_RATIO   = 0.3               # % overlap between slices
MIN_VISIBILITY  = 0.1               # keep a bbox in a slice if at least this
                                    # fraction of its area survives clipping
# ─────────────────────────────────────────────

# helper functions for bbox format conversions and clipping since SAHI gives us the tile coordinates and we need to convert bboxes to tile-local and back to YOLO format for saving
def yolo_to_abs(cx, cy, w, h, img_w, img_h): # yolo bboxes to unnormalised absolute values
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2

def abs_to_yolo(x1, y1, x2, y2, tile_w, tile_h): # unnormalised absolute values to yolo
    cx = (x1 + x2) / 2 / tile_w
    cy = (y1 + y2) / 2 / tile_h
    w  = (x2 - x1) / tile_w
    h  = (y2 - y1) / tile_h
    return cx, cy, w, h

def clip_bbox_to_tile(x1, y1, x2, y2, tx, ty, tw, th):
    # Intersect with tile
    ix1 = max(x1, tx)
    iy1 = max(y1, ty)
    ix2 = min(x2, tx + tw)
    iy2 = min(y2, ty + th)

    if ix2 <= ix1 or iy2 <= iy1:
        return None  # no overlap

    orig_area = max((x2 - x1) * (y2 - y1), 1e-6)
    clip_area = (ix2 - ix1) * (iy2 - iy1)

    if clip_area / orig_area < MIN_VISIBILITY:
        return None  # too little of the box survived

    # Shift to tile-local coordinates
    return ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty

def load_labels(label_path):
    # returns list of (class_id, cx, cy, w, h) tuples from a YOLO label file, or empty list if file doesn't exist or is empty/invalid
    annotations = []
    if not label_path.exists():
        return annotations
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            annotations.append((cls, cx, cy, w, h))
    return annotations


def process_dataset():
    data_path   = Path(DATA_DIR)
    images_path = data_path / "images"
    labels_path = data_path / "labels"
    # create output folders
    out_path        = Path(OUTPUT_DIR)
    out_images_path = out_path / "images"
    out_labels_path = out_path / "labels"
    out_images_path.mkdir(parents=True, exist_ok=True)
    out_labels_path.mkdir(parents=True, exist_ok=True)
    # gather image files
    image_files = sorted(
        p for p in images_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    )
    # stats because why not
    total       = len(image_files)
    sliced      = 0
    skipped     = 0
    total_tiles = 0

    print(f"\n{'─'*50}")
    print(f"  YOLO Dataset Slicer")
    print(f"{'─'*50}")
    print(f"  Source   : {data_path.resolve()}")
    print(f"  Output   : {out_path.resolve()}")
    print(f"  Slice    : {SLICE_SIZE}×{SLICE_SIZE}  |  Overlap: {OVERLAP_RATIO*100:.0f}%")
    print(f"  Images   : {total}")
    print(f"{'─'*50}\n")

    for img_path in image_files:  # loop over images to find corresponding labels and slice
        label_path  = labels_path / (img_path.stem + ".txt")
        annotations = load_labels(label_path)

        if not annotations: 
            # if no labels, skip slicing since we are not saving background tiles and it would be a waste of time and disk space to slice them just to throw them away
            skipped += 1
            print(f"  [SKIP]  {img_path.name}  (no labels)")
            continue

        # load image
        image    = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # slice with SAHI
        slice_result = slice_image(
            image            = str(img_path),
            slice_height     = SLICE_SIZE,
            slice_width      = SLICE_SIZE,
            overlap_height_ratio = OVERLAP_RATIO,
            overlap_width_ratio  = OVERLAP_RATIO,
        )

        tiles_saved = 0

        for idx, sliced_img in enumerate(slice_result.sliced_image_list):
            # SAHI may return numpy array or PIL Image depending on version
            # mine returns numpy
            raw      = sliced_img.image
            tile_pil = Image.fromarray(raw) if not isinstance(raw, Image.Image) else raw
            sx       = sliced_img.starting_pixel[0]   # top-left x in original
            sy       = sliced_img.starting_pixel[1]   # top-left y in original
            tw, th   = tile_pil.size

            tile_labels = []
            for (cls, cx, cy, bw, bh) in annotations: 
                # for each original bbox, convert to absolute coords, clip to tile, convert back to yolo relative coords
                x1, y1, x2, y2 = yolo_to_abs(cx, cy, bw, bh, img_w, img_h)
                clipped = clip_bbox_to_tile(x1, y1, x2, y2, sx, sy, tw, th)
                if clipped is None:
                    continue
                lx1, ly1, lx2, ly2 = clipped
                ncx, ncy, nw, nh   = abs_to_yolo(lx1, ly1, lx2, ly2, tw, th)
                # Safety clamp to [0,1]
                ncx = max(0.0, min(1.0, ncx))
                ncy = max(0.0, min(1.0, ncy))
                nw  = max(0.0, min(1.0, nw))
                nh  = max(0.0, min(1.0, nh))
                tile_labels.append((cls, ncx, ncy, nw, nh))

            # Only save tiles that actually contain at least one bbox
            # if it is background without label I am not saving it since its already way too fucking big
            if not tile_labels:
                continue
            
            # save tile image and labels
            stem     = f"{img_path.stem}_tile{idx:04d}"
            out_img  = out_images_path / (stem + ".jpg")
            out_lbl  = out_labels_path / (stem + ".txt")

            tile_pil.save(out_img, quality=95)

            with open(out_lbl, "w") as f: # write tile labels in YOLO format
                for (cls, ncx, ncy, nw, nh) in tile_labels:
                    f.write(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")

            tiles_saved  += 1
            total_tiles  += 1

        sliced += 1
        # keeping track of things
        print(f"  [DONE]  {img_path.name}  →  {tiles_saved} tiles saved")

    print(f"\n{'─'*50}")
    print(f"  Finished!")
    print(f"  Images sliced : {sliced}")
    print(f"  Images skipped: {skipped}  (no labels)")
    print(f"  Total tiles   : {total_tiles}")
    print(f"  Output folder : {out_path.resolve()}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    process_dataset()