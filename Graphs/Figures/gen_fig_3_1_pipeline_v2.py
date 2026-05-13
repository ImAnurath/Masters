"""
Figure 3.1 — End-to-end pipeline diagram with real images.
Redesigned: cleaner layout, correct 768-dim / 3-layer / 6-head values,
AR_v2_dataset crops.

Run:  python gen_fig_3_1_pipeline_v2.py
Out:  fig_3_1_pipeline_v2.png
"""
import os, glob, re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image as PILImage
import supervision as sv
from ultralytics import YOLO

# ── config ────────────────────────────────────────────────────────────────────
BASE      = r'D:\Projects\Masters'
YOLO_PATH = rf'{BASE}\Trained_Models\DET_medium-best.pt'
FRAME_PATH = rf'{BASE}\Data\AR_dataset\fanning\train\images\20230609c_00228.jpg'
CROPS_DIR  = rf'{BASE}\Data\AR_v2_dataset\train\fanning'
LABEL_TEXT  = 'FANNING'
LABEL_COLOR = '#00dc64'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_3_1_pipeline_v2.png')
DPI = 200

# ── stage definitions ─────────────────────────────────────────────────────────
STAGE_COLORS = [
    ('#D6EAF8', '#2471A3'),   # 1 input          blue
    ('#D5F5E3', '#1A8754'),   # 2 SAHI tiling     green
    ('#D5F5E3', '#1A8754'),   # 3 detection        green
    ('#FDEBD0', '#D35400'),   # 4 tracking         orange
    ('#E8DAEF', '#7D3C98'),   # 5 DINOv2           purple
    ('#E8DAEF', '#7D3C98'),   # 6 buffer           purple
    ('#FADBD8', '#C0392B'),   # 7 classifier       red
    ('#D5F5E3', '#1A8754'),   # 8 output           green
]

STAGE_INFO = [
    ('Video Frame',                      '1920 × 1080 px  ·  source video'),
    ('SAHI Tiling',                      '640 × 640 tiles  ·  25 % overlap'),
    ('YOLOv11-medium Detection',         'conf ≥ 0.45  ·  NMS IoU = 0.60'),
    ('ByteTracker Identity',             'lost_track_buffer = 30 frames'),
    ('DINOv2-small Feature Extraction',  '768-d (CLS ∥ mean-patch) per crop'),
    ('Rolling Buffer',                   'T = 16 feature vectors per track'),
    ('Temporal Transformer Classifier',  '3 encoder layers  ·  6 heads  ·  768→256→3'),
    ('Behavior Label',                   'softmax confidence ≥ 0.70'),
]

ARROW_NOTES = [
    'full-resolution frame',
    'overlapping 640×640 tiles',
    'bounding boxes + confidence scores',
    'track IDs + 224×224 padded crops',
    '768-d feature vectors',
    '16 × 768 feature tensor',
    'softmax distribution',
]

SOFTMAX_PROBS   = [0.94, 0.03, 0.03]
SOFTMAX_CLASSES = ['Fanning', 'Neutral', 'Trophallaxis']
SOFTMAX_COLORS  = ['#00dc64', '#64b4ff', '#ff7800']
C_DARK = '#2C3E50'


# ── image helpers ─────────────────────────────────────────────────────────────

def load_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f'Cannot load: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_tile_grid(frame_rgb, tile=640, overlap=0.25):
    h, w  = frame_rgb.shape[:2]
    step  = int(tile * (1 - overlap))
    vis   = frame_rgb.copy()
    col   = (30, 210, 90)
    thick = max(2, h // 300)
    for x in range(0, w, step):
        cv2.line(vis, (x, 0), (x, h - 1), col, thick)
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (w - 1, y), col, thick)
    return vis


def make_detected_tile(frame_path, yolo_path, tile=640, overlap=0.25):
    print('  Running YOLO on tiles …')
    model     = YOLO(yolo_path)
    frame_bgr = cv2.imread(str(frame_path))
    h, w      = frame_bgr.shape[:2]
    step      = int(tile * (1 - overlap))
    best_tile, best_boxes, best_n = None, [], 0
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            crop = frame_bgr[y0:min(y0+tile,h), x0:min(x0+tile,w)]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                continue
            res = model(crop, conf=0.45, verbose=False)[0]
            n   = len(res.boxes)
            if n > best_n:
                best_n = n; best_tile = crop.copy()
                best_boxes = res.boxes.xyxy.cpu().numpy()
    if best_tile is None:
        best_tile = frame_bgr[:tile, :tile].copy()
    ann = best_tile.copy()
    for box in best_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(ann, (x1,y1), (x2,y2), (0,210,90), 2)
    return cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)


def make_tracked_frame(frame_path, yolo_path):
    print('  Running YOLO + ByteTracker …')
    model     = YOLO(yolo_path)
    frame_bgr = cv2.imread(str(frame_path))
    res       = model(frame_bgr, conf=0.45, verbose=False)[0]
    xyxy, confs = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
    dets    = sv.Detections(xyxy=xyxy, confidence=confs, class_id=np.zeros(len(xyxy), int))
    tracker = sv.ByteTrack()
    dets    = tracker.update_with_detections(dets)
    palette = [(0,210,90),(255,120,0),(50,140,255),(210,50,210),(0,200,210),(200,200,0)]
    ann     = frame_bgr.copy()
    for box, tid in zip(dets.xyxy, dets.tracker_id):
        x1,y1,x2,y2 = map(int,box)
        col = palette[int(tid)%len(palette)]
        cv2.rectangle(ann,(x1,y1),(x2,y2),col,2)
        lbl = f'#{tid}'
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
        cv2.rectangle(ann,(x1,y1-th-6),(x1+tw+4,y1),col,-1)
        cv2.putText(ann,lbl,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
    return cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)


def find_bee_sequence(crops_dir, n=5, video_id=None):
    pat    = re.compile(r'fanning_(.+?)_(\d+)_(\d+)\.jpg$')
    groups = {}
    for f in glob.glob(os.path.join(crops_dir, '*.jpg')):
        m = pat.search(os.path.basename(f))
        if m:
            key = (m.group(1), m.group(3))
            groups.setdefault(key, []).append((int(m.group(2)), f))
    candidates = sorted(groups.items(), key=lambda x: (
        0 if (video_id and x[0][0].startswith(video_id)) else 1, -len(x[1])
    ))
    for _, paths in candidates:
        if len(paths) >= n:
            return [p for _, p in sorted(paths)[:n]]
    return sorted(glob.glob(os.path.join(crops_dir, '*.jpg')))[:n]


def make_buffer_strip(paths, n=5, cell=88):
    imgs = [np.array(PILImage.open(p).convert('RGB').resize((cell,cell), PILImage.Resampling.LANCZOS))
            for p in paths[:n]]
    spacer = np.ones((cell, 18, 3), dtype=np.uint8) * 235
    cv2.putText(spacer, '...', (1, cell//2+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)
    return np.concatenate(imgs[:3] + [spacer] + imgs[3:], axis=1)


# ── load all stage images ─────────────────────────────────────────────────────
print('Loading images …')
frame_rgb   = load_rgb(FRAME_PATH)
tiled_rgb   = draw_tile_grid(frame_rgb)
det_rgb     = make_detected_tile(FRAME_PATH, YOLO_PATH)
tracked_rgb = make_tracked_frame(FRAME_PATH, YOLO_PATH)

_vid = re.split(r'[_.]', os.path.basename(FRAME_PATH))[0]
seq_paths    = find_bee_sequence(CROPS_DIR, video_id=_vid)
crop_rgb     = np.array(PILImage.open(seq_paths[0]).convert('RGB'))
buffer_strip = make_buffer_strip(seq_paths)
print('Images ready — composing figure …')

# ── compose figure ────────────────────────────────────────────────────────────
N     = len(STAGE_INFO)
FIG_W = 12.0
ROW_H = 2.2   # image row height (inches)
LBL_H = ROW_H

# Build a gridspec: alternating [stage-row, arrow-row, stage-row, ...]
n_rows   = 2 * N - 1
h_ratios = []
for i in range(N):
    h_ratios.append(ROW_H * 0.50 if i == N - 1 else ROW_H)
    if i < N - 1:
        h_ratios.append(0.28)

fig = plt.figure(figsize=(FIG_W, sum(h_ratios) + 0.2))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    len(h_ratios), 2,
    figure=fig,
    height_ratios=h_ratios,
    width_ratios=[4.5, 2.5],
    hspace=0.0, wspace=0.06,
    left=0.01, right=0.99, top=0.998, bottom=0.002,
)

IMAGES = [
    frame_rgb, tiled_rgb, det_rgb, tracked_rgb,
    crop_rgb, buffer_strip,
    None,   # softmax bars
    None,   # FANNING badge
]

for i in range(N):
    gs_row = i * 2   # gridspec row for this stage
    face, edge = STAGE_COLORS[i]
    title, sub = STAGE_INFO[i]

    # ── image panel ──────────────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[gs_row, 0])
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(1.4)
    ax_img.set_facecolor(face)

    img = IMAGES[i]
    if img is not None:
        ax_img.imshow(img, aspect='auto', interpolation='lanczos')

    elif i == 6:   # softmax bar chart
        y_pos = range(len(SOFTMAX_CLASSES))
        bars  = ax_img.barh(list(y_pos), SOFTMAX_PROBS,
                            color=SOFTMAX_COLORS, edgecolor='white', height=0.55)
        for bar, p, cls in zip(bars, SOFTMAX_PROBS, SOFTMAX_CLASSES):
            ax_img.text(p + 0.04, bar.get_y() + bar.get_height()/2,
                        f'{p:.2f}', va='center', fontsize=11, fontweight='bold',
                        color=C_DARK)
        ax_img.set_yticks(list(y_pos))
        ax_img.set_yticklabels(SOFTMAX_CLASSES, fontsize=11, fontweight='bold')
        ax_img.set_xlim(0, 1.30); ax_img.set_ylim(-0.6, 2.6)
        ax_img.tick_params(left=False, bottom=False, labelbottom=False)
        for sp in ax_img.spines.values(): sp.set_visible(False)

    else:   # FANNING badge
        ax_img.set_facecolor(LABEL_COLOR)
        ax_img.text(0.5, 0.5, LABEL_TEXT,
                    ha='center', va='center', transform=ax_img.transAxes,
                    fontsize=22, fontweight='bold', color='white',
                    fontfamily='monospace')
        for sp in ax_img.spines.values(): sp.set_visible(False)

    # ── label panel ──────────────────────────────────────────────────────────
    ax_lbl = fig.add_subplot(gs[gs_row, 1])
    ax_lbl.axis('off')
    ax_lbl.set_facecolor('white')

    # full-panel coloured box
    ax_lbl.add_patch(FancyBboxPatch(
        (0.10, 0.08), 0.80, 0.84,
        boxstyle='round,pad=0.04',
        facecolor=face, edgecolor=edge, linewidth=1.6,
        transform=ax_lbl.transAxes, clip_on=False,
    ))

    # stage number badge
    ax_lbl.text(0.5, 0.82, str(i + 1),
                ha='center', va='center', transform=ax_lbl.transAxes,
                fontsize=11, fontweight='bold', color=face,
                bbox=dict(boxstyle='circle,pad=0.22', facecolor=edge, edgecolor='none'))

    # stage title
    ax_lbl.text(0.5, 0.52, title,
                ha='center', va='center', transform=ax_lbl.transAxes,
                fontsize=8.2, fontweight='bold', color=C_DARK,
                multialignment='center')

    # subtitle
    ax_lbl.text(0.5, 0.20, sub,
                ha='center', va='center', transform=ax_lbl.transAxes,
                fontsize=6.8, color='#555555',
                multialignment='center')

    # ── arrow row (between stages) ────────────────────────────────────────────
    if i < N - 1:
        ax_arr = fig.add_subplot(gs[gs_row + 1, 0])
        ax_arr.axis('off')

        # downward arrow centred in the image column
        ax_arr.annotate('', xy=(0.5, 0.05), xytext=(0.5, 0.95),
                        xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>', color=C_DARK,
                                        lw=1.8, mutation_scale=16))

        note = ARROW_NOTES[i] if i < len(ARROW_NOTES) else ''
        if note:
            ax_arr.text(0.53, 0.50, note,
                        ha='left', va='center', transform=ax_arr.transAxes,
                        fontsize=7.5, color='#777', style='italic',
                        bbox=dict(boxstyle='round,pad=0.20', fc='white',
                                  ec='#cccccc', lw=0.8, alpha=0.9))

        # blank right column in arrow rows
        ax_blank = fig.add_subplot(gs[gs_row + 1, 1])
        ax_blank.axis('off')

plt.savefig(OUT, dpi=DPI, bbox_inches='tight', pad_inches=0.05, facecolor='white')
plt.close()
print(f'\nSaved: {OUT}')
