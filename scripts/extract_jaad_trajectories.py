"""
ProxiSense — JAAD Clip Trajectory Extractor
============================================
Runs YOLOv8 on every JAAD video clip to detect pedestrians frame-by-frame,
tracks them with a simple IoU-based tracker, then applies heuristic labelling
to assign intent classes:

    0 = STATIONARY   (very low total displacement)
    1 = WILL_CROSS   (strong lateral / crossing movement)
    2 = ERRATIC      (high direction-change variance)
    3 = LANE_CHANGE  (strong horizontal movement — vehicle class)

Output: data/datasets/jaad/trajectories.npz  (X, y arrays ready for LSTM)

Usage:
    python scripts/extract_jaad_trajectories.py
    python scripts/extract_jaad_trajectories.py --clips-dir data/datasets/jaad/JAAD_clips --max-clips 50
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
SEQ_LEN       = 15        # frames per trajectory window fed to LSTM
FRAME_SKIP    = 2         # process every Nth frame (speeds up extraction)
MIN_SEQ_LEN   = 10        # discard tracks shorter than this
IOU_THRESHOLD = 0.35      # IoU to match boxes across frames
CONF_THRESH   = 0.40      # YOLO detection confidence cutoff
IMG_W, IMG_H  = 1920, 1080  # normalisation resolution


# ── IoU helper ────────────────────────────────────────────────────────────────
def iou(a, b):
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


# ── Simple IoU tracker ────────────────────────────────────────────────────────
class SimpleTracker:
    def __init__(self):
        self.tracks   = {}   # track_id -> [cx, cy] history
        self.boxes    = {}   # track_id -> last box
        self.next_id  = 0
        self.max_lost = 10   # frames before track is dropped
        self.lost     = {}   # track_id -> frames since last seen

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2] boxes
        Returns: list of (track_id, cx, cy) for each matched/new detection
        """
        results = []
        matched_tracks = set()

        for det in detections:
            best_iou, best_id = 0, None
            for tid, box in self.boxes.items():
                if tid in matched_tracks:
                    continue
                score = iou(det, box)
                if score > best_iou:
                    best_iou, best_id = score, tid

            if best_iou >= IOU_THRESHOLD and best_id is not None:
                tid = best_id
                matched_tracks.add(tid)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = []
                self.lost[tid]   = 0

            cx = (det[0] + det[2]) / 2.0
            cy = (det[1] + det[3]) / 2.0
            self.tracks[tid].append([cx, cy])
            self.boxes[tid]  = det
            self.lost[tid]   = 0
            results.append((tid, cx, cy))

        # age unmatched tracks
        for tid in list(self.boxes.keys()):
            if tid not in matched_tracks:
                self.lost[tid] = self.lost.get(tid, 0) + 1
                if self.lost[tid] > self.max_lost:
                    del self.boxes[tid]
                    del self.lost[tid]

        return results

    def get_finished_tracks(self):
        """Return tracks that have been lost (completed)."""
        done = {}
        for tid in list(self.tracks.keys()):
            if tid not in self.boxes and len(self.tracks[tid]) >= MIN_SEQ_LEN:
                done[tid] = self.tracks.pop(tid)
        return done

    def flush(self):
        """Return all remaining tracks at clip end."""
        done = {}
        for tid, traj in self.tracks.items():
            if len(traj) >= MIN_SEQ_LEN:
                done[tid] = traj
        self.tracks.clear()
        return done


# ── Heuristic intent labeller ─────────────────────────────────────────────────
def heuristic_label(traj: list) -> int:
    """
    traj: list of [cx, cy] pixel coordinates (unnormalised).
    Returns intent class 0-3.
    """
    pts = np.array(traj, dtype=np.float32)
    pts[:, 0] /= IMG_W
    pts[:, 1] /= IMG_H

    if len(pts) < 3:
        return 0  # STATIONARY

    displacements = np.diff(pts, axis=0)            # (N-1, 2)
    total_disp    = np.linalg.norm(pts[-1] - pts[0])
    speed_std     = np.std(np.linalg.norm(displacements, axis=1))
    lateral_disp  = abs(pts[-1, 0] - pts[0, 0])     # horizontal (x)
    vertical_disp = abs(pts[-1, 1] - pts[0, 1])     # vertical   (y)

    # Direction-change variance (erratic)
    if len(displacements) >= 3:
        angles = np.arctan2(displacements[:, 1], displacements[:, 0])
        angle_changes = np.abs(np.diff(angles))
        angle_changes = np.where(angle_changes > np.pi,
                                 2*np.pi - angle_changes, angle_changes)
        angle_var = float(np.mean(angle_changes))
    else:
        angle_var = 0.0

    # ── Decision rules ──────────────────────────────────────────────────────
    if total_disp < 0.02:                            # barely moved
        return 0  # STATIONARY

    if angle_var > 0.6 and speed_std > 0.01:        # random direction changes
        return 2  # ERRATIC

    if vertical_disp > 0.12 and vertical_disp > lateral_disp:
        return 1  # WILL_CROSS  (moving toward/away from camera = crossing)

    if lateral_disp > 0.12 and lateral_disp >= vertical_disp:
        return 3  # LANE_CHANGE (moving sideways)

    return 1  # Default: some movement = will cross


# ── Feature extractor (matches train.py format) ───────────────────────────────
def extract_features(traj: list, max_len: int = SEQ_LEN) -> np.ndarray:
    """
    traj: list of [cx, cy] pixel coords.
    Returns flat feature vector of length max_len * 6:
        [x, y, vx, vy, ax, ay] per timestep
    """
    pts = np.array(traj[-max_len:], dtype=np.float32)
    pts[:, 0] /= IMG_W
    pts[:, 1] /= IMG_H

    features = []
    for i in range(len(pts)):
        pos = pts[i]
        vel = pts[i] - pts[i-1] if i > 0 else np.zeros(2)
        acc = vel - (pts[i-1] - pts[i-2]) if i > 1 else np.zeros(2)
        features.extend([pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]])

    target_len = max_len * 6
    features = np.array(features, dtype=np.float32)
    if len(features) < target_len:
        features = np.pad(features, (0, target_len - len(features)))
    else:
        features = features[:target_len]

    return features


# ── Sliding-window segmenter ──────────────────────────────────────────────────
def segment_trajectory(traj: list, window: int = SEQ_LEN, step: int = 5):
    """Slide a window over a long track to produce multiple training samples."""
    segments = []
    if len(traj) < window:
        segments.append(traj)
    else:
        for start in range(0, len(traj) - window + 1, step):
            segments.append(traj[start:start + window])
    return segments


# ── Main extractor ────────────────────────────────────────────────────────────
def extract(clips_dir: Path, output_path: Path, max_clips: int = None,
            yolo_model_path: str = "yolov8n.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    print(f"[ProxiSense] Loading YOLOv8 model: {yolo_model_path}")
    model = YOLO(yolo_model_path)

    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No .mp4 clips found in {clips_dir}")

    if max_clips:
        clips = clips[:max_clips]

    print(f"[ProxiSense] Processing {len(clips)} clips from {clips_dir}")

    all_X, all_y = [], []
    label_counts  = defaultdict(int)
    LABEL_NAMES   = {0: "STATIONARY", 1: "WILL_CROSS", 2: "ERRATIC", 3: "LANE_CHANGE"}

    # YOLO pedestrian/cyclist/car class IDs (COCO)
    PERSON_CLASS = 0
    CYCLE_CLASS  = 1
    CAR_CLASSES  = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    t0 = time.time()
    for clip_idx, clip_path in enumerate(clips):
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"  [SKIP] Cannot open {clip_path.name}")
            continue

        tracker   = SimpleTracker()
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % FRAME_SKIP != 0:
                continue

            h, w = frame.shape[:2]
            results = model(frame, conf=CONF_THRESH, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls  = int(box.cls[0])
                if cls not in {PERSON_CLASS, CYCLE_CLASS} | CAR_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # scale to standard resolution
                x1 = x1 * IMG_W / w; x2 = x2 * IMG_W / w
                y1 = y1 * IMG_H / h; y2 = y2 * IMG_H / h
                detections.append([x1, y1, x2, y2])

            tracker.update(detections)

            # collect completed tracks
            for tid, traj in tracker.get_finished_tracks().items():
                for seg in segment_trajectory(traj):
                    if len(seg) >= MIN_SEQ_LEN:
                        label  = heuristic_label(seg)
                        feats  = extract_features(seg)
                        all_X.append(feats)
                        all_y.append(label)
                        label_counts[label] += 1

        cap.release()

        # flush remaining tracks at clip end
        for tid, traj in tracker.flush().items():
            for seg in segment_trajectory(traj):
                if len(seg) >= MIN_SEQ_LEN:
                    label  = heuristic_label(seg)
                    feats  = extract_features(seg)
                    all_X.append(feats)
                    all_y.append(label)
                    label_counts[label] += 1

        elapsed = time.time() - t0
        print(f"  [{clip_idx+1:03d}/{len(clips)}] {clip_path.name} | "
              f"Samples so far: {len(all_X)} | Elapsed: {elapsed:.1f}s")

    if not all_X:
        print("[ERROR] No trajectories extracted! Check clip path or YOLO model.")
        return

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), X=X, y=y)

    print(f"\n[ProxiSense] Extraction complete!")
    print(f"  Total samples : {len(X)}")
    print(f"  Feature shape : {X.shape}")
    print(f"  Label distribution:")
    for cls_id, name in LABEL_NAMES.items():
        count = label_counts[cls_id]
        pct   = 100.0 * count / max(len(X), 1)
        print(f"    {name:12s}: {count:5d}  ({pct:.1f}%)")
    print(f"  Saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract pedestrian trajectories from JAAD clips using YOLOv8"
    )
    parser.add_argument(
        "--clips-dir", type=str,
        default="data/datasets/jaad/JAAD_clips",
        help="Directory containing JAAD .mp4 clips"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/datasets/jaad/trajectories.npz",
        help="Output .npz file path"
    )
    parser.add_argument(
        "--max-clips", type=int, default=None,
        help="Limit number of clips (for quick testing)"
    )
    parser.add_argument(
        "--yolo-model", type=str, default="yolov8n.pt",
        help="Path to YOLOv8 weights"
    )
    args = parser.parse_args()

    extract(
        clips_dir        = Path(args.clips_dir),
        output_path      = Path(args.output),
        max_clips        = args.max_clips,
        yolo_model_path  = args.yolo_model,
    )


if __name__ == "__main__":
    main()
