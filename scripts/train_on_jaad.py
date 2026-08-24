"""
ProxiSense — Train LSTM on JAAD dataset
========================================
Loading priority:
  1. Pre-extracted trajectories.npz  (from scripts/extract_jaad_trajectories.py)
  2. annotations.json                (official JAAD annotations, if present)
  3. Synthetic data fallback

Usage:
    python scripts/train_on_jaad.py
    python scripts/train_on_jaad.py --data-dir data/datasets/jaad --epochs 50
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Ensure the project root (parent of this script's directory) is on sys.path
# so that `src.*` imports work regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Feature extractor (annotation JSON path) ─────────────────────────────────
def _extract_features(traj: np.ndarray, max_len: int = 15) -> np.ndarray:
    traj = traj[-max_len:]
    traj_normalized = traj / np.array([1920, 1080])

    features = []
    for i in range(len(traj_normalized)):
        pos = traj_normalized[i]
        vel = traj_normalized[i] - traj_normalized[i-1] if i > 0 else np.zeros(2)
        acc = vel - (traj_normalized[i-1] - traj_normalized[i-2]) if i > 1 else np.zeros(2)
        features.extend([pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]])

    target_len = max_len * 6
    features = np.array(features)
    if len(features) < target_len:
        features = np.pad(features, (0, target_len - len(features)))
    else:
        features = features[:target_len]
    return features


def _map_action_to_intent(action: str) -> int:
    mapping = {
        "standing":    0,
        "walking":     1,
        "crossing":    1,
        "running":     2,
        "stopping":    0,
        "lane_change": 3,
    }
    return mapping.get(action.lower(), 1)


def load_annotation_json(data_dir: Path):
    """Parse official JAAD annotations.json if present."""
    annotations_file = data_dir / "annotations.json"
    if not annotations_file.exists():
        return None

    import json
    print(f"  Reading {annotations_file}")
    with open(annotations_file) as f:
        annotations = json.load(f)

    X, y = [], []
    for seq in annotations:
        for ped in seq.get("pedestrians", []):
            traj = [[f.get("x", 0), f.get("y", 0)] for f in ped.get("frames", [])]
            if len(traj) >= 15:
                feats = _extract_features(np.array(traj))
                X.append(feats)
                y.append(_map_action_to_intent(ped.get("action", "walking")))

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64)) if X else None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train LSTM on JAAD trajectories")
    parser.add_argument("--data-dir", type=str, default="data/datasets/jaad",
                        help="Root JAAD data directory")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data     = None
    LABEL_NAMES = {0: "STATIONARY", 1: "WILL_CROSS", 2: "ERRATIC", 3: "LANE_CHANGE"}

    # ── Priority 1: Pre-extracted .npz ────────────────────────────────────
    npz_path = data_dir / "trajectories.npz"
    if npz_path.exists():
        print(f"[ProxiSense] Loading pre-extracted trajectories: {npz_path}")
        npz  = np.load(str(npz_path))
        X, y = npz["X"], npz["y"]
        print(f"  Samples      : {len(X)}")
        print(f"  Feature size : {X.shape[1]}")
        print(f"  Label distribution:")
        for cls_id, name in LABEL_NAMES.items():
            count = int((y == cls_id).sum())
            pct   = 100.0 * count / max(len(y), 1)
            print(f"    {name:12s}: {count:5d}  ({pct:.1f}%)")
        data = (X, y)

    # ── Priority 2: Annotation JSON ───────────────────────────────────────
    elif data_dir.exists():
        print(f"[ProxiSense] No trajectories.npz found. Trying annotations.json...")
        data = load_annotation_json(data_dir)
        if data is None:
            print("  No annotations.json found either.")

    # ── Priority 3: Synthetic fallback ────────────────────────────────────
    if data is None:
        print("[ProxiSense] Falling back to synthetic data (5 000 samples).")
        print("  Tip: Run  python scripts/extract_jaad_trajectories.py  first")
        print("       to train on your real JAAD clips instead.")
        from src.prediction.train import generate_synthetic_data
        X, y = generate_synthetic_data(5000)
    else:
        X, y = data

    # ── Shuffle & split ───────────────────────────────────────────────────
    np.random.seed(42)
    idx  = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split       = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"\n[ProxiSense] Train: {len(X_train)} | Validation: {len(X_val)}")
    print(f"[ProxiSense] Starting LSTM training ({args.epochs} epochs)...\n")

    import argparse as ap
    train_args = ap.Namespace(
        epochs      = args.epochs,
        batch_size  = 32,
        hidden_size = 64,
        num_layers  = 2,
        lr          = 0.001,
        cpu         = True,
        output_dir  = "models/prediction",
    )

    from src.prediction.train import train_lstm
    train_lstm(X_train, y_train, X_val, y_val, train_args)

    print("\n[ProxiSense] Next step — export model to ONNX:")
    print("  python src/prediction/export_onnx.py "
          "--model models/prediction/lstm_intent_best.pt")


if __name__ == "__main__":
    main()