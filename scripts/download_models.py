"""
Download pre-trained YOLOv8 model and train initial LSTM intent model.

Usage:
    python scripts/download_models.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_yolo():
    print("[1/2] Downloading YOLOv8-nano...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model_path = Path("models/detection/yolov8n.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"  -> YOLOv8-nano saved to {model_path}")
        return True
    except ImportError:
        print("  -> ultralytics not installed. Run: pip install ultralytics")
        return False


def train_lstm():
    print("\n[2/2] Training LSTM Intent Prediction Model on synthetic data...")
    try:
        from src.prediction.train import generate_synthetic_data, train_lstm
        import argparse

        class Args:
            epochs = 30
            batch_size = 32
            hidden_size = 64
            num_layers = 2
            lr = 0.001
            cpu = True
            output_dir = "models/prediction"
            num_samples = 5000
            val_split = 0.2

        args = Args()

        print("  Generating synthetic trajectory data...")
        X, y = generate_synthetic_data(args.num_samples)

        split = int(len(X) * (1 - args.val_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        print(f"  Training samples: {len(X_train)}, Validation: {len(X_val)}")
        train_lstm(X_train, y_train, X_val, y_val, args)

        print("  -> LSTM model trained and saved to models/prediction/")
        return True
    except Exception as e:
        print(f"  -> LSTM training failed: {e}")
        return False


def main():
    print("=" * 50)
    print("ProxiSense - Model Setup")
    print("=" * 50)

    if download_yolo():
        print("\n✅ YOLOv8-nano ready")
    else:
        print("\n⚠️  YOLO download skipped")

    if train_lstm():
        print("\n✅ LSTM intent model ready")
    else:
        print("\n⚠️  LSTM training skipped (will use heuristic fallback)")

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nTo start the pipeline:")
    print("  streamlit run src/dashboard/app.py")
    print("  python -m src.dashboard.app")
    print("=" * 50)


if __name__ == "__main__":
    main()
