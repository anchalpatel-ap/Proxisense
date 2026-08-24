"""
Training script for the LSTM Intent Prediction model.
Supports training on JAAD/ETH datasets or synthetic data from CARLA.

Usage:
    python -m src.prediction.train --epochs 50 --data data/datasets/jaad
"""

import numpy as np
import argparse
import json
from pathlib import Path


def generate_synthetic_data(num_samples: int = 5000, seq_length: int = 15) -> tuple:
    np.random.seed(42)
    X, y = [], []

    for _ in range(num_samples):
        intent = np.random.randint(0, 4)

        if intent == 0:
            t = np.zeros((seq_length, 2))
            t[:, 0] = np.random.uniform(0.3, 0.7)
            t[:, 1] = np.random.uniform(0.3, 0.7)

        elif intent == 1:
            start_y = np.random.uniform(0.2, 0.4)
            t = np.zeros((seq_length, 2))
            t[:, 0] = np.random.uniform(0.3, 0.7)
            t[:, 1] = np.linspace(start_y, start_y + np.random.uniform(0.3, 0.6), seq_length)

        elif intent == 2:
            t = np.zeros((seq_length, 2))
            t[0] = [np.random.uniform(0.3, 0.7), np.random.uniform(0.2, 0.6)]
            for i in range(1, seq_length):
                t[i] = t[i-1] + np.random.randn(2) * 0.05

        else:
            start_x = np.random.uniform(0.2, 0.4)
            t = np.zeros((seq_length, 2))
            t[:, 0] = np.linspace(start_x, start_x + np.random.uniform(0.3, 0.5), seq_length)
            t[:, 1] = np.random.uniform(0.3, 0.7)

        features = []
        for i in range(seq_length):
            window = t[max(0, i - 10):i + 1]
            if len(window) >= 2:
                vel = np.diff(window, axis=0).mean(axis=0) if len(window) > 1 else np.zeros(2)
            else:
                vel = np.zeros(2)
            if len(window) >= 3:
                acc = np.diff(window, n=2, axis=0).mean(axis=0) if len(window) > 2 else np.zeros(2)
            else:
                acc = np.zeros(2)

            features.extend([t[i, 0], t[i, 1], vel[0], vel[1], acc[0], acc[1]])

        features = np.array(features).flatten()
        target_len = seq_length * 6
        if len(features) < target_len:
            features = np.pad(features, (0, target_len - len(features)))
        else:
            features = features[:target_len]

        X.append(features)
        y.append(intent)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_lstm(X_train, y_train, X_val, y_val, args):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("PyTorch is required for training. pip install torch")

    input_size = X_train.shape[1]

    class LSTMIntentModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=4):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.2 if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            out = self.fc(hn[-1])
            return out

    model = LSTMIntentModel(input_size, args.hidden_size, args.num_layers, 4)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).unsqueeze(1),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).unsqueeze(1),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Path(args.output_dir) / "lstm_intent_best.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'num_classes': 4,
                'val_acc': val_acc,
                'epoch': epoch
            }, str(model_path))
            print(f"  -> Saved best model ({val_acc:.2f}%) to {model_path}")

    final_path = Path(args.output_dir) / "lstm_intent_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_classes': 4
    }, str(final_path))

    with open(Path(args.output_dir) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Intent Prediction Model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to real dataset (uses synthetic if not provided)")
    parser.add_argument("--output-dir", type=str, default="models/prediction")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    print("Generating training data...")
    X, y = generate_synthetic_data(args.num_samples)

    split = int(len(X) * (1 - args.val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Input feature size: {X.shape[1]}")

    model = train_lstm(X_train, y_train, X_val, y_val, args)

    print("\nTo export to ONNX:")
    print("  python -m src.prediction.export_onnx")


if __name__ == "__main__":
    main()
