"""
Export trained PyTorch LSTM model to ONNX for edge inference.

Usage:
    python -m src.prediction.export_onnx --model models/prediction/lstm_intent_best.pt
"""

import argparse
from pathlib import Path


def export_to_onnx(model_path: str, output_path: str = None):
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch required. pip install torch")

    class LSTMIntentModel(nn.Module):
        def __init__(self, input_size=90, hidden_size=64, num_layers=2, num_classes=4):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.2 if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.softmax(self.fc(hn[-1]))

    checkpoint = torch.load(model_path, map_location="cpu")
    input_size = checkpoint.get("input_size", 90)
    hidden_size = checkpoint.get("hidden_size", 64)
    num_layers = checkpoint.get("num_layers", 2)
    num_classes = checkpoint.get("num_classes", 4)

    model = LSTMIntentModel(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    dummy_input = torch.randn(1, 1, input_size)

    if output_path is None:
        output_path = Path(model_path).with_suffix(".onnx")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False
    )

    print(f"Model exported to ONNX: {output_path}")

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX validated. Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"Outputs: {[o.name for o in onnx_model.graph.output]}")


def main():
    parser = argparse.ArgumentParser(description="Export LSTM to ONNX")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to PyTorch model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path (default: same path with .onnx)")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)


if __name__ == "__main__":
    main()
