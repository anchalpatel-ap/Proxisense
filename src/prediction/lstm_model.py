import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class LSTMPredictor:
    def __init__(self, model_path: Optional[str] = None,
                 input_size: int = 90, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 4,
                 confidence_threshold: float = 0.75, device: str = "cpu"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.onnx_session = None

        self.intent_labels = {
            0: "STATIONARY",
            1: "WILL_CROSS",
            2: "ERRATIC",
            3: "LANE_CHANGE"
        }

        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_path.suffix == ".onnx":
            self._load_onnx(model_path)
        else:
            self._load_torch(model_path)

    def _load_onnx(self, model_path: Path):
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"] if self.device == "cpu" \
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.onnx_session = ort.InferenceSession(
            str(model_path), providers=providers
        )
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def _load_torch(self, model_path: Path):
        try:
            import torch
            import torch.nn as nn

            class LSTMIntentModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, num_classes):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                        batch_first=True, dropout=0.2 if num_layers > 1 else 0)
                    self.fc = nn.Linear(hidden_size, num_classes)

                def forward(self, x):
                    _, (hn, _) = self.lstm(x)
                    return self.fc(hn[-1])

            self.model = LSTMIntentModel(
                self.input_size, self.hidden_size,
                self.num_layers, self.num_classes
            )
            state = torch.load(str(model_path), map_location=self.device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)
            self.model.eval()
        except ImportError:
            raise ImportError("PyTorch not installed for model loading")

    def predict(self, features: np.ndarray, return_all: bool = False) -> Tuple[int, float]:
        if self.onnx_session:
            return self._predict_onnx(features, return_all)
        return self._predict_torch(features, return_all)

    def _predict_torch(self, features: np.ndarray, return_all: bool) -> Tuple[int, float]:
        import torch
        with torch.no_grad():
            inp = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            inp = inp.to(self.device)
            outputs = self.model(inp)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

        if return_all:
            return pred_class, confidence, probs.tolist()
        return pred_class, confidence

    def _predict_onnx(self, features: np.ndarray, return_all: bool) -> Tuple[int, float]:
        inp = features.astype(np.float32).reshape(1, 1, -1)
        outputs = self.onnx_session.run(
            [self.output_name], {self.input_name: inp}
        )[0]
        probs = outputs[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        if return_all:
            return pred_class, confidence, probs.tolist()
        return pred_class, confidence

    def should_alert(self, intent_class: int, confidence: float) -> bool:
        return (intent_class > 0 and confidence >= self.confidence_threshold)

    def get_label(self, intent_class: int) -> str:
        return self.intent_labels.get(intent_class, "UNKNOWN")
