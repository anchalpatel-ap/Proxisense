import numpy as np
from typing import List, Optional, Tuple


class TrajectoryEncoder:
    def __init__(self, input_len: int = 15, predict_len: int = 10):
        self.input_len = input_len
        self.predict_len = predict_len

    def encode(self, trajectory: np.ndarray, image_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        trajectory = np.asarray(trajectory, dtype=np.float32)
        if len(trajectory) < 2:
            return np.array([])

        raw = trajectory[-self.input_len:].copy()
        raw[:, 0] /= image_size[0]
        raw[:, 1] /= image_size[1]

        padded = np.pad(raw, ((0, max(0, self.input_len - len(raw))), (0, 0)), mode='edge')[:self.input_len]

        features = []
        for i in range(self.input_len):
            pos = padded[i]
            window = padded[max(0, i - 10):i + 1]
            if len(window) >= 2:
                vel = np.mean(np.diff(window, axis=0), axis=0)
            else:
                vel = np.zeros(2, dtype=np.float32)
            if len(window) >= 3:
                acc = np.mean(np.diff(window, n=2, axis=0), axis=0)
            else:
                acc = np.zeros(2, dtype=np.float32)
            features.extend([pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]])

        return np.array(features, dtype=np.float32)

    def decode(self, sequence: np.ndarray) -> np.ndarray:
        return sequence

    def compute_heading(self, trajectory: np.ndarray) -> float:
        if len(trajectory) < 2:
            return 0.0
        dx = trajectory[-1, 0] - trajectory[0, 0]
        dy = trajectory[-1, 1] - trajectory[0, 1]
        return np.degrees(np.arctan2(dy, dx))

    def compute_speed(self, trajectory: np.ndarray, fps: float = 30.0) -> float:
        if len(trajectory) < 2:
            return 0.0
        displacements = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        return float(np.mean(displacements) * fps)


class IntentClassifier:
    INTENT_LABELS = {0: "STATIONARY", 1: "WILL_CROSS", 2: "ERRATIC", 3: "LANE_CHANGE"}

    @staticmethod
    def classify_from_heuristics(trajectory: np.ndarray, fps: float = 30.0) -> Tuple[int, float]:
        if len(trajectory) < 5:
            return 0, 0.0

        speed = TrajectoryEncoder().compute_speed(trajectory, fps)
        heading = TrajectoryEncoder().compute_heading(trajectory)
        displacements = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        heading_deltas = np.abs(np.diff(np.arctan2(
            np.diff(trajectory[:, 1]), np.diff(trajectory[:, 0])
        )))

        direction_change = float(np.std(heading_deltas)) if len(heading_deltas) > 0 else 0.0
        speed_variance = float(np.var(displacements)) if len(displacements) > 0 else 0.0

        if speed < 0.5:
            return 0, 0.85

        if direction_change > 0.5 or speed_variance > 0.3:
            return 2, min(0.9, 0.5 + direction_change)

        if abs(heading) > 30 and abs(heading) < 150:
            return 1, min(0.9, 0.6 + abs(heading) / 300)

        return 3, 0.6
