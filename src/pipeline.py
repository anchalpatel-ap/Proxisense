"""
ProxiSense Main Pipeline
Orchestrates Perception -> Prediction -> Alert across all layers.
"""

import cv2
import numpy as np
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ProxiSensePipeline:
    def __init__(self, config_path: str = "config/config.yaml",
                 use_webcam: bool = False, video_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.use_webcam = use_webcam
        self.video_path = video_path
        self.cap = None
        self.prev_frame_time = 0
        self.fps_values = []
        self.frame_size = tuple(self.config["model"]["detection"]["input_size"])

        self._init_layers()
        self._init_video()

    def _load_config(self, path: str) -> dict:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def _init_layers(self):
        from src.perception.detector import YOLODetector
        from src.perception.tracker import ByteTrack
        from src.prediction.trajectory import TrajectoryEncoder, IntentClassifier
        from src.prediction.lstm_model import LSTMPredictor
        from src.alert.ttc import TTCCalculator
        from src.alert.alert_engine import AlertEngine

        det_cfg = self.config.get("model", {}).get("detection", {})
        pred_cfg = self.config.get("model", {}).get("prediction", {})
        track_cfg = self.config.get("tracking", {})
        alert_cfg = self.config.get("alert", {})

        self.detector = YOLODetector(
            model_path=det_cfg.get("name", "yolov8n.pt"),
            conf_threshold=det_cfg.get("conf_threshold", 0.5),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            input_size=tuple(det_cfg.get("input_size", [640, 640])),
            half_precision=det_cfg.get("half_precision", False)
        )

        self.tracker = ByteTrack(
            track_thresh=track_cfg.get("track_thresh", 0.5),
            match_thresh=track_cfg.get("match_thresh", 0.8),
            max_lost=track_cfg.get("max_lost", 30)
        )

        self.traj_encoder = TrajectoryEncoder(
            input_len=pred_cfg.get("input_len", 15),
            predict_len=pred_cfg.get("predict_len", 10)
        )

        self.intent_classifier = IntentClassifier()

        model_path = Path("models/prediction/lstm_intent_best.onnx")
        if not model_path.exists():
            model_path = Path("models/prediction/lstm_intent_best.pt")
        if model_path.exists():
            self.lstm_model = LSTMPredictor(
                model_path=str(model_path),
                confidence_threshold=pred_cfg.get("confidence_threshold", 0.75)
            )
        else:
            self.lstm_model = None
            print("[ProxiSense] No trained LSTM model found — using heuristic fallback")

        self.ttc_calc = TTCCalculator()
        self.alert_engine = AlertEngine(
            ttc_red=alert_cfg.get("ttc_thresholds", {}).get("red", 2.0),
            ttc_amber=alert_cfg.get("ttc_thresholds", {}).get("amber", 4.0),
            ttc_green=alert_cfg.get("ttc_thresholds", {}).get("green", 8.0),
            braking_threshold=alert_cfg.get("braking_signal_threshold", 2.0),
            confidence_threshold=pred_cfg.get("confidence_threshold", 0.75)
        )

        self.prev_bboxes = {}
        self._last_alerts  = []
        self._last_metrics = {"fps": 0, "inference_time": 0, "detections": 0, "tracks": 0}

    def _init_video(self):
        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)
        elif self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(0)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Could not open video source")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])

    def process_frame(self, run_detection: bool = True) -> Tuple[np.ndarray, List, Dict]:
        ret, frame = self.cap.read()
        if not ret:
            return None, [], {}

        frame = cv2.resize(frame, self.frame_size)

        # ── Skip frame: reuse last detection results, only render overlay ──
        if not run_detection:
            current_time = time.time()
            fps = 1.0 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
            self.prev_frame_time = current_time
            self.fps_values.append(fps)
            avg_fps = float(np.mean(self.fps_values[-30:]))
            metrics = dict(self._last_metrics)
            metrics["fps"] = round(avg_fps, 1)
            metrics["inference_time"] = 0.0
            return frame, self._last_alerts, metrics

        start_time = time.perf_counter()
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)

        alerts = []
        for track_id, track in tracks.items():
            intent_label = "STATIONARY"
            intent_conf = 0.0

            if self.lstm_model and len(track.trajectory) >= 5:
                features = self.traj_encoder.encode(
                    np.array(track.trajectory), self.frame_size
                )
                if features.size > 0:
                    pred_class, intent_conf = self.lstm_model.predict(features)
                    intent_label = self.lstm_model.get_label(pred_class)
            else:
                pred_class, intent_conf = self.intent_classifier.classify_from_heuristics(
                    np.array(track.trajectory)
                )
                intent_label = self.intent_classifier.INTENT_LABELS[pred_class]

            prev_bbox = self.prev_bboxes.get(track_id)
            ttc, approach_speed = self.ttc_calc.calculate(
                track.tlbr, prev_bbox, fps=30.0, track_id=track_id
            )

            alert = self.alert_engine.evaluate(
                track_id=track_id,
                class_name=track.det_class,
                bbox=track.tlbr,
                intent_label=intent_label,
                intent_confidence=intent_conf,
                ttc=ttc
            )
            alerts.append(alert)
            self.prev_bboxes[track_id] = track.tlbr

        inference_time = (time.perf_counter() - start_time) * 1000
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = current_time
        self.fps_values.append(fps)
        avg_fps = np.mean(self.fps_values[-30:]) if self.fps_values else 0

        metrics = {
            "fps": round(avg_fps, 1),
            "inference_time": round(inference_time, 1),
            "detections": len(detections),
            "tracks": len(tracks)
        }

        # Cache for skip frames
        self._last_alerts  = alerts
        self._last_metrics = metrics

        return frame, alerts, metrics

    def render_alerts(self, frame: np.ndarray, alerts: List) -> np.ndarray:
        overlay = frame.copy()

        for alert in alerts:
            x1, y1, x2, y2 = map(int, alert.bbox)
            color = self.alert_engine.get_zone_color(alert.zone)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if alert.braking_signal:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

            label = f"ID:{alert.track_id} {alert.intent_label}"
            conf_text = f"{alert.intent_confidence:.2f}"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ttc_text = f"TTC:{alert.ttc:.1f}s" if alert.ttc != float('inf') else "TTC:∞"
            cv2.putText(frame, ttc_text, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if alert.braking_signal:
                cv2.putText(frame, "BRAKE", (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        status = self.alert_engine.get_system_status(alerts)
        status_color = {"SAFE": (0, 255, 0), "GREEN": (0, 255, 0),
                        "AMBER": (0, 165, 255), "RED": (0, 0, 255)}
        cv2.putText(frame, f"STATUS: {status['status']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color.get(status['status'], (255, 255, 255)), 2)

        return frame

    def render_trajectories(self, frame: np.ndarray) -> np.ndarray:
        trajectories = self.tracker.get_trajectories()
        for t_id, traj in trajectories.items():
            for i in range(1, len(traj)):
                alpha = i / len(traj)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                pt1 = tuple(map(int, traj[i - 1]))
                pt2 = tuple(map(int, traj[i]))
                cv2.line(frame, pt1, pt2, color, 2)
        return frame

    def render_zone_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        center_x, center_y = w // 2, h // 2

        for zone_name, color, distance in [
            ("SAFE", (255, 255, 255), h),
            ("GREEN", (0, 255, 0), int(h * 0.7)),
            ("AMBER", (0, 165, 255), int(h * 0.4)),
            ("RED", (0, 0, 255), int(h * 0.2))
        ]:
            pts = np.array([
                [center_x - distance, h],
                [center_x + distance, h],
                [center_x + distance, h],
                [center_x - distance, h]
            ], np.int32)

        return frame

    def get_fps(self) -> float:
        return np.mean(self.fps_values[-30:]) if self.fps_values else 0

    def release(self):
        if self.cap:
            self.cap.release()
