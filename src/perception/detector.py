import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional


class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45, input_size: tuple = (640, 640),
                 half_precision: bool = True):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.half_precision = half_precision

        model_path = Path(model_path)
        self._load_model(model_path)

        self.class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            5: "bus", 7: "truck"
        }

    def _load_model(self, model_path: Path):
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            if self.half_precision:
                self.model.to("cpu").half()
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def detect(self, frame: np.ndarray) -> List[dict]:
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            cls_id = class_ids[i]
            if cls_id not in self.class_names:
                continue

            detections.append({
                "bbox": boxes[i].tolist(),
                "confidence": float(scores[i]),
                "class_id": int(cls_id),
                "class_name": self.class_names[cls_id]
            })

        return detections

    def __call__(self, frame: np.ndarray) -> List[dict]:
        return self.detect(frame)
