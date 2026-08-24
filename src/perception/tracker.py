import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional


class TrackState:
    NEW = 0
    ACTIVE = 1
    LOST = 2


class Track:
    def __init__(self, track_id: int, bbox: List[float], det_class: str, confidence: float):
        self.track_id = track_id
        self.det_class = det_class
        self.confidence = confidence
        self.state = TrackState.NEW
        self.hits = 1
        self.no_losses = 0
        self.age = 1

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.tlwh = [bbox[0], bbox[1], w, h]
        self.tlbr = bbox
        self.centroid = np.array([cx, cy], dtype=np.float32)
        self.trajectory = [self.centroid.copy()]
        self.bbox_history = [self.tlbr]

    def update(self, bbox: List[float], confidence: float):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.tlwh = [bbox[0], bbox[1], w, h]
        self.tlbr = bbox
        self.centroid = np.array([cx, cy], dtype=np.float32)
        self.confidence = confidence
        self.hits += 1
        self.no_losses = 0
        self.age += 1
        self.state = TrackState.ACTIVE
        self.trajectory.append(self.centroid.copy())
        self.bbox_history.append(self.tlbr)

    def mark_lost(self):
        self.state = TrackState.LOST
        self.no_losses += 1


class ByteTrack:
    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8,
                 max_lost: int = 30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}

    def _iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _match(self, detections: List[dict], track_list: Dict[int, Track]) -> tuple:
        matches, unmatched_dets, unmatched_tracks = [], [], []

        if not detections or not track_list:
            return matches, list(range(len(detections))), list(track_list.keys())

        track_ids = list(track_list.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))

        for d_idx, det in enumerate(detections):
            for t_idx, t_id in enumerate(track_ids):
                iou_matrix[d_idx, t_idx] = self._iou(
                    det["bbox"], track_list[t_id].tlbr
                )

        for d_idx in range(len(detections)):
            best_t_idx = iou_matrix[d_idx].argmax()
            best_iou = iou_matrix[d_idx, best_t_idx]
            if best_iou >= self.match_thresh:
                matches.append((d_idx, track_ids[best_t_idx]))
            else:
                unmatched_dets.append(d_idx)

        matched_tracks = set(t_id for _, t_id in matches)
        unmatched_tracks = [t_id for t_id in track_ids
                            if t_id not in matched_tracks]

        return matches, unmatched_dets, unmatched_tracks

    def update(self, detections: List[dict]) -> Dict[int, Track]:
        high_score_dets = [d for d in detections
                           if d["confidence"] >= self.track_thresh]
        low_score_dets = [d for d in detections
                          if d["confidence"] < self.track_thresh]

        matches, unmatched_high, unmatched_active = self._match(
            high_score_dets, self.tracks
        )

        for d_idx, t_id in matches:
            self.tracks[t_id].update(
                high_score_dets[d_idx]["bbox"],
                high_score_dets[d_idx]["confidence"]
            )

        for t_id in unmatched_active:
            if self.tracks[t_id].hits >= 1:
                self.tracks[t_id].mark_lost()
                self.lost_tracks[t_id] = self.tracks[t_id]

        for d_idx in unmatched_high:
            det = high_score_dets[d_idx]
            track = Track(self.next_id, det["bbox"], det["class_name"],
                          det["confidence"])
            self.tracks[self.next_id] = track
            self.next_id += 1

        lost_matches, _, _ = self._match(
            low_score_dets, self.lost_tracks
        )
        for d_idx, t_id in lost_matches:
            if t_id not in self.lost_tracks:
                continue   # already recovered — skip to avoid KeyError
            self.tracks[t_id] = self.lost_tracks.pop(t_id)
            self.tracks[t_id].update(
                low_score_dets[d_idx]["bbox"],
                low_score_dets[d_idx]["confidence"]
            )

        expired = [e_id for e_id, track in self.lost_tracks.items()
                   if track.no_losses > self.max_lost]
        for e_id in expired:
            del self.lost_tracks[e_id]

        active = {a_id: track for a_id, track in self.tracks.items()
                  if track.state != TrackState.LOST}
        return active

    def get_trajectories(self) -> Dict[int, np.ndarray]:
        return {
            t_id: np.array(track.trajectory)
            for t_id, track in self.tracks.items()
            if track.state != TrackState.LOST and len(track.trajectory) >= 2
        }
