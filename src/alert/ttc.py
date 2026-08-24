import numpy as np
from typing import Optional, Tuple


class TTCCalculator:
    """
    Time-To-Collision estimator using bounding-box optical expansion.

    Key fixes over naive version:
    - Minimum change threshold to ignore sub-pixel noise
    - Smoothed TTC using exponential moving average per track
    - Clamps unrealistically small TTC values caused by jitter
    """

    # Minimum bbox width/area change (pixels) to count as real approach
    MIN_WIDTH_DELTA = 1.5    # px
    MIN_AREA_DELTA  = 80.0   # px²
    TTC_CLAMP_LOW   = 0.5    # Never report TTC < 0.5 s (avoids noise spikes)
    TTC_CLAMP_HIGH  = 30.0   # Cap at 30 s — beyond that treat as SAFE

    def __init__(self, focal_length: float = 700.0, assumed_width: float = 0.5,
                 ema_alpha: float = 0.35):
        self.focal_length   = focal_length
        self.assumed_width  = assumed_width
        self.ema_alpha      = ema_alpha          # smoothing factor (0=more smooth)
        self._ttc_history: dict[int, float] = {}  # track_id → smoothed TTC

    def calculate(self, bbox_current: list, bbox_previous: Optional[list] = None,
                  fps: float = 30.0, track_id: int = -1) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox_current
        width  = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return float('inf'), 0.0

        cx   = (x1 + x2) / 2
        area = width * height

        if bbox_previous is None:
            self._ttc_history.pop(track_id, None)
            return float('inf'), 0.0

        px1, py1, px2, py2 = bbox_previous
        p_width = px2 - px1
        p_area  = p_width * (py2 - py1)
        p_cx    = (px1 + px2) / 2

        dw = width  - p_width
        da = area   - p_area
        dx = cx     - p_cx

        # ── Width-based TTC ───────────────────────────────────────────────
        if dw > self.MIN_WIDTH_DELTA:
            width_rate = dw * fps
            ttc_width  = float(width / (width_rate + 1e-6))
        else:
            ttc_width  = float('inf')

        # ── Area-based TTC ────────────────────────────────────────────────
        if da > self.MIN_AREA_DELTA:
            area_rate = da * fps
            ttc_area  = float(area / (area_rate + 1e-6))
        else:
            ttc_area = float('inf')

        raw_ttc = min(ttc_width, ttc_area)

        # Clamp extremes
        if raw_ttc != float('inf'):
            raw_ttc = float(np.clip(raw_ttc, self.TTC_CLAMP_LOW, self.TTC_CLAMP_HIGH))

        # Exponential moving average — smooth out per-frame jitter
        if track_id in self._ttc_history and raw_ttc != float('inf'):
            prev_smooth = self._ttc_history[track_id]
            if prev_smooth == float('inf'):
                smoothed = raw_ttc
            else:
                smoothed = self.ema_alpha * raw_ttc + (1 - self.ema_alpha) * prev_smooth
        else:
            smoothed = raw_ttc

        self._ttc_history[track_id] = smoothed
        approach_speed = abs(dx) * fps if abs(dx) > 0.5 else 0.0

        return smoothed, approach_speed

    def get_zone(self, ttc: float) -> str:
        if ttc <= 2.0:
            return "RED"
        elif ttc <= 4.0:
            return "AMBER"
        elif ttc <= 8.0:
            return "GREEN"
        return "SAFE"

    def clear_track(self, track_id: int):
        self._ttc_history.pop(track_id, None)
