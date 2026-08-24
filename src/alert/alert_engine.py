import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Alert:
    track_id: int
    class_name: str
    intent_label: str
    intent_confidence: float
    ttc: float
    zone: str
    bbox: List[float]
    braking_signal: bool = False


class AlertEngine:
    """
    Graduated alert engine: GREEN / AMBER / RED.

    Gating rules (both must be true to escalate beyond GREEN):
    1. TTC must be within the threshold band
    2. Intent must be WILL_CROSS or ERRATIC
    3. LSTM confidence must be >= confidence_threshold (default 0.75)

    STATIONARY or LANE_CHANGE agents never trigger AMBER/RED regardless of TTC,
    because they are not on a collision course with the ego vehicle.
    """

    # Intent classes that can actually cause a collision with the ego vehicle
    DANGEROUS_INTENTS = {"WILL_CROSS", "ERRATIC"}

    def __init__(self, ttc_red: float = 2.0, ttc_amber: float = 4.0,
                 ttc_green: float = 8.0, braking_threshold: float = 2.0,
                 confidence_threshold: float = 0.75):
        self.ttc_red              = ttc_red
        self.ttc_amber            = ttc_amber
        self.ttc_green            = ttc_green
        self.braking_threshold    = braking_threshold
        self.confidence_threshold = confidence_threshold

    def evaluate(self, track_id: int, class_name: str, bbox: List[float],
                 intent_label: str, intent_confidence: float, ttc: float) -> Alert:

        zone    = self._get_zone(intent_label, intent_confidence, ttc)
        braking = self._should_brake(intent_label, intent_confidence, ttc)

        return Alert(
            track_id         = track_id,
            class_name       = class_name,
            intent_label     = intent_label,
            intent_confidence= intent_confidence,
            ttc              = ttc,
            zone             = zone,
            bbox             = bbox,
            braking_signal   = braking
        )

    def _get_zone(self, intent_label: str, confidence: float, ttc: float) -> str:
        """
        Zone logic:
        - STATIONARY / LANE_CHANGE / low-confidence → cap at GREEN
        - WILL_CROSS / ERRATIC + high confidence → full RED/AMBER/GREEN scale
        """
        ttc_zone = self._raw_ttc_zone(ttc)

        # Low confidence or non-threatening intent → never worse than GREEN
        is_dangerous = (intent_label in self.DANGEROUS_INTENTS
                        and confidence >= self.confidence_threshold)

        if not is_dangerous:
            # Downgrade RED/AMBER to GREEN for non-threatening agents
            if ttc_zone in ("RED", "AMBER"):
                return "GREEN"
            return ttc_zone

        return ttc_zone

    def _raw_ttc_zone(self, ttc: float) -> str:
        if ttc <= self.ttc_red:
            return "RED"
        elif ttc <= self.ttc_amber:
            return "AMBER"
        elif ttc <= self.ttc_green:
            return "GREEN"
        return "SAFE"

    def _should_brake(self, intent_label: str, confidence: float, ttc: float) -> bool:
        if ttc > self.braking_threshold:
            return False
        if intent_label in self.DANGEROUS_INTENTS and confidence >= self.confidence_threshold:
            return True
        if ttc <= 1.0:   # imminent regardless of intent
            return True
        return False

    def get_zone_color(self, zone: str) -> Tuple[int, int, int]:
        return {
            "RED"  : (0,   0,   255),
            "AMBER": (0,   165, 255),
            "GREEN": (0,   255, 0  ),
            "SAFE" : (200, 200, 200)
        }.get(zone, (200, 200, 200))

    def get_system_status(self, alerts: List[Alert]) -> Dict:
        if not alerts:
            return {"status": "SAFE", "level": 0, "message": "No agents detected"}

        level_map = {"SAFE": 0, "GREEN": 1, "AMBER": 2, "RED": 3}
        max_level = 0
        status    = "SAFE"

        for alert in alerts:
            level = level_map.get(alert.zone, 0)
            if level > max_level:
                max_level = level
                status    = alert.zone

        dangerous = [a for a in alerts if a.zone in ("AMBER", "RED")]
        msg = (f"{len(dangerous)} high-risk agent(s) — {status}"
               if dangerous else
               f"{len(alerts)} agent(s) tracked — {status}")

        return {"status": status, "level": max_level, "message": msg}
