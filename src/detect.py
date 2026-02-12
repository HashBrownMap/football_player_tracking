"""Player detection (YOLOv8) and tracking (ByteTrack)."""

import numpy as np
import supervision as sv
from ultralytics import YOLO


class Detector:
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence: float = 0.3,
        device: str = "mps",
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        """Run detection + tracking on a single frame.

        Returns sv.Detections with tracker_id populated.
        """
        results = self.model(
            frame, device=self.device, conf=self.confidence, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        # Filter to person class only (COCO class 0)
        detections = detections[detections.class_id == 0]
        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
        return detections

    def reset(self):
        """Reset tracker state (e.g., between sampling and full processing)."""
        self.tracker.reset()
