"""Team classification via jersey color clustering."""

from collections import defaultdict

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self, n_sample_frames: int = 30, referee_threshold: float = 55.0):
        """
        Args:
            n_sample_frames: Number of frames to sample for initial clustering.
            referee_threshold: HSV distance above which a detection is labeled
                               as referee/other (-1) rather than a team.
        """
        self.n_sample_frames = n_sample_frames
        self.referee_threshold = referee_threshold
        self._color_samples: list[np.ndarray] = []
        self.cluster_centers: np.ndarray | None = None
        self._track_votes: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._track_assignments: dict[int, int] = {}
        self._vote_lock_threshold = 5  # lock after this many consistent votes

    # ------------------------------------------------------------------
    # Sampling phase
    # ------------------------------------------------------------------

    def collect_sample(self, frame: np.ndarray, detections: sv.Detections):
        """Extract torso colors from detections and store for later clustering."""
        for bbox in detections.xyxy:
            color = self._extract_torso_color(frame, bbox)
            if color is not None:
                self._color_samples.append(color)

    # ------------------------------------------------------------------
    # Clustering phase
    # ------------------------------------------------------------------

    def fit(self):
        """Run KMeans on collected samples. Call after sampling phase."""
        if len(self._color_samples) < 2:
            raise ValueError(
                f"Need at least 2 color samples to cluster, got {len(self._color_samples)}"
            )
        X = np.array(self._color_samples)
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        km.fit(X)
        self.cluster_centers = km.cluster_centers_

    # ------------------------------------------------------------------
    # Per-frame classification
    # ------------------------------------------------------------------

    def classify(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Classify each detection as team 0, team 1, or referee (-1).

        Returns:
            (N,) int array of team labels.
        """
        if self.cluster_centers is None:
            raise RuntimeError("Must call fit() before classify()")

        labels = np.full(len(detections), -1, dtype=int)
        tracker_ids = detections.tracker_id

        for idx, bbox in enumerate(detections.xyxy):
            tid = int(tracker_ids[idx]) if tracker_ids is not None else None

            # Use locked assignment if available
            if tid is not None and tid in self._track_assignments:
                labels[idx] = self._track_assignments[tid]
                continue

            color = self._extract_torso_color(frame, bbox)
            if color is None:
                # If torso is mostly yellow/green (all pixels excluded),
                # this is likely a referee — keep label as -1
                continue

            # Bright yellow detection: H 20-34, S >= 80 in HSV
            if 20 <= color[0] <= 34 and color[1] >= 80:
                labels[idx] = -1
                if tid is not None:
                    self._track_assignments[tid] = -1
                continue

            # Distance to each cluster center
            dists = np.linalg.norm(self.cluster_centers - color, axis=1)
            nearest = int(np.argmin(dists))
            min_dist = dists[nearest]

            if min_dist > self.referee_threshold:
                label = -1  # referee / other
            else:
                label = nearest

            labels[idx] = label

            # Vote tracking for stability
            if tid is not None and label >= 0:
                self._track_votes[tid][label] += 1
                votes = self._track_votes[tid]
                majority = max(votes, key=votes.get)
                if votes[majority] >= self._vote_lock_threshold:
                    self._track_assignments[tid] = majority
                    labels[idx] = majority

        return labels

    # ------------------------------------------------------------------
    # Color extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_torso_color(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        """Crop upper-middle portion of bbox (torso), mask green, return mean HSV."""
        x1, y1, x2, y2 = bbox.astype(int)
        h = y2 - y1
        w = x2 - x1
        if h < 10 or w < 5:
            return None

        # Upper 40-60% of the bbox (torso region)
        torso_top = y1 + int(h * 0.15)
        torso_bot = y1 + int(h * 0.55)
        # Narrow horizontally to avoid background bleed
        margin = int(w * 0.15)
        crop = frame[torso_top:torso_bot, x1 + margin : x2 - margin]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Mask out green pixels (field bleed-through)
        # and bright yellow pixels (referee kit)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        green_mask = (h_channel >= 35) & (h_channel <= 85)
        yellow_mask = (h_channel >= 20) & (h_channel <= 34) & (s_channel >= 80)
        exclude = green_mask | yellow_mask
        kept = hsv[~exclude]

        if len(kept) < 10:
            return None

        return kept.mean(axis=0)
