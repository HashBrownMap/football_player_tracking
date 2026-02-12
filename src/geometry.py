"""Homography transforms and pairwise distance computation."""

from itertools import combinations

import cv2
import numpy as np
import supervision as sv

from .field import FIELD_LENGTH, FIELD_WIDTH

# Players projecting more than this far outside the pitch are ignored.
_OOB_MARGIN = 10.0


def get_foot_positions(detections: sv.Detections) -> np.ndarray:
    """Extract foot position (bottom-center of bbox) for each detection.

    Returns:
        (N, 2) array of pixel (x, y) positions.
    """
    x_center = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
    y_bottom = detections.xyxy[:, 3]
    return np.column_stack([x_center, y_bottom])


def pixel_to_field(pixel_points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transform pixel coordinates to field coordinates using homography.

    Args:
        pixel_points: (N, 2) pixel (x, y) positions.
        H: 3x3 homography matrix (pixel → field meters).

    Returns:
        (N, 2) field (x, y) positions in meters.
    """
    pts = pixel_points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def is_in_bounds(field_positions: np.ndarray) -> np.ndarray:
    """Return boolean mask for positions within the pitch (with margin).

    Args:
        field_positions: (N, 2) field coordinates in meters.

    Returns:
        (N,) boolean array.
    """
    x, y = field_positions[:, 0], field_positions[:, 1]
    return (
        (x >= -_OOB_MARGIN)
        & (x <= FIELD_LENGTH + _OOB_MARGIN)
        & (y >= -_OOB_MARGIN)
        & (y <= FIELD_WIDTH + _OOB_MARGIN)
    )


def compute_proximity_pairs(
    field_positions: np.ndarray,
    team_labels: np.ndarray,
    threshold_meters: float = 15.0,
    obstruction_tolerance: float = 1.2,
    team_id: int | None = None,
) -> list[tuple[int, int]]:
    """Find pairs of same-team players within threshold distance.

    Skips a pair (i, j) if a teammate k lies roughly between them,
    i.e. dist(i,k) + dist(k,j) < dist(i,j) * obstruction_tolerance.

    Args:
        field_positions: (N, 2) field coordinates in meters.
        team_labels: (N,) array with values 0, 1, or -1 (referee).
        threshold_meters: distance threshold.
        obstruction_tolerance: factor for the "between" check (1.0 = exactly
            on the line, 1.2 = 20% slack).
        team_id: If set (0 or 1), only compute pairs for that team.
                 If None, compute for both teams.

    Returns:
        List of (i, j) index pairs into the detections array.
    """
    in_bounds = is_in_bounds(field_positions)
    pairs = []
    teams = [team_id] if team_id is not None else [0, 1]

    for team_id in teams:
        team_mask = (team_labels == team_id) & in_bounds
        team_indices = np.where(team_mask)[0]
        if len(team_indices) < 2:
            continue

        # Precompute pairwise distances for this team
        n = len(team_indices)
        dist_map: dict[tuple[int, int], float] = {}
        for a in range(n):
            for b in range(a + 1, n):
                i, j = team_indices[a], team_indices[b]
                dist_map[(i, j)] = np.linalg.norm(
                    field_positions[i] - field_positions[j]
                )

        # Collect candidate pairs within threshold
        candidates = [
            (i, j) for (i, j), d in dist_map.items() if d < threshold_meters
        ]

        # Filter out pairs where a teammate is between them
        for i, j in candidates:
            d_ij = dist_map[(i, j)]
            obstructed = False
            for idx in team_indices:
                if idx == i or idx == j:
                    continue
                d_ik = dist_map.get((min(i, idx), max(i, idx)), float("inf"))
                d_kj = dist_map.get((min(idx, j), max(idx, j)), float("inf"))
                if d_ik + d_kj < d_ij * obstruction_tolerance:
                    obstructed = True
                    break
            if not obstructed:
                pairs.append((i, j))

    return pairs
