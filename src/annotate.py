"""Draw proximity lines and optional debug overlays on video frames."""

import cv2
import numpy as np
import supervision as sv

from .geometry import get_foot_positions

# Team colors for debug overlays (BGR)
_TEAM_COLORS = {
    0: (255, 150, 50),   # blue-ish for team 0
    1: (50, 150, 255),   # orange-ish for team 1
    -1: (180, 180, 180), # gray for referee
}


def draw_proximity_lines(
    frame: np.ndarray,
    detections: sv.Detections,
    pairs: list[tuple[int, int]],
    team_labels: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    debug: bool = False,
    team_id: int | None = None,
) -> np.ndarray:
    """Draw red lines between proximate teammates.

    Args:
        frame: BGR image.
        detections: Player detections with bounding boxes.
        pairs: List of (i, j) index pairs to connect.
        team_labels: (N,) team assignment per detection.
        color: BGR color for proximity lines.
        thickness: Line thickness in pixels.
        debug: If True, draw bboxes and tracker IDs.

    Returns:
        Annotated copy of the frame.
    """
    annotated = frame.copy()
    foot_positions = get_foot_positions(detections)

    # Draw team-colored bounding boxes around each player
    for idx in range(len(detections)):
        label = int(team_labels[idx])
        if team_id is not None and label != team_id:
            continue
        x1, y1, x2, y2 = detections.xyxy[idx].astype(int)
        box_color = _TEAM_COLORS.get(label, (180, 180, 180))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        if debug and detections.tracker_id is not None:
            tid = int(detections.tracker_id[idx])
            text = f"#{tid} T{label}"
            cv2.putText(
                annotated, text, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA,
            )

    # Draw proximity lines with transparency
    overlay = annotated.copy()
    for i, j in pairs:
        pt1 = tuple(foot_positions[i].astype(int))
        pt2 = tuple(foot_positions[j].astype(int))
        cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

    return annotated
