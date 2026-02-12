"""Interactive homography calibration via OpenCV GUI.

The user clicks known field landmarks on a reference frame to build
a pixel-to-field-meters homography matrix.
"""

import cv2
import numpy as np

from .field import LANDMARKS


def calibrate(video_path: str, frame_index: int = 0) -> np.ndarray:
    """Open interactive GUI for homography calibration.

    Args:
        video_path: Path to the input video.
        frame_index: Which frame to show for calibration.

    Returns:
        3x3 homography matrix (pixel → field meters).
    """
    cap = cv2.VideoCapture(video_path)
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    display = frame.copy()
    clicked_point = [None]  # mutable container for closure

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point[0] = (x, y)

    window_name = "Calibration - click landmarks, press 'q' when done"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame.shape[1], 1280), min(frame.shape[0], 720))
    cv2.setMouseCallback(window_name, on_mouse)

    landmark_names = list(LANDMARKS.keys())
    pixel_points = []
    field_points = []

    print("\n=== Homography Calibration ===")
    print("For each landmark, click its location on the image.")
    print("Press 's' to skip a landmark you can't see.")
    print("Press 'q' to finish early (need at least 4 points).\n")

    for name in landmark_names:
        field_xy = LANDMARKS[name]
        print(f"  Click: {name}  (field coords: {field_xy})  [s=skip, q=done]")

        clicked_point[0] = None
        while True:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                print(f"    Skipped {name}")
                break

            if clicked_point[0] is not None:
                px = clicked_point[0]
                pixel_points.append(px)
                field_points.append(field_xy)

                # Visual feedback
                cv2.circle(display, px, 6, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    name.replace("_", " "),
                    (px[0] + 10, px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                print(f"    Marked at pixel {px}")
                break

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(pixel_points) < 4:
        raise ValueError(
            f"Need at least 4 calibration points, got {len(pixel_points)}"
        )

    pixel_arr = np.array(pixel_points, dtype=np.float64)
    field_arr = np.array(field_points, dtype=np.float64)

    H, mask = cv2.findHomography(pixel_arr, field_arr, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography computation failed. Try different points.")

    inliers = int(mask.sum()) if mask is not None else len(pixel_points)
    print(f"\nHomography computed with {inliers}/{len(pixel_points)} inliers.")
    return H
