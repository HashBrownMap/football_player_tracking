"""Soccer game proximity annotation tool.

Usage:
    # First run (interactive calibration):
    python main.py game.mp4 output.mp4 --save-homography calib.npy

    # Subsequent runs (reuse calibration):
    python main.py game.mp4 output.mp4 --load-homography calib.npy

    # With debug overlays:
    python main.py game.mp4 output.mp4 --load-homography calib.npy --debug

    # Faster processing (skip every other frame):
    python main.py game.mp4 output.mp4 --load-homography calib.npy --frame-skip 2
"""

import argparse

import cv2
import numpy as np
import supervision as sv

from src.annotate import draw_proximity_lines
from src.calibration import calibrate
from src.detect import Detector
from src.geometry import compute_proximity_pairs, get_foot_positions, pixel_to_field
from src.teams import TeamClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soccer proximity annotation tool")
    p.add_argument("input", help="Path to input .mp4 video")
    p.add_argument("output", help="Path to output annotated .mp4 video")
    p.add_argument("--model", default="yolov8m.pt", help="YOLOv8 model path")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument(
        "--distance", type=float, default=15.0, help="Proximity threshold (meters)"
    )
    p.add_argument(
        "--team", type=int, default=None, choices=[0, 1],
        help="Only draw lines for this team (0 or 1). Omit for both teams.",
    )
    p.add_argument(
        "--frame-skip", type=int, default=1, help="Process every Nth frame (1=all)"
    )
    p.add_argument(
        "--calibration-frame", type=int, default=0, help="Frame index for calibration"
    )
    p.add_argument("--save-homography", help="Save homography matrix to .npy")
    p.add_argument("--load-homography", help="Load homography matrix from .npy")
    p.add_argument("--debug", action="store_true", help="Show debug overlays")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Step 1: Homography calibration ---
    if args.load_homography:
        H = np.load(args.load_homography)
        print(f"Loaded homography from {args.load_homography}")
    else:
        H = calibrate(args.input, frame_index=args.calibration_frame)
        if args.save_homography:
            np.save(args.save_homography, H)
            print(f"Saved homography to {args.save_homography}")

    # --- Step 2: Initialize detector and team classifier ---
    detector = Detector(
        model_path=args.model, confidence=args.confidence, device=args.device
    )
    classifier = TeamClassifier(n_sample_frames=30)

    # --- Step 3: Sample team colors from first N frames ---
    print("Sampling team colors...")
    cap = cv2.VideoCapture(args.input)
    frame_idx = 0
    while frame_idx < classifier.n_sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect_and_track(frame)
        if len(detections) > 0:
            classifier.collect_sample(frame, detections)
        frame_idx += 1
    cap.release()
    classifier.fit()
    print(f"Team clustering complete. Centers (HSV): {classifier.cluster_centers}")

    # --- Step 4: Reset tracker for clean full-video pass ---
    detector.reset()

    # --- Step 5: Process full video ---
    # State shared across frames for frame-skip
    state = {
        "last_detections": None,
        "last_labels": None,
        "last_pairs": [],
    }

    def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
        # Frame skip: reuse previous results for non-processed frames
        if args.frame_skip > 1 and index % args.frame_skip != 0:
            if state["last_detections"] is not None and len(state["last_detections"]) > 0:
                return draw_proximity_lines(
                    frame,
                    state["last_detections"],
                    state["last_pairs"],
                    state["last_labels"],
                    debug=args.debug,
                    team_id=args.team,
                )
            return frame

        # Detect + track
        detections = detector.detect_and_track(frame)
        if len(detections) == 0:
            state["last_detections"] = detections
            state["last_labels"] = np.array([], dtype=int)
            state["last_pairs"] = []
            return frame

        # Classify teams
        team_labels = classifier.classify(frame, detections)

        # Transform to field coordinates
        foot_pixels = get_foot_positions(detections)
        field_positions = pixel_to_field(foot_pixels, H)

        # Compute proximity pairs
        pairs = compute_proximity_pairs(
            field_positions, team_labels, args.distance, team_id=args.team,
        )

        # Cache for frame-skip
        state["last_detections"] = detections
        state["last_labels"] = team_labels
        state["last_pairs"] = pairs

        return draw_proximity_lines(
            frame, detections, pairs, team_labels,
            debug=args.debug, team_id=args.team,
        )

    print(f"Processing video: {args.input}")
    sv.process_video(
        source_path=args.input,
        target_path=args.output,
        callback=process_frame,
    )
    print(f"Done. Output saved to {args.output}")


if __name__ == "__main__":
    main()
