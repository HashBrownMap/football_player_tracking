# Soccer Game Proximity Annotation Tool

Annotate soccer game footage with visual overlays showing spatial relationships between players. The tool detects players using YOLOv8, classifies them into two teams by jersey color, and draws red lines between same-team players who are within a configurable distance threshold (default 15 meters).

## Features

- **Player detection and tracking** via YOLOv8 + ByteTrack with persistent IDs across frames
- **Automatic team classification** using K-Means clustering on jersey colors (HSV)
- **Referee filtering** — bright yellow kits are excluded automatically
- **Real-world distance estimation** via manual homography calibration (pixel-to-field-meters transform)
- **Obstruction-aware lines** — skips drawing a line between two players if a teammate is between them
- **Single-team mode** — optionally restrict annotations to one team only
- **Frame-skip** for faster processing on long videos
- **Reusable calibration** — save/load homography matrices for repeated runs on the same camera angle

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (for video conversion)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (for YouTube downloads)

```bash
brew install ffmpeg yt-dlp
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Download a YouTube video

```bash
python tools/download.py <url> <output.mp4> [--resolution N]
```

| Argument | Description |
|---|---|
| `url` | YouTube video URL |
| `output` | Output .mp4 file path |
| `--resolution` | Max video height in pixels (default: 720) |

### 2. Convert .mov/.qt to .mp4

```bash
python tools/convert.py <input> <output.mp4> [--crf N]
```

| Argument | Description |
|---|---|
| `input` | Input .mov or .qt file |
| `output` | Output .mp4 file path |
| `--crf` | Quality level: 0 = lossless, 23 = default, 51 = worst |

### 3. Annotate a game

```bash
python main.py <input.mp4> <output.mp4> [options]
```

| Argument | Description |
|---|---|
| `input` | Path to input .mp4 video |
| `output` | Path to output annotated .mp4 video |
| `--model` | YOLOv8 model path (default: `yolov8m.pt`) |
| `--device` | Inference device: `mps`, `cpu`, or `cuda` (default: `mps`) |
| `--confidence` | Detection confidence threshold (default: 0.3) |
| `--distance` | Proximity threshold in meters (default: 15.0) |
| `--team` | Only draw lines/boxes for this team: `0` or `1` (default: both) |
| `--frame-skip` | Process every Nth frame (default: 1) |
| `--calibration-frame` | Frame index to use for calibration (default: 0) |
| `--save-homography` | Save homography matrix to a .npy file |
| `--load-homography` | Load a previously saved homography matrix |
| `--debug` | Show debug overlays (tracker IDs, team labels) |

**Examples:**

```bash
# First run — interactive calibration, save for reuse
python main.py game.mp4 annotated.mp4 --save-homography calib.npy

# Reuse saved calibration
python main.py game.mp4 annotated.mp4 --load-homography calib.npy

# Single team with debug overlays
python main.py game.mp4 annotated.mp4 --load-homography calib.npy --team 0 --debug

# Faster processing (every other frame)
python main.py game.mp4 annotated.mp4 --load-homography calib.npy --frame-skip 2
```

## Calibration

On the first run (without `--load-homography`), an interactive calibration window opens:

1. A frame from the video is displayed.
2. The terminal prompts field landmark names one at a time (e.g., "center_circle_center", "left_penalty_spot").
3. For each landmark: **click its pixel location** on the frame, or press **`s`** to skip.
4. Press **`q`** when done (minimum 4 points required, 6-8 recommended).
5. The homography matrix is computed and optionally saved.

**Tips:**
- Choose a frame where as many field markings are visible as possible.
- Spread calibration points across the visible field area.
- Pick points you can identify with confidence (corners, penalty spots, center mark).

## How It Works

1. **Detection** — YOLOv8 detects all persons in each frame.
2. **Tracking** — ByteTrack assigns persistent IDs across frames.
3. **Team classification** — Jersey colors are sampled from the first 30 frames, then K-Means (k=2) clusters players into two teams. Referees (bright yellow) are filtered out.
4. **Homography** — Player foot positions (bottom-center of bounding box) are projected to real-world field coordinates using the calibrated homography matrix.
5. **Proximity** — For each team, pairwise distances are computed. Pairs within the threshold get a semi-transparent red line drawn between them. Lines are suppressed when a teammate lies between the two players.

## Limitations

- **Goalkeepers** may be misclassified if their jersey differs significantly from outfield players.
- **Camera panning** — a single homography assumes a mostly static camera. Significant panning will reduce distance accuracy near frame edges.
- **Similar jerseys** — K-Means struggles when both teams wear similar colors.
