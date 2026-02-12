"""Convert a .mov/.qt video file to .mp4.

Usage:
    python tools/convert.py input.mov output.mp4
    python tools/convert.py input.qt output.mp4 --crf 20
"""

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Convert .mov/.qt to .mp4")
    p.add_argument("input", help="Input .mov or .qt file")
    p.add_argument("output", help="Output .mp4 file path")
    p.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Quality (0=lossless, 23=default, 51=worst)",
    )
    args = p.parse_args()

    cmd = [
        "ffmpeg",
        "-i", args.input,
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-c:a", "aac",
        "-movflags", "+faststart",
        args.output,
    ]

    print(f"Converting: {args.input} -> {args.output}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install it with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed (exit code {e.returncode})", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
