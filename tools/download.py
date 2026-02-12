"""Download a YouTube video as .mp4.

Usage:
    python tools/download.py "https://www.youtube.com/watch?v=..." output.mp4
    python tools/download.py "https://www.youtube.com/watch?v=..." output.mp4 --resolution 480
"""

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Download YouTube video as .mp4")
    p.add_argument("url", help="YouTube video URL")
    p.add_argument("output", help="Output .mp4 file path")
    p.add_argument(
        "--resolution",
        type=int,
        default=720,
        help="Max video height in pixels (default: 720)",
    )
    args = p.parse_args()

    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={args.resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={args.resolution}][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", args.output,
        args.url,
    ]

    print(f"Downloading: {args.url}")
    print(f"Resolution:  <={args.resolution}p")
    print(f"Output:      {args.output}\n")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: yt-dlp not found. Install it with: brew install yt-dlp", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Download failed (exit code {e.returncode})", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Saved to {args.output}")


if __name__ == "__main__":
    main()
