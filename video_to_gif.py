#!/usr/bin/env python3
"""
video_to_gif.py - Convert a video file into an animated GIF.

Install dependencies:
    pip install opencv-python pillow

Usage:
    python video_to_gif.py                      # opens a file picker window
    python video_to_gif.py "C:\\path\\clip.mp4"   # pass a path directly
    python video_to_gif.py clip.mp4 --start 5 --duration 4 --fps 12 --width 480
    python video_to_gif.py clip.mp4 -o output.gif

If no path is given, a file-picker dialog opens. If that isn't available
(e.g. headless machine), you'll be asked to type or paste the path instead.
"""

import argparse
import os
import sys

try:
    import cv2
    from PIL import Image
except ImportError:
    sys.exit("Missing dependencies. Run:  pip install opencv-python pillow")


def pick_video_path() -> str:
    """Ask the user for a video via file dialog, falling back to typed input."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.flv *.wmv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if path:
            return path
    except Exception:
        pass  # No GUI available; fall back to typing a path

    return input("Enter or paste the path to your video: ")


def clean_path(path: str) -> str:
    """Strip quotes/whitespace (common when drag-dropping or copying paths)."""
    return os.path.expanduser(path.strip().strip('"').strip("'"))


def video_to_gif(
    video_path: str,
    output_path: str,
    start: float = 0.0,
    duration: float | None = None,
    fps: int = 10,
    width: int | None = 480,
    colors: int = 256,
    loop: int = 0,
) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    video_length = total_frames / src_fps if total_frames else None

    if video_length is not None and start >= video_length:
        cap.release()
        raise ValueError(
            f"Start time ({start}s) is past the end of the video ({video_length:.1f}s)."
        )

    end = start + duration if duration else None
    if end is None and video_length is not None:
        end = video_length

    # Jump to the start position
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    frame_interval = 1.0 / fps
    next_capture_time = start
    frames = []

    print(f"Reading '{os.path.basename(video_path)}' "
          f"(source {src_fps:.1f} fps) ...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end is not None and t > end:
            break

        # Only keep frames at the target GIF frame rate
        if t + 1e-6 < next_capture_time:
            continue
        next_capture_time += frame_interval

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        if width and img.width > width:
            new_h = int(img.height * (width / img.width))
            img = img.resize((width, new_h), Image.LANCZOS)

        # Reduce palette for smaller file size
        img = img.convert("P", palette=Image.ADAPTIVE, colors=colors)
        frames.append(img)

    cap.release()

    if not frames:
        raise RuntimeError("No frames were captured. Check --start/--duration values.")

    print(f"Captured {len(frames)} frames. Saving GIF ...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=loop,
        optimize=True,
        disposal=2,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! Saved to: {output_path} ({size_mb:.2f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert a video to an animated GIF.")
    parser.add_argument("video", nargs="?", help="Path to the video (omit to open a file picker)")
    parser.add_argument("-o", "--output", help="Output GIF path (default: <video name>.gif)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Length of the GIF in seconds (default: until end of video)")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second (default: 10)")
    parser.add_argument("--width", type=int, default=480,
                        help="Output width in pixels, 0 keeps original (default: 480)")
    parser.add_argument("--colors", type=int, default=256, help="Palette size 2-256 (default: 256)")
    parser.add_argument("--loop", type=int, default=0,
                        help="Number of loops, 0 = forever (default: 0)")
    args = parser.parse_args()

    video_path = clean_path(args.video) if args.video else clean_path(pick_video_path())

    if not video_path or not os.path.isfile(video_path):
        sys.exit(f"File not found: {video_path!r}")

    output_path = args.output or os.path.splitext(video_path)[0] + ".gif"

    try:
        video_to_gif(
            video_path,
            output_path,
            start=args.start,
            duration=args.duration,
            fps=args.fps,
            width=args.width or None,
            colors=max(2, min(256, args.colors)),
            loop=args.loop,
        )
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
