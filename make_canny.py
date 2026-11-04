"""Turn any video into a canny-edge control video (512x512, 81 frames, 16 fps)."""
import argparse
import subprocess

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("video", help="input video")
p.add_argument("out", help="output .mp4")
p.add_argument("--low", type=int, default=80)
p.add_argument("--high", type=int, default=200)
p.add_argument("--size", type=int, default=512)
p.add_argument("--frames", type=int, default=81)
p.add_argument("--fps", type=int, default=16)
args = p.parse_args()

cap = cv2.VideoCapture(args.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total < 1:
    raise SystemExit(f"can't read {args.video}")

ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
     "-s", f"{args.size}x{args.size}", "-r", str(args.fps), "-i", "pipe:0",
     "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out],
    stdin=subprocess.PIPE)

for idx in np.linspace(0, total - 1, args.frames, dtype=int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        break
    # fit inside a square canvas, black bars on the sides
    h, w = frame.shape[:2]
    s = args.size / max(h, w)
    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args.low, args.high)
    canvas = np.zeros((args.size, args.size), np.uint8)
    y, x = (args.size - edges.shape[0]) // 2, (args.size - edges.shape[1]) // 2
    canvas[y:y + edges.shape[0], x:x + edges.shape[1]] = edges
    ffmpeg.stdin.write(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR).tobytes())

ffmpeg.stdin.close()
ffmpeg.wait()
print("saved", args.out)
