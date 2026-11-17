"""Run 차재혁's ball_tracking/ scripts as they are, once per video, for trackview --ball-model.

The chain is his README's: detect_video2.py at imgsz 1280 and 1920 (stride: about 30 frames a second) -> merge_raw.py
-> filter_static.py (cam2 only, as his README says) -> track_ball.py. His scripts take the whole video and write CSVs;
they stay in <out>/ball_<cam>/ as a cache (the detection takes about 2 hours per match on an RTX 5060), and only the
final track goes on to pipeline.ball, which puts the ball into tracks.json.

His times are frame index / CAP_PROP_FPS. A phone video with a variable frame rate (11/3 cam1: up to 4.6 s by the
end) is off by that much, so the track is re-timed with each frame's own timestamp (the clock the players and
HOGEUN's calibration use): Ultralytics' vid_stride s keeps frame i * s + s - 1 for result i."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .. import nearfar

BALL_TRACKING = Path(__file__).resolve().parents[3] / "ball_tracking"


def stride_for(fps: float) -> int:
    """About 30 frames a second, as in his README (60 fps -> 2, 30 fps -> 1)."""
    return max(1, int(round(fps / 30.0)))


def frame_times(video: str) -> np.ndarray:
    """Every frame's own timestamp in seconds (what pipeline/detect.py reads per frame)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    out = []
    while cap.grab():
        out.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    return np.array(out, float)


def retime(t: np.ndarray, fps: float, stride: int, pts: np.ndarray) -> np.ndarray:
    """His t = i * stride / fps (offset 0) -> the timestamp of the frame Ultralytics read for result i."""
    i = np.round(np.asarray(t, float) * fps / stride).astype(int)
    f = np.clip(i * stride + stride - 1, 0, len(pts) - 1)
    return pts[f]


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=BALL_TRACKING)


def ball_track_csv(video: str, cam: str, model: str, out_dir: str) -> str:
    """His chain for one camera's whole video, cached in <out_dir>/ball_<cam>/; returns the re-timed track CSV."""
    video, model = os.path.abspath(video), os.path.abspath(model)        # his scripts run in ball_tracking/
    folder = Path(out_dir).resolve() / f"ball_{cam}"
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = stride_for(fps)
    want = {"video": nearfar.file_name(video), "video_bytes": os.path.getsize(video), "model": nearfar.file_name(model),
            "model_bytes": os.path.getsize(model), "stride": stride, "filter_static": cam == "cam2"}
    final, meta_path = folder / "track_pts.csv", folder / "meta.json"
    if final.exists() and meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) == want:
        print(f"  {cam}: 저장된 공 추적 결과 사용 ({final})")
        return str(final)
    raw = {}
    for imgsz in (1280, 1920):
        raw[imgsz] = folder / f"raw_{imgsz}.csv"
        print(f"  {cam}: 공 검출 (ball_tracking/detect_video2.py, imgsz {imgsz}, 영상 전체)...", flush=True)
        _run("detect_video2.py", video, str(raw[imgsz]), "0", str(stride), str(imgsz), model)
    merged = folder / "merged.csv"
    _run("merge_raw.py", str(raw[1280]), str(raw[1920]), str(merged))
    if want["filter_static"]:
        _run("filter_static.py", str(merged), str(folder / "merged_ns.csv"))
        merged = folder / "merged_ns.csv"
    track = folder / "track.csv"
    _run("track_ball.py", str(merged), str(track))
    with open(track, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows:
        t = retime(np.array([float(r["t"]) for r in rows]), fps, stride, frame_times(video))
        for r, tt in zip(rows, t):
            r["t"] = f"{tt:.4f}"
    with open(final, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "status", "x1", "y1", "x2", "y2", "conf", "track_id"])
        w.writeheader()
        w.writerows(rows)
    meta_path.write_text(json.dumps(want, ensure_ascii=False), encoding="utf-8")
    return str(final)
