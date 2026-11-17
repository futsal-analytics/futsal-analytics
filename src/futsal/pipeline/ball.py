"""The ball on the players' map: 차재혁's ball_tracking/ result (track_ball.py CSV, one per camera) mapped through
HOGEUN's calibration exactly like the players' feet: the same calibration timeline, clock, grid and cross-camera
alignment, so the ball and the players share one pitch frame. ball_tracking/to_court.py is not used here.

A CSV row is t (seconds on the clock of the video detect_video2.py read), status (detected / predicted / missing)
and the box x1, y1, x2, y2; the box's bottom centre is the ball's ground point (ball_tracking/README.md)."""
from __future__ import annotations

import csv

import numpy as np

from .. import nearfar
from ..court import Court
from .align import aligned
from .boxes import _pitch


def read_track(path: str) -> list[dict]:
    """track_ball.py rows that have a box (detected or predicted). Other CSVs (to_court.py's, the raw detections) are
    refused, so a wrong file stops the run before the tracking, not after it."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        if not {"t", "status", "x1", "y1", "x2", "y2"} <= set(rows.fieldnames or ()):
            raise ValueError(f"{nearfar.file_name(path)}: ball_tracking/track_ball.py 결과가 아닙니다 "
                             "(t, status, x1, y1, x2, y2 칸이 있어야 합니다)")
        return [r for r in rows if r.get("status") in ("detected", "predicted")]


def clock_shift(ball_video: str, player_video: str) -> float:
    """Seconds to add to a time on `ball_video` to get the same moment on `player_video`. Both must be the same
    camera of the same match (nearfar.find_video: original or start-aligned copy), or both unknown (same clock):
    a wrong file would move the ball by the cut (145.3 s for 10/8 cam1)."""
    a, b = nearfar.find_video(ball_video), nearfar.find_video(player_video)
    if (a is None) != (b is None) or (a is not None and a[:2] != b[:2]):
        raise ValueError(f"공 영상 {nearfar.file_name(ball_video)} 와 선수 영상 {nearfar.file_name(player_video)} 이 "
                         "같은 경기·같은 카메라가 아닙니다 (--ball-video 확인)")
    if a is None and nearfar.file_name(ball_video).lower() != nearfar.file_name(player_video).lower():
        raise ValueError(f"공 영상 {nearfar.file_name(ball_video)} 와 선수 영상 {nearfar.file_name(player_video)} 의 "
                         "시각 차를 알 수 없습니다 (sync.json 에 없는 영상은 같은 파일이어야 합니다)")
    return b[2] - a[2] if a else 0.0


def ball_grid(path: str, ball_video: str, player_video: str, cal, offset: float, drift: float, start: float,
              rate: float, n: int, W=None, court: Court = Court()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One camera's CSV -> (n, 2) pitch xy, (n,) sigma (m), (n,) detected, on the players' grid
    k = round((t + offset + drift * t - start) * rate) (boxes.grid_frames). `cal`, `offset`, `drift` are the players'
    for `player_video` (after any clock correction: align.corrected_sync), W its alignment field. Per grid frame the
    row nearest in time; rows off the window, missing rows and stretches where HOGEUN found no court stay NaN.
    Off-court points are kept (a ball in the air projects past its ground point)."""
    rows = read_track(path)
    t = np.array([float(r["t"]) for r in rows]) + clock_shift(ball_video, player_video)
    uv = np.array([[(float(r["x1"]) + float(r["x2"])) / 2, float(r["y2"])] for r in rows], float).reshape(-1, 2)
    det = np.array([r["status"] == "detected" for r in rows], bool)
    x = (t + offset + drift * t - start) * rate
    k = np.round(x).astype(int)
    order = np.lexsort((np.abs(x - k), k))
    keep = order[np.r_[True, k[order][1:] != k[order][:-1]]] if len(order) else order
    keep = keep[(k[keep] >= 0) & (k[keep] < n)]
    xy, sig, got = np.full((n, 2), np.nan), np.full(n, np.nan), np.zeros(n, bool)
    xy[k[keep]], sig[k[keep]] = _pitch(cal, t[keep], uv[keep])
    got[k[keep]] = det[keep]
    if W is not None:
        xy = aligned(xy, W, court)
    return xy, sig, got


def choose(per_cam: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """One ball per grid frame and the camera it came from (1, 2; 0 = none): a detection beats a prediction, then the
    nearer camera (smaller sigma). Never the mean: an airborne ball's two ground points are on neither camera's ray."""
    n = len(next(iter(per_cam.values()))[0])
    xy, cam, best = np.full((n, 2), np.nan), np.zeros(n, int), np.full(n, np.inf)
    for c, (p, s, d) in per_cam.items():
        score = np.where(d, s, s + 1e6)
        win = np.isfinite(score) & np.isfinite(p).all(1) & (score < best)
        xy[win], cam[win], best[win] = p[win], int(c.removeprefix("cam")), score[win]
    return xy, cam


def ball_track(csvs: dict[str, str], ball_videos: dict[str, str], player_videos: dict[str, str], cals: dict,
               offsets: dict[str, tuple[float, float]], start: float, rate: float, n: int, alignment: dict | None,
               court: Court = Court()) -> tuple[np.ndarray, np.ndarray, dict]:
    """ball_grid for every camera with a CSV, then choose. Returns xy, cam and a record for summary.json."""
    coef = (alignment or {}).get("coef") or {}
    bv = {cam: ball_videos.get(cam) or player_videos[cam] for cam in csvs}
    per = {cam: ball_grid(path, bv[cam], player_videos[cam], cals[cam], *offsets.get(cam, (0.0, 0.0)), start, rate, n,
                          coef.get(cam), court) for cam, path in csvs.items()}
    xy, cam = choose(per)
    gap = None                                   # the two cameras' ground points where both detected the ball
    if len(per) == 2:
        (pa, _, da), (pb, _, db) = per.values()
        m = da & db & np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
        gap = round(float(np.median(np.linalg.norm(pa[m] - pb[m], axis=1))), 3) if m.any() else None
    record = {"cameras": {c: {"csv": nearfar.file_name(csvs[c]), "video": nearfar.file_name(bv[c]),
                              "clock_shift_s": round(clock_shift(bv[c], player_videos[c]), 4),
                              "frames": int(np.isfinite(per[c][0]).all(1).sum())} for c in per},
              "frames_with_ball": round(float(np.mean(cam > 0)), 3) if n else 0.0,
              "from_camera": {f"cam{i}": int(np.sum(cam == i)) for i in sorted(set(cam.tolist()) - {0})},
              "both_detected_median_m": gap}
    return xy, cam, record
