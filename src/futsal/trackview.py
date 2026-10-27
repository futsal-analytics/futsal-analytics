"""두 영상 → 선수 추적 한 번에: 검출(detection/) → ReID → 좌표 변환(calibration/) → 시간 맞추기 → 추적 v2 → 결과

  python -m futsal.trackview --video cam1.mp4 cam2.mp4 --offset 0 124.27 --start 600 --duration 180 --out trackview

단계와 코드:
  검출       detection/detect_person.py (차재혁) 의 모델·설정 (YOLO26x, conf 0.3, ByteTrack) 을 그대로 읽어서 모든
             프레임에 돌림 (pipeline/yolo_track.py). 추적에는 초당 --rate 장만 씀
  ReID       pipeline/reid.py (OSNet), 추적에 쓰는 프레임의 박스마다
  좌표 변환  calibration/ (HOGEUN) 의 court_map.load / pixel_to_court 를 그대로 부르고 우리 좌표(0~40, 0~20 m)로 옮김
             (futsal/nearfar.py). 10/8 영상은 기록된 카메라 움직임을 자동 보정, 다른 영상은 --auto-moves
  시간 맞추기 --offset (python -m futsal.sync 결과). 10분 이상이면 선수 위치로 두 폰 시계 어긋남을 한 번 더 보정
  추적       추적 v2 (아래 --ids)

결과 (--out 폴더):
  tracks.json    선수별 위치 (web/viewer 로 재생 가능)
  summary.json   선수별 이동거리·속도·보인 비율, 쓴 검출 설정과 보정 파일
  events.csv     ID가 경기장 한가운데서 새로 생기거나 끊긴 순간 (= ID가 바뀌었을 가능성이 큰 곳) 목록
  trackview.mp4  확인 영상. 위: 두 카메라 화면에 선수 ID, 아래: 2D 지도 + ID 사건 목록 (--no-video 면 안 만듦)
  det_cam1.json  검출 결과 캐시 (같은 설정으로 다시 돌리면 검출을 건너뜀; ReID 특징은 det_cam1.json.reid.npy)

--start 는 cam1 영상 기준 초, --offset 은 cam1 시각 = 그 카메라 시각 + offset (python -m futsal.sync 결과).
검출에는 Ultralytics 가 필요합니다: pip install -e ".[vision]". 모든 프레임을 검출하므로 GPU가 없으면 매우 느립니다.
예전 방식(우리 검출기 pipeline/detect.py, 우리 탭 보정 calib/<cam>/calib.json)은 main() 과 run() 에 주석으로 남겨 두었습니다.

추적 방식 (--ids):
  v2          (기본) 상자 단위 추적: 카메라별 조각 → 두 카메라 짝짓기 → 팀당 5+1 자리에 배정. 경기 전체에서
              선수 한 명을 한 ID로 가장 오래 유지합니다 (53분 경기에서 5~6명이 80% 넘게 같은 ID). 쉬는 시간처럼
              코트 위 사람이 바뀌는 구간에서는 자리에 억지로 끼워 넣다가 ID가 바뀌기도 합니다.
              사람 재식별(ReID) 특징이 필요합니다 (설치: futsal/pipeline/reid.py 설명). 이 숫자는 우리 검출(conf 0.1)과
              우리 탭 보정으로 잰 값입니다 (같은 검출에 calibration/ 보정이면 4명).
  v2gta       같은 조각을 정원 없이 하나씩 이어 붙이는 방식. 6분 구간에서는 같은 팀 선수끼리 엇갈릴 때 ID가 바뀌는
              일이 가장 적었지만, 경기 전체에서는 ID가 잘게 나뉩니다 (53분에 86개). 짧은 구간 확인용. ReID 필요.
  두 방식 모두 10분 이상 구간이면 선수 위치로 두 폰의 시계 어긋남을 먼저 재서 보정합니다 (10/8 두 폰은 어긋남 없음).
  appearance  예전 방식 (발 위치 + 색 특징). ReID를 쓸 수 없으면 v2 대신 자동으로 이 방식을 씁니다.
  motion      움직임만 보는 가장 오래된 방식.
--no-reid 는 ReID 계산을 건너뛰고 예전 방식(appearance)으로 추적합니다.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os

import cv2
import numpy as np

from . import nearfar
from .court import Court
from .homography import Calibration, CalibrationTimeline, calibration_at, timeline_from_json
from .pipeline import detect, yolo_track
from .pipeline.align import corrected_sync
from .pipeline.run import IDS, LINKERS, build_tracks
from .tracks import TrackSet

PANEL_W, PANEL_H = 960, 540


def color_of(pid: int) -> tuple[int, int, int]:
    h = int((pid * 0.618034 % 1.0) * 180)
    c = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def load_calibration(path: str, video: str | None = None, auto_moves: bool = False, cache_dir: str | None = None
                     ) -> Calibration | CalibrationTimeline:
    """calib.json (calibtool fit) or a timeline list. With `auto_moves`, camera knocks found by futsal.camshift
    over the whole video turn a single calibration into a timeline (cached as moves_<name>.json)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cal = timeline_from_json(d)
    if auto_moves and isinstance(cal, Calibration) and video:
        cal = with_moves(cal, video, float(d.get("at", 0.0)), cache_dir)
    return cal


def load_team_calibration(n: int, video: str | None = None, auto_moves: bool = False, cache_dir: str | None = None,
                          out_dir: str = str(nearfar.TEAM_OUT)) -> Calibration | CalibrationTimeline:
    """Camera n of HOGEUN's calibration (calibration/court_map.load: H_cam<n>_refined.json, else H_cam<n>.json in
    `out_dir`) in our pitch coordinates (futsal.nearfar). Camera knocks: with `auto_moves` futsal.camshift measures
    them against its frame (t_sec), as load_calibration does for a tapped one; otherwise the measured poses of the 10/8
    test videos are used when `video` is one of them (nearfar.known_knocks, relative to t_sec), and none for other
    videos (with a notice)."""
    cal = nearfar.load_team(n, out_dir)
    at_s = cal.team.get("t_sec")
    cam = nearfar.camera_20261008(video) if video else None
    if cam is not None and cam != f"cam{n}":          # videos given in the wrong order: cam1's calibration on cam2's video
        raise ValueError(f"{os.path.basename(video)} 은 10/8 경기 {cam} 영상인데 cam{n} 자리에 들어왔습니다 "
                         "(--video 는 cam1 영상, cam2 영상 순서)")
    if auto_moves and video:
        return with_moves(cal, video, float(at_s or 0.0), cache_dir)
    if cam is None or at_s is None:
        if video:
            print(f"  {os.path.basename(video)}: 기록된 카메라 움직임이 없어 촬영 내내 카메라가 그대로였다고 보고 계산합니다"
                  " (움직였으면 --auto-moves)")
            made_on = os.path.basename(str(cal.team.get("video", "")))
            if cam is None and made_on and made_on != os.path.basename(video):
                print(f"  주의: 이 보정은 HOGEUN이 {made_on} 영상에서 찍은 것입니다. 다른 경기이거나 카메라 자리가 바뀌었으면"
                      " calibration/ 에서 그 영상으로 다시 찍어야 합니다")
        return cal
    if nearfar.unsettled_20261008(cam, float(at_s)):
        print(f"  주의: HOGEUN 보정 프레임({_fmt(float(at_s))})은 {cam} 카메라가 흔들리던 때라 기록된 움직임 보정이 맞지 않습니다")
    knocks = nearfar.knocks_20261008(cam, float(at_s))
    print(f"  {os.path.basename(video)}: 기록된 카메라 움직임 보정 (보정 프레임 {_fmt(float(at_s))} 기준) "
          + ", ".join(f"{_fmt(t)}부터 ({dx:+.1f}, {dy:+.1f})px" for t, (dx, dy) in knocks))
    return nearfar.knock_timeline(cal, knocks)


def with_moves(cal: Calibration, video: str, at_s: float, cache_dir: str | None = None) -> Calibration | CalibrationTimeline:
    """`cal` (made on the frame at `at_s`) + the camera knocks futsal.camshift finds over the whole video -> a
    timeline, or `cal` itself when the camera never moved. The scan is cached as <cache_dir>/moves_<name>.json."""
    from . import camshift
    cache = os.path.join(cache_dir or ".", f"moves_{os.path.splitext(os.path.basename(video))[0]}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            moves = json.load(f)
    else:
        print(f"  {os.path.basename(video)}: 카메라가 움직였는지 훑는 중 (영상 전체, 몇 분)...")
        moves = camshift.camera_moves(video)["moves"]
        with open(cache, "w") as f:
            json.dump(moves, f)
    if not moves:
        return cal
    for line in camshift.describe({"moves": moves}):
        print("   ", line)
    return camshift.timeline_from_moves(cal, moves, at_s)


class FrameReader:
    """Frames of one video by time, read forward (seeks only when going back or jumping far ahead)."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.t, self.frame = -1e9, None

    def _read(self) -> bool:
        ok, f = self.cap.read()
        if ok:
            self.frame, self.t = f, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        return ok

    def at(self, t: float) -> np.ndarray | None:
        if t < self.t - 0.5 or t > self.t + 5:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(t - 0.5, 0) * 1000)
            self.t = -1e9
        half = 0.5 / self.fps
        while self.t < t - half:
            if not self._read():
                break
        return self.frame

    def close(self):
        self.cap.release()


def find_events(ts: TrackSet, court: Court, edge_m: float = 2.5, settle_s: float = 2.0) -> list[dict]:
    """IDs that start or end in the middle of the pitch (not at the window edges, not at the touchlines where
    players really come and go): the likely places where an identity was lost or swapped."""
    ev = []
    n, fps = ts.n_frames, ts.fps
    for p in ts.players:
        seen = np.where(np.all(np.isfinite(p.xy), axis=1))[0]
        if len(seen) == 0:
            continue
        for kind, k in (("new", int(seen[0])), ("lost", int(seen[-1]))):
            if (kind == "new" and k < settle_s * fps) or (kind == "lost" and k > n - 1 - settle_s * fps):
                continue
            x, y = p.xy[k]
            if min(x, y, court.length - x, court.width - y) < edge_m:
                continue
            ev.append({"t": k / fps, "frame": k, "id": p.id, "kind": kind, "x": round(float(x), 1), "y": round(float(y), 1)})
    return sorted(ev, key=lambda e: e["t"])


def _court_map(court: Court, w: int, h: int, margin: int = 30):
    s = min((w - 2 * margin) / court.length, (h - 2 * margin) / court.width)
    ox, oy = (w - court.length * s) / 2, (h - court.width * s) / 2

    def P(xy):
        xy = np.atleast_2d(xy)
        return np.stack([ox + xy[:, 0] * s, oy + (court.width - xy[:, 1]) * s], 1)
    base = np.full((h, w, 3), (38, 92, 52), np.uint8)
    for m in court.markings():
        cv2.polylines(base, [np.round(P(m)).astype(np.int32)], False, (235, 235, 235), 2, cv2.LINE_AA)
    for gx in (0.0, court.length):
        a, b = P([[gx, court.width / 2 - 1.5], [gx, court.width / 2 + 1.5]])
        cv2.line(base, tuple(np.int32(a)), tuple(np.int32(b)), (255, 255, 255), 5)
    return base, P


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _rect(d) -> tuple[float, float, float, float]:
    """A detection's box in calibrated pixels: the stored box, or one rebuilt from the foot point and height for
    caches written before boxes were kept."""
    if d.box is not None:
        return d.box
    return d.u - 0.22 * d.h_px, d.v - d.h_px, d.u + 0.22 * d.h_px, d.v


def frame_size(path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(path)
    wh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return wh


def render(out_path: str, videos: dict[str, str], cals: dict, offsets: dict, ts: TrackSet, dets: dict[str, list],
           court: Court, start: float, events: list[dict], fps_out: float | None = None, tail_s: float = 2.0) -> int:
    """One output frame per tracking step: camera panels on top, 2D map + event list below. Returns frames written."""
    fps_out = fps_out or ts.fps
    readers = {c: FrameReader(v) for c, v in videos.items()}
    det_times = {c: sorted({d.t for d in ds}) for c, ds in dets.items()}
    det_by_t = {c: {} for c in dets}
    for c, ds in dets.items():
        for d in ds:
            det_by_t[c].setdefault(d.t, []).append(d)
    map_w, map_h = 1180, PANEL_H
    base_map, P = _court_map(court, map_w, map_h)
    W, H = PANEL_W * 2, PANEL_H * 2
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (W, H))
    cams = list(videos)[:2]
    n_out = int(ts.n_frames / ts.fps * fps_out)
    for j in range(n_out):
        t_rel = j / fps_out
        k = min(int(round(t_rel * ts.fps)), ts.n_frames - 1)
        canvas = np.zeros((H, W, 3), np.uint8)
        pos = {p.id: p.xy[k] for p in ts.players if np.all(np.isfinite(p.xy[k]))}
        for ci, cam in enumerate(cams):
            off, drift = offsets.get(cam, (0.0, 0.0))
            t_cam = (start + t_rel - off) / (1 + drift)
            frame = readers[cam].at(t_cam)
            panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            if frame is not None:
                fh, fw = frame.shape[:2]
                sc = min(PANEL_W / fw, PANEL_H / fh)
                small = cv2.resize(frame, (int(fw * sc), int(fh * sc)))
                panel[:small.shape[0], :small.shape[1]] = small
                cal = calibration_at(cals[cam], t_cam)
                i = bisect.bisect_left(det_times[cam], t_cam)
                near = [det_times[cam][q] for q in (i - 1, i) if 0 <= q < len(det_times[cam])]
                boxes = det_by_t[cam].get(min(near, key=lambda q: abs(q - t_cam)), []) if near else []
                feet = np.array([(d.u, d.v) for d in boxes]).reshape(-1, 2)
                taken = set()
                if pos:
                    ids = list(pos)
                    uv = cal.to_image(np.array([pos[i_] for i_ in ids]))
                    for pid, (u, v) in zip(ids, uv):
                        if not (np.isfinite(u) and -50 < u < fw + 50 and -50 < v < fh + 50):
                            continue
                        col = color_of(pid)
                        b = None
                        if len(feet):
                            dist = np.hypot(feet[:, 0] - u, feet[:, 1] - v)
                            q = int(np.argmin(dist))
                            if dist[q] < max(25.0, 0.4 * boxes[q].h_px) and q not in taken:
                                b, _ = boxes[q], taken.add(q)
                        if b is not None:
                            x0, y0, x1, y1 = (int(c * sc) for c in _rect(b))
                            cv2.rectangle(panel, (x0, y0), (x1, y1), col, 2)
                            tx, ty = x0, y0 - 4
                        else:                       # tracked here from the other camera / smoothing, no box
                            cv2.circle(panel, (int(u * sc), int(v * sc)), 5, col, 2)
                            tx, ty = int(u * sc) + 6, int(v * sc) - 6
                        cv2.putText(panel, str(pid), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(panel, str(pid), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                for q, b in enumerate(boxes):           # detections no track claimed: thin grey
                    if q not in taken:
                        x0, y0, x1, y1 = (int(c * sc) for c in _rect(b))
                        cv2.rectangle(panel, (x0, y0), (x1, y1), (160, 160, 160), 1)
            label = f"{cam}  {_fmt(t_cam)}"
            cv2.putText(panel, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(panel, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            canvas[:PANEL_H, ci * PANEL_W:(ci + 1) * PANEL_W] = panel
        m = base_map.copy()
        k0 = max(0, k - int(tail_s * ts.fps))
        for p in ts.players:                      # trail: only connect consecutive seen frames, never across gaps
            seg = p.xy[k0:k + 1]
            ok = np.all(np.isfinite(seg), axis=1)
            runs = np.split(np.arange(len(seg)), np.where(np.diff(ok.astype(int)) != 0)[0] + 1)
            for r in runs:
                if len(r) >= 2 and ok[r[0]]:
                    cv2.polylines(m, [np.round(P(seg[r])).astype(np.int32)], False, color_of(p.id), 2, cv2.LINE_AA)
        for pid, xy in pos.items():
            (x, y), = P(xy)
            cv2.circle(m, (int(x), int(y)), 8, color_of(pid), -1, cv2.LINE_AA)
            cv2.putText(m, str(pid), (int(x) + 9, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        for e in events:
            if 0 <= t_rel - e["t"] < 2.5:            # flash the event for 2.5 s
                (x, y), = P([e["x"], e["y"]])
                cv2.circle(m, (int(x), int(y)), 22, (0, 0, 255) if e["kind"] == "lost" else (0, 220, 255), 3, cv2.LINE_AA)
        canvas[PANEL_H:, :map_w] = m
        side = canvas[PANEL_H:, map_w:]
        side[:] = (28, 28, 28)
        cv2.putText(side, f"{_fmt(start + t_rel)}  ids:{len(pos)}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side, "ID events (new / lost mid-pitch)", (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        past = [e for e in events if e["t"] <= t_rel][-14:]
        for r, e in enumerate(reversed(past)):
            fresh = t_rel - e["t"] < 2.5
            col = ((0, 0, 255) if e["kind"] == "lost" else (0, 220, 255)) if fresh else (200, 200, 200)
            txt = f"{_fmt(start + e['t'])}  #{e['id']} {e['kind']}  ({e['x']:.0f},{e['y']:.0f})"
            cv2.putText(side, txt, (14, 102 + 30 * r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2 if fresh else 1, cv2.LINE_AA)
        vw.write(canvas)
        if j and j % 300 == 0:
            print(f"    영상 {j}/{n_out}")
    vw.release()
    for r in readers.values():
        r.close()
    return n_out


def run(videos: dict[str, str], cals: dict, offsets: dict[str, tuple[float, float]], start: float, duration: float,
        out_dir: str, detector_factory, court: Court, rate_hz: float = 10.0, fps_out: float = 10.0,
        det_tag: str = "", embedder=None, ids: str = "v2", video: bool = True, summary: dict | None = None) -> dict:
    """Detect (cached per camera), track with build_tracks(ids=...), write tracks.json / events.csv / summary.json and,
    with `video`, trackview.mp4. `detector_factory(cam)` gives the camera's detector, which sees every frame of the
    window in order (pipeline.yolo_track.detection_tracker: detection/'s YOLO + ByteTrack; a plain detect.Detector
    works too); its boxes are kept at `rate_hz` (yolo_track.track_video). `embedder` (pipeline.reid.reid_embedder)
    adds a ReID embedding to every kept box, which ids="v2" / "v2gta" need; without it they fall back to
    "appearance". The detection cache key is `det_tag` plus whether ReID was computed. `summary` (optional) goes into
    summary.json next to the run's own numbers (per-player distance, speed, time seen)."""
    os.makedirs(out_dir, exist_ok=True)
    tag = det_tag + ("+reid" if embedder is not None else "")
    dets = {}
    for cam, path in videos.items():
        off, drift = offsets.get(cam, (0.0, 0.0))
        s_cam = (start - off) / (1 + drift)
        cache = os.path.join(out_dir, f"det_{cam}.json")
        want = {"video": os.path.basename(path), "start": round(s_cam, 3), "duration": duration, "rate": rate_hz, "tag": tag}
        # the file size tells two videos of the same name apart (HOGEUN's are camera1.mp4 / camera2.mp4 for any match);
        # a cache written before it was recorded still counts
        size = os.path.getsize(path) if os.path.isfile(path) else None
        if os.path.exists(cache):
            d, meta = detect.load(cache)
            if all(meta.get(k) == v for k, v in want.items()) and meta.get("video_bytes", size) == size:
                dets[cam] = d
                print(f"  {cam}: 저장된 검출 결과 사용 ({len(d)}개)")
                continue
        print(f"  {cam}: 선수 검출 중 ({_fmt(s_cam)}부터 {duration:.0f}초)...")
        # To switch back to our detector on the 10 Hz frames (pipeline/detect.py): uncomment these, drop track_video.
        # d, fps, _ = detect.detect_video(path, detector_factory(cam), rate_hz=rate_hz, start_s=max(s_cam, 0.0),
        #                                 end_s=s_cam + duration, progress=True, embedder=embedder)
        d, fps, _ = yolo_track.track_video(path, detector_factory(cam), rate_hz=rate_hz, start_s=max(s_cam, 0.0),
                                           end_s=s_cam + duration, progress=True, embedder=embedder)
        detect.save(d, cache, **want, video_bytes=size)
        dets[cam] = detect.load(cache)[0]       # what a rerun reads (float16 ReID), so both give the same tracks
    # reference clock for tracking: 0 = window start on cam1's clock
    sync = {cam: {"offset": offsets.get(cam, (0.0, 0.0))[0] - start, "drift": offsets.get(cam, (0.0, 0.0))[1]} for cam in videos}
    info: dict = {}
    extra = {"frame_wh": {cam: frame_size(p) for cam, p in videos.items()}} if ids in LINKERS else {}
    ts = build_tracks(court, dets, cals, sync, fps_out=rate_hz, ids=ids, info=info, **extra)
    if info.get("fallback"):
        print("  ReID 특징이 없어 예전 방식(appearance)으로 추적합니다")
    clk = info.get("clock")
    if clk and clk["applied"]:          # the corrected camera is read on the players' clock, like its boxes
        offsets = dict(offsets)
        offsets[clk["camera"]] = corrected_sync(*offsets.get(clk["camera"], (0.0, 0.0)), clk, start)
        print(f"  {clk['camera']} 시계 보정: {clk['offset_s']:+.3f}초, {clk['drift'] * 1e6:+.0f} ppm")
    al = info.get("alignment")
    if info.get("ids") in LINKERS:
        print(f"  추적 {info['ids']}: 조각 {info['tracklets']}개 → 선수 {info['identities']}명"
              + (f", 두 카메라 위치 차 {al['median_before_m']} → {al['median_after_m']} m" if al else ""))
    n = int(round(duration * rate_hz))
    for p in ts.players:                                   # clip to the window
        xy = np.full((n, 2), np.nan)
        m = min(n, len(p.xy))
        xy[:m] = p.xy[:m]
        p.xy = xy
    ts.players = [p for p in ts.players if np.any(np.isfinite(p.xy[:, 0]))]
    ts.start = start
    ts.compute_stats()
    ts.dump(os.path.join(out_dir, "tracks.json"))
    events = find_events(ts, court)
    with open(os.path.join(out_dir, "events.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["시각(cam1)", "ID", "종류", "x(m)", "y(m)"])
        for e in events:
            w.writerow([_fmt(start + e["t"]), e["id"], "새로 생김" if e["kind"] == "new" else "끊김", e["x"], e["y"]])
    seen = {p.id: float(np.mean(np.isfinite(p.xy[:, 0]))) for p in ts.players}
    res = {"ids": len(ts.players), "ids_over_half": sum(v > 0.5 for v in seen.values()), "events": len(events),
           "frames": 0, "detections": {c: len(d) for c, d in dets.items()}, "tracker": info.get("ids", ids)}
    record = {**(summary or {}), "window_cam1_s": [start, start + duration], "rate_hz": rate_hz, "offsets": offsets,
              **{k: res[k] for k in ("tracker", "detections", "ids", "ids_over_half", "events")},
              "tracklets": info.get("tracklets"), "clock": clk,
              "alignment": al and {k: al[k] for k in ("pairs", "median_before_m", "median_after_m") if k in al},
              "players": [{"id": p.id, "team": p.team, "seen_share": round(seen[p.id], 3), **p.stats} for p in ts.players]}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1, default=str)
    if video:
        print(f"  영상 만드는 중...")
        res["frames"] = render(os.path.join(out_dir, "trackview.mp4"), videos, cals, offsets, ts, dets, court, start,
                               events, fps_out)
    return res


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, nearfar.REPO).replace(os.sep, "/")
    except ValueError:                                     # another drive
        return str(path)


def calibration_record(n: int, cal: Calibration | CalibrationTimeline, out_dir: str, tapped: str | None = None) -> dict:
    """What summary.json says about camera n's calibration: HOGEUN's file, the frame it was clicked on and the knock
    shifts applied to it; for a tapped calibration (load_calibration) its calib.json `tapped` and the timeline starts."""
    tl = cal if isinstance(cal, list) else [(0.0, cal)]
    if not isinstance(tl[0][1], nearfar.NearFarCalibration):
        return {"file": _rel(tapped) if tapped else None, "timeline": [{"from_s": t} for t, _ in tl]}
    team = tl[0][1].team
    return {"file": _rel(nearfar.team_file(n, out_dir)), "video": os.path.basename(str(team.get("video", ""))),
            "t_sec": team.get("t_sec"),
            "timeline": [{"from_s": t, "shift_px": [round(v, 2) for v in c.shift]} for t, c in tl]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m futsal.trackview", description="두 영상 → 선수 추적 (tracks.json, summary.json, 확인 영상)")
    ap.add_argument("--video", nargs="+", required=True, help="cam1 영상 [cam2 영상]")
    # To switch back to our tapped calib/<cam>/calib.json: uncomment --calib, its check and load_calibration below
    # (and drop the H_cam file check and the load_team_calibration lines); summary.json then names those files.
    # ap.add_argument("--calib", nargs="+", required=True, help="영상마다 calib.json (python -m futsal.calibtool)")
    ap.add_argument("--team-calib", default=str(nearfar.TEAM_OUT), metavar="DIR",
                    help="HOGEUN 좌표 변환 결과 폴더 (기본 calibration/out). 카메라 N은 H_camN_refined.json, 없으면 H_camN.json")
    ap.add_argument("--offset", nargs="+", type=float, default=None, help="영상마다 시간 차 (cam1은 0, cam2는 futsal.sync 결과)")
    ap.add_argument("--drift", nargs="+", type=float, default=None)
    ap.add_argument("--start", type=float, required=True, help="cam1 영상 기준 시작 초")
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--court", default="40x20")
    # To switch back to our detector: uncomment these with the detect.yolo_detector run(...) call below.
    # ap.add_argument("--model", default="yolo26s.pt"); ap.add_argument("--imgsz", type=int, default=1280)
    # ap.add_argument("--conf", type=float, default=None, help="검출 신뢰도 하한 (기본: --ids v2 / v2gta 는 0.1, 그 외 0.3)")
    ap.add_argument("--rate", type=float, default=10.0, help="추적에 쓰는 초당 프레임 수 (검출·ByteTrack은 모든 프레임)")
    ap.add_argument("--ids", choices=IDS, default="v2", help="추적 방식: v2 (기본, 팀당 5+1 자리 배정, ReID 필요) / v2gta (정원 없는 연결, ReID 필요) / appearance (예전 방식) / motion")
    ap.add_argument("--no-reid", action="store_true", help="ReID 계산을 건너뛰고 예전 방식(appearance)으로 추적")
    ap.add_argument("--auto-moves", action="store_true", help="카메라가 움직인 순간을 자동으로 찾아 보정 (futsal.camshift)")
    ap.add_argument("--no-video", action="store_true", help="확인 영상(trackview.mp4)을 만들지 않음")
    ap.add_argument("--out", default="trackview")
    a = ap.parse_args(argv)
    # if len(a.calib) != len(a.video):
    #     ap.error("--calib 는 --video 와 개수가 같아야 합니다")
    for opt, vals in (("--offset", a.offset), ("--drift", a.drift)):     # a missing one would silently become 0
        if vals is not None and len(vals) != len(a.video):
            ap.error(f"{opt} 는 --video 와 개수가 같아야 합니다 (영상마다 하나)")
    if a.duration <= 0 or a.rate <= 0:
        ap.error("--duration 과 --rate 는 0보다 커야 합니다")
    for i in range(len(a.video)):
        if not os.path.exists(nearfar.team_file(i + 1, a.team_calib)):
            ap.error(f"{a.team_calib} 에 cam{i + 1} 보정 파일(H_cam{i + 1}.json)이 없습니다")
    names = [f"cam{i + 1}" for i in range(len(a.video))]
    offs = a.offset or [0.0] * len(names)
    drifts = a.drift or [0.0] * len(names)
    court = Court.parse(a.court)
    os.makedirs(a.out, exist_ok=True)
    videos = dict(zip(names, a.video))
    # cals = {n: load_calibration(c, v, a.auto_moves, a.out) for n, c, v in zip(names, a.calib, a.video)}
    try:
        cals = {n: load_team_calibration(i + 1, v, a.auto_moves, a.out, a.team_calib) for i, (n, v) in enumerate(videos.items())}
    except ValueError as e:                                # a 10/8 video in the other camera's place
        ap.error(str(e))
    team = any(isinstance(c[0][1] if isinstance(c, list) else c, nearfar.NearFarCalibration) for c in cals.values())
    if team and (court.length, court.width) != (40.0, 20.0):     # calibration/court.py: L, W = 40, 20
        ap.error("calibration/ (HOGEUN) 좌표 변환은 40x20 코트만 됩니다 (--court 40x20)")
    offsets = {n: (o, d) for n, o, d in zip(names, offs, drifts)}
    ids = "appearance" if a.no_reid and a.ids in LINKERS else a.ids
    embedder = None
    if ids in LINKERS:
        from .pipeline.reid import reid_embedder
        try:
            embedder = reid_embedder()
        except ImportError as e:
            print(f"ReID를 쓸 수 없어 예전 방식(appearance)으로 추적합니다: {e}")
            ids = "appearance"
    tapped = getattr(a, "calib", None) or [None] * len(names)      # a.calib exists only with --calib uncommented
    summary = {"calibration": {n: calibration_record(i + 1, cals[n], a.team_calib, tapped[i]) for i, n in enumerate(videos)}}
    # To switch back to our detector (pipeline/detect.py): uncomment these, --model/--conf above, detect_video in run(),
    # and drop the settings / print / summary["detection"] / run(...) lines below them (one run() call only).
    # conf = a.conf if a.conf is not None else (0.1 if ids in LINKERS else 0.3)
    # res = run(videos, cals, offsets, a.start, a.duration, a.out,
    #           lambda cam: detect.yolo_detector(a.model, conf=conf, imgsz=a.imgsz), court, a.rate,
    #           det_tag=f"{a.model}@{a.imgsz}/{conf}", embedder=embedder, ids=ids, video=not a.no_video, summary=summary)
    settings = yolo_track.read_settings()                  # detection/detect_person.py (차재혁): model and track() settings
    print(f"검출: {_rel(settings['file'])} → {settings['model']} {yolo_track.track_kwargs(settings)}")
    summary["detection"] = {"file": _rel(settings["file"]), "model": settings["model"], "track": yolo_track.track_kwargs(settings)}
    res = run(videos, cals, offsets, a.start, a.duration, a.out, lambda cam: yolo_track.detection_tracker(settings),
              court, a.rate, det_tag=yolo_track.tag(settings), embedder=embedder, ids=ids, video=not a.no_video,
              summary=summary)
    print(f"\n추적 방식 {res['tracker']}: ID {res['ids']}개 (구간 절반 이상 보인 ID {res['ids_over_half']}개), "
          f"경기장 한가운데서 생기거나 끊긴 ID {res['events']}건")
    outs = ["tracks.json", "summary.json", "events.csv"] + ([] if a.no_video else ["trackview.mp4"])
    print("→ " + ", ".join(os.path.join(a.out, f) for f in outs))


if __name__ == "__main__":
    main()
