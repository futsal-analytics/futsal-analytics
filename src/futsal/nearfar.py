"""HOGEUN's court calibration (calibration/ in this repository) as the pipeline's Calibration.

calibration/ gives each camera two homographies (calibration/calibrate.py, refine.py): H_near for the camera's own
half, from 5 clicks, and H_far beyond the halfway line, fitted to the white lines and tied to H_near so that both send
the halfway line to the same pixels. calibration/court_map.py maps pixels to court metres with them (load,
pixel_to_court). NearFarCalibration runs that code as it is and adds what the pipeline needs on top: our pitch
coordinates, the inverse map (drawing only), the metres-per-pixel scale, camera knocks (shifted) and JSON for
calibration timelines. Nothing in calibration/ is changed or written.

Coordinates. HOGEUN's (calibration/court.py): origin at the centre spot, x along the court towards the goal cam1 faces
("east", x = +20), y towards cam1's left sideline (y = +10), metres. Ours (futsal.court): origin at corner_bl, x 0..length,
y 0..width. cam1 stands outside our corner (0, 0) facing the x = length goal and cam2 at the opposite corner, so the
axes agree and the map is a pure translation (no rotation, no flip):

    x_ours = x_team + length / 2,   y_ours = y_team + width / 2

Found on the 10/8 videos: of the 8 axis-aligned maps with a fitted translation the identity fits our tapped
calibration best for both cameras (translation within 0.3 m of (20, 10), the next map 3.5 m or more off), and
HOGEUN's clicks land on our keypoints of the same marks (tests/test_nearfar.py).

Loading calibration/ code. court_map.py imports its neighbours as top-level modules (`from calibrate import
HALF_SIGN`, and calibrate imports `court`). It is loaded by file path once, with calibration/ on sys.path for the
import only and bytecode writing off (no __pycache__ inside calibration/); court_map, calibrate and court are then
removed from sys.modules (any module of the same name that was there before is put back), so these names cannot
shadow anything. The functions keep working through their own module globals.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from .court import Court
from .homography import Calibration, CalibrationTimeline, image_to_pitch

REPO = Path(__file__).resolve().parents[2]
TEAM_DIR = REPO / "calibration"          # HOGEUN's folder (only read)
TEAM_OUT = TEAM_DIR / "out"              # H_cam<N>.json (5 clicks) and H_cam<N>_refined.json (white lines)
TOP_LEVEL = ("court_map", "calibrate", "court")    # what importing court_map.py pulls in as top-level modules

# Camera poses of the 10/8 test videos: [(from_s on the camera's own clock, (dx, dy) px the picture is slid against a
# reference frame)], one row per settled stretch. A calibration made on the frame at t (HOGEUN's t_sec) gets the knocks
# pose - pose(t) (knocks_20261008), as Calibration.shifted takes them.
# Measured on the court: edge phase correlation (camshift._edges) in 320x270 px tiles against the 600 s frame, a
# shift + rotation + scale fit, its median shift over the court's pixels.
# cam1 (reference 600 s) was knocked at 2:40: the 60 s and 150 s pictures are (+5.8, +9.3) / (+5.8, +9.2) px from the
# 600 s one (rotation 0.07 deg), and 175 s onwards stays within 2 px of 600 s.
# cam2 (reference 60 s, within 0.6 px of the 600 s picture): after the 46:13 knock (2800 s, 3000 s) the court slid
# (+12.1, -0.8) / (+12.4, -1.8) px and turned 0.2 deg, which a shift follows within about 4 px.
POSES_20261008 = {"cam1": [(0.0, (5.8, 9.2)), (159.9, (0.0, 0.0))],
                  "cam2": [(0.0, (0.0, 0.0)), (2773.0, (12.2, -1.3))]}
# Where no row fits (single full-frame edge phase correlations against the 60 s frame): cam1 starts up to 130 px off and
# is within 3 px from about 15 s; its knock starts between 159.4 and 159.5 s and it shakes (40 px and more) until about
# 172 s. cam2 starts up to 120 px off and settles at about 30 s; its knock starts between 2773 and 2774 s, within 2 px of
# the new pose from about 2776 s. Boxes there are still mapped with their row's pose: metres off on the far half.
# Within a row the picture drifts by about 2 px (cam1 after its knock, cam2 before it): 0.1-0.4 m on the far half.
UNSETTLED_20261008 = {"cam1": [(0.0, 20.0), (159.4, 172.0)], "cam2": [(0.0, 30.0), (2773.0, 2776.0)]}
# File name and frame count of each camera's video (a renamed copy is found by its frame count)
VIDEOS_20261008 = {"cam1": ("20261008_170919.mp4", 198967), "cam2": ("20261008_171122.mp4", 96222)}

_court_map: ModuleType | None = None


def load_court_map(folder: str | os.PathLike = TEAM_DIR) -> ModuleType:
    """calibration/court_map.py as a module object (see the module docstring for what is and is not left behind)."""
    folder = str(folder)
    saved = {n: sys.modules.pop(n) for n in TOP_LEVEL if n in sys.modules}
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location("court_map", os.path.join(folder, "court_map.py"))
        if spec is None or spec.loader is None:
            raise ImportError(f"no court_map.py in {folder}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(folder)
        sys.path_importer_cache.pop(folder, None)
        sys.dont_write_bytecode = bytecode
        for n in TOP_LEVEL:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return mod


def court_map() -> ModuleType:
    """HOGEUN's calibration/court_map.py, loaded on first use."""
    global _court_map
    if _court_map is None:
        _court_map = load_court_map()
    return _court_map


def team_to_pitch(centre: tuple[float, float] = Court().centre) -> np.ndarray:
    """3x3 map from HOGEUN's centre-origin court coordinates to ours (see the module docstring)."""
    cx, cy = centre
    return np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])


def _inverse(H: np.ndarray) -> np.ndarray:
    Hi = np.linalg.inv(H)
    return Hi / Hi[2, 2]


@dataclass(kw_only=True)
class NearFarCalibration(Calibration):
    """One camera of HOGEUN's calibration in our pitch coordinates.

    `team` is the calibration as calibration/court_map.load returns it (half, H_near, H_far in HOGEUN's court metres,
    the clicks), with both homographies moved by shifted() after a camera knock. to_pitch is court_map.pixel_to_court
    on it plus the translation; a pixel is on the far half when H_near puts it beyond the halfway line, and both
    homographies agree on that line, so positions are continuous across it. `H` is the near half in our coordinates
    (Calibration.H_inv and homography.framing use the near half only). identity's covariance Jacobian takes central
    differences of to_pitch, so it is the far homography's on the far half too."""
    team: dict
    centre: tuple[float, float] = (20.0, 10.0)      # HOGEUN's origin in our coordinates
    shift: tuple[float, float] = (0.0, 0.0)         # (dx, dy) px of the knocks applied since HOGEUN's frame (record only)

    @property
    def sign(self) -> int:
        """+1: the camera faces the x = length goal, so its far half is x > centre; -1: the other way."""
        return int(court_map().HALF_SIGN[self.team["half"]])

    @property
    def H_far(self) -> np.ndarray:
        """image (u, v) -> our pitch (x, y), beyond the halfway line."""
        return team_to_pitch(self.centre) @ np.asarray(self.team["H_far"], float)

    def beyond(self, xy) -> np.ndarray:
        """Pitch points on this camera's far half (beyond the halfway line, seen from the camera)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        return self.sign * (xy[:, 0] - self.centre[0]) > 0

    def to_pitch(self, uv) -> np.ndarray:
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        if not len(uv):
            return np.zeros((0, 2))
        xy = court_map().pixel_to_court(self.team, uv).astype(np.float64) + self.centre
        xy[~np.isfinite(uv).all(1)] = np.nan          # cv2 would put a NaN pixel on the centre spot, in court
        return xy

    def to_image(self, xy) -> np.ndarray:
        """Pitch -> pixels with the homography of the pitch point's half (drawing only; calibration/ has no inverse)."""
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        uv = image_to_pitch(self.H_inv, xy)
        far = self.beyond(xy)
        if far.any():
            uv[far] = image_to_pitch(_inverse(self.H_far), xy[far])
        return uv

    def metres_per_pixel(self, uv) -> np.ndarray:
        """homography.metres_per_pixel through to_pitch: the largest singular value of the 1 px forward-difference
        Jacobian, so each pixel gets its own half's scale."""
        uv = np.atleast_2d(np.asarray(uv, float))
        g0 = self.to_pitch(uv)
        J = np.stack([self.to_pitch(uv + [1.0, 0.0]) - g0, self.to_pitch(uv + [0.0, 1.0]) - g0], axis=2)
        out = np.full(len(uv), np.nan)                 # NaN for a NaN pixel (svd does not take one)
        ok = np.isfinite(J).all((1, 2))
        if ok.any():
            out[ok] = np.linalg.svd(J[ok], compute_uv=False)[:, 0]
        return out

    def shifted(self, dx: float, dy: float) -> "NearFarCalibration":
        """Calibration.shifted for both homographies (they still share the halfway line). The clicks (img_pts) and
        t_sec in `team` stay those of the clicked frame; `shift` says how far the homographies were moved since."""
        T = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]])
        team = {**self.team, **{k: (np.asarray(self.team[k], float) @ T).tolist() for k in ("H_near", "H_far")}}
        H = team_to_pitch(self.centre) @ np.asarray(team["H_near"], float)      # as from_team builds it (JSON round trip)
        return replace(self, H=H, used=list(self.used), team=team, shift=(self.shift[0] + float(dx), self.shift[1] + float(dy)))

    def to_json(self) -> dict:
        """Calibration's keys (H: the near half) plus H_far in our coordinates and HOGEUN's calibration itself, which
        from_json rebuilds the camera from."""
        return {**super().to_json(), "H_far": self.H_far.tolist(), "centre": list(self.centre), "shift_px": list(self.shift),
                "team": self.team}

    @classmethod
    def from_json(cls, d: dict) -> "NearFarCalibration":
        """Our to_json, or one of HOGEUN's calibration/out/H_cam<N>[_refined].json files (they have "H_near")."""
        if "H_near" in d:
            return cls.from_team(d)
        cal = cls.from_team(d["team"], tuple(float(v) for v in d["centre"]))
        return replace(cal, rms_px=float(d["rms_px"]), used=list(d.get("used", [])),
                       shift=tuple(float(v) for v in d.get("shift_px", (0.0, 0.0))))

    @classmethod
    def from_team(cls, d: dict, centre: tuple[float, float] = Court().centre) -> "NearFarCalibration":
        """HOGEUN's calibration of one camera (court_map.load) -> our pitch coordinates (`centre`: Court.centre).
        rms_px is the 5 clicks against their court points through the near homography: about 1 px for the click-only
        file, a few px for the refined one (fitted to the white lines with the clicks as soft anchors)."""
        H = team_to_pitch(centre) @ np.asarray(d["H_near"], float)     # bottom row (0, 0, 1): H[2, 2] stays 1
        img = np.asarray(d["img_pts"], float)
        world = np.array([d["world"]["near"][str(i)] for i in range(len(img))], float) + centre
        rms = float(np.sqrt(np.mean(np.sum((image_to_pitch(_inverse(H), world) - img) ** 2, 1))))
        return cls(H=H, rms_px=rms, used=list(d.get("names", [])), team=dict(d), centre=tuple(centre))


def team_file(n: int, out_dir: str | os.PathLike = TEAM_OUT) -> str:
    """The file calibration/court_map.load(n, out_dir) reads: the refined one if it exists (for the record only)."""
    refined = os.path.join(out_dir, f"H_cam{n}_refined.json")
    return refined if os.path.exists(refined) else os.path.join(out_dir, f"H_cam{n}.json")


def load_team(n: int, out_dir: str | os.PathLike = TEAM_OUT, court: Court = Court()) -> NearFarCalibration:
    """Camera n through calibration/court_map.load. `out_dir` is passed on because court_map.load defaults to "out"
    relative to the working directory."""
    return NearFarCalibration.from_team(court_map().load(n, str(out_dir)), court.centre)


def knock_timeline(base: Calibration, knocks: list[tuple[float, tuple[float, float]]]) -> CalibrationTimeline:
    """[(from_s, (dx, dy))] -> [(from_s, base.shifted(dx, dy))], e.g. knocks_20261008(cam, t_sec)."""
    return [(float(t), base.shifted(dx, dy)) for t, (dx, dy) in knocks]


def knocks_20261008(cam: str, at_s: float) -> list[tuple[float, tuple[float, float]]]:
    """POSES_20261008[cam] for a calibration made on the camera's frame at `at_s`: [(from_s, pose - pose at at_s)]."""
    poses = POSES_20261008[cam]
    x0, y0 = next((p for t, p in reversed(poses) if t <= at_s), poses[0][1])
    return [(t, (dx - x0, dy - y0)) for t, (dx, dy) in poses]


def unsettled_20261008(cam: str, t: float) -> bool:
    """True when camera `cam` of the 10/8 test videos is starting up or shaking at `t` (UNSETTLED_20261008)."""
    return any(a <= t < b for a, b in UNSETTLED_20261008[cam])


def camera_20261008(video: str) -> str | None:
    """"cam1" / "cam2" when `video` is one of the 10/8 test videos (by file name, or by frame count), else None."""
    name = os.path.basename(str(video).replace("\\", "/"))    # a Windows path names the file on Linux too
    cam = next((c for c, (v, _) in VIDEOS_20261008.items() if v == name), None)
    if cam is None and os.path.isfile(video):
        cap = cv2.VideoCapture(video)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        cam = next((c for c, (_, n) in VIDEOS_20261008.items() if n == frames), None)
    return cam


def known_knocks(video: str, at_s: float) -> list[tuple[float, tuple[float, float]]] | None:
    """knocks_20261008 of a 10/8 test video (camera_20261008) for a calibration made on its frame at `at_s`, else None."""
    cam = camera_20261008(video)
    return None if cam is None else knocks_20261008(cam, at_s)


KNOCKS_20261008 = {cam: knocks_20261008(cam, 60.0) for cam in POSES_20261008}    # HOGEUN's files: the 60 s frame


def timeline_json(timeline: CalibrationTimeline) -> list[dict]:
    """A timeline as homography.timeline_from_json reads it back."""
    return [{"from": t, **c.to_json()} for t, c in timeline]


def save_timeline(timeline: CalibrationTimeline, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timeline_json(timeline), f, indent=1)
