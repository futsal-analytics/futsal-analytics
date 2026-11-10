"""HOGEUN's calibration (calibration/out, calibration/court_map.py) as the pipeline's calibration: futsal.nearfar.

calibration/ is only read: court_map.py is loaded with bytecode off and its top-level modules are dropped afterwards
(tested below). The cross-checks against our tapped calibration need the local calib/ folder and are skipped without
it."""
import importlib
import json
import os
import sys
import types

import numpy as np
import pytest

from futsal import nearfar
from futsal.court import Court
from futsal.homography import Calibration, calibration_at, image_to_pitch, metres_per_pixel, timeline_from_json
from futsal.nearfar import NearFarCalibration, knock_timeline, load_team, team_file, timeline_json
from futsal.pipeline import identity
from futsal.pipeline.boxes import build_cam_boxes
from futsal.pipeline.detect import Detection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_DIR = os.path.join(ROOT, "calibration")
TEAM = os.path.join(TEAM_DIR, "out")
CALIB = os.path.join(ROOT, "calib")
COURT = Court()
FILES = [(n, kind) for n in (1, 2) for kind in ("", "_refined")]


def team_json(n, kind):
    with open(os.path.join(TEAM, f"H_cam{n}{kind}.json"), encoding="utf-8") as f:
        return json.load(f)


def pitch_grid(cal, step=0.5, margin=0.0):
    """Pixels of an in-court pitch grid (both halves, `margin` m kept off the halfway line), and the grid points."""
    xs, ys = np.arange(0.25, COURT.length, step), np.arange(0.25, COURT.width, step)
    xy = np.array([(x, y) for x in xs for y in ys])
    xy = xy[np.abs(xy[:, 0] - COURT.length / 2) > margin]
    uv = cal.to_image(xy)
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
    return uv[keep], xy[keep]


def homography_jacobian(H, uv):
    """(n, 2, 2) exact d(pitch)/d(pixel) of one homography."""
    uv1 = np.hstack([uv, np.ones((len(uv), 1))])
    w = uv1 @ H[2]
    p = image_to_pitch(H, uv)
    return (H[None, :2, :2] - p[:, :, None] * H[None, 2, :2][:, None, :]) / w[:, None, None]


def listing(folder):
    """Everything under `folder` (names and sizes): nothing may be added or changed by loading teammate code."""
    return sorted((os.path.relpath(os.path.join(d, f), folder), os.path.getsize(os.path.join(d, f)))
                  for d, _, fs in os.walk(folder) for f in fs)


# ---------------------------------------------------------------------------------------------------------------
# Loading calibration/ code
# ---------------------------------------------------------------------------------------------------------------

def test_loading_court_map_writes_nothing_and_leaves_no_modules(monkeypatch):
    before = listing(TEAM_DIR)
    stand_in = types.ModuleType("court")                      # someone else's top-level `court` must survive
    monkeypatch.setitem(sys.modules, "court", stand_in)
    path, bytecode = list(sys.path), sys.dont_write_bytecode
    mod = nearfar.load_court_map()
    assert sys.modules["court"] is stand_in
    assert not {"court_map", "calibrate"} & set(sys.modules)
    assert sys.path == path and sys.dont_write_bytecode == bytecode
    assert listing(TEAM_DIR) == before and not os.path.exists(os.path.join(TEAM_DIR, "__pycache__"))
    # the functions still work through their own globals (HOGEUN's calibrate.HALF_SIGN, court constants)
    assert mod.HALF_SIGN == {"east": 1, "west": -1}
    xy = mod.pixel_to_court(mod.load(1, TEAM), [[960.0, 700.0]])
    assert xy.shape == (1, 2) and np.all(np.isfinite(xy))


def test_load_team_reads_what_court_map_load_reads():
    cm = nearfar.load_court_map()
    for n in (1, 2):
        assert team_file(n, TEAM) == os.path.join(TEAM, f"H_cam{n}_refined.json")     # refined first, as court_map.load
        cal = load_team(n, TEAM)
        assert cal.team == cm.load(n, TEAM) == team_json(n, "_refined")
        assert cal.team["half"] == ("east" if n == 1 else "west") and cal.sign == (1 if n == 1 else -1)


# ---------------------------------------------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,kind", FILES)
def test_equals_court_map_plus_translation(n, kind):
    """to_pitch is HOGEUN's pixel_to_court (loaded separately here) + (20, 10), and its near / far rule is the one in
    calibration/court_map.py: H_near decides, H_far maps the pixels it puts beyond the halfway line."""
    d = team_json(n, kind)
    cal = NearFarCalibration.from_team(d)
    cm = nearfar.load_court_map()
    rng = np.random.default_rng(n)
    px = rng.uniform([0, 0], [1920, 1080], (20000, 2)).astype(np.float32).astype(float)
    ours = cal.to_pitch(px)
    assert np.array_equal(ours, cm.pixel_to_court(d, px).astype(float) + COURT.centre)
    keep = COURT.contains(ours, 10.0)                              # above the horizon both blow up
    near = image_to_pitch(cal.H, px)
    far = cal.sign * (near[:, 0] - COURT.length / 2) > 0
    assert (keep & far).sum() > 1000 and (keep & ~far).sum() > 1000
    assert np.abs(ours[keep & ~far] - near[keep & ~far]).max() < 1e-3                  # float32 inside court_map
    assert np.abs(ours[keep & far] - image_to_pitch(cal.H_far, px[keep & far])).max() < 1e-3


# HOGEUN's 5 clicks (calibration/calibrate.py LANDMARKS; a west-facing camera sees them turned by 180 degrees) and our
# keypoints of the same marks. Click 3 is any point on the right sideline: only its y is known.
CLICKED = {"east": ["halfway_top", "circle_bottom", "halfway_bottom", 0.0, "second_penalty_left"],
           "west": ["halfway_bottom", "circle_top", "halfway_top", COURT.width, "second_penalty_right"]}


@pytest.mark.parametrize("n,kind", FILES)
def test_clicks_land_on_our_keypoints(n, kind):
    """In our coordinates HOGEUN's clicks land on our keypoints of the same marks: the pure translation is right.
    The click-only file fits its 5 clicks (8 constraints, about 1 px): within 0.1 m. The refined file was fitted to the
    white lines with the clicks as soft anchors (rms 4.6 / 5.3 px, at most 8.3 px): within 10 px in the image."""
    d = team_json(n, kind)
    cal = NearFarCalibration.from_team(d)
    kp = COURT.keypoints()
    clicks = np.asarray(d["img_pts"], float)
    xy = cal.to_pitch(clicks)
    for i, mark in enumerate(CLICKED[d["half"]]):
        if not isinstance(mark, str):
            assert abs(xy[i, 1] - mark) < 0.1
        elif kind == "":
            assert np.hypot(*(xy[i] - kp[mark])) < 0.1, mark
        else:
            assert np.hypot(*(cal.to_image([kp[mark]])[0] - clicks[i])) < 10.0, mark
    assert cal.rms_px < (2.0 if kind == "" else 6.0)


@pytest.mark.parametrize("n,kind", FILES)
def test_continuous_across_the_halfway_line(n, kind):
    cal = NearFarCalibration.from_team(team_json(n, kind))
    seam = np.stack([np.full(41, COURT.length / 2), np.linspace(0, COURT.width, 41)], 1)
    uv0 = image_to_pitch(cal.H_inv, seam)
    assert np.allclose(image_to_pitch(np.linalg.inv(cal.H_far), seam), uv0, atol=1e-6)   # both: the same pixels
    into_far = cal.to_image(seam + [cal.sign * 0.05, 0.0]) - uv0
    d = 0.01 * into_far / np.linalg.norm(into_far, axis=1, keepdims=True)
    near, far = cal.to_pitch(uv0 - d), cal.to_pitch(uv0 + d)                # 0.02 px apart, one on each side
    assert np.all(cal.beyond(far)) and not np.any(cal.beyond(near))
    assert np.abs(far - near).max() < 0.01                                  # no jump at the seam


@pytest.mark.parametrize("n,kind", FILES)
def test_to_image_inverts_to_pitch_on_both_halves(n, kind):
    cal = NearFarCalibration.from_team(team_json(n, kind))
    uv, xy = pitch_grid(cal, margin=0.25)
    far = cal.beyond(xy)
    assert far.sum() > 100 and (~far).sum() > 100
    assert np.abs(cal.to_pitch(uv) - xy).max() < 1e-3
    if (n, kind) == (1, "_refined"):                               # H_far really differs here: the far half needs it
        assert np.hypot(*(image_to_pitch(cal.H, uv[far]) - xy[far]).T).max() > 0.5


@pytest.mark.parametrize("n,kind", FILES)
def test_metres_per_pixel_and_jacobian_follow_each_half(n, kind):
    """identity's covariance Jacobian (1 px central differences of to_pitch) and metres_per_pixel against the exact
    Jacobian of the half's own homography."""
    cal = NearFarCalibration.from_team(team_json(n, kind))
    uv, xy = pitch_grid(cal, margin=1.5)
    far = cal.beyond(xy)
    J = identity._pitch_jacobian(cal, uv)
    rel = lambda A, B: np.linalg.norm(A - B, axis=(1, 2)) / np.linalg.norm(B, axis=(1, 2))
    for H, m in ((cal.H, ~far), (cal.H_far, far)):
        assert rel(J[m], homography_jacobian(H, uv[m])).max() < 2e-3
        assert np.allclose(cal.metres_per_pixel(uv[m]), metres_per_pixel(H, uv[m]), rtol=2e-3)
    if (n, kind) == (1, "_refined"):
        assert np.median(rel(J[far], homography_jacobian(cal.H, uv[far]))) > 1e-2


def test_shifted_moves_both_homographies():
    cal = load_team(1, TEAM)
    moved = cal.shifted(-5.8, -9.2)
    assert isinstance(moved, NearFarCalibration) and moved.shift == (-5.8, -9.2) and cal.shift == (0.0, 0.0)
    uv, xy = pitch_grid(cal, margin=0.5)
    far = cal.beyond(xy)
    assert np.abs(moved.to_pitch(uv - [5.8, 9.2]) - cal.to_pitch(uv)).max() < 1e-3        # both halves
    T = np.array([[1.0, 0.0, 5.8], [0.0, 1.0, 9.2], [0.0, 0.0, 1.0]])
    for k in ("H_near", "H_far"):
        assert np.allclose(np.asarray(moved.team[k]), np.asarray(cal.team[k]) @ T)
    assert np.allclose(moved.H_far, cal.H_far @ T) and np.allclose(moved.H, cal.H @ T)
    assert np.allclose(moved.to_image(xy[far]), cal.to_image(xy[far]) - [5.8, 9.2])
    assert moved.shifted(5.8, 9.2).shift == (0.0, 0.0)


def test_timeline_json_round_trip_keeps_both_homographies():
    cal = load_team(2, TEAM)
    tl = knock_timeline(cal, [(2773.0, (12.2, -1.3)), (0.0, (0.0, 0.0))])
    back = timeline_from_json(json.loads(json.dumps(timeline_json(tl))))
    assert [t for t, _ in back] == [0.0, 2773.0]                   # sorted by start, like any timeline
    uv, _ = pitch_grid(cal, step=1.0)
    for (t, a), (_, b) in zip(sorted(tl, key=lambda p: p[0]), back):
        assert type(b) is NearFarCalibration and b.shift == a.shift and b.rms_px == a.rms_px and b.used == a.used
        assert np.array_equal(b.H_far, a.H_far) and np.array_equal(b.H, a.H) and b.team == a.team
        assert np.array_equal(b.to_pitch(uv), a.to_pitch(uv))
    assert calibration_at(back, 3000.0).shift == (12.2, -1.3)
    raw = timeline_from_json(team_json(2, "_refined"))             # HOGEUN's file itself reads as one calibration
    assert type(raw) is NearFarCalibration and np.array_equal(raw.to_pitch(uv), cal.to_pitch(uv))
    assert type(timeline_from_json(Calibration(np.eye(3), 1.0).to_json())) is Calibration     # ours: as before


def test_boxes_and_covariances_with_a_team_timeline():
    """The tracker's inputs (pitch position, sigma, covariance) from a knocked HOGEUN timeline."""
    cal = load_team(1, TEAM)
    tl = knock_timeline(cal, nearfar.KNOCKS_20261008["cam1"])
    dx, dy = nearfar.KNOCKS_20261008["cam1"][1][1]
    uv, _ = pitch_grid(cal, step=4.0)
    dets = [Detection(i, t, float(u) + (dx if t >= 159.9 else 0.0), float(v) + (dy if t >= 159.9 else 0.0), 0.9, None,
                      80.0, None) for i, (u, v) in enumerate(uv) for t in (60.0, 600.0)]
    c = build_cam_boxes("cam1", dets, tl, 0.0, 0.0, 0.0, 10.0, COURT)
    assert np.abs(c.xy[0::2] - c.xy[1::2]).max() < 1e-3          # same spot before and after the knock
    assert np.abs(c.xy[0::2] - cal.to_pitch(uv)).max() < 1e-9
    assert np.allclose(c.sigma[0::2], 3.0 * cal.metres_per_pixel(uv))
    R = identity.box_covariances({"cam1": c}, "cam1")
    assert np.all(np.isfinite(R)) and np.all(np.linalg.eigvalsh(R) > 0)


def test_known_knocks_by_video_and_calibration_frame():
    """The 10/8 knocks are the measured poses relative to the frame the calibration was clicked on (t_sec)."""
    assert nearfar.known_knocks(r"C:\videos\20261008_170919.mp4", 60.0) == nearfar.KNOCKS_20261008["cam1"]
    assert nearfar.KNOCKS_20261008["cam1"] == [(0.0, (0.0, 0.0)), (159.9, (-5.8, -9.2))]   # HOGEUN's files: 60 s
    assert nearfar.known_knocks("20261008_171122.mp4", 60.0) == [(0.0, (0.0, 0.0)), (2773.0, (12.2, -1.3))]
    assert nearfar.known_knocks("other.mp4", 60.0) is None
    # clicked after the cam1 knock (600 s, like our taps): the picture before 2:40 is shifted, after it is not
    assert nearfar.knocks_20261008("cam1", 600.0) == [(0.0, (5.8, 9.2)), (159.9, (0.0, 0.0))]
    assert nearfar.knocks_20261008("cam2", 3000.0) == [(0.0, (-12.2, 1.3)), (2773.0, (0.0, 0.0))]
    assert nearfar.unsettled_20261008("cam1", 165.0) and nearfar.unsettled_20261008("cam2", 10.0)
    assert not nearfar.unsettled_20261008("cam1", 60.0) and not nearfar.unsettled_20261008("cam2", 600.0)


def test_trackview_knocks_follow_the_calibration_frame(tmp_path, capsys):
    """trackview.load_team_calibration reads t_sec from HOGEUN's file: the same 10/8 video gets other knocks when the
    calibration was clicked on another frame, and an unknown video gets none, with a notice."""
    from futsal import trackview
    video = r"C:\videos\20261008_170919.mp4"
    tl = trackview.load_team_calibration(1, video)
    assert [(t, c.shift) for t, c in tl] == nearfar.KNOCKS_20261008["cam1"]
    d = team_json(1, "_refined")
    d["t_sec"] = 600.0
    (tmp_path / "H_cam1_refined.json").write_text(json.dumps(d), encoding="utf-8")
    tl = trackview.load_team_calibration(1, video, out_dir=str(tmp_path))
    assert [(t, c.shift) for t, c in tl] == [(0.0, (5.8, 9.2)), (159.9, (0.0, 0.0))]
    d["t_sec"] = 165.0
    (tmp_path / "H_cam1_refined.json").write_text(json.dumps(d), encoding="utf-8")
    capsys.readouterr()
    trackview.load_team_calibration(1, video, out_dir=str(tmp_path))
    assert "주의" in capsys.readouterr().out                      # clicked while the camera was shaking
    cal = trackview.load_team_calibration(1, r"D:\x\camera1.mp4")
    out = capsys.readouterr().out
    assert isinstance(cal, NearFarCalibration) and "그대로였다고" in out and "다시 찍어야" not in out
    trackview.load_team_calibration(1, r"D:\x\match2_cam1.mp4")       # not the video HOGEUN clicked on: say so
    assert "camera1.mp4 영상에서 찍은 것" in capsys.readouterr().out


def test_trackview_refuses_videos_in_the_wrong_order_or_with_missing_offsets(tmp_path):
    """cam2's 10/8 video in cam1's place would get cam1's calibration with cam2's knocks; an --offset or --drift
    list shorter than --video would leave a camera at 0 s without a word. Both stop before anything runs."""
    from futsal import trackview
    with pytest.raises(ValueError, match="cam2"):
        trackview.load_team_calibration(1, r"C:\videos\20261008_171122.mp4")
    run = ["--start", "0", "--out", str(tmp_path / "o")]
    for args in (["--video", r"C:\v\20261008_171122.mp4", r"C:\v\20261008_170919.mp4"],
                 ["--video", "a.mp4", "b.mp4", "--offset", "124.27"],
                 ["--video", "a.mp4", "b.mp4", "--drift", "0"],
                 ["--video", "a.mp4", "--duration", "0"],
                 ["--video", "a.mp4", "b.mp4", "--team-calib", str(tmp_path)]):          # no H_cam1.json there
        with pytest.raises(SystemExit):
            trackview.main(args + run)


def test_renamed_copy_of_a_test_video_is_found_by_frame_count(tmp_path, monkeypatch):
    """A copy under another name (HOGEUN's are camera1.mp4 / camera2.mp4) is found by its frame count (a stand-in
    VideoCapture reports it here); other frame counts and missing files are not 10/8 videos."""
    frames = {}

    class Capture:
        def __init__(self, path):
            self.n = frames[os.path.basename(path)]

        def get(self, prop):
            return float(self.n) if prop == nearfar.cv2.CAP_PROP_FRAME_COUNT else 0.0

        def release(self):
            pass
    monkeypatch.setattr(nearfar.cv2, "VideoCapture", Capture)
    for name, n in (("camera1.mp4", 198967), ("camera2.mp4", 96222), ("other.mp4", 198966)):
        (tmp_path / name).write_bytes(b"")
        frames[name] = n
    assert [nearfar.camera_20261008(str(tmp_path / f)) for f in ("camera1.mp4", "camera2.mp4", "other.mp4")] == \
        ["cam1", "cam2", None]
    assert nearfar.camera_20261008(str(tmp_path / "missing.mp4")) is None


def test_summary_names_the_calibration_actually_used():
    """trackview's summary.json record: HOGEUN's file, its frame and knocks, or a tapped calib.json after switching back."""
    from futsal import trackview
    rec = trackview.calibration_record(1, trackview.load_team_calibration(1, r"C:\videos\20261008_170919.mp4"), TEAM)
    assert rec["file"] == "calibration/out/H_cam1_refined.json" and rec["t_sec"] == 60.0 and rec["video"] == "camera1.mp4"
    assert [r["shift_px"] for r in rec["timeline"]] == [[0.0, 0.0], [-5.8, -9.2]]
    tapped = knock_timeline(Calibration(np.eye(3), 1.0), [(0.0, (41.0, 19.0)), (159.9, (0.0, 0.0))])
    rec = trackview.calibration_record(1, tapped, TEAM, os.path.join(ROOT, "calib", "cam1", "calib.json"))
    assert rec == {"file": "calib/cam1/calib.json", "timeline": [{"from_s": 0.0}, {"from_s": 159.9}]}


def test_nan_pixel_maps_to_nan():
    """A NaN foot pixel stays NaN (cv2 alone would put it on the centre spot, in court), and so does its scale."""
    cal = load_team(1, TEAM)
    uv = [[np.nan, 500.0], [960.0, np.nan], [960.0, 700.0]]
    xy = cal.to_pitch(uv)
    assert np.isnan(xy[:2]).all() and np.isfinite(xy[2]).all()
    mpp = cal.metres_per_pixel(uv)
    assert np.isnan(mpp[:2]).all() and mpp[2] == cal.metres_per_pixel([uv[2]])[0]
    assert cal.metres_per_pixel(np.zeros((0, 2))).shape == (0,)


@pytest.mark.skipif(not os.path.exists(os.path.join(CALIB, "cam1", "calib_timeline.json")),
                    reason="needs the local calib/ taps")
def test_full_match_timelines_agree_with_our_calibration(monkeypatch):
    """experiments/full_match.py now runs on HOGEUN's calibration; with its 10/8 knocks it agrees in metres with our
    tapped timeline (calib/<cam>/calib_timeline.json, only read) where both are settled: cam1 after its 2:40 knock,
    cam2 on both sides of 46:13. A wrong knock sign or a flipped axis is metres off."""
    monkeypatch.syspath_prepend(os.path.join(ROOT, "experiments"))
    full_match = importlib.import_module("full_match")
    for cam, times in (("cam1", (600.0, 3000.0)), ("cam2", (600.0, 3000.0))):
        with open(os.path.join(CALIB, cam, "calib_timeline.json"), encoding="utf-8") as f:
            ours = timeline_from_json(json.load(f))
        team = full_match.timeline(cam)
        assert all(isinstance(c, NearFarCalibration) for _, c in team)
        for t in times:
            o, m = calibration_at(ours, t), calibration_at(team, t)
            uv, xy = pitch_grid(m, step=1.0)
            err = np.hypot(*(o.to_pitch(uv) - m.to_pitch(uv)).T)
            near = ~m.beyond(xy)
            assert np.median(err[near]) < 0.5 and np.median(err) < 0.8, (cam, t, np.median(err[near]), np.median(err))
