"""HOGEUN's calibration (calibration/out/<match>, calibration/court_map.py) as the pipeline's calibration: futsal.nearfar.

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
from futsal.nearfar import NearFarCalibration, knock_timeline, load_team, team_file, team_timeline, timeline_json
from futsal.pipeline import identity
from futsal.pipeline.boxes import build_cam_boxes
from futsal.pipeline.detect import Detection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_DIR = os.path.join(ROOT, "calibration")
TEAM = os.path.join(TEAM_DIR, "out", "1008")
TEAM_1103 = os.path.join(TEAM_DIR, "out", "1103")
CUT_1008 = {1: 145.3047, 2: 21.0476}            # calibration/out/1008/sync.json: the originals' time of the copies' 0 s
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
    assert not {"court_map", "calib_core", "calibrate"} & set(sys.modules)
    assert sys.path == path and sys.dont_write_bytecode == bytecode
    assert listing(TEAM_DIR) == before and not os.path.exists(os.path.join(TEAM_DIR, "__pycache__"))
    # the functions still work through their own globals (HOGEUN's calib_core.HALF_SIGN, court constants)
    assert mod.HALF_SIGN == {"east": 1, "west": -1}
    xy = mod.pixel_to_court(mod.load_anchor(1, TEAM), [[960.0, 700.0]])
    assert xy.shape == (1, 2) and np.all(np.isfinite(xy))
    xy = mod.pixel_to_court(mod.load(1, TEAM), [[960.0, 700.0]], t=600.0)      # the whole-match file needs a time
    assert xy.shape == (1, 2) and np.all(np.isfinite(xy))


def test_load_team_reads_what_court_map_load_anchor_reads():
    cm = nearfar.load_court_map()
    for n in (1, 2):
        assert team_file(n, TEAM) == os.path.join(TEAM, f"H_cam{n}_refined.json")     # refined first, as court_map
        cal = load_team(n, TEAM)
        assert cal.team == cm.load_anchor(n, TEAM) == team_json(n, "_refined")
        assert cal.team["half"] == ("east" if n == 1 else "west") and cal.sign == (1 if n == 1 else -1)
    with pytest.raises(ValueError, match="team_timeline"):           # the whole-match file is no single calibration
        NearFarCalibration.from_team(cm.load(1, TEAM))


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
    """In our coordinates HOGEUN's clicks land on our keypoints of the same marks: the pure translation is right (a
    flipped axis or a wrong offset is metres off). His 11/11 files fit the 5 near clicks by least squares with the
    sideline point free to slide (calib_core.solve_half): rms 3.3 / 3.8 px, the centre-circle crossing up to 7.3 px
    (0.27 m) off; the refined file fits the white lines with the clicks as soft anchors (rms 3.5 / 4.2 px). Both:
    within 0.5 m on the court and 10 px in the image."""
    d = team_json(n, kind)
    cal = NearFarCalibration.from_team(d)
    kp = COURT.keypoints()
    clicks = np.asarray(d["img_pts"][:len(CLICKED[d["half"]])], float)
    xy = cal.to_pitch(clicks)
    for i, mark in enumerate(CLICKED[d["half"]]):
        if not isinstance(mark, str):
            assert abs(xy[i, 1] - mark) < 0.1
        else:
            assert np.hypot(*(xy[i] - kp[mark])) < 0.5, mark
            assert np.hypot(*(cal.to_image([kp[mark]])[0] - clicks[i])) < 10.0, mark
    assert cal.rms_px < 6.0


def test_reads_files_from_hogeuns_newer_click_tool():
    """His tool since hoya3 bacfd0e adds two optional check clicks and writes None for the ones skipped (also after a
    plain 7 clicks + Enter): the same calibration, through from_team and the JSON round trip."""
    d = team_json(1, "_refined")
    new = {**d, "img_pts": d["img_pts"] + [None, None], "names": d["names"] + ["CHECK arc", "CHECK sideline"]}
    a, b = NearFarCalibration.from_team(d), NearFarCalibration.from_team(new)
    assert np.allclose(a.H, b.H) and a.rms_px == pytest.approx(b.rms_px)
    assert np.allclose(NearFarCalibration.from_json(b.to_json()).H_far, a.H_far)


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


def sources(tl):
    return [(t, c.team["source"], c.resolved) for t, c in tl]


def test_trackview_takes_hogeuns_whole_match_file_for_his_videos(tmp_path, capsys):
    """trackview.load_team_calibration: a video of a calibrated match (original or start-aligned copy, 10/8 or 11/3)
    gets HOGEUN's whole-match file on its own clock, without knocks (he followed the camera). A folder with only a
    single-frame file keeps the knocks relative to its frame (t_sec): other knocks when it was clicked on another
    frame. An unknown video gets the single-frame 10/8 calibration as it is, with notices."""
    from futsal import trackview
    video = r"C:\videos\20261008_170919.mp4"
    tl = trackview.load_team_calibration(1, video)
    assert sources(tl) == sources(team_timeline(1, TEAM, CUT_1008[1]))
    assert sources(tl)[:2] == [(0.0, "before", False), (CUT_1008[1], "tracked", True)]
    copy = trackview.load_team_calibration(2, r"D:\footsal\synced_20261008\20261008_171122_sync.mp4")
    assert sources(copy) == sources(team_timeline(2, TEAM, 0.0)) and copy[0][0] == 0.0 and copy[0][1].resolved
    t1103 = trackview.load_team_calibration(1, r"D:\footsal\synced\20261103_165422_sync.mp4")
    assert sources(t1103) == sources(team_timeline(1, TEAM_1103, 0.0))
    assert t1103[0][1].team["video"].endswith("20261103_165422_sync-001.mp4")
    d = team_json(1, "_refined")
    copy = d["video"]                       # he clicked the start-aligned copy: t_sec + the cut on the original
    after = [(0.0, (5.8, 9.2)), (159.9, (0.0, 0.0))]                 # clicked after the cam1 knock at 2:40
    for clicked, at, knocks in ((copy, 60.0, after), ("camera1.mp4", 60.0, nearfar.KNOCKS_20261008["cam1"]),
                                ("camera1.mp4", 600.0, after)):     # camera1.mp4: his pre-11/11 file, original clock
        d.update(video=clicked, t_sec=at)
        (tmp_path / "H_cam1_refined.json").write_text(json.dumps(d), encoding="utf-8")
        tl = trackview.load_team_calibration(1, video, out_dir=str(tmp_path))
        assert [(t, c.shift) for t, c in tl] == knocks
    for clicked, at in (("camera1.mp4", 165.0), (copy, 20.0)):      # 20 s of the copy = 165.3 s of the original
        d.update(video=clicked, t_sec=at)
        (tmp_path / "H_cam1_refined.json").write_text(json.dumps(d), encoding="utf-8")
        capsys.readouterr()
        trackview.load_team_calibration(1, video, out_dir=str(tmp_path))
        assert "주의" in capsys.readouterr().out                  # clicked while the camera was shaking
    cal = trackview.load_team_calibration(1, r"D:\x\match2_cam1.mp4")       # no calibrated match: say so
    out = capsys.readouterr().out
    assert isinstance(cal, NearFarCalibration) and "그대로였다고" in out and "20261008_170919_sync.mp4 영상에서 찍은 것" in out


def test_trackview_refuses_videos_in_the_wrong_order_or_with_missing_offsets(tmp_path):
    """cam2's 10/8 video in cam1's place would get cam1's calibration with cam2's knocks; an --offset or --drift
    list shorter than --video would leave a camera at 0 s without a word. Both stop before anything runs."""
    from futsal import trackview
    with pytest.raises(ValueError, match="cam2"):
        trackview.load_team_calibration(1, r"C:\videos\20261008_171122.mp4")
    with pytest.raises(ValueError, match="1103 경기 cam1"):
        trackview.load_team_calibration(2, r"D:\footsal\synced\20261103_165422_sync.mp4")
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
    assert rec["file"] == "calibration/out/1008/calib_cam1.json" and rec["t_sec"] == 60.0
    assert rec["video"] == "20261008_170919_sync.mp4"                 # HOGEUN's clock; ours is the original's
    assert [(r["from_s"], r["source"], r["resolved"]) for r in rec["timeline"][:2]] == \
        [(0.0, "before", False), (round(CUT_1008[1], 3), "tracked", True)]
    rec = trackview.calibration_record(1, load_team(1, TEAM), TEAM)
    assert rec == {"file": "calibration/out/1008/H_cam1_refined.json", "video": "20261008_170919_sync.mp4", "t_sec": 60.0,
                   "timeline": [{"from_s": 0.0, "shift_px": [0.0, 0.0]}]}
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


# ---------------------------------------------------------------------------------------------------------------
# HOGEUN's whole-match calibration (calibration/out/<match>/calib_cam<N>.json)
# ---------------------------------------------------------------------------------------------------------------

def test_find_video_by_name_and_frame_count(tmp_path, monkeypatch):
    """Originals get the cut as clock shift, start-aligned copies (ours: <name>_sync.mp4, HOGEUN's: <name>_sync-001.mp4)
    none; a renamed copy is found by its frame count (the original's or the copy's); anything else is no match."""
    def f(video):
        found = nearfar.find_video(video)
        return found and (os.path.basename(found[0]), found[1], found[2])
    assert f(r"C:\videos\20261008_170919.mp4") == ("1008", 1, CUT_1008[1])
    assert f("/mnt/d/x/20261008_171122.mp4") == ("1008", 2, CUT_1008[2])
    assert f(r"D:\footsal\synced_20261008\20261008_170919_sync.mp4") == ("1008", 1, 0.0)
    assert f("20261103_165422_sync-001.mp4") == ("1103", 1, 0.0)
    assert f(r"D:\footsal\20261103_165851.mp4") == ("1103", 2, 33.0288)
    assert f("a.mp4") is None and f("20261008_170918.mp4") is None
    frames = {}

    class Capture:
        def __init__(self, path):
            self.n = frames[os.path.basename(path)]

        def get(self, prop):
            return float(self.n) if prop == nearfar.cv2.CAP_PROP_FRAME_COUNT else 0.0

        def release(self):
            pass
    monkeypatch.setattr(nearfar.cv2, "VideoCapture", Capture)
    for name, n in (("camera1.mp4", 198967), ("camera2_cut.mp4", 95592), ("other.mp4", 198966)):
        (tmp_path / name).write_bytes(b"")
        frames[name] = n
    assert f(str(tmp_path / "camera1.mp4")) == ("1008", 1, CUT_1008[1])
    assert f(str(tmp_path / "camera2_cut.mp4")) == ("1008", 2, 0.0)
    assert f(str(tmp_path / "other.mp4")) is None


@pytest.mark.parametrize("match,n,shift", [("1008", 1, CUT_1008[1]), ("1008", 2, 0.0), ("1103", 1, 0.0),
                                           ("1103", 2, 33.0288)])
def test_team_timeline_is_court_map_on_the_video_clock(match, n, shift):
    """At any time on the video's clock the timeline maps pixels as HOGEUN's court_map.pixel_to_court does at that
    time minus the cut, plus (20, 10): NaN at the same times (before his first stretch, unresolved, after the last),
    the stretch's own homographies, and his drift held per sample gap (exact in the middle of each gap, under 0.8 px
    off anywhere)."""
    out_dir = os.path.join(TEAM_DIR, "out", match)
    cm = nearfar.load_court_map()
    cal = cm.load(n, out_dir)
    tl = team_timeline(n, out_dir, shift)
    assert tl[0][0] == 0.0 and [t for t, _ in tl] == sorted(t for t, _ in tl)
    rng = np.random.default_rng(n)
    end = max(s["t1"] for s in cal["segments"])
    px = rng.uniform([0, 300], [1920, 1080], (30, 2))
    for t in np.concatenate([rng.uniform(0, end + shift + 20, 400), [s["t0"] + shift for s in cal["segments"]]]):
        ours = calibration_at(tl, float(t)).to_pitch(px)
        his = cm.pixel_to_court(cal, px, t=float(t) - shift).astype(float) + COURT.centre
        assert np.array_equal(np.isnan(ours), np.isnan(his)), t
        ok = np.isfinite(his).all(1)
        if ok.any():
            mpp = calibration_at(tl, float(t)).metres_per_pixel(px[ok])
            assert np.all(np.hypot(*(ours[ok] - his[ok]).T) <= 0.8 * mpp + 1e-6), t
    for seg in cal["segments"]:
        d = [row[0] for row in seg.get("drift") or [] if seg["t0"] < row[0] < seg["t1"]]
        for a, b in zip(d, d[1:]):
            mid = (a + b) / 2
            assert np.allclose(calibration_at(tl, mid + shift).to_pitch(px),
                               cm.pixel_to_court(cal, px, t=mid).astype(float) + COURT.centre, rtol=1e-9, atol=1e-6)


def test_team_timeline_boxes_get_no_position_where_hogeun_found_no_court():
    """Boxes before his first stretch (the original's first 145.3 s) and in an unresolved one (10-20 s on the copy's
    clock) keep their pixels but get NaN positions and sigma and are out of court; the timeline still draws there
    (to_image) and survives the JSON round trip full_match.py makes."""
    tl = team_timeline(1, TEAM, CUT_1008[1])
    uv, _ = pitch_grid(calibration_at(tl, 600.0), step=8.0)
    dets = [Detection(int(t * 10), t, float(u), float(v), 0.9, None, 80.0, None) for u, v in uv
            for t in (130.0, 160.0, 600.0)]                       # one source frame per instant
    c = build_cam_boxes("cam1", dets, tl, 0.0, 0.0, 0.0, 10.0, COURT)
    dark = np.array([d.t for d in dets]) < 600.0
    assert np.isnan(c.xy[dark]).all() and np.isnan(c.sigma[dark]).all() and not c.in_court[dark].any()
    assert np.isfinite(c.xy[~dark]).all() and c.in_court[~dark].all()
    assert (c.k[dark] == -1).all() and (c.k[~dark] >= 0).all()          # off the grid, as if not there
    assert sorted(np.concatenate(list(c.by_k.values()))) == list(np.flatnonzero(~dark))
    assert np.isfinite(calibration_at(tl, 160.0).to_image([[20.0, 10.0]])).all()
    back = timeline_from_json(json.loads(json.dumps(timeline_json(tl))))
    assert sources(back) == sources(tl)
    for t in (130.0, 160.0, 600.0, 3000.0):
        a, b = calibration_at(tl, t), calibration_at(back, t)
        assert np.array_equal(a.to_pitch(uv), b.to_pitch(uv), equal_nan=True) and (b.rms_px, b.shift) == (a.rms_px, a.shift)


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
