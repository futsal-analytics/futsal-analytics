"""ball_tracking/ (차재혁) results on the players' map: futsal.pipeline.ball with HOGEUN's real 10/8 calibration."""
import csv
import os

import numpy as np
import pytest

from futsal import nearfar
from futsal.homography import calibration_at
from futsal.pipeline.ball import ball_track, clock_shift, read_track
from futsal.tracks import PlayerTrack, TrackSet

TEAM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "out", "1008")
CUT = {1: 145.3047, 2: 21.0476}                     # calibration/out/1008/sync.json: the originals' time of the copies' 0 s


def _csv(path, cal, cut, rows):
    """track_ball.py's columns: boxes whose bottom centre is the pitch point xy at time t of the start-aligned copy."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "status", "x1", "y1", "x2", "y2", "conf", "track_id"])
        for t, status, xy in rows:
            if status == "missing":
                w.writerow([t, "missing", "", "", "", "", "", ""])
                continue
            u, v = calibration_at(cal, t + cut).to_image([xy])[0]
            w.writerow([t, status, u - 6, v - 12, u + 6, v, "" if status == "predicted" else 0.8, 1])
    return str(path)


def test_ball_lands_on_the_players_grid_and_pitch_frame(tmp_path):
    """CSVs made on the copies (HOGEUN's clock), players run on the originals from 125 s at 10 Hz: a detection comes
    back at its pitch point on grid frame round((t + cut - start) * rate); off-court points stay; HOGEUN's unresolved
    stretch (cam1 10-20 s of the copy) and missing rows give nothing; a detection beats a prediction and, both
    detected, the nearer camera wins (never their mean)."""
    start, rate, n = 125.0, 10.0, 31850
    tl = {c: nearfar.team_timeline(c, TEAM, CUT[c]) for c in (1, 2)}          # the players' timelines
    csvs = {"cam1": _csv(tmp_path / "c1.csv", tl[1], CUT[1], [
                (600.0, "detected", (30.0, 10.0)), (900.0, "detected", (-1.0, 5.0)), (15.0, "detected", (25.0, 10.0)),
                (1200.0, "predicted", (12.0, 6.0)), (1300.0, "missing", None)]),
            "cam2": _csv(tmp_path / "c2.csv", tl[2], CUT[2], [(600.0, "detected", (30.0, 10.0)),
                                                              (1200.0, "detected", (12.0, 6.0))])}
    videos = {"cam1": "20261008_170919.mp4", "cam2": "20261008_171122.mp4"}
    copies = {"cam1": "20261008_170919_sync.mp4", "cam2": "20261008_171122_sync.mp4"}
    offsets = {"cam1": (0.0, 0.0), "cam2": (CUT[1] - CUT[2], 0.0)}
    xy, cam, rec = ball_track(csvs, copies, videos, {"cam1": tl[1], "cam2": tl[2]}, offsets, start, rate, n, None)
    k = lambda t: int(round((t + CUT[1] - start) * rate))
    assert np.allclose(xy[k(600.0)], (30.0, 10.0), atol=0.02) and cam[k(600.0)] == 2     # x = 30: cam2's near half
    assert np.allclose(xy[k(900.0)], (-1.0, 5.0), atol=0.02) and cam[k(900.0)] == 1      # behind the goal line: kept
    assert np.allclose(xy[k(1200.0)], (12.0, 6.0), atol=0.02) and cam[k(1200.0)] == 2    # cam2 detected, cam1 predicted
    assert np.isfinite(xy).all(1).sum() == 3 and (cam > 0).sum() == 3                    # 15 s: HOGEUN found no court
    assert rec["cameras"]["cam1"]["clock_shift_s"] == pytest.approx(CUT[1])
    assert rec["both_detected_median_m"] < 0.05
    with pytest.raises(ValueError):                    # cam2's copy given for cam1: the ball would move by the cut
        clock_shift(copies["cam2"], videos["cam1"])
    (tmp_path / "ball_cam1.csv").write_text("t,status,track_id,u,v,conf,x_m,y_m\n600.0,detected,1,900,500,0.8,10,5\n")
    with pytest.raises(ValueError):                    # to_court.py's output instead of track_ball.py's: refused
        read_track(str(tmp_path / "ball_cam1.csv"))
    ts = TrackSet(40, 20, rate, [PlayerTrack(1, "A", np.zeros((n, 2)))], ball=xy, ball_cam=cam)
    back = TrackSet.from_json(ts.to_json())
    assert np.allclose(back.ball, xy, atol=0.006, equal_nan=True) and (back.ball_cam == cam).all()   # cm in the file
    assert "ball" not in TrackSet(40, 20, rate, []).to_json()                          # no ball: format unchanged
