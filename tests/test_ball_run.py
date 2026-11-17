"""trackview --ball-model: 차재혁's ball_tracking/ scripts run as they are (futsal.pipeline.ball_run)."""
import csv
import os
import sys
import textwrap

import cv2
import numpy as np
import pytest

from futsal import trackview
from futsal.pipeline import ball_run


def test_retime_uses_the_frame_ultralytics_read():
    """His t = i * stride / fps; Ultralytics' vid_stride keeps frame i * s + s - 1, whose own timestamp counts. A
    variable frame rate (here frames 0.04 s apart after frame 4) makes the index time drift; the timestamp does not."""
    pts = np.array([0.0, 1, 2, 3, 4, 4.04, 4.08, 4.12, 4.16, 4.2]) / 1.0
    fps, s = 25.0, 2
    t = np.array([0, 1, 2, 3, 4]) * s / fps                      # results 0..4
    assert np.allclose(ball_run.retime(t, fps, s, pts), pts[[1, 3, 5, 7, 9]])
    assert ball_run.stride_for(60.02) == 2 and ball_run.stride_for(30.0) == 1


def test_his_chain_runs_once_per_video_and_is_re_timed(tmp_path, monkeypatch):
    """Stand-ins for his five scripts (same arguments, same CSV columns) in a stand-in ball_tracking/: detection at
    1280 and 1920, merge, the static filter for cam2 only, the Kalman track; the track comes back on frame timestamps;
    a second call reads the cache and runs nothing."""
    bt = tmp_path / "ball_tracking"
    bt.mkdir()
    log = tmp_path / "log.txt"
    head = f"import csv, sys\nopen(r'{log}', 'a').write(__file__.split('\\\\')[-1].split('/')[-1] + ' ' + ' '.join(sys.argv[1:]) + '\\n')\n"
    (bt / "detect_video2.py").write_text(head + textwrap.dedent("""
        video, out, off, stride = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t", "x1", "y1", "x2", "y2", "conf"])
            for i in range(5):
                w.writerow([round(off + i * stride / 30.0, 4), 10, 10, 20, 20, 0.9])
    """))
    for name in ("merge_raw.py", "filter_static.py", "track_ball.py"):
        (bt / name).write_text(head + textwrap.dedent("""
            rows = list(csv.DictReader(open(sys.argv[1])))
            track = sys.argv[0].endswith("track_ball.py")
            with open(sys.argv[-1], "w", newline="") as f:
                cols = ["t", "status", "x1", "y1", "x2", "y2", "conf", "track_id"] if track else ["t", "x1", "y1", "x2", "y2", "conf"]
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
                for r in rows:
                    w.writerow({**r, "status": "detected", "track_id": 1} if track else r)
        """))
    monkeypatch.setattr(ball_run, "BALL_TRACKING", bt)
    video = str(tmp_path / "cam2.mp4")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(12):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    model = tmp_path / "best.pt"
    model.write_bytes(b"weights")
    out = ball_run.ball_track_csv(video, "cam2", str(model), str(tmp_path / "out"))
    ran = log.read_text().split("\n")
    assert [r.split()[0] for r in ran if r] == ["detect_video2.py", "detect_video2.py", "merge_raw.py",
                                                  "filter_static.py", "track_ball.py"]
    assert ran[0].split()[3:6] == ["0", "1", "1280"] and ran[1].split()[5] == "1920"
    with open(out, newline="") as f:
        t = [float(r["t"]) for r in csv.DictReader(f)]
    pts = ball_run.frame_times(video)
    assert np.allclose(t, pts[:5], atol=1e-4)                    # stride 1 at 30 fps: frames 0..4
    log.write_text("")
    assert ball_run.ball_track_csv(video, "cam2", str(model), str(tmp_path / "out")) == out
    assert log.read_text() == ""                                 # cached: nothing ran


def test_offsets_come_from_sync_json_for_known_matches(monkeypatch, tmp_path, capsys):
    """Without --offset, two videos of one calibrated match get the offset sync.json implies (the start-aligned copies
    share a clock): 10/8 originals 145.3047 - 21.0476 s, copies 0, an original with a copy the cut of the original;
    unknown videos or the wrong order stay None (0, as before). --offset given wins."""
    assert trackview.sync_offsets(["20261008_170919.mp4", "20261008_171122.mp4"]) == [0.0, 124.2571]
    assert trackview.sync_offsets(["20261008_170919_sync.mp4", "20261008_171122_sync.mp4"]) == [0.0, 0.0]
    assert trackview.sync_offsets(["20261008_170919.mp4", "20261008_171122_sync.mp4"]) == [0.0, 145.3047]
    assert trackview.sync_offsets(["a.mp4", "b.mp4"]) is None
    assert trackview.sync_offsets(["20261008_171122.mp4", "20261008_170919.mp4"]) is None
    from futsal.pipeline import reid
    seen = {}
    monkeypatch.setattr(reid, "reid_embedder", lambda: "embedder")
    monkeypatch.setattr(trackview, "run", lambda *a, **kw: seen.update(offsets=a[2]) or {
        "tracker": "v2", "ids": 0, "ids_over_half": 0, "events": 0})
    trackview.main(["--video", r"C:\v\20261008_170919.mp4", r"C:\v\20261008_171122.mp4", "--start", "600",
                    "--out", str(tmp_path)])
    assert seen["offsets"] == {"cam1": (0.0, 0.0), "cam2": (124.2571, 0.0)} and "sync.json" in capsys.readouterr().out
    trackview.main(["--video", r"C:\v\20261008_170919.mp4", r"C:\v\20261008_171122.mp4", "--start", "600",
                    "--offset", "0", "124.27", "--out", str(tmp_path)])
    assert seen["offsets"]["cam2"] == (124.27, 0.0)
