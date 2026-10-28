"""Whole-match tracking from the full cache (experiments/cache_full, 10/8 test, ~53 min of overlap) with the src v2 tracker.

python experiments/full_match.py [--render] [--cache experiments/cache_full26] [--out experiments/full_match_team] [--linker closed|gta]
Writes <out>/tracks.json (TrackSet, web/viewer format), events.csv, summary.json (per-player distance,
speed, time seen), calib_timeline_<cam>.json and, with --render, trackview.mp4 (two camera panels + 2D map).
Coordinates: HOGEUN's calibration (calibration/out through calibration/court_map.py, futsal.nearfar), with the camera
knocks of the 10/8 videos taken relative to the frame it was clicked on (t_sec, 60 s: nearfar.knocks_20261008; cam1
knocked at 2:40 after that frame, cam2 at 46:13).
Detections: the cache's own boxes (experiments/build_cache.py: our YOLO26x predict, conf 0.1, 10 Hz, OSNet ReID), not
detection/; python -m futsal.trackview runs detection/ on the videos. Nothing is written into calib/ or calibration/.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from futsal.court import Court
from futsal.homography import timeline_from_json
from futsal.nearfar import NearFarCalibration, knock_timeline, knocks_20261008, load_team, save_timeline, team_file
from futsal.pipeline.align import corrected_sync
from futsal.pipeline.boxes import cam_boxes_from_cache
from futsal.pipeline.detect import Detection
from futsal.pipeline.run import tracks_to_trackset, track_cams
from futsal.trackview import _fmt, find_events, render

CACHE = os.path.join(ROOT, "experiments", "cache_full26")     # YOLO26x (cache_full: the older YOLO11x run)
OUT = os.path.join(ROOT, "experiments", "full_match_team")    # experiments/full_match*: earlier runs with our calibration
# Knocks for our tapped calibration (600 s frame), measured 1 s before / 1.5 s after each knock (experiments/shift_sign.py).
# 1.5 s after the cam1 knock the camera is still shaking: the settled pose before 2:40 is about (+5.8, +9.2) px from the
# 600 s frame (nearfar.POSES_20261008). Kept as it was, so this path reproduces experiments/full_match26.
KNOCKS = {"cam1": [(0.0, (41.0, 19.0)), (159.9, (0.0, 0.0))], "cam2": [(0.0, (0.0, 0.0)), (2773.0, (13.8, 1.3))]}


def timeline(cam: str) -> list:
    """[(from_s, calibration)] of one camera on its own clock: HOGEUN's calibration with the 10/8 knocks."""
    # To switch back to our tapped calibration (calib/<cam>/calib.json): uncomment these two, drop the last two.
    # base = timeline_from_json(json.load(open(os.path.join(ROOT, "calib", cam, "calib.json"), encoding="utf-8")))
    # return knock_timeline(base, KNOCKS[cam])
    base = load_team(int(cam.removeprefix("cam")))
    return knock_timeline(base, knocks_20261008(cam, float(base.team["t_sec"])))


def calib_record(cam: str, tl: list) -> dict:
    """summary.json's note of which calibration a camera used, and its knock shifts."""
    if isinstance(tl[0][1], NearFarCalibration):
        return {"file": os.path.relpath(team_file(int(cam.removeprefix("cam"))), ROOT).replace(os.sep, "/"),
                "knocks": [(t, c.shift) for t, c in tl]}
    return {"file": f"calib/{cam}/calib.json", "knocks": KNOCKS[cam]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--linker", default="closed", choices=("closed", "gta"))
    a = ap.parse_args()
    CACHE, OUT = a.cache, a.out
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    court, rate, start = Court(), float(meta["rate"]), float(meta["start"])
    t0 = time.time()
    cams, cals = {}, {}
    for cam in meta["videos"]:
        path = os.path.join(OUT, f"calib_timeline_{cam}.json")     # was calib/<cam>/calib_timeline.json: <out> only now
        save_timeline(timeline(cam), path)
        with open(path, encoding="utf-8") as f:              # read back: the run uses what the file holds
            cals[cam] = timeline_from_json(json.load(f))
        cams[cam] = cam_boxes_from_cache(os.path.join(CACHE, f"{cam}.npz"), cam, cals[cam], float(meta["offsets"][cam]),
                                         start, rate, court)
        print(f"{cam}: {len(cams[cam])} boxes ({time.time() - t0:.0f}s)", flush=True)
    info = {}
    ids, slot_teams = track_cams(cams, court, rate, info=info, linker=a.linker)      # also corrects cam2's clock
    clock = info["clock"]
    if clock is not None:
        print(f"clock: {clock['camera']} offset {clock['offset_s']:+.3f} s, drift {clock['drift'] * 1e6:+.0f} ppm "
              f"(applied {clock['applied']}); shifts {clock['shift_s']}", flush=True)
    st = info["solver"]
    print(f"tracked: {info['tracklets']} tracklets -> {len(ids)} identities; alignment {info['alignment'] and {k: info['alignment'][k] for k in ('pairs', 'median_before_m', 'median_after_m')}}; "
          f"teams {info['teams'].get('mode')}; windows {getattr(st, 'windows', '?')} ({time.time() - t0:.0f}s)", flush=True)
    n = 1 + max(max(c.by_k) for c in cams.values())
    ts = tracks_to_trackset(ids, cams, court, rate, n, slot_teams or None)
    ts.start = start
    ts.compute_stats()
    ts.dump(os.path.join(OUT, "tracks.json"))
    events = find_events(ts, court)
    with open(os.path.join(OUT, "events.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time(cam1)", "ID", "kind", "x(m)", "y(m)"])
        for e in events:
            w.writerow([_fmt(start + e["t"]), e["id"], e["kind"], e["x"], e["y"]])
    seen = {p.id: float(np.isfinite(p.xy[:, 0]).mean()) for p in ts.players}
    summary = {"detector": meta.get("model"), "detections": f"{os.path.basename(CACHE)} (experiments/build_cache.py, conf {meta.get('conf')})",
               "calib": {cam: calib_record(cam, cals[cam]) for cam in cams},
               "linker": a.linker, "clock": clock, "window_cam1_s": [start, start + n / rate], "identities": len(ts.players), "events": len(events),
               "events_per_min": round(len(events) / (n / rate / 60), 2), "tracklets": info["tracklets"],
               "alignment": info["alignment"] and {k: info["alignment"][k] for k in ("pairs", "median_before_m", "median_after_m")},
               "teams": {k: v for k, v in info["teams"].items() if k in ("mode", "pairs", "pair_agreement", "counts")},
               "windowed": {k: getattr(st, k) for k in ("windows", "new_ids", "ended", "agreement") if hasattr(st, k)},
               "players": [{"id": p.id, "team": p.team, "seen_share": round(seen[p.id], 3), **p.stats} for p in ts.players]}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    long_ids = sum(1 for v in seen.values() if v > 0.5)
    print(f"{len(ts.players)} identities ({long_ids} seen > half the match), {len(events)} mid-pitch events "
          f"({summary['events_per_min']}/min) -> {OUT} ({time.time() - t0:.0f}s)", flush=True)
    if a.render:
        dets = {}
        for cam, c in cams.items():
            m = c.conf >= 0.3
            dets[cam] = [Detection(int(c.fid[i]), float(c.t[i]), float(c.foot[i, 0]), float(c.foot[i, 1]), float(c.conf[i]), None,
                                   float(c.h[i]), box=tuple(float(v) for v in c.xyxy[i])) for i in np.flatnonzero(m)]
        offsets = {cam: (float(meta["offsets"][cam]), 0.0) for cam in cams}
        if clock is not None and clock["applied"]:
            offsets[clock["camera"]] = corrected_sync(*offsets[clock["camera"]], clock, start)
        frames = render(os.path.join(OUT, "trackview.mp4"), meta["videos"], cals, offsets, ts, dets, court, start, events)
        print(f"rendered {frames} frames ({time.time() - t0:.0f}s)", flush=True)
