"""Ground truth for scoring the tracker: PTS-matched clips, SAM 3 pre-labels, our boxes, TrackEval scores (cam1 only).

python experiments/sam3_gt.py clip  --name cam1_0900 --start 540 --duration 120     # cam1 clock, seconds
python experiments/sam3_gt.py sam3  --name cam1_0900 --model sam3.pt [--seconds 30] [--imgsz 644]   # facebook/sam3 weights
python experiments/sam3_gt.py ours  --name cam1_0900 cam1_2900                       # re-runs the 53-min v2 (~6 min CPU)
python experiments/sam3_gt.py score --name cam1_0900 --gt labeling/cam1_0900/cvat/gt/gt.txt [--drop-nonplayer]
python experiments/sam3_gt.py dedup --in export_mot.zip --out labeling/cam1_0900/dedup_mot.zip     # re-import into CVAT

Everything goes to labeling/<name>/ (git-ignored: video frames of people). Clip frame n (1-based in the MOT files) is
grid frame k = k0 + n - 1 of the full-match run (experiments/cache_full26, start 125 s, 10 Hz). Frames are picked by
the cache's own PTS (frame_t), so the ground truth and our boxes come from the same source frames; an ffmpeg fps=10
cut would be off by up to 50 ms (60 fps sampled every 6th frame runs 0.0444 % ahead of the 0.1 s grid).
MOT rows: frame,id,x,y,w,h,conf,class,visibility. Classes follow the CVAT labels order: 1 player, 2 nonplayer
(TrackEval's MOT17 'person_on_vehicle', i.e. a distractor: matched tracker boxes are neither FP nor TP).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from futsal.court import Court
from futsal.homography import calibration_at, timeline_from_json
from futsal.nearfar import save_timeline
from futsal.pipeline.boxes import COURT_MARGIN_M, cam_boxes_from_cache, grid_frames

CACHE = os.path.join(ROOT, "experiments", "cache_full26")
LAB = os.path.join(ROOT, "labeling")
CAM = "cam1"        # shortcut: cam1 only; cam2 needs the run's clock correction for its k (align.refine_clock)


def meta() -> dict:
    with open(os.path.join(CACHE, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def calib_cam1(out_dir: str):
    """cam1's calibration timeline exactly as full_match.py builds and reads it back (HOGEUN + 10/8 knocks)."""
    from full_match import timeline
    path = os.path.join(out_dir, f"calib_timeline_{CAM}.json")
    save_timeline(timeline(CAM), path)
    with open(path, encoding="utf-8") as f:
        return timeline_from_json(json.load(f))


def frames_of(name: str) -> list[dict]:
    with open(os.path.join(LAB, name, "frames.csv"), encoding="utf-8") as f:
        return [{"n": int(r["n"]), "k": int(r["k"]), "fid": int(r["fid"]), "pts": float(r["pts"])} for r in csv.DictReader(f)]


def cmd_clip(a):
    import cv2
    m = meta()
    z = np.load(os.path.join(CACHE, f"{CAM}.npz"))
    start, rate = float(m["start"]), float(m["rate"])
    k_box = grid_frames(z["fid"], z["t"], float(m["offsets"][CAM]), 0.0, start, rate)     # the run's own grid rule
    k_of_fid = {int(f): int(k) for f, k in zip(z["fid"], k_box) if k >= 0}
    k0 = int(round((a.start - start) * rate))
    ks = range(k0, k0 + int(round(a.duration * rate)))
    fid_of_k = {k: f for f, k in k_of_fid.items() if k in ks}
    missing = [k for k in ks if k not in fid_of_k]
    assert not missing, f"no cache sample with boxes for grid frames {missing[:10]} ({len(missing)})"
    want = [(k, fid_of_k[k], float(z["frame_t"][fid_of_k[k]])) for k in ks]
    out = os.path.join(LAB, a.name)
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(m["videos"][CAM])
    cap.set(cv2.CAP_PROP_POS_MSEC, (want[0][2] - 3.0) * 1000)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", str(rate), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                            "-pix_fmt", "yuv420p", os.path.join(out, "clip.mp4")], stdin=subprocess.PIPE)
    j, t0 = 0, time.time()
    while j < len(want) and cap.grab():
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        assert pts < want[j][2] + 0.0015, f"grid frame {want[j][0]}: no source frame at PTS {want[j][2]:.4f} (next {pts:.4f})"
        if abs(pts - want[j][2]) <= 0.0015:
            ok, img = cap.retrieve()
            assert ok and img.shape[:2] == (h, w)
            enc.stdin.write(img.tobytes())
            j += 1
    enc.stdin.close()
    assert enc.wait() == 0 and j == len(want), f"wrote {j}/{len(want)} frames"
    with open(os.path.join(out, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["n", "k", "fid", "pts"])
        wr.writerows([(i + 1, k, fid, f"{pts:.6f}") for i, (k, fid, pts) in enumerate(want)])
    print(f"{a.name}: {len(want)} frames k {k0}..{k0 + len(want) - 1} -> {out} ({time.time() - t0:.0f}s)")


def cmd_sam3(a):
    import torch
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    out = os.path.join(LAB, a.name)
    rows, court = frames_of(a.name), Court()
    cal = calib_cam1(out)
    n_max = int(a.seconds * 10) if a.seconds else len(rows)
    pred = SAM3VideoSemanticPredictor(
        overrides=dict(model=a.model, quantize=16, imgsz=a.imgsz, save=False, verbose=False),
        max_num_objects=40,
        # a track gains +1 per matched frame up to 30 (keeps its id through ~3 s hidden); a one-off false
        # detection starts at 10 and is gone within 1 s (unconfirmed tracks are not filtered from the output)
        init_trk_keep_alive=10, max_trk_keep_alive=30)
    lines, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for i, r in enumerate(pred(source=os.path.join(out, "clip.mp4"), text=["person"], stream=True)):
        if i == 0:
            t0 = time.time()                             # the model is built on the first next(), not in the constructor
        if i >= n_max:
            break
        b = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 7))
        if len(b):
            foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
            on = court.contains(calibration_at(cal, rows[i]["pts"]).to_pitch(foot), COURT_MARGIN_M)
            for (x1, y1, x2, y2, tid, _, _), p in zip(b, on):
                # a CVAT track has one label for its whole life: on-court and off-court stretches get separate ids
                lines.append(f"{i + 1},{2 * int(tid) + (1 if p else 2)},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,{1 if p else 2},1.0")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_max} frames, {i / (time.time() - t0 + 1e-9):.2f} fps, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    gt = os.path.join(out, "sam3", "gt")
    os.makedirs(gt, exist_ok=True)
    with open(os.path.join(gt, "gt.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(gt, "labels.txt"), "w") as f:
        f.write("player\nnonplayer\n")
    with zipfile.ZipFile(os.path.join(out, "sam3_mot.zip"), "w") as zf:     # CVAT: Upload annotations -> MOT 1.1
        zf.write(os.path.join(gt, "gt.txt"), "gt/gt.txt")
        zf.write(os.path.join(gt, "labels.txt"), "gt/labels.txt")
    ids = {ln.split(",")[1] for ln in lines}
    print(f"{a.name}: {min(i + 1, n_max)} frames, {len(lines)} boxes, {len(ids)} ids, {time.time() - t0:.0f}s, "
          f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB -> {out}/sam3_mot.zip")


def cmd_ours(a):
    """Re-run full_match.py's v2 (same cache, calibration, defaults) and dump cam1 boxes with the final ids."""
    from futsal.pipeline.run import tracks_to_trackset, track_cams
    from futsal.pipeline.identity import team_vote
    from futsal.tracks import TrackSet
    from full_match import timeline
    m = meta()
    court, rate, start = Court(), float(m["rate"]), float(m["start"])
    run_dir = os.path.join(LAB, "_ours_run")
    os.makedirs(run_dir, exist_ok=True)
    cams, t0 = {}, time.time()
    for cam in m["videos"]:
        path = os.path.join(run_dir, f"calib_timeline_{cam}.json")
        save_timeline(timeline(cam), path)
        with open(path, encoding="utf-8") as f:
            cal = timeline_from_json(json.load(f))
        cams[cam] = cam_boxes_from_cache(os.path.join(CACHE, f"{cam}.npz"), cam, cal, float(m["offsets"][cam]), start, rate, court)
    ids, slot_teams = track_cams(cams, court, rate, info={}, linker="closed")
    n = 1 + max(max(c.by_k) for c in cams.values())
    ts = tracks_to_trackset(ids, cams, court, rate, n, slot_teams or None)
    ref = TrackSet.load(os.path.join(ROOT, "experiments", "full_match_team", "tracks.json"))
    same = len(ref.players) == len(ts.players) and all(
        np.allclose(p.xy, q.xy, atol=0.02, equal_nan=True) for p, q in zip(ref.players, ts.players))
    print(f"re-run: {len(ts.players)} identities ({time.time() - t0:.0f}s); same as full_match_team/tracks.json: {same}")
    # tracks_to_trackset's numbering (team A first, then first appearance), checked against its output below
    rows = []
    for q, tr in enumerate(ids):
        ks = [k for k in tr if 0 <= k < n]
        if ks:
            team = team_vote(tr, cams)[0]
            if team not in ("A", "B"):
                team = slot_teams[q] if slot_teams and slot_teams[q] in ("A", "B") else "?"
            rows.append((team, min(ks), q))
    rows.sort(key=lambda r: ({"A": 0, "B": 1}.get(r[0], 2), r[1]))
    pid = {q: i + 1 for i, (_, _, q) in enumerate(rows)}
    for q, i in pid.items():
        k = min(k for k in ids[q] if 0 <= k < n)
        assert np.allclose(ts.players[i - 1].xy[k], ids[q][k].xy), f"id numbering differs for track {q}"
    c = cams[CAM]
    for name in a.name:
        fr = {r["k"]: r["n"] for r in frames_of(name)}
        out, seen = [], set()
        for q, tr in enumerate(ids):
            if q not in pid:
                continue
            for k, p in tr.items():
                if k not in fr:
                    continue
                for cam, i in p.boxes:
                    if cam == CAM:
                        assert (fr[k], pid[q]) not in seen, f"two cam1 boxes for id {pid[q]} at frame {fr[k]}"
                        seen.add((fr[k], pid[q]))
                        x1, y1, x2, y2 = c.xyxy[i]
                        out.append((fr[k], pid[q], f"{fr[k]},{pid[q]},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{c.conf[i]:.3f},-1,-1,-1"))
        with open(os.path.join(LAB, name, "ours.txt"), "w") as f:
            f.write("\n".join(r for *_, r in sorted(out)) + "\n")
        print(f"{name}: {len(out)} cam1 boxes, {len({r[1] for r in out})} ids -> labeling/{name}/ours.txt")


def dedup_rows(r: np.ndarray, hi: float = 0.6, lo: float = 0.3) -> tuple[np.ndarray, list[str]]:
    """Remove SAM 3's duplicate tracks from MOT rows, before or between hand fixes.
    A short track that lies inside a longer one (>= 90 % of its frames shared, its box mostly the same) is dropped: losing
    it costs little. Two tracks of one label whose boxes coincide nearly every time both exist (a hand-over, or on-off
    doubles) are merged, and the longer track's box wins on shared frames. Players who cross pass the merge test only if
    they also share almost no other frames, so every merge is listed for a look by eye."""
    ids, cnt = np.unique(r[:, 1].astype(int), return_counts=True)
    length = dict(zip(ids.tolist(), cnt.tolist()))
    label = {int(x[1]): int(x[7]) for x in r}
    stats, seen = {}, {}
    for f in np.unique(r[:, 0]):
        a = r[r[:, 0] == f]
        x1, y1, x2, y2 = a[:, 2], a[:, 3], a[:, 2] + a[:, 4], a[:, 3] + a[:, 5]
        w = np.maximum(0, np.minimum.outer(x2, x2) - np.maximum.outer(x1, x1))
        h = np.maximum(0, np.minimum.outer(y2, y2) - np.maximum.outer(y1, y1))
        area = a[:, 4] * a[:, 5]
        iou = w * h / np.maximum(area[:, None] + area[None] - w * h, 1e-9)
        for i, j in zip(*np.triu_indices(len(a), 1)):
            k = tuple(sorted((int(a[i, 1]), int(a[j, 1]))))
            st = stats.setdefault(k, [0, 0, 0])
            st[0] += 1; st[1] += iou[i, j] >= hi; st[2] += iou[i, j] >= lo
            seen.setdefault(k, []).append(int(f))
    drop, parent, report = set(), {}, []
    find = lambda x: x if parent.get(x, x) == x else find(parent[x])
    for (a, b), (both, n_hi, n_lo) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        short, long_ = (a, b) if length[a] < length[b] else (b, a)
        if short in drop or long_ in drop or both < 3:
            continue
        inside = both / length[short] >= 0.9
        same = n_hi / both >= 0.6 or (label[short] == 2 and n_lo / both >= 0.85)
        if inside and same and not (label[short] == 1 and label[long_] == 2):
            drop.add(short)
            report.append(f"drop  id {short} ({length[short]} frames) inside id {long_}, frames {seen[(a, b)][0]}-{seen[(a, b)][-1]}")
        elif label[a] == label[b] and both >= 5 and n_hi / both >= 0.75:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
                report.append(f"merge id {a} + id {b} ({both} shared frames), frames {seen[(a, b)][0]}-{seen[(a, b)][-1]}")
    order = np.argsort([-length[int(i)] for i in r[:, 1]], kind="stable")      # the longer track's box comes first
    out, used = [], set()
    for x in r[order]:
        tid = int(x[1])
        if tid in drop:
            continue
        key = (int(x[0]), find(tid))
        if key not in used:
            used.add(key)
            y = x.copy(); y[1] = key[1]
            out.append(y)
    out = np.array(out)
    out = out[np.lexsort((out[:, 1], out[:, 0]))]
    assert len({(int(x[0]), int(x[1])) for x in out}) == len(out), "two boxes for one id in a frame"
    return out, report


def cmd_dedup(a):
    with zipfile.ZipFile(a.input) as z:
        names = z.namelist()
        gt = next(n for n in names if n.endswith("gt.txt"))
        lab = next(n for n in names if n.endswith("labels.txt"))
        r = np.loadtxt(z.open(gt), delimiter=",", ndmin=2)
        labels = z.read(lab).decode()
    out, report = dedup_rows(r)
    print(*report, sep=chr(10))
    print(f"tracks {len(set(r[:, 1]))} -> {len(set(out[:, 1]))}, rows {len(r)} -> {len(out)}; "
          f"dropped {sum(x.startswith('drop') for x in report)}, merged {sum(x.startswith('merge') for x in report)}")
    if a.output:
        with zipfile.ZipFile(a.output, "w") as z:
            z.writestr("gt/gt.txt", "".join(f"{int(x[0])},{int(x[1])},{x[2]:.1f},{x[3]:.1f},{x[4]:.1f},{x[5]:.1f},1,{int(x[7])},1.0{chr(10)}" for x in out))
            z.writestr("gt/labels.txt", labels)
        print("->", a.output)


def cmd_score(a):
    import trackeval
    n = len(frames_of(a.name))
    ev_dir = os.path.join(LAB, "_eval", a.name + ("_drop" if a.drop_nonplayer else ""))
    shutil.rmtree(ev_dir, ignore_errors=True)
    gt_dir, tr_dir = os.path.join(ev_dir, "gt", a.name, "gt"), os.path.join(ev_dir, "trackers", "ours", "data")
    os.makedirs(gt_dir)
    os.makedirs(tr_dir)
    with open(a.gt) as f:
        gt = [ln for ln in f.read().splitlines() if ln.strip()]
    with open(os.path.join(os.path.dirname(a.gt), "labels.txt"), encoding="utf-8") as f:
        assert f.read().split() == ["player", "nonplayer"], "CVAT task labels must be exactly: player, nonplayer (in that order)"
    # CVAT exports boxes drawn as shapes (not tracks) with id -1; TrackEval would merge or reject them
    assert all(int(ln.split(",")[1]) > 0 for ln in gt), "GT has track_id -1 (shape boxes): convert them to tracks in CVAT"
    assert max(int(ln.split(",")[0]) for ln in gt) == n, "GT does not cover the whole clip: our later boxes would all be FP"
    if a.drop_nonplayer:
        gt = [ln for ln in gt if ln.split(",")[7].strip() != "2"]
    with open(os.path.join(gt_dir, "gt.txt"), "w") as f:
        f.write("\n".join(gt) + "\n")
    shutil.copy(a.ours or os.path.join(LAB, a.name, "ours.txt"), os.path.join(tr_dir, f"{a.name}.txt"))
    ev = trackeval.Evaluator({"USE_PARALLEL": False, "PLOT_CURVES": False, "LOG_ON_ERROR": None, "PRINT_CONFIG": False,
                              "PRINT_RESULTS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False})
    ds = trackeval.datasets.MotChallenge2DBox({"GT_FOLDER": os.path.join(ev_dir, "gt"), "TRACKERS_FOLDER": os.path.join(ev_dir, "trackers"),
                                               "OUTPUT_FOLDER": os.path.join(ev_dir, "out"), "SKIP_SPLIT_FOL": True,
                                               "SEQ_INFO": {a.name: n}, "TRACKERS_TO_EVAL": ["ours"], "PRINT_CONFIG": False})
    res, _ = ev.evaluate([ds], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
    r = res["MotChallenge2DBox"]["ours"]["COMBINED_SEQ"]["pedestrian"]
    h, c, i = r["HOTA"], r["CLEAR"], r["Identity"]
    s = {"HOTA": h["HOTA"].mean(), "DetA": h["DetA"].mean(), "AssA": h["AssA"].mean(), "LocA": h["LocA"].mean(),
         "IDF1": i["IDF1"], "IDP": i["IDP"], "IDR": i["IDR"], "MOTA": c["MOTA"], "MODA": c["MODA"], "IDSW": c["IDSW"],
         "precision": c["CLR_Pr"], "recall": c["CLR_Re"], "FP_per_frame": c["FP_per_frame"], "MOTP_IoU": c["MOTP"],
         "frames": n}
    s = {k: float(v) for k, v in s.items()}
    with open(os.path.join(ev_dir, "scores.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("clip"); p.add_argument("--name", required=True); p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, default=120.0)
    p = sub.add_parser("sam3"); p.add_argument("--name", required=True); p.add_argument("--model", required=True)
    p.add_argument("--seconds", type=float, default=0.0); p.add_argument("--imgsz", type=int, default=644)
    p = sub.add_parser("ours"); p.add_argument("--name", nargs="+", required=True)
    p = sub.add_parser("score"); p.add_argument("--name", required=True); p.add_argument("--gt", required=True)
    p.add_argument("--ours"); p.add_argument("--drop-nonplayer", action="store_true")
    p = sub.add_parser("dedup"); p.add_argument("--in", dest="input", required=True); p.add_argument("--out", dest="output")
    a = ap.parse_args()
    {"clip": cmd_clip, "sam3": cmd_sam3, "ours": cmd_ours, "score": cmd_score, "dedup": cmd_dedup}[a.cmd](a)
