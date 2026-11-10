"""Per-camera player boxes on a common frame grid: the input of the box-level tracker.

The older pipeline (run.to_observations) keeps only a foot point per detection and fuses cameras frame by frame.
The box-level tracker needs more: the whole box (overlaps, crops, image-space motion), the ReID embedding and the
colour histograms, and a way to go back from any track point to the pixels it was built from. CamBoxes keeps all
boxes of one camera as flat arrays indexed by box index; a TrackPoint refers to boxes as (camera, box index).

Frame grid: k = round((t + offset + drift * t - start) * rate) on the reference clock, so k = 0 is the window start.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..court import Court
from ..homography import Calibration, CalibrationTimeline, calibration_at, timeline_from_json
from .detect import FEAT_LEN, HUE_BINS, Detection

DETECTOR_JITTER_PX = 3.0      # foot-point noise of the detector (same as run.DETECTOR_JITTER_PX)
COURT_MARGIN_M = 1.0          # in_court keeps players just outside the lines (throw-ins, keeper behind the line)
UPPER = HUE_BINS + 3          # upper-body half of the appearance histogram (pipeline.detect.appearance)


@dataclass
class TrackPoint:
    xy: np.ndarray                                                  # (2,) metres
    boxes: list[tuple[str, int]] = field(default_factory=list)      # (camera, box index into CamBoxes arrays)


Track = dict[int, TrackPoint]     # {grid frame k: TrackPoint}


@dataclass
class CamBoxes:
    """All boxes of one camera, flat arrays indexed by box index.

    `reid` / `color` rows of boxes that have no embedding / histogram are zero: a zero ReID row has cosine
    similarity 0 with everything (never a confident match) and a zero histogram is skipped by assign_teams."""
    name: str
    fid: np.ndarray               # (n,) int   source frame index (as stored by the detector)
    t: np.ndarray                 # (n,) camera clock, s
    k: np.ndarray                 # (n,) int   grid frame on the reference clock (-1: surplus source frame, off the grid)
    xyxy: np.ndarray              # (n, 4) pixels (calibrated resolution)
    conf: np.ndarray              # (n,)
    reid: np.ndarray | None       # (n, d) float32, L2-normalised (to the float16 precision it is stored with)
    color: np.ndarray | None      # (n, 30) in [0, 1], upper + lower body histograms
    foot: np.ndarray              # (n, 2) pixels, bottom centre of the box
    h: np.ndarray                 # (n,) box height, px
    w: np.ndarray                 # (n,) box width, px
    xy: np.ndarray                # (n, 2) pitch metres (camera-aligned once align.align_cameras ran)
    xy_raw: np.ndarray            # (n, 2) pitch metres straight from the homography
    sigma: np.ndarray             # (n,) foot-point uncertainty, metres
    in_court: np.ndarray          # (n,) bool, inside the pitch + COURT_MARGIN_M
    team: np.ndarray              # (n,) str 'A' / 'B' / '' (see assign_teams)
    cal: Calibration | CalibrationTimeline
    frame_wh: tuple[int, int] = (1920, 1080)
    by_k: dict[int, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    def boxes_at(self, k: int, min_conf: float = 0.0, court_only: bool = True) -> np.ndarray:
        """Box indices of grid frame k (ascending)."""
        idx = self.by_k.get(k, np.zeros(0, int))
        if len(idx) == 0:
            return idx
        m = self.conf[idx] >= min_conf
        if court_only:
            m &= self.in_court[idx]
        return idx[m]


def one_source_frame_per_k(keys: list, x: list[float], legacy: bool = False) -> dict:
    """Grid frame of each source frame (`keys`) at grid position `x` (float, in grid frames): round(x), at most one
    source frame per grid frame. When two consecutive source frames round onto one k, the later one moves to the next
    k if that is still within half a grid frame of it (a tie: 60 fps sampled every 6th frame lands exactly between
    two grid frames at x.x5 s) and is dropped (k = -1) otherwise: the camera simply has more frames than the grid.

    Pushing it on regardless (legacy=True, the rule until 2026-10-22 and in the research results) made every later
    frame follow it for good: the 10/8 S26 sampled every 0.0999556 s, so its grid ran ahead of its own clock by
    444 ppm, 1.3 s at the end of the match, which looked like the two phones' clocks drifting apart."""
    k_of, last = {}, None
    for key, xx in sorted(zip(keys, x), key=lambda kx: (kx[1], kx[0])):
        kk = int(np.round(xx))
        if last is not None and kk <= last:
            if not legacy and last + 1 - xx > 0.5 + 1e-6:
                k_of[key] = -1
                continue
            kk = last + 1
        k_of[key] = last = kk
    return k_of


def grid_frames(fid: np.ndarray, t: np.ndarray, offset_s: float, drift: float, start_s: float, rate: float,
                legacy: bool = False) -> np.ndarray:
    """Grid frame k = round((t + offset + drift * t - start) * rate) of every box, at most one source frame per grid
    frame (one_source_frame_per_k: a tie moves to the next k, a surplus frame gets k = -1 and is left off the grid)."""
    frames = sorted({(int(f), float(tt)) for f, tt in zip(fid, t)})
    k_of = one_source_frame_per_k(frames, [(tt + offset_s + drift * tt - start_s) * rate for _, tt in frames], legacy)
    return np.array([k_of[(int(f), float(tt))] for f, tt in zip(fid, t)], int).reshape(-1)


def _pitch(cal: Calibration | CalibrationTimeline, t: np.ndarray, foot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Foot pixels -> pitch metres and their sigma, with the calibration in force at each box's time
    (a knocked tripod gives a timeline)."""
    xy, sig = np.zeros((len(foot), 2)), np.zeros(len(foot))
    if not len(foot):
        return xy, sig
    if isinstance(cal, Calibration):
        groups = [(cal, np.ones(len(foot), bool))]
    else:
        which = np.array([id(calibration_at(cal, float(tt))) for tt in t])
        groups = [(c, which == id(c)) for _, c in cal]
    for c, m in groups:
        if m.any():
            xy[m] = c.to_pitch(foot[m])
            sig[m] = DETECTOR_JITTER_PX * c.metres_per_pixel(foot[m])
    return xy, sig


def _index_by_k(k: np.ndarray) -> dict[int, np.ndarray]:
    """Box indices per grid frame; boxes off the grid (k < 0) are left out."""
    out: dict[int, list[int]] = {}
    for i in np.argsort(k, kind="stable"):
        if k[i] < 0:
            continue
        out.setdefault(int(k[i]), []).append(int(i))
    return {kk: np.array(v, int) for kk, v in out.items()}


def _cam_boxes(name: str, fid, t, xyxy, conf, reid, color, cal, offset_s: float, drift: float, start_s: float,
               rate: float, court: Court, frame_wh=(1920, 1080), foot=None, legacy_grid: bool = False) -> CamBoxes:
    fid = np.asarray(fid).astype(int).reshape(-1)
    t = np.asarray(t, float).reshape(-1)
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    conf = np.asarray(conf, float).reshape(-1)
    if foot is None:
        foot = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], 1)
    xy, sigma = _pitch(cal, t, foot)
    k = grid_frames(fid, t, offset_s, drift, start_s, rate, legacy_grid)
    return CamBoxes(name=name, fid=fid, t=t, k=k, xyxy=xyxy, conf=conf, reid=reid, color=color, foot=foot,
                    h=xyxy[:, 3] - xyxy[:, 1], w=xyxy[:, 2] - xyxy[:, 0], xy=xy, xy_raw=xy.copy(), sigma=sigma,
                    in_court=court.contains(xy, COURT_MARGIN_M) if len(xy) else np.zeros(0, bool),
                    team=np.full(len(t), "", dtype="<U1"), cal=cal, frame_wh=(int(frame_wh[0]), int(frame_wh[1])),
                    by_k=_index_by_k(k))


def build_cam_boxes(name: str, dets: list[Detection], cal: Calibration | CalibrationTimeline, offset_s: float,
                    drift: float, start_s: float, rate: float, court: Court, frame_wh=(1920, 1080)) -> CamBoxes:
    """Detections of one camera -> CamBoxes on the grid k = round((t + offset + drift*t - start) * rate).

    Detections saved before boxes were kept (no `box`) get a box rebuilt from the foot point and height with a
    typical player aspect (w = 0.4 h); the foot point itself stays exact. ReID rows are used as stored (the
    embedder normalised them), not renormalised, so this matches the research cache box for box."""
    n = len(dets)
    foot = np.array([[d.u, d.v] for d in dets], float).reshape(-1, 2)
    xyxy = np.zeros((n, 4))
    for i, d in enumerate(dets):
        if d.box is not None:
            xyxy[i] = d.box
        else:
            hw = 0.2 * d.h_px
            xyxy[i] = (d.u - hw, d.v - d.h_px, d.u + hw, d.v)
    reid = None
    dims = {len(d.reid) for d in dets if d.reid is not None}
    if dims:
        if len(dims) > 1:
            raise ValueError(f"{name}: ReID embeddings of different sizes {sorted(dims)}")
        reid = np.zeros((n, dims.pop()), np.float32)
        for i, d in enumerate(dets):
            if d.reid is not None and np.all(np.isfinite(d.reid)):
                reid[i] = d.reid
    color = None
    if any(d.feat is not None for d in dets):
        color = np.zeros((n, FEAT_LEN))
        for i, d in enumerate(dets):
            if d.feat is not None:
                color[i] = np.asarray(d.feat, float) / 255.0
    return _cam_boxes(name, [d.frame for d in dets], [d.t for d in dets], xyxy, [d.conf for d in dets], reid, color,
                      cal, offset_s, drift, start_s, rate, court, frame_wh, foot=foot)


def cam_boxes_from_cache(npz_path: str, name: str, cal: Calibration | CalibrationTimeline, offset: float, start: float,
                         rate: float, court: Court, drift: float = 0.0, frame_wh=(1920, 1080),
                         legacy_grid: bool = False) -> CamBoxes:
    """CamBoxes from a research detection cache (experiments/build_cache.py: fid, t, xyxy, conf, reid float16,
    color uint8), without the video. ReID is used as stored, like the research harness. Box i is the cache's box i
    (boxes of a surplus source frame keep their index with k = -1). legacy_grid: grid_frames' old rule, which the
    results in experiments/results were made with."""
    z = np.load(npz_path)
    n = len(z["t"])
    reid = z["reid"].astype(np.float32).reshape(n, -1) if "reid" in z and z["reid"].size else None
    color = z["color"].astype(float).reshape(n, -1) / 255.0 if "color" in z and z["color"].size else None
    return _cam_boxes(name, z["fid"], z["t"], z["xyxy"], z["conf"], reid, color, cal, offset, drift, start, rate, court,
                      frame_wh, legacy_grid=legacy_grid)


def cams_from_cache(cache_dir: str, root: str = ".", court: Court | None = None,
                    legacy_grid: bool = False) -> tuple[dict[str, CamBoxes], dict]:
    """All cameras of a research cache directory (meta.json + <cam>.npz); calibration paths in meta.json are
    relative to `root` (the repository). Returns (cams, meta); no alignment, no teams."""
    with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    court = court or Court()
    cams = {}
    for cam in meta["videos"]:
        with open(os.path.join(root, meta["calib"][cam]), encoding="utf-8") as f:
            cal = timeline_from_json(json.load(f))
        cams[cam] = cam_boxes_from_cache(os.path.join(cache_dir, f"{cam}.npz"), cam, cal, float(meta["offsets"][cam]),
                                         float(meta["start"]), float(meta["rate"]), court, legacy_grid=legacy_grid)
    return cams, meta


# ---------------------------------------------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------------------------------------------

def _kmeans2(F: np.ndarray, rng: np.random.Generator, n_init: int = 5, iters: int = 100) -> np.ndarray:
    """Two-centre k-means (k-means++ seeding, best of n_init by inertia)."""
    best, best_cost = None, np.inf
    for _ in range(n_init):
        c0 = F[rng.integers(len(F))]
        d0 = ((F - c0) ** 2).sum(1)
        c1 = F[rng.choice(len(F), p=d0 / d0.sum())] if d0.sum() > 0 else F[rng.integers(len(F))]
        C = np.stack([c0, c1])
        for _ in range(iters):
            a = ((F[:, None] - C[None]) ** 2).sum(2).argmin(1)
            if (a == 0).all() or (a == 1).all():
                break
            new = np.stack([F[a == j].mean(0) for j in range(2)])
            if np.allclose(new, C):
                break
            C = new
        cost = ((F[:, None] - C[None]) ** 2).sum(2).min(1).sum()
        if cost < best_cost:
            best, best_cost = C, cost
    return best


def _cross_camera_pairs(a: CamBoxes, b: CamBoxes, min_conf: float = 0.5, max_d: float = 2.5, min_sim: float = 0.7,
                        step: int = 5) -> np.ndarray:
    """(i, j) box pairs that are the same person in cameras a and b: mutual nearest on the pitch (raw positions, the
    cameras are not aligned yet), within `max_d`, and alike by ReID. Every `step`-th frame is enough to vote."""
    from scipy.optimize import linear_sum_assignment
    if a.reid is None or b.reid is None:
        return np.zeros((0, 2), int)
    out = []
    for k in sorted(set(a.by_k) & set(b.by_k))[::step]:
        ia, ib = a.boxes_at(k, min_conf), b.boxes_at(k, min_conf)
        if not len(ia) or not len(ib):
            continue
        D = np.linalg.norm(a.xy_raw[ia][:, None] - b.xy_raw[ib][None], axis=2)
        S = a.reid[ia] @ b.reid[ib].T
        r, c = linear_sum_assignment(D - 0.5 * S)
        for i, j in zip(r, c):
            if D[i, j] < max_d and S[i, j] > min_sim and D[i].argmin() == j and D[:, j].argmin() == i:
                out.append((ia[i], ib[j]))
    return np.array(out, int).reshape(-1, 2)


_SPREAD_FLOOR = 0.05       # each side's spread of the bib share counts as at least this part of the gap between the sides


def _fit_pairs(X: np.ndarray, rng: np.random.Generator, n_init: int = 20, floor: float = 1 / 3) -> np.ndarray:
    """Two centres for samples that hold both cameras' views of one person: the smallest inertia among restarts that
    leave at least `floor` of the samples on the smaller side (two teams of five), else the smallest overall."""
    best, key = None, None
    for _ in range(n_init):
        C = _kmeans2(X, rng, n_init=1)
        d = ((X[:, None] - C[None]) ** 2).sum(2)
        side = float((d.argmin(1) == 0).mean())
        k = (min(side, 1 - side) >= floor, -float(d.min(1).sum()))
        if key is None or k > key:
            best, key = C, k
    return best


def _bib_bins(bib: np.ndarray, other: np.ndarray) -> list[int]:
    """One camera's bib colour: the hue bin where the bib side's mean histogram exceeds the other side's most, plus a
    neighbouring bin that carries at least half as much (a colour on a bin edge)."""
    d = bib[:HUE_BINS] - other[:HUE_BINS]
    top = int(np.argmax(d))
    if d[top] <= 0:
        return [top]
    return sorted([top] + [nb for nb in ((top - 1) % HUE_BINS, (top + 1) % HUE_BINS) if d[nb] >= 0.5 * d[top]])


def _bib_rule(F: np.ndarray, lab: np.ndarray, side: int, min_side: int) -> dict | None:
    """How well cluster `side`'s own colour tells it from the other cluster in one camera's fitted samples: its bins,
    both sides' mean and spread of that colour's share, the share of samples a midpoint threshold puts on the right
    side ('fit'), and whether the two sides are well apart ('separates'). None when a side has under min_side."""
    on = lab == side
    if min(int(on.sum()), int((~on).sum())) < min_side:
        return None
    bins = _bib_bins(F[on].mean(0), F[~on].mean(0))
    s = F[:, bins].sum(1)
    m1, m0, s1, s0 = float(s[on].mean()), float(s[~on].mean()), float(s[on].std()), float(s[~on].std())
    return {"bins": bins, "m1": m1, "m0": m0, "s1": s1, "s0": s0,
            "fit": float(((np.abs(s - m1) < np.abs(s - m0)) == on).mean()), "separates": abs(m1 - m0) >= 2 * max(s1, s0)}


def assign_teams(cams: dict[str, CamBoxes], min_h: float = 40, min_conf: float = 0.4, label_min_h: float = 30,
                 margin: float = 0.95, odds: float = 9.0, max_fit: int = 20000, seed: int = 0, per_camera: bool = True,
                 min_pairs: int = 50, min_side: int = 10) -> dict:
    """Split the boxes into the bib team 'A' and the team without bibs 'B' and write cam.team ('A' / 'B' / '').

    In our matches one side wears bibs over its own mixed shirts, so the teams are "bib or not", found in two steps.
    1. Two clusters of the upper-body histogram (shares per hue / black / grey / white bin). With two cameras and at
       least `min_pairs` box pairs that are one person (mutual nearest on the pitch and alike by ReID, every 5th
       frame; per_camera=True), one k-means runs over the pairs, each pair one sample holding both views. A camera on
       its own is not enough: the phones render colours differently (a yellow bib looked lime in one phone of the
       10/8 test), and one camera's largest colour gap need not be the teams (cam2 of the 10/8 match on its own split
       red shirts from the rest: 0.69 agreement with cam1). Restarts that leave under a third of the pairs on one side
       are passed over (5 v 5). Without pairs, or when the pairs do not give a bib split (all from one team): one
       joint fit on the confident, big, in-court boxes of every camera.
    2. The bib side is the cluster whose own colour best tells it from the other (a bib is one colour on every
       player of its side; a mixed side's largest colour group is only part of it). In each camera its colour is the
       hue bin where its mean histogram exceeds the other side's most, plus a neighbour with at least half that. A
       box at least `label_min_h` px tall is labelled by its share of that colour, each side with its own mean and
       spread (so a bib seen in part still reads as bib), a share beyond a side's mean counting as that mean. It
       stays '' when the two likelihoods are within `odds` of each other, or when its share is about as far from both
       means (within `margin`, as for a box half-way between two kits). Where that colour does not keep the sides
       apart (two kits differing only in black / white, too few samples of one side in a camera), the box takes the
       nearer cluster centre instead (the rule before this one).
    Returns a summary: mode ('paired' / 'joint'), n_fit (the confident, big, in-court boxes), pairs and their label
    agreement, label counts, per camera the bib bins, the two sides' mean bib share and which rule labelled it, and
    the side means as 'centres'; {'n_fit': n} with every team '' when there is too little colour to fit.

    On the 10/8 match, against the players of two AI-corrected, not yet human-checked cam1 clips (cam2 through the
    boxes paired with them): cam1 0.949 -> 0.98 and cam2 0.742 -> 0.99 of the labels right, cameras agreeing on
    0.98 of the pairs (was 0.69)."""
    for c in cams.values():
        c.team = np.full(len(c), "", dtype="<U1")
    rng = np.random.default_rng(seed)
    feats = {}
    for name, c in cams.items():
        if c.color is None or not len(c):
            continue
        up = c.color[:, :UPPER]
        s = up.sum(1)
        valid = s > 0
        F = up / np.maximum(s, 1e-9)[:, None]
        feats[name] = (F, valid, valid & (c.h >= min_h) & (c.conf >= min_conf) & c.in_court)
    names = list(feats)
    n_fit = int(sum(feats[n][2].sum() for n in names))
    P = np.zeros((0, 2), int)
    if per_camera and len(names) == 2:
        P = _cross_camera_pairs(cams[names[0]], cams[names[1]])
        P = P[feats[names[0]][1][P[:, 0]] & feats[names[1]][1][P[:, 1]]] if len(P) else P

    def paired():                   # {camera: (fitted histograms, their cluster 0 / 1)}, {camera: the two centres}
        if len(P) < min_pairs:
            return None
        a, b = names
        X = np.hstack([feats[a][0][P[:, 0]], feats[b][0][P[:, 1]]])                # one person, both views, one sample
        C = _fit_pairs(X, rng)
        lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
        return {a: (feats[a][0][P[:, 0]], lab), b: (feats[b][0][P[:, 1]], lab)}, {a: C[:, :UPPER], b: C[:, UPPER:]}

    def joint():
        F_fit = np.vstack([feats[n][0][feats[n][2]] for n in names]) if names else np.zeros((0, UPPER))
        if len(F_fit) < 20:
            return None
        C = _kmeans2(F_fit[rng.choice(len(F_fit), max_fit, replace=False)] if len(F_fit) > max_fit else F_fit, rng)
        return {n: (feats[n][0][feats[n][2]], ((feats[n][0][feats[n][2]][:, None] - C[None]) ** 2).sum(2).argmin(1))
                for n in names}, {n: C for n in names}

    chosen = None
    for mode, make in (("paired", paired), ("joint", joint)):
        fit = make()
        if fit is None:
            continue
        fitted, centre = fit
        rules = [{n: _bib_rule(*fitted[n], j, min_side) for n in names} for j in (0, 1)]
        weight = lambda j: [(r, len(fitted[n][1])) for n, r in rules[j].items() if r]
        score = [(sum(r["fit"] * w for r, w in weight(j)) / max(1, sum(w for _, w in weight(j))),
                  sum(abs(r["m1"] - r["m0"]) * w for r, w in weight(j))) for j in (0, 1)]
        bib = int(score[1] > score[0])
        chosen = (mode, fitted, centre, bib, rules[bib])
        if weight(bib) and all(r["separates"] for r, _ in weight(bib)):
            break
    if chosen is None:
        return {"n_fit": n_fit}
    mode, fitted, centre, bib, rules = chosen
    per_cam = {}
    for n in names:
        F, valid, _ = feats[n]
        r = rules[n]
        if r is None or not r["separates"]:
            D = np.sqrt(((F[:, None] - centre[n][None]) ** 2).sum(2))
            ok = valid & (D.min(1) < margin * D.max(1)) & (cams[n].h >= label_min_h)
            cams[n].team = np.where(ok, np.where(D.argmin(1) == bib, "A", "B"), "").astype("<U1")
            per_cam[n] = {"bib_bins": r["bins"] if r else [], "labels": "nearest centre"}
            continue
        m1, m0 = r["m1"], r["m0"]
        s1, s0 = (max(s, _SPREAD_FLOOR * abs(m1 - m0), 1e-3) for s in (r["s1"], r["s0"]))
        x = np.clip(F[:, r["bins"]].sum(1), min(m0, m1), max(m0, m1))     # beyond a side's mean counts as that mean
        llr = (-0.5 * ((x - m1) / s1) ** 2 - np.log(s1)) - (-0.5 * ((x - m0) / s0) ** 2 - np.log(s0))
        d1, d0 = np.abs(x - m1), np.abs(x - m0)
        ok = valid & (np.abs(llr) >= np.log(odds)) & (np.minimum(d1, d0) < margin * np.maximum(d1, d0)) & (cams[n].h >= label_min_h)
        cams[n].team = np.where(ok, np.where(llr > 0, "A", "B"), "").astype("<U1")
        per_cam[n] = {"bib_bins": r["bins"], "bib_share": [round(m1, 4), round(m0, 4)], "labels": "bib share"}
    info = {"mode": mode, "n_fit": n_fit}
    if len(P):
        la, lb = cams[names[0]].team[P[:, 0]], cams[names[1]].team[P[:, 1]]
        both = (la != "") & (lb != "")
        info |= {"pairs": int(both.sum()), "pair_agreement": round(float((la[both] == lb[both]).mean()), 4) if both.any() else 0.0}
    means = {n: [F[lab == j].mean(0) if (lab == j).any() else np.zeros(UPPER) for j in (0, 1)] for n, (F, lab) in fitted.items()}
    return info | {"counts": {n: {t: int((cams[n].team == t).sum()) for t in ("A", "B", "")} for n in names}, "cameras": per_cam,
                   "centres": {n: {"A": means[n][bib].round(4).tolist(), "B": means[n][1 - bib].round(4).tolist()} for n in names}}
