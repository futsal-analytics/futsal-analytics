"""The pipeline's detection step: 차재혁's detection/detect_person.py (YOLO + ByteTrack through Ultralytics' model.track).

detection/detect_person.py is a script, not a module: at import it builds YOLO(<model>) and runs model.track(source=
<a fixed video>, ..., save=True), which only writes an annotated video. read_settings executes the file with a stand-in
`ultralytics` module whose YOLO records the model name and the track() keyword arguments and does nothing else (run as
__main__, so a main guard changes nothing; anything else the script calls on Ultralytics or on the results does
nothing; the script sees its own file name as sys.argv; bytecode off, so nothing is written into detection/; the real
ultralytics package and its submodules, if loaded, are set aside and put back). Other code in the file (file or video
access, prints) does run. detection_tracker
then runs the real YOLO(<model>).track(source=<frame>, stream=True, save=False, persist=True, **<those arguments>) on
our video, so a change of model / conf / imgsz / tracker in that file reaches the pipeline. Only what picks the input
or writes / shows output (source, save, show, ...) is replaced; device falls back to the CPU without CUDA.

track_video feeds every frame of the requested window to the tracker (ByteTrack needs them all) and keeps the boxes
of every round(fps / rate_hz)-th one: the frames, timestamps (each frame's own), colour histograms and ReID embeddings
are exactly those of detect.detect_video(rate_hz=...), which it runs with stride 1. The result is a list of
detect.Detection (saved and loaded with detect.save / load); the ByteTrack id goes to Detection.track_id.

Against our own detector (detect.yolo_detector, which the research caches and the validated runs used):
  - the boxes are ByteTrack's track states (Kalman-smoothed, new tracks hidden until confirmed), not raw detections
  - conf 0.3 instead of 0.1: v2 was tuned on 0.1 boxes, and ByteTrack's low-score second stage (0.1) gets nothing
  - the detector runs on every frame (6x the work for a 60 fps camera at 10 Hz)
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from . import detect
from .detect import Detection, Detector, Embedder

REPO = Path(__file__).resolve().parents[3]
DETECT_PERSON = REPO / "detection" / "detect_person.py"
# track() arguments that pick the input or write / show results: the pipeline sets its own
OURS = ("source", "stream", "persist", "save", "save_txt", "save_conf", "save_crop", "save_frames", "show", "verbose",
        "project", "name")


class _Nothing:
    """Whatever else the script touches under the stand-in (ultralytics.settings, model.names, the results of track()):
    calling it gives `ret` (itself by default), any attribute or item is another _Nothing, and it iterates as empty."""

    def __init__(self, ret=None):
        self._ret = ret

    def __call__(self, *args, **kwargs):
        return self if self._ret is None else self._ret

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Nothing()

    def __getitem__(self, key):
        return _Nothing()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


def _anything(name: str):
    """Module __getattr__ of the stand-in ultralytics: any public name is a _Nothing (`__path__` and the like are not,
    so `import ultralytics.<sub>` still fails)."""
    if name.startswith("__"):
        raise AttributeError(name)
    return _Nothing()


def read_settings(path: str | os.PathLike = DETECT_PERSON) -> dict:
    """{"file", "model", "track"} of detection/detect_person.py without running YOLO (see the module docstring):
    "track" is the keyword arguments of the last model.track() call and "model" the model that call was made on."""
    calls: list = []

    class YOLO:                                   # Ultralytics' YOLO(model="yolo26n.pt", task=None, verbose=False)
        def __init__(self, model="yolo26n.pt", *args, **kwargs):
            self.model_name = str(model)
            calls.append(("model", self.model_name))

        def track(self, *args, **kwargs):
            if args:                              # Model.track(source=None, stream=False, persist=False, **kwargs)
                kwargs = {**dict(zip(("source", "stream", "persist"), args)), **kwargs}
            calls.append(("track", (self.model_name, kwargs)))
            return _Nothing()                     # the results: empty, and res[0].boxes does not fail

        def __getattr__(self, name):              # model.to("cuda"), model.fuse(), ...: return the model, as Ultralytics does
            if name.startswith("__"):
                raise AttributeError(name)
            return _Nothing(self)

    stub = ModuleType("ultralytics")
    stub.YOLO = YOLO
    stub.__getattr__ = _anything                  # `from ultralytics import YOLO, settings`
    # the real package and its submodules step aside: `from ultralytics.models import YOLO` must not reach the real one
    ours = lambda k: k == "ultralytics" or k.startswith("ultralytics.")
    real = {k: sys.modules.pop(k) for k in [k for k in sys.modules if ours(k)]}
    bytecode, argv = sys.dont_write_bytecode, sys.argv
    sys.modules["ultralytics"], sys.dont_write_bytecode = stub, True
    sys.argv = [str(path)]                        # an argparse in the script must not see the pipeline's own arguments
    try:
        runpy.run_path(str(path), run_name="__main__")     # as `python detect_person.py`: a __main__ guard runs too
    finally:
        sys.dont_write_bytecode, sys.argv = bytecode, argv
        for k in [k for k in sys.modules if ours(k)]:
            del sys.modules[k]
        sys.modules.update(real)
    tracks = [c[1] for c in calls if c[0] == "track"]
    if not tracks:
        raise ValueError(f"{path}: expected YOLO(<model>) and model.track(...), got {calls}")
    model, kwargs = tracks[-1]                    # the model the last track() ran on (a file may build more than one)
    return {"file": str(path), "model": model, "track": kwargs}


def track_kwargs(settings: dict, cuda: bool | None = None) -> dict:
    """The file's track() arguments minus the input / output ones (OURS); device 'cpu' when CUDA is unavailable."""
    kw = {k: v for k, v in settings["track"].items() if k not in OURS}
    if cuda is None:
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:                       # no torch: Ultralytics cannot run either, CPU is the honest answer
            cuda = False
    if not cuda and str(kw.get("device", "cpu")) != "cpu":
        kw["device"] = "cpu"
    return kw


def model_path(name: str) -> str:
    """The weights file as given if it exists (absolute or from the working directory), else the repository root's
    copy, else the bare name (Ultralytics downloads its own models)."""
    for p in (Path(name), REPO / name):
        if p.is_file():
            return str(p)
    return name


def tag(settings: dict) -> str:
    """Detection cache key (trackview's det_<cam>.json): any change of the file's settings re-runs detection."""
    kw = track_kwargs(settings, cuda=True)
    kw.pop("device", None)
    return f"detection/{Path(settings['file']).name}:{settings['model']} " + ",".join(f"{k}={kw[k]}" for k in sorted(kw))


def detection_tracker(settings: dict | None = None, cuda: bool | None = None) -> Detector:
    """frame -> [(x1, y1, x2, y2, conf, None, track_id)], stateful: call it on every frame of one video, in order
    (a new tracker per video). Needs Ultralytics (pip install -e ".[vision]")."""
    s = settings or read_settings()                # before importing ultralytics (read_settings sets it aside anyway)
    from ultralytics import YOLO
    net = YOLO(model_path(s["model"]))
    kw = track_kwargs(s, cuda)

    def track(frame: np.ndarray) -> list:
        out = []
        for r in net.track(source=frame, stream=True, save=False, persist=True, verbose=False, **kw):
            b = r.boxes
            # no ids: Ultralytics passed the raw detections through (the tracker returned nothing and had no
            # unconfirmed track to hide), as detect_person.py would draw them; they are kept without a track id
            ids = [int(i) for i in b.id.int().cpu().numpy()] if b.id is not None else [None] * len(b)
            for (x1, y1, x2, y2), c, i in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), ids):
                out.append((float(x1), float(y1), float(x2), float(y2), float(c), None, i))
        return out
    return track


def track_video(path: str, tracker: Detector, rate_hz: float = 10.0, start_s: float = 0.0, end_s: float | None = None,
                progress: bool = False, embedder: Embedder | None = None, feat_every: int = 2
                ) -> tuple[list[Detection], float, int]:
    """detect.detect_video(rate_hz=...) with a tracker that sees every frame (module docstring). `tracker` may also be
    a plain Detector (boxes without a 7th element get track_id None). Returns detections, the video fps and the
    number of frames read."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, int(round(fps / rate_hz)))
    n, ids = [0], []

    def every_frame(frame):
        boxes = tracker(frame)
        keep, n[0] = n[0] % stride == 0, n[0] + 1
        if not keep:
            return []
        ids.extend(b[6] if len(b) > 6 else None for b in boxes)
        return [tuple(b[:6]) for b in boxes]

    # frame i is kept when i % stride == 0, as in detect_video(rate_hz); appearance on every feat_every-th kept frame
    dets, fps, frames = detect.detect_video(path, every_frame, stride=1, feat_every=feat_every * stride, start_s=start_s,
                                            end_s=end_s, progress=progress, embedder=embedder)
    for d, i in zip(dets, ids, strict=True):
        d.track_id = i
    return dets, fps, frames
