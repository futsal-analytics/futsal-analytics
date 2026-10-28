"""차재혁's detection/detect_person.py as the pipeline's detector: futsal.pipeline.yolo_track.

Nothing here runs YOLO: the settings are read with a stand-in ultralytics module, and the tracker is tested with a fake
YOLO whose track() yields Ultralytics-shaped results."""
import ast
import os
import sys
import types

import cv2
import numpy as np
import pytest

from futsal.pipeline import detect, yolo_track

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTION = os.path.join(ROOT, "detection")


def listing(folder):
    return sorted((os.path.relpath(os.path.join(d, f), folder), os.path.getsize(os.path.join(d, f)))
                  for d, _, fs in os.walk(folder) for f in fs)


def written_settings(path=yolo_track.DETECT_PERSON):
    """The YOLO(...) model and the track(...) keywords as written in the file, read with ast: the expected values follow
    whatever 차재혁 puts there."""
    calls = [n for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())) if isinstance(n, ast.Call)]
    yolo = next(c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "YOLO")
    model = yolo.args[0] if yolo.args else next(k.value for k in yolo.keywords if k.arg == "model")
    track = next(c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "track")
    return ast.literal_eval(model), {k.arg: ast.literal_eval(k.value) for k in track.keywords}


def test_reads_detect_person_settings_without_running_it(monkeypatch):
    before = listing(DETECTION)

    class Untouchable(types.ModuleType):                    # a real ultralytics module must not even be looked at
        def __getattr__(self, name):
            raise AssertionError(f"ultralytics.{name} used")
    real = Untouchable("ultralytics")
    monkeypatch.setitem(sys.modules, "ultralytics", real)
    s = yolo_track.read_settings()
    assert sys.modules["ultralytics"] is real and "detect_person" not in sys.modules
    assert (s["model"], s["track"]) == written_settings()
    monkeypatch.delitem(sys.modules, "ultralytics")
    yolo_track.read_settings()
    assert "ultralytics" not in sys.modules                 # nothing left behind when it was not loaded before
    assert listing(DETECTION) == before and not os.path.exists(os.path.join(DETECTION, "__pycache__"))


def test_track_arguments_follow_the_file(tmp_path):
    """Whatever the file passes to track() reaches the pipeline, except what picks the input or writes output."""
    script = tmp_path / "detect_person.py"
    script.write_text('from ultralytics import YOLO\nm = YOLO(model="best.pt")\n'
                      'm.track("game.mp4", classes=[0], conf=0.2, imgsz=960, tracker="botsort.yaml", device=0, save=True,'
                      ' show=True, half=True)\n', encoding="utf-8")
    s = yolo_track.read_settings(script)
    assert s["model"] == "best.pt" and s["track"]["source"] == "game.mp4"
    kw = yolo_track.track_kwargs(s, cuda=True)
    assert kw == {"classes": [0], "conf": 0.2, "imgsz": 960, "tracker": "botsort.yaml", "device": 0, "half": True}
    assert yolo_track.track_kwargs(s, cuda=False)["device"] == "cpu"         # no CUDA: the CPU, nothing else changes
    assert yolo_track.tag(s) == "detection/detect_person.py:best.pt classes=[0],conf=0.2,half=True,imgsz=960,tracker=botsort.yaml"
    script.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        yolo_track.read_settings(script)


def test_main_guard_and_real_submodules(tmp_path, monkeypatch):
    """A `__main__` guard around the calls still gives the settings, and with the real Ultralytics loaded an import from
    one of its submodules fails loudly instead of building the real model; sys.modules is put back either way."""
    guarded = tmp_path / "guarded.py"
    guarded.write_text('from ultralytics import YOLO\n\nif __name__ == "__main__":\n    m = YOLO("best.pt")\n'
                       '    m.track(source="game.mp4", conf=0.2)\n', encoding="utf-8")
    assert yolo_track.read_settings(guarded)["track"] == {"source": "game.mp4", "conf": 0.2}
    built = []
    real, models = types.ModuleType("ultralytics"), types.ModuleType("ultralytics.models")
    real.__path__, models.YOLO = [], lambda *a, **kw: built.append(a)
    monkeypatch.setitem(sys.modules, "ultralytics", real)
    monkeypatch.setitem(sys.modules, "ultralytics.models", models)
    sub = tmp_path / "sub.py"
    sub.write_text('from ultralytics.models import YOLO\nYOLO("yolo26x.pt").track(source="game.mp4")\n', encoding="utf-8")
    with pytest.raises(ImportError):
        yolo_track.read_settings(sub)
    assert not built and sys.modules["ultralytics"] is real and sys.modules["ultralytics.models"] is models


def test_settings_survive_ordinary_edits_of_the_script(tmp_path, monkeypatch):
    """Edits a detection script often gets: more imports from ultralytics, model.to(...), a second model, an argparse,
    using the results. The settings are still read, and they are those of the track() call with its own model."""
    script = tmp_path / "detect_person.py"
    script.write_text(
        "import argparse\nfrom ultralytics import YOLO, settings\n"
        "settings.update({'runs_dir': 'runs'})\n"
        "ap = argparse.ArgumentParser()\nap.add_argument('--src', default='game.mp4')\na = ap.parse_args()\n"
        "det = YOLO('yolo26x.pt').to('cuda')\npose = YOLO('yolo26n-pose.pt')\n"
        "res = det.track(source=a.src, conf=0.1, imgsz=1280, tracker='bytetrack.yaml')\n"
        "print(len(res[0].boxes), det.names)\nfor r in res:\n    print(r)\n", encoding="utf-8")
    argv = ["python -m futsal.trackview", "--video", "a.mp4"]
    monkeypatch.setattr(sys, "argv", argv)
    s = yolo_track.read_settings(script)
    assert s["model"] == "yolo26x.pt"                                    # not the pose model built after it
    assert s["track"] == {"source": "game.mp4", "conf": 0.1, "imgsz": 1280, "tracker": "bytetrack.yaml"}
    assert sys.argv is argv and "ultralytics" not in sys.modules


class _T:
    """Just enough of a torch tensor: .cpu().numpy(), .int()."""
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def int(self):
        return _T(self.a.astype(int))


class _Boxes:
    def __init__(self, xyxy, conf, ids):
        self.xyxy, self.conf, self.id = _T(np.reshape(xyxy, (-1, 4))), _T(conf), None if ids is None else _T(ids)

    def __len__(self):
        return len(self.conf.a)


class FakeYOLO:
    """YOLO(model).track(source=frame, ...) -> one result per call: a player walking right (ByteTrack id 7), from the
    4th frame on a second one (id 9), and on the 7th frame raw boxes without ids (Ultralytics passes the detections
    through when the tracker returns nothing and has no unconfirmed track)."""
    made: list = []

    def __init__(self, model):
        self.model, self.calls = model, []
        FakeYOLO.made.append(self)

    def track(self, source=None, **kw):
        assert isinstance(source, np.ndarray) and source.ndim == 3
        self.calls.append(kw)
        n = len(self.calls)
        boxes = [[10.0 + n, 20.0, 30.0 + n, 60.0]]
        ids, conf = [7], [0.8]
        if n >= 4:
            boxes.append([100.0, 30.0, 115.0, 70.0]); conf.append(0.4); ids.append(9)
        yield types.SimpleNamespace(boxes=_Boxes(boxes, conf, ids if n != 7 else None))


def _video(path, frames=30, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    for i in range(frames):
        vw.write(np.full((90, 160, 3), 60 + 4 * i, np.uint8))
    vw.release()


def test_tracker_sees_every_frame_and_detections_match_detect_video(tmp_path, monkeypatch):
    path = tmp_path / "v.mp4"
    _video(path)
    FakeYOLO.made = []
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    s = yolo_track.read_settings()
    tracker = yolo_track.detection_tracker(s, cuda=False)
    net, = FakeYOLO.made
    assert os.path.basename(net.model) == os.path.basename(s["model"])
    embedded = []

    def embedder(frame, xyxy):
        embedded.append(len(xyxy))
        return np.tile(np.eye(4)[0], (len(xyxy), 1))
    dets, fps, n = yolo_track.track_video(str(path), tracker, rate_hz=10, embedder=embedder)
    assert fps == 30 and n == 30 and len(net.calls) == 30          # ByteTrack got every frame ...
    want = {"stream": True, "save": False, "persist": True, "verbose": False,
            **{k: v for k, v in written_settings()[1].items() if k not in yolo_track.OURS}}
    if "device" in want:
        want["device"] = "cpu"                                      # cuda=False above
    assert all(kw == want for kw in net.calls)
    frames = sorted({d.frame for d in dets})
    assert frames == list(range(0, 30, 3))                         # ... the pipeline keeps every 3rd (10 Hz at 30 fps)
    assert len(embedded) == 10 and sum(embedded) == len(dets)      # ReID once per kept frame, every box
    first = [d for d in dets if d.box[0] < 50]
    assert [d.box[0] for d in first] == [11.0 + f for f in frames]  # the tracker's own box of that frame
    assert [d.track_id for d in first] == [None if f == 6 else 7 for f in frames]
    assert [(d.frame, d.track_id) for d in dets if d.box[0] >= 50] == [(f, None if f == 6 else 9) for f in frames if f >= 3]
    assert all(d.u == (d.box[0] + d.box[2]) / 2 and d.v == d.box[3] and d.h_px == 40.0 for d in dets)
    assert all(d.reid is not None and d.team is None for d in dets)
    # frames, timestamps and colour histograms are exactly those of detect_video at 10 Hz
    plain = lambda frame: [(10.0, 20.0, 30.0, 60.0, 0.8, None)]
    ref, _, _ = detect.detect_video(str(path), plain, rate_hz=10)
    assert [(d.frame, d.t, d.feat is None) for d in first] == [(d.frame, d.t, d.feat is None) for d in ref]
    assert sum(d.feat is not None for d in first) == 5
    # the det JSON keeps the ByteTrack id; files without one read as before
    out = str(tmp_path / "det.json")
    detect.save(dets, out)
    back, _ = detect.load(out)
    assert back == dets and [d.track_id for d in back] == [d.track_id for d in dets]
    detect.save(ref, out)
    assert all("track_id" not in x for x in __import__("json").load(open(out))["detections"])


def test_plain_detector_through_track_video(tmp_path):
    """A detector without a tracker (6-tuples, e.g. the synthetic one in the trackview test) works too."""
    path = tmp_path / "v.mp4"
    _video(path, frames=12)
    calls = []
    dets, _, _ = yolo_track.track_video(str(path), lambda f: calls.append(1) or [(1.0, 2.0, 11.0, 32.0, 0.9, "A")], rate_hz=10)
    assert len(calls) == 12 and [d.frame for d in dets] == [0, 3, 6, 9]
    assert all(d.track_id is None and d.team == "A" for d in dets)
