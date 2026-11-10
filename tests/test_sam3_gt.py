"""SAM 3 duplicate clean-up: real duplicates go, players who only cross stay."""
import importlib.util
import os

import numpy as np

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "sam3_gt.py")
_s = importlib.util.spec_from_file_location("sam3_gt", _p)
sam3_gt = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sam3_gt)


def rows(tid, frames, x, cls=1, y=100.0):
    return [[f, tid, x(f) if callable(x) else x, y, 50, 100, 1, cls, 1.0] for f in frames]


def test_short_track_inside_a_long_one_is_dropped():
    r = np.array(rows(1, range(1, 101), 10) + rows(2, range(30, 50), 12), float)
    out, rep = sam3_gt.dedup_rows(r)
    assert set(out[:, 1]) == {1} and len(out) == 100 and rep[0].startswith("drop")


def test_hand_over_with_coinciding_boxes_is_merged_without_doubles():
    r = np.array(rows(1, range(1, 61), 10) + rows(2, range(51, 121), 11), float)
    out, rep = sam3_gt.dedup_rows(r)
    assert set(out[:, 1]) == {1} and len(out) == 120 and rep[0].startswith("merge")


def test_crossing_players_stay_apart():
    r = np.array(rows(1, range(1, 101), lambda f: 10.0 + 5 * f) + rows(2, range(1, 101), lambda f: 510.0 - 5 * f), float)
    out, rep = sam3_gt.dedup_rows(r)
    assert set(out[:, 1]) == {1, 2} and len(out) == 200 and not rep


def test_different_labels_are_not_merged():
    r = np.array(rows(1, range(1, 61), 10) + rows(2, range(51, 121), 11, cls=2), float)
    out, _ = sam3_gt.dedup_rows(r)
    assert set(out[:, 1]) == {1, 2}
