"""tools/app_server: tracks.json -> app server position samples (no server needed)."""
import math

from tools.app_server.tracks_to_positions import samples, server_distance


def _tracks():
    # 10 fps, 6 frames: track 1 walks 1 m per frame along x, track 2 has a null and an out-of-court point, track 3 unmapped
    return {
        "fps": 10, "start": 600.0,
        "players": [
            {"id": 1, "team": "A", "x": [1, 2, 3, 4, 5, 6], "y": [5, 5, 5, 5, 5, 5]},
            {"id": 2, "team": "A", "x": [1, None, 41.0, 4, 5, 6], "y": [1, None, 1, 1, 1, 1]},
            {"id": 3, "team": "B", "x": [9] * 6, "y": [9] * 6},
        ],
    }


def test_samples_rate_mapping_and_filters():
    pts, dropped = samples(_tracks(), {1: 101, 2: 102}, hz=5, length=40, width=20)
    # 5 Hz from 10 fps keeps frames 0, 2, 4; timestamps are ms from the start of tracks.json
    assert [(p, t) for p, t, _, _ in pts if p == 101] == [(101, 0), (101, 200), (101, 400)]
    # track 2: frame 2 is outside the 40 m court -> dropped; frame 1 (null) is not sampled at 5 Hz
    assert [(t, x) for p, t, x, _ in pts if p == 102] == [(0, 1.0), (400, 5.0)]
    assert dropped == {"unmapped_tracks": 1, "null": 0, "outside": 1}
    assert all(0 <= x <= 40 and 0 <= y <= 20 for _, _, x, y in pts)


def test_server_distance_matches_server_formula():
    pts, _ = samples(_tracks(), {1: 101}, hz=10, length=40, width=20)
    # the server sums straight lines between consecutive samples in time order (routes.py summary)
    assert math.isclose(server_distance(pts)[101], 5.0)
    shuffled = list(reversed(pts))
    assert math.isclose(server_distance(shuffled)[101], 5.0)
