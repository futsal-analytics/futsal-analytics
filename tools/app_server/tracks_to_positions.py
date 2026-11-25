"""Send a pipeline tracks.json to the Playbook app server as position samples (POST /api/v1/matches/{id}/positions).

Reference converter for the app <-> pipeline connection test. Standard library only.

  python tracks_to_positions.py --tracks tracks.json --mapping mapping.csv --server http://127.0.0.1:8000 --match 1 --dry-run
  python tracks_to_positions.py --tracks tracks.json --mapping mapping.csv --server http://127.0.0.1:8000 --match 1

mapping.csv: header `track_id,player_id` (pipeline track id -> app Player.id). Unmapped tracks are skipped.
Password: environment variable PLAYBOOK_ADMIN_PASSWORD, otherwise it is asked for (never on the command line).

Server rules this follows (backend/app/api/routes.py): camera_id one capital letter, timestamp_ms >= 0, x_m/y_m >= 0 and
within the match's court length/width (one bad point rejects the whole batch with 422), player_id must be a player of the
match's team, 1..20,000 points per request, body <= 8 MB. There is no delete API, so a second upload doubles distances:
the script stops if the match already has samples (unless --force).
"""
import argparse, csv, getpass, json, math, os, sys, urllib.error, urllib.request


def http(method, url, body=None, token=None):
    data = json.dumps(body).encode('utf-8') if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise SystemExit(f'{method} {url} -> {e.code} {e.read().decode("utf-8", "replace")[:300]}')


def samples(tracks, mapping, hz, length, width):
    """[(player_id, timestamp_ms, x, y)] every 1/hz s for mapped tracks, inside the court, null frames dropped."""
    fps = float(tracks['fps'])
    step = max(1, round(fps / hz))
    out, dropped = [], {'unmapped_tracks': 0, 'null': 0, 'outside': 0}
    for p in tracks['players']:
        pid = mapping.get(int(p['id']))
        if pid is None:
            dropped['unmapped_tracks'] += 1
            continue
        for i in range(0, len(p['x']), step):
            x, y = p['x'][i], p['y'][i]
            if x is None or y is None or (isinstance(x, float) and math.isnan(x)):
                dropped['null'] += 1
                continue
            if not (0 <= x <= length and 0 <= y <= width):
                dropped['outside'] += 1
                continue
            out.append((pid, int(round(i / fps * 1000)), round(float(x), 2), round(float(y), 2)))
    return out, dropped


def server_distance(points):
    """The server's /summary formula: per player, straight-line distance between consecutive samples in time order."""
    by = {}
    for pid, t, x, y in sorted(points, key=lambda s: (s[0], s[1])):
        by.setdefault(pid, []).append((x, y))
    return {pid: sum(math.dist(a, b) for a, b in zip(v, v[1:])) for pid, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--server', required=True, help='e.g. http://127.0.0.1:8000 (no /api/v1)')
    ap.add_argument('--match', type=int, required=True)
    ap.add_argument('--hz', type=float, default=5.0, help='samples per second sent (default 5)')
    ap.add_argument('--user', default='admin')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='upload even if the match already has samples')
    a = ap.parse_args()

    tracks = json.load(open(a.tracks, encoding='utf-8'))
    mapping = {int(r['track_id']): int(r['player_id']) for r in csv.DictReader(open(a.mapping, encoding='utf-8'))}
    api = a.server.rstrip('/') + '/api/v1'
    pw = os.environ.get('PLAYBOOK_ADMIN_PASSWORD') or getpass.getpass('admin password: ')
    token = http('POST', f'{api}/auth/login', {'username': a.user, 'password': pw})['access_token']
    match = http('GET', f'{api}/matches/{a.match}', token=token)
    pts, dropped = samples(tracks, mapping, a.hz, match['court_length_m'], match['court_width_m'])
    print(f"match {a.match} court {match['court_length_m']}x{match['court_width_m']} | samples {len(pts)} at {a.hz} Hz | dropped {dropped}")

    # what the app will show vs the pipeline's own (smoothed) distance
    app_m = server_distance(pts)
    stats = {int(p['id']): p.get('stats', {}).get('distance_m') for p in tracks['players']}
    for tid, pid in sorted(mapping.items()):
        print(f'  track {tid:>3} -> player {pid:>3}: app distance {app_m.get(pid, 0):7.1f} m | pipeline distance {stats.get(tid) or 0:7.1f} m')
    if a.dry_run:
        return
    summary = http('GET', f'{api}/matches/{a.match}/summary', token=token)
    if summary.get('position_samples', 0) and not a.force:
        raise SystemExit(f"match {a.match} already has {summary['position_samples']} samples; not uploading again (no delete API)")
    for k in range(0, len(pts), 20000):
        batch = [{'camera_id': 'A', 'player_id': pid, 'timestamp_ms': t, 'x_m': x, 'y_m': y, 'confidence': 1.0}
                 for pid, t, x, y in pts[k:k + 20000]]
        r = http('POST', f'{api}/matches/{a.match}/positions', {'positions': batch}, token=token)
        print('  sent', r)
    summary = http('GET', f'{api}/matches/{a.match}/summary', token=token)
    print('server summary:', json.dumps({k: summary[k] for k in ('position_samples', 'analysis_ready')}, ensure_ascii=False))
    for row in summary['players']:
        print(f"  {row['name']}: {row['distance_m']} m")


if __name__ == '__main__':
    sys.exit(main())
