"""Put one pipeline result on the Playbook server as a new match with clearly labelled players, then upload positions.

  python upload_to_server.py --server <server url> --tracks tracks.json \
      --team test --tracks-ids 1,2,3,4,5 --opponent "11/3 분석 시연" --label "11/3 트랙"

The admin password is typed by the person running it (getpass, not echoed, never stored).
It shows what it will create and only writes after you type y. The server has no delete API, so it refuses to
run if a match with the same opponent name already exists for that team.
"""
import argparse, getpass, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracks_to_positions import http, samples, server_distance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', required=True)
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--team', required=True, help='existing team name on the server')
    ap.add_argument('--preset', choices=['1103'], help='1103: track ids 1-5, opponent "11/3 분석 시연", label "11/3 트랙"')
    ap.add_argument('--tracks-ids', help='pipeline track ids to upload, e.g. 1,2,3,4,5')
    ap.add_argument('--opponent', help='opponent name of the new match (used as its title)')
    ap.add_argument('--label', help='player name prefix, e.g. "11/3 트랙"')
    ap.add_argument('--minutes', type=int, default=10)
    ap.add_argument('--hz', type=float, default=5.0)
    ap.add_argument('--user', default='admin')
    a = ap.parse_args()
    if a.preset == '1103':
        a.tracks_ids = a.tracks_ids or '1,2,3,4,5'
        a.opponent = a.opponent or '11/3 분석 시연'
        a.label = a.label or '11/3 트랙'
    if not (a.tracks_ids and a.opponent and a.label):
        raise SystemExit('need --tracks-ids, --opponent and --label (or --preset)')

    api = a.server.rstrip('/') + '/api/v1'
    tracks = json.load(open(a.tracks, encoding='utf-8'))
    ids = [int(x) for x in a.tracks_ids.split(',')]
    known = {int(p['id']) for p in tracks['players']}
    missing = [i for i in ids if i not in known]
    if missing:
        raise SystemExit(f'track ids not in tracks.json: {missing}')

    print(f'server: {a.server}  (first request can take ~30-60 s if the free server is asleep)')
    pw = os.environ.get('PLAYBOOK_ADMIN_PASSWORD') or getpass.getpass(f'{a.user} password (not shown): ')
    token = http('POST', f'{api}/auth/login', {'username': a.user, 'password': pw})['access_token']
    del pw
    teams = [t for t in http('GET', f'{api}/teams', token=token)['teams'] if t['name'] == a.team]
    if len(teams) != 1:
        raise SystemExit(f'team "{a.team}" found {len(teams)} times; stop')
    team = teams[0]
    matches = http('GET', f'{api}/matches', token=token)['matches']
    if any(m.get('team_id') == team['id'] and m.get('opponent_name') == a.opponent for m in matches):
        raise SystemExit(f'a match "vs {a.opponent}" already exists for team {a.team}; not creating another (no delete API)')

    names = [f'{a.label}{n}' for n in range(1, len(ids) + 1)]
    fake_map = {tid: n for n, tid in enumerate(ids, 1)}
    pts, dropped = samples(tracks, fake_map, a.hz, 40.0, 20.0)
    dist = server_distance(pts)
    print('\nThis will PERMANENTLY add to the server (there is no delete):')
    print(f'  team "{team["name"]}" (id {team["id"]}): new match "vs {a.opponent}", court 40 x 20 m')
    for n, (tid, name) in enumerate(zip(ids, names), 1):
        print(f'  new player "{name}" <- track {tid}: about {dist.get(n, 0) / 1000:.2f} km, {a.minutes} min')
    print(f'  {len(pts)} position samples at {a.hz} Hz (dropped {dropped})')
    if input('\nType y to upload, anything else to stop: ').strip().lower() != 'y':
        raise SystemExit('stopped, nothing was written')

    pids = []
    for name in names:
        p = http('POST', f'{api}/teams/{team["id"]}/players', {'name': name}, token=token)
        pids.append(p['id'])
    m = http('POST', f'{api}/matches', {'team_id': team['id'], 'opponent_name': a.opponent,
                                         'court_length_m': 40, 'court_width_m': 20}, token=token)
    http('PUT', f'{api}/matches/{m["id"]}/players',
         {'players': [{'player_id': pid, 'played_minutes': a.minutes} for pid in pids]}, token=token)
    real_map = dict(zip(ids, pids))
    pts, _ = samples(tracks, real_map, a.hz, m['court_length_m'], m['court_width_m'])
    for k in range(0, len(pts), 20000):
        batch = [{'camera_id': 'A', 'player_id': pid, 'timestamp_ms': t, 'x_m': x, 'y_m': y, 'confidence': 1.0}
                 for pid, t, x, y in pts[k:k + 20000]]
        print('  sent', http('POST', f'{api}/matches/{m["id"]}/positions', {'positions': batch}, token=token))
    s = http('GET', f'{api}/matches/{m["id"]}/summary', token=token)
    print(f'\ndone: match id {m["id"]}, samples {s["position_samples"]}, analysis_ready {s["analysis_ready"]}')
    for row in s['players']:
        print(f"  {row['name']}: {row['distance_m']} m")


if __name__ == '__main__':
    main()
