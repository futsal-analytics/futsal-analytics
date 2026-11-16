import csv, sys
import numpy as np

inp, outp = sys.argv[1], sys.argv[2]

# ---- 설정 (여기 숫자를 바꿔 가며 실험) ----
INIT_CONF = 0.5     # 새 궤적을 시작할 최소 신뢰도
MIN_CONF = 0.25      # 궤적에 붙일 후보의 최소 신뢰도
CONFIRM_N = 3       # 연속 몇 프레임 이어져야 확정 궤적인가
GATE_BASE = 60      # 예측 위치에서 허용하는 거리(px)
GATE_GROW = 20      # 놓친 프레임마다 허용 거리를 넓히는 양(px)
REACQ_CONF = 0.5
GATE_MAX = 120
MAX_COAST = 15      # 예측만으로 유지하는 최대 프레임 수 (30fps 기준 0.5초)

by_t, times = {}, []
for r in csv.DictReader(open(inp)):
    t = float(r["t"])
    if t not in by_t:
        by_t[t] = []
        times.append(t)
    if r["conf"] != "":
        x1, y1, x2, y2, c = (float(r[k]) for k in ("x1", "y1", "x2", "y2", "conf"))
        if c >= MIN_CONF:
            by_t[t].append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, c))
times.sort()

class KF:  # 상태: [x, y, vx, vy]
    def __init__(self, cx, cy):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([9.0, 9.0, 40000.0, 40000.0])
    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        Q = np.diag([4.0, 4.0, 90000.0 * dt, 90000.0 * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
    def update(self, cx, cy):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        S = H @ self.P @ H.T + np.diag([4.0, 4.0])
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([cx, cy]) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

rec_by_t = {}
kf = None
hits = misses = seg = 0
confirmed = False
pending = []
last_t = None
last_wh = (0.0, 0.0)

for t in times:
    cands = by_t[t]
    out = None
    if kf is not None:
        kf.predict(t - last_t)
        gate = min(GATE_BASE + GATE_GROW * misses, GATE_MAX)
        best, best_cost = None, 1e9
        for c in cands:
            d = ((c[0] - kf.x[0]) ** 2 + (c[1] - kf.x[1]) ** 2) ** 0.5
            if d <= gate and (misses == 0 or c[4] >= REACQ_CONF):
                cost = d / gate - c[4]
                if cost < best_cost:
                    best, best_cost = c, cost
        if best is not None:
            kf.update(best[0], best[1])
            hits += 1
            misses = 0
            last_wh = (best[2], best[3])
            out = (t, "detected", best[0], best[1], best[2], best[3], best[4], seg)
        else:
            misses += 1
            if (not confirmed) or misses > MAX_COAST:
                kf, confirmed, pending = None, False, []
            else:
                out = (t, "predicted", kf.x[0], kf.x[1], last_wh[0], last_wh[1], "", seg)
    if kf is None and out is None:
        starters = [c for c in cands if c[4] >= INIT_CONF]
        if starters:
            c = max(starters, key=lambda c: c[4])
            seg += 1
            kf = KF(c[0], c[1])
            hits, misses, confirmed = 1, 0, False
            last_wh = (c[2], c[3])
            out = (t, "detected", c[0], c[1], c[2], c[3], c[4], seg)
    last_t = t
    if out is not None:
        if confirmed:
            rec_by_t[t] = out
        else:
            pending.append(out)
            if hits >= CONFIRM_N:
                confirmed = True
                for p in pending:
                    rec_by_t[p[0]] = p
                pending = []

# 궤적 끝에 남은 "예측만" 구간은 검증이 안 되므로 missing 처리
seg_times = {}
for t in sorted(rec_by_t):
    seg_times.setdefault(rec_by_t[t][7], []).append(t)
for s, ts_ in seg_times.items():
    for t in reversed(ts_):
        if rec_by_t[t][1] == "predicted":
            del rec_by_t[t]
        else:
            break

n_cand_frames = sum(1 for t in times if by_t[t])
with open(outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "status", "x1", "y1", "x2", "y2", "conf", "track_id"])
    det = pred = miss = run = longest = 0
    for t in times:
        r = rec_by_t.get(t)
        if r is None:
            w.writerow([t, "missing", "", "", "", "", "", ""])
            miss += 1
            run += 1
            longest = max(longest, run)
        else:
            _, st, cx, cy, bw, bh, cf, sid = r
            w.writerow([t, st, round(cx - bw / 2, 1), round(cy - bh / 2, 1),
                        round(cx + bw / 2, 1), round(cy + bh / 2, 1), cf, sid])
            det += st == "detected"
            pred += st == "predicted"
            run = 0
dt = (times[-1] - times[0]) / max(1, len(times) - 1)
tracks = len({r[7] for r in rec_by_t.values()})
print(f"frames={len(times)}  detected={det}  predicted={pred}  missing={miss}")
print(f"raw frames with candidates={n_cand_frames}  tracks={tracks}")
print(f"longest missing run={longest} frames ({longest * dt:.2f} s)")

