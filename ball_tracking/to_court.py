import csv, os, sys
import numpy as np

# 사용법: python to_court.py 카메라번호(1/2) 추적csv 출력csv
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")  # 이 저장소의 calibration/ 폴더
sys.path.insert(0, CALIB)
from court_map import load, pixel_to_court

cam_n, inp, outp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
cal = load(cam_n, out_dir=os.path.join(CALIB, "out"))

rows = list(csv.DictReader(open(inp)))
idx = [i for i, r in enumerate(rows) if r["status"] != "missing"]
if not idx:
    sys.exit("추적된 공이 없어요")
px = [[(float(rows[i]["x1"]) + float(rows[i]["x2"])) / 2,
       float(rows[i]["y2"])] for i in idx]
ts = [float(rows[i]["t"]) for i in idx]
try:
    xy = pixel_to_court(cal, px, t=ts)   # 시간에 따라 보정이 바뀌는 새 버전 court_map
except TypeError:
    xy = pixel_to_court(cal, px)         # 시간 인자가 없는 옛 버전 court_map

res = {i: xy[k] for k, i in enumerate(idx)}
with open(outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "status", "track_id", "u", "v", "conf", "x_m", "y_m"])
    for i, r in enumerate(rows):
        if i in res:
            k = idx.index(i)
            w.writerow([r["t"], r["status"], r["track_id"], round(px[k][0], 1), round(px[k][1], 1), r["conf"],
                        round(float(res[i][0]), 2), round(float(res[i][1]), 2)])
        else:
            w.writerow([r["t"], "missing", "", "", "", "", "", ""])

nan = int(np.isnan(xy[:, 0]).sum())
ok = ~np.isnan(xy[:, 0])
out_court = int(((np.abs(xy[ok, 0]) > 21) | (np.abs(xy[ok, 1]) > 11)).sum())
print(f"변환한 점={len(idx)}  NaN={nan}  코트 밖(여유 1m)={out_court}")
if ok.any():
    print(f"x 범위 {xy[ok,0].min():.1f} ~ {xy[ok,0].max():.1f} m / y 범위 {xy[ok,1].min():.1f} ~ {xy[ok,1].max():.1f} m")

speeds = []
for a, b in zip(idx[:-1], idx[1:]):
    if rows[a]["track_id"] == rows[b]["track_id"] and not (np.isnan(res[a][0]) or np.isnan(res[b][0])):
        dt = float(rows[b]["t"]) - float(rows[a]["t"])
        if 0 < dt < 0.2:
            speeds.append(float(np.hypot(*(res[b] - res[a]))) / dt)
if speeds:
    s = np.array(speeds)
    print(f"프레임 간 속도: 중앙값 {np.median(s):.1f} m/s, 최대 {s.max():.1f} m/s, 30 m/s 초과 {int((s > 30).sum())}개")

