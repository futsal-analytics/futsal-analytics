import csv, sys, bisect, cv2, numpy as np

# 사용법: python render_dual.py cam1영상 cam1추적csv cam2영상 cam2추적csv 출력.mp4 오프셋(초) [시작(초) 끝(초)]
c1_clip, c1_csv, c2_clip, c2_csv, out_path = sys.argv[1:6]
offset = float(sys.argv[6])
start = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0   # 그리기 시작할 시간(입력 영상 기준 초). 안 적으면 처음부터
end = float(sys.argv[8]) if len(sys.argv) > 8 else 1e9      # 그리기 끝낼 시간(초). 안 적으면 끝까지
W, H = 960, 540                      # 카메라 한 개를 줄여서 넣을 크기 (원본의 절반)
COLOR = {"detected": (0, 255, 0), "predicted": (0, 165, 255)}   # 초록 / 주황 (BGR 순서)

def load_track(path):
    rows = list(csv.DictReader(open(path)))
    return rows, [float(r["t"]) for r in rows]

def nearest(rows, ts, t):
    j = bisect.bisect_left(ts, t)
    ks = [k for k in (j - 1, j) if 0 <= k < len(ts)]
    k = min(ks, key=lambda k: abs(ts[k] - t))
    return rows[k] if abs(ts[k] - t) <= 0.04 else None

def draw(fr, r, cam):
    status = r["status"] if r else "missing"
    if status in COLOR:
        x1, y1, x2, y2 = (int(float(r[c])) for c in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(fr, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), COLOR[status], 3)
    fr = cv2.resize(fr, (W, H))
    label = {"detected": "DETECT", "predicted": "PREDICT", "missing": "MISSING"}[status]
    cv2.putText(fr, f"cam{cam}  {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                COLOR.get(status, (0, 0, 255)), 3)
    return fr

tr1, tr2 = load_track(c1_csv), load_track(c2_csv)
cap1, cap2 = cv2.VideoCapture(c1_clip), cv2.VideoCapture(c2_clip)
fps1, fps2 = cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS)
vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps2, (W * 2, H + 60))

j = round(start * fps1)              # 지금 들고 있는 cam1 프레임 번호 (시작 시각으로 이동)
cap1.set(cv2.CAP_PROP_POS_FRAMES, j)
ok1, fr1 = cap1.read()
i = round(start * fps2)              # cam2 프레임 번호 (시작 시각으로 이동)
cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
n_out = 0                            # 출력한 프레임 수
while True:
    ok2, fr2 = cap2.read()
    if not ok2:
        break
    t = i / fps2                     # 입력 영상 기준 시간(초)
    if t > end:                      # 끝 시간을 넘으면 그만 그림
        break
    while ok1 and (j + 1) / fps1 <= t + 0.5 / fps1:   # cam1을 같은 시각까지 앞으로 감기
        ok1, nxt = cap1.read()
        if ok1:
            fr1, j = nxt, j + 1
    left = draw(fr1.copy(), nearest(*tr1, offset + t), 1)
    right = draw(fr2, nearest(*tr2, offset + t), 2)
    bar = np.zeros((60, W * 2, 3), np.uint8)
    tt = offset + t                  # 원본 영상 기준 시간 (분:초 표시용)
    cv2.putText(bar, f"t = {t:06.3f} s  ({int(tt // 60):02d}:{tt % 60:06.3f})   (frame {i})", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    vw.write(np.vstack([bar, np.hstack([left, right])]))
    i += 1
    n_out += 1
cap1.release(); cap2.release(); vw.release()
print("saved:", out_path, f"frames={n_out}")
